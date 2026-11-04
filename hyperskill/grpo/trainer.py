"""
HyperSkillGRPOTrainer: Training-Free Group Relative Policy Optimization
using the omni-modal weighted hypergraph skill library.

Instead of gradient updates, each GRPO iteration:
  1. Retrieves relevant skills for each question (in-context policy)
  2. Generates N rollouts with skill-augmented prompts
  3. Computes rewards and GRPO advantages
  4. Evolves the skill library from high-advantage trajectories

No model parameters are updated — the skill library IS the evolving policy.
"""

from __future__ import annotations
import asyncio
import copy
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from ..config import HyperSkillConfig
from ..datasets.loaders import DataBatch
from ..prompts import build_skill_augmented_prompt
from ..hypergraph.skill_updater import SkillUpdater
from ..skill.library import SkillLibrary
from .rewards import compute_f1_reward


def _extract_keyframes(video_path: str, max_frames: int = 3) -> List[str]:
    """Extract keyframes from video as base64 JPEG strings. Returns [] if OpenCV unavailable."""
    import base64
    try:
        import cv2
    except ImportError:
        return []
    try:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []
        indices = [int(i * total / max_frames) for i in range(max_frames)]
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                frames.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
        cap.release()
        return frames
    except Exception:
        return []


_KEYFRAME_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "keyframes"


def install_keyframe_cache() -> None:
    """Memoise _extract_keyframes on disk (cache/keyframes): a pure function of the
    video file and max_frames, so the frames are identical; they are extracted once
    per video instead of once per rollout."""
    global _extract_keyframes
    if getattr(_extract_keyframes, "_cached", False):
        return
    orig = _extract_keyframes

    def cached(video_path: str, max_frames: int = 3):
        try:
            st = os.stat(video_path)
            k = hashlib.sha256(f"{os.path.realpath(video_path)}|{st.st_size}|{st.st_mtime_ns}|{max_frames}".encode()).hexdigest()
        except OSError:
            return orig(video_path, max_frames)
        f = _KEYFRAME_CACHE_DIR / k[:2] / f"{k}.json"
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:
                pass
        frames = orig(video_path, max_frames)
        if frames:
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                tmp = f.with_name(f"{f.name}.{os.getpid()}.tmp")
                tmp.write_text(json.dumps(frames)); os.replace(tmp, f)
            except OSError:
                pass
        return frames
    cached._cached = True
    _extract_keyframes = cached


def _extract_audio_from_video(video_path: str) -> Optional[bytes]:
    """Extract the audio track of a video file as WAV bytes via ffmpeg.

    Used only for WorldSense (resource_type "video_audio") as a fallback when
    the dataset's .wav is missing or empty.
    Returns None if ffmpeg is unavailable or the file has no audio track.
    """
    import subprocess, tempfile, os
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", out_path,
        ]
        subprocess.run(cmd, check=True, timeout=30, capture_output=True)
        with open(out_path, "rb") as f:
            data = f.read()
        os.unlink(out_path)
        # Cap at 10 MB (Gemini input_audio limit ~varies; be conservative)
        return data[:10 * 1024 * 1024] if data else None
    except Exception:
        return None


class HyperSkillGRPOTrainer:
    """
    Training-Free GRPO Trainer with omni-modal weighted hypergraph skill library.

    The trainer maintains:
      - A SkillLibrary (the evolving policy)
    """

    def __init__(self, cfg: HyperSkillConfig):
        self.cfg = cfg

        # Initialize skill library + CRUD updater
        self.skill_library = SkillLibrary(cfg.hypergraph, cfg.llm)
        self.skill_updater = SkillUpdater(self.skill_library.builder, cfg.llm)

        self._llm_cfg = cfg.llm

        # Protocol skill (strategies/protocol.py): failure-type counts of the
        # training rollouts; at evaluation they are loaded from the library.
        from ..strategies.protocol import ProtocolStats
        hcfg = cfg.hypergraph
        self.protocol = ProtocolStats(hcfg.protocol_min_count)
        if cfg.mode == "test":
            loaded = ProtocolStats.load(hcfg.working_dir, hcfg.protocol_min_count)
            if loaded is not None:
                self.protocol = loaded
            elif hcfg.protocol_skill:
                raise RuntimeError(f"protocol_skill is on but {hcfg.working_dir} has no protocol_skill.json")

    def _protocol_ctx(self, item) -> str:
        """Protocol-skill text for `item` ("" when disabled or nothing learned yet)."""
        if not self.cfg.hypergraph.protocol_skill:
            return ""
        from ..agents.tools import TOOL_PROFILES
        from ..strategies.protocol import protocol_text, answer_form
        hcfg = self.cfg.hypergraph
        if hcfg.protocol_answer_form not in ("short", "data"):
            raise ValueError(f"protocol_answer_form must be short|data, not {hcfg.protocol_answer_form}")
        if hcfg.protocol_answer_form == "data" and getattr(self, "_gold_form", None) is None:
            from ..datasets.loaders import HyperSkillDataset
            ds = self.cfg.data.dataset_names[0]
            self._gold_form = answer_form([it.gold_answers[0] if it.gold_answers else ""
                                           for it in HyperSkillDataset(ds, self.cfg.data.train_split)])
        form = self._gold_form if hcfg.protocol_answer_form == "data" else None
        tags = list(TOOL_PROFILES.get(item.task_type, []))
        return protocol_text(self.protocol.active, item.task_type, tags,
                             self.cfg.rollout.text_search_limit, self.cfg.rollout.max_new_tokens, form)

    def _is_mc(self) -> bool:
        """>= 50% of the training golds are option letters."""
        if getattr(self, "_gold_form", None) is None:
            from ..datasets.loaders import HyperSkillDataset
            from ..strategies.protocol import answer_form
            ds = self.cfg.data.dataset_names[0]
            self._gold_form = answer_form([it.gold_answers[0] if it.gold_answers else ""
                                           for it in HyperSkillDataset(ds, self.cfg.data.train_split)])
        return (self._gold_form or {}).get("mc", 0.0) >= 0.5

    def _build_prompt(self, item, skills, tool_desc: str, skills_context) -> str:
        """Initial prompt (strategies/prompting.py); with every option off this is
        build_skill_augmented_prompt unchanged. Lines after the question: answer_suffix,
        and at evaluation the locate_pages block and the time_grounding / option_check lines."""
        from ..strategies.prompting import build_prompt, answer_suffix, join_suffix
        hcfg = self.cfg.hypergraph
        suffix = ""
        if hcfg.answer_suffix:
            suffix = answer_suffix(item.task_type, item.task_type != "math" and self._is_mc())
        if self.cfg.mode == "test":
            from ..strategies import locate_pages, video
            lp = locate_pages.lp_suffix(self, item)
            suffix = f"{lp}\n\n{suffix}" if lp and suffix else (lp or suffix)
            suffix = join_suffix(suffix, video.prompt_suffix(self, item))
        return build_prompt(question=item.question, modality=self.cfg.data.modality, skills=skills,
                            tool_desc=tool_desc, skills_context=skills_context,
                            v2=hcfg.concrete_tool_prompt, suffix=suffix)

    async def _skill_context(self, question: str, skills, task_type: str = "") -> Optional[str]:
        """P7 hint synthesis (App. A.3) when enabled and |S_q| > 0; None
        means "use the default terse format" (or no injection if no skills)."""
        if not skills or not self.cfg.hypergraph.hint_synthesis:
            return None
        from ..llm import async_llm_call
        from ..prompts import SKILL_SUMMARY_PROMPT, SKILL_SUMMARY_SCHEMA, SKILL_SUMMARY_SCHEMA_TOOLS
        from ..agents.tools import TOOL_PROFILES
        hcfg = self.cfg.hypergraph
        has_tools = bool(task_type and TOOL_PROFILES.get(task_type))
        raw = "\n".join(f"- {getattr(s, 'name', '')}: {getattr(s, 'description', '')}" for s in skills)
        p7 = SKILL_SUMMARY_PROMPT.format(question=question[:2000], skills_raw=raw)
        tail = "\nStrategy hint (2-3 sentences, actionable):"
        if hcfg.schema_constrained_hint:
            clause = SKILL_SUMMARY_SCHEMA_TOOLS.format(limit=self.cfg.rollout.text_search_limit) if has_tools else ""
            p7 = p7.replace(tail, SKILL_SUMMARY_SCHEMA.format(tool_clause=clause) + tail)
        try:
            hint = await async_llm_call(
                prompt=p7,
                cfg=self._llm_cfg, use_cache=True, max_tokens=200, temperature=0.0)
        except Exception as exc:
            logger.warning("hint synthesis failed: %s", exc)
            return None
        hint = " ".join((hint or "").split())
        return hint or None

    async def _train_split_embeddings(self):
        """(normalised TRAIN question embeddings, gold word-lengths), cached per trainer.
        TRAIN is the split the library was trained on (cfg.data.train_split)."""
        if getattr(self, "_train_emb", None) is None:
            from ..datasets.loaders import HyperSkillDataset
            from ..llm import async_embed_texts
            ds = self.cfg.data.dataset_names[0]
            items = list(HyperSkillDataset(ds, self.cfg.data.train_split))
            Q = np.asarray(await async_embed_texts([it.question for it in items], self._llm_cfg), dtype=np.float32)
            Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8
            lens = [len(str(it.gold_answers[0]).split()) if it.gold_answers else 1 for it in items]
            mc = [bool(re.match(r"^\(?[A-J]\)?[.)]?$", str(it.gold_answers[0]).strip())) if it.gold_answers else False
                  for it in items]
            self._train_emb = (Q, np.asarray(lens, dtype=np.float32))
            self._train_mc = np.asarray(mc, dtype=bool)
        return self._train_emb

    async def _answer_forms(self, k: int = 5) -> None:
        """Answer-form descriptors of every skill from its k nearest training
        questions: median gold length (words) and whether their golds are
        option letters."""
        vdb = self.skill_library.builder.vdb_skills
        self._ans_len, self._ans_mc = {}, {}
        if len(vdb) == 0 or vdb._vecs is None:
            return
        Q, lens = await self._train_split_embeddings()
        S = np.asarray(vdb._vecs, dtype=np.float32)
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
        sim = S @ Q.T                                    # (skills, train)
        top = np.argsort(-sim, axis=1)[:, :k]
        self._ans_mc = {sid: bool(self._train_mc[t].mean() >= 0.5) for sid, t in zip(vdb._ids, top)}
        self._ans_len = {sid: float(np.median(lens[t])) for sid, t in zip(vdb._ids, top)}

    def _form_line(self, skills, task_type: str) -> str:
        """Answer-form line appended to the injected skills (answer_form_line)."""
        from ..agents.tools import TOOL_PROFILES
        hcfg = self.cfg.hypergraph
        budget = (f"Use at most {self.cfg.rollout.text_search_limit} searches; "
                  f"after that, give your best final answer.")
        if not hcfg.answer_form_line:
            return ""
        L = [self._ans_len[s.skill_id] for s in skills if getattr(s, "skill_id", None) in self._ans_len]
        L = float(np.median(L)) if L else 3.0
        mc = [self._ans_mc[s.skill_id] for s in skills if getattr(s, "skill_id", None) in self._ans_mc]
        if task_type == "math":
            line = ("Answer form: reason step by step but concisely so that the whole response fits in about "
                    "1,000 tokens, then give the final integer inside <answer>...</answer>.")
        elif mc and sum(mc) * 2 >= len(mc):
            line = "Answer form: give the option letter of your choice inside <answer>...</answer> (e.g. <answer>C</answer>)."
        elif L <= 3:
            line = (f"Answer form: give only the final answer inside <answer>...</answer> as a short phrase "
                    f"of at most {int(L) + 2} words, not a full sentence.")
        else:
            line = "Answer form: always end with your final answer inside <answer>...</answer>."
        if TOOL_PROFILES.get(task_type):
            line += " " + budget
        return line

    async def _hub_means(self) -> Dict[str, float]:
        """mu_v = mean cosine of every skill node to the dataset's TRAIN questions
        (the split the library was trained on, cfg.data.train_split)."""
        from ..datasets.loaders import HyperSkillDataset
        from ..llm import async_embed_texts
        vdb = self.skill_library.builder.vdb_skills
        if len(vdb) == 0 or vdb._vecs is None:
            return {}
        ds = self.cfg.data.dataset_names[0]
        qs = [it.question for it in HyperSkillDataset(ds, self.cfg.data.train_split)]
        Q = np.asarray(await async_embed_texts(qs, self._llm_cfg), dtype=np.float32)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8
        S = np.asarray(vdb._vecs, dtype=np.float32)
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
        mu = (Q @ S.T).mean(0)
        return {sid: float(m) for sid, m in zip(vdb._ids, mu)}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    async def train(self, dataloader) -> Dict[str, float]:
        """
        Run the full training-free GRPO loop over a dataloader.

        Returns a dict of aggregate metrics.
        """
        total_batches = len(dataloader)
        total_f1 = 0.0
        batch_count = 0

        # ─── Early-stopping bookkeeping ───────────────────────────────────
        es = self.cfg.early_stop
        f1_history: List[float] = []
        best_window_f1 = -1.0
        no_improve_streak = 0
        no_skill_streak = 0
        no_variance_streak = 0
        stop_reason: Optional[str] = None

        max_retries = 3
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            t0 = time.time()
            metrics = None
            for attempt in range(max_retries):
                try:
                    metrics = await self.train_batch(batch)
                    break
                except Exception as exc:
                    wait = 2 ** attempt
                    print(
                        f"  [WARN] Batch {batch_idx+1} attempt {attempt+1}/{max_retries} "
                        f"failed: {exc}. Retrying in {wait}s..."
                    )
                    await asyncio.sleep(wait)
            if metrics is None:
                print(f"  [SKIP] Batch {batch_idx+1}/{total_batches} failed after {max_retries} retries.")
                continue
            elapsed = time.time() - t0

            total_f1 += metrics.get("avg_f1", 0.0)
            batch_count += 1

            metric_label = "Acc" if self.cfg.data.dataset_names and \
                any(n in ("AIME2024", "AIME2025") for n in self.cfg.data.dataset_names) else "F1"
            print(
                f"[Batch {batch_idx+1}/{total_batches}] "
                f"{metric_label}={metrics['avg_f1']:.4f}  "
                f"SkillInserted={metrics.get('skills_inserted', 0)}  "
                f"Elapsed={elapsed:.1f}s"
            )

            # ─── Early stopping ──────────────────────────────────────────
            if es.enabled:
                f1_history.append(metrics["avg_f1"])
                if metrics.get("skills_inserted", 0) > 0:
                    no_skill_streak = 0
                else:
                    no_skill_streak += 1
                # "no variance" means every group in this batch had identical
                # rewards across rollouts — no contrastive signal at all.
                if metrics.get("groups_no_variance", 0) >= len(batch):
                    no_variance_streak += 1
                else:
                    no_variance_streak = 0

                if batch_count >= es.min_batches:
                    # F1 plateau: take the last `f1_window` batches' mean
                    if len(f1_history) >= es.f1_window:
                        window_mean = sum(f1_history[-es.f1_window:]) / es.f1_window
                        if window_mean > best_window_f1 + es.f1_min_delta:
                            best_window_f1 = window_mean
                            no_improve_streak = 0
                        else:
                            no_improve_streak += 1

                    if es.f1_patience and no_improve_streak >= es.f1_patience:
                        stop_reason = (
                            f"F1 plateau ({es.f1_patience} batches without "
                            f">+{es.f1_min_delta} window improvement)"
                        )
                    elif es.skill_patience and no_skill_streak >= es.skill_patience:
                        stop_reason = (
                            f"no skill insertions for {es.skill_patience} consecutive batches"
                        )
                    elif es.novariance_patience and no_variance_streak >= es.novariance_patience:
                        stop_reason = (
                            f"all rollouts produced identical rewards for "
                            f"{es.novariance_patience} consecutive batches"
                        )

                if stop_reason is not None:
                    print(f"[EarlyStop] batch {batch_idx+1}/{total_batches}: {stop_reason}")
                    break

        # Save final library (and the protocol-skill statistics next to it)
        self.skill_library.save()
        self.protocol.save(self.cfg.hypergraph.working_dir)
        print(f"  [protocol] {self.protocol.n_rollouts} rollouts, counts={self.protocol.counts}, "
              f"active={self.protocol.active}", flush=True)

        return {
            "avg_f1": total_f1 / max(1, batch_count),
            "batches_run": batch_count,
            "early_stop": stop_reason or "",
            **self.skill_library.stats(),
        }

    async def train_batch(self, batch: List[DataBatch]) -> Dict[str, float]:
        """
        Process one batch:
          1. Retrieve skills for each question
          2. Generate N rollouts per question
          3. Compute rewards and advantages
          4. Evolve skill library
        """
        n = self.cfg.rollout.n

        all_rewards: List[float] = []

        # For each question, we need N rollouts
        expanded: List[Dict[str, Any]] = []
        for item in batch:
            for _ in range(n):
                expanded.append(copy.deepcopy(item))

        # -- Step 1: Retrieve skills for each query ----------------------
        queries = [item.question for item in expanded]
        image_bytes_list = [
            item.images[0] if item.images else None
            for item in expanded
        ]
        try:
            skills_per_query = await self.skill_library.retrieve_for_batch(
                queries=queries,
                top_k=self.cfg.hypergraph.top_k,
                image_bytes_list=image_bytes_list,
                min_similarity=self.cfg.hypergraph.train_min_similarity,
            )
        except Exception as exc:
            logger.warning("Skill retrieval failed: %s — using empty skills", exc)
            skills_per_query = [[] for _ in queries]

        # -- Step 2: Build augmented prompts and generate rollouts -------
        from ..agents.tools import build_tools, format_available_tools

        # P7: one hint per question (its n rollouts share the retrieved set)
        group_ctx = await asyncio.gather(*[
            self._skill_context(expanded[i].question, skills_per_query[i], expanded[i].task_type)
            for i in range(0, len(expanded), n)])
        rollout_prompts = []
        for idx_e, (item, skills) in enumerate(zip(expanded, skills_per_query)):
            # Determine available tools for this item and format for prompt
            tools = build_tools(
                task_type=item.task_type,
                dataset_name=item.source,
                corpus_dir=item.dataset_dir,
            )
            tool_desc = format_available_tools(tools)
            ctx = group_ctx[idx_e // n]
            pctx = self._protocol_ctx(item)
            if pctx:
                from ..prompts import format_skills_context
                ctx = (ctx or (format_skills_context(skills) if skills else ""))
                ctx = f"{ctx}\n{pctx}" if ctx else pctx
            prompt = self._build_prompt(item, skills, tool_desc, ctx)
            rollout_prompts.append(prompt)

        rollouts = await self._generate_rollouts(
            expanded, rollout_prompts, rollout_ids=[i % n for i in range(len(expanded))])
        for roll in rollouts:
            if "fail" in roll:              # errored rollouts carry no failure type
                self.protocol.add(roll["fail"], roll.get("shape"))

        # -- Step 3: Compute rewards per group ---------------------------
        updater_groups: List[Dict[str, Any]] = []

        for i in range(0, len(rollouts), n):
            group_rollouts = rollouts[i: i + n]
            group_items = expanded[i: i + n]
            gold_answers = group_items[0].gold_answers if group_items else []

            group_rewards = []
            group_trajs = []
            for roll in group_rollouts:
                full_response = roll.get("full_response", "")
                reward = compute_f1_reward(
                    prediction=full_response,
                    gold_answers=gold_answers,
                )
                group_rewards.append(reward)
                group_trajs.append({"text": full_response, "reward": reward})

            all_rewards.extend(group_rewards)

            # Collect group for CRUD-based skill evolution
            updater_groups.append({
                "question": group_items[0].question if group_items else "",
                "gold_answer": ", ".join(gold_answers) if gold_answers else "",
                "trajectories": group_trajs,
            })

        # -- Step 4: counterfactually verified step-level distillation -----
        groups_no_var = 0
        try:
            counts = await self.skill_updater.update(updater_groups, concurrency=4)
            total_inserted = counts.get("added", 0)
            groups_no_var = counts.get("groups_no_variance", 0)
        except Exception as exc:
            import traceback
            print(f"  [SkillUpdater FAIL] {type(exc).__name__}: {exc}")
            traceback.print_exc()
            total_inserted = 0

        # -- Step 5: Retrieval feedback → advantage update + co-occurrence edges
        self._retrieval_advantage_feedback(
            skills_per_query, all_rewards, n
        )

        # -- Step 6: Prune low-advantage and stale skills
        builder = self.skill_library.builder
        builder._update_count += 1
        prune_every = max(1, self.cfg.hypergraph.prune_interval // 50)
        if builder._update_count % prune_every == 0:
            pruned = builder.prune(self.cfg.hypergraph.prune_min_advantage)
            if pruned > 0:
                print(f"  [Prune] removed {pruned} low-advantage skills")

        return {
            "avg_f1": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "skills_inserted": total_inserted,
            "groups_no_variance": groups_no_var,
        }

    # ------------------------------------------------------------------
    # Retrieval feedback: EMA advantage update + co-occurrence hyperedges
    # ------------------------------------------------------------------

    def _retrieval_advantage_feedback(
        self,
        skills_per_query: List[List[Any]],
        all_rewards: List[float],
        n: int,
    ) -> None:
        """
        After rollouts, update advantages of retrieved skills based on reward:
          - High reward → advantage ↑  (skill was helpful)
          - Low reward  → advantage ↓  (skill was useless/harmful)

        Also creates co-occurrence hyperedges when ≥2 skills are retrieved
        together for a high-reward question.
        """
        from ..hypergraph.skill_updater import make_edge_id
        builder = self.skill_library.builder

        num_questions = len(all_rewards) // n if n > 0 else 0
        if num_questions == 0:
            return

        # Batch-level stats for normalization
        batch_mean = sum(all_rewards) / len(all_rewards)
        batch_var = sum((r - batch_mean) ** 2 for r in all_rewards) / len(all_rewards)

        for q_idx in range(num_questions):
            idx = q_idx * n
            retrieved = skills_per_query[idx]
            if not retrieved:
                continue

            # Average reward for this question's rollouts
            q_rewards = all_rewards[idx: idx + n]
            avg_reward = sum(q_rewards) / len(q_rewards)

            # Eq. (4): advantage signal A = tanh((r - mean) / (std + 1e-8)) in [-1, 1]
            adv_signal = float(np.tanh((avg_reward - batch_mean) / (batch_var ** 0.5 + 1e-8)))

            # EMA update for each retrieved skill
            skill_ids = []
            for skill in retrieved:
                sid = getattr(skill, "skill_id", None)
                if sid and builder.graph.has_node(sid):
                    builder._apply_advantage_update(sid, adv_signal)
                    skill_ids.append(sid)

            # Co-occurrence hyperedge: ≥2 skills + high reward
            if len(skill_ids) >= 2 and avg_reward >= self.cfg.hypergraph.rho:
                edge_id = make_edge_id(skill_ids[:5])
                existing = builder.kv_hyperedges.get(edge_id)
                if existing:
                    old_adv = existing.get("advantage", 0.0)
                    new_adv = builder._compute_new_advantage(old_adv, adv_signal)
                    existing["advantage"] = new_adv
                    builder.kv_hyperedges.set(edge_id, existing)
                    builder.graph.upsert_node(edge_id, advantage=new_adv)
                else:
                    desc = " + ".join(getattr(s, "name", "?") for s in retrieved[:5])
                    full_desc = f"Co-occurrence: {desc}"
                    builder.kv_hyperedges.set(edge_id, {
                        "edge_id": edge_id,
                        "description": full_desc,
                        "node_ids": skill_ids[:5],
                        "advantage": max(0.0, adv_signal),
                        "edge_type": "cooccurrence",
                    })
                    builder.graph.upsert_node(
                        edge_id, node_type="hyperedge",
                        description=full_desc,
                        advantage=max(0.0, adv_signal),
                    )
                    for sid in skill_ids[:5]:
                        builder.graph.upsert_edge(edge_id, sid, weight=0.1)
                    # Write vector to VDB so retrieval can find this hyperedge
                    if not builder.vdb_hyperedges.has(edge_id):
                        try:
                            from ..llm import sync_embed_texts
                            vecs = sync_embed_texts([full_desc], self._llm_cfg)
                            builder.vdb_hyperedges.upsert_batch([(edge_id, vecs[0])])
                        except Exception as _exc:
                            logger.warning("Co-occurrence hyperedge VDB embed failed: %s", _exc)

    # ------------------------------------------------------------------
    # Rollout generation
    # ------------------------------------------------------------------

    async def _generate_rollouts(
        self,
        items: List[DataBatch],
        prompts: List[str],
        rollout_ids: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate agentic rollouts for all items concurrently.
        Handles multi-turn search loops (websearch tasks).
        Failed rollouts return an empty-response placeholder.
        `rollout_ids[i]` (j-th rollout of its question) salts the sampling
        seed so that seeded rollouts of the same question stay distinct.
        """
        from ..llm import SEED_SALT
        max_concurrent = min(len(items), 40)
        semaphore = asyncio.Semaphore(max_concurrent)
        rollout_ids = rollout_ids or [0] * len(items)

        async def _generate_one(item: DataBatch, prompt: str, rid: int) -> Dict[str, Any]:
            SEED_SALT.set(f"rollout{rid}")   # task-local (each gather task has its own context)
            async with semaphore:
                try:
                    return await self._single_rollout(item, prompt)
                except Exception as exc:
                    logger.warning("Rollout failed: %s", exc)
                    return {"full_response": "", "response_list": []}

        results = await asyncio.gather(
            *[_generate_one(item, prompt, rid) for item, prompt, rid in zip(items, prompts, rollout_ids)]
        )
        return list(results)

    async def _single_rollout(
        self,
        item: DataBatch,
        initial_prompt: str,
        llm_cfg=None,
    ) -> Dict[str, Any]:
        """
        Execute one agentic rollout with unified multi-turn tool dispatch.
        `llm_cfg` (sampled vote rollouts) replaces self._llm_cfg for the rollout's calls.

        Tools are determined by dataset type (see agents/tools.py).
        Skills are re-injected at each search turn for context-adaptive augmentation.
        """
        from ..llm import async_llm_call, async_llm_call_with_images
        from ..agents.tools import build_tools
        from ..prompts import format_skills_context, AFTER_SEARCH_PROMPT
        import base64

        llm_cfg = llm_cfg or self._llm_cfg
        prompt = initial_prompt
        response_list: List[str] = []
        image_b64_list = [
            base64.b64encode(img).decode("utf-8") for img in (item.images or [])
        ]

        # For Omni: extract video keyframes as images. 8 frames give much
        # better temporal coverage than 3 — most video QA needs more than
        # first/middle/last to localise events.
        if item.video_path and not image_b64_list:
            keyframes = _extract_keyframes(item.video_path, max_frames=self.cfg.rollout.max_frames)
            image_b64_list.extend(keyframes)

        # Audio path: use the dataset's .wav (WorldSense only); fall back to
        # extracting the audio track via ffmpeg if the .wav is missing.
        audio_bytes_to_send = item.audio
        audio_fmt = "wav"
        if audio_bytes_to_send and item.resource_paths:
            for rp in item.resource_paths:
                ext = rp.rsplit(".", 1)[-1].lower() if "." in rp else ""
                if ext in ("wav", "mp3", "flac", "m4a", "ogg"):
                    audio_fmt = ext
                    break
        elif item.video_path and item.resource_type == "video_audio" and not audio_bytes_to_send:
            audio_bytes_to_send = _extract_audio_from_video(item.video_path)
            audio_fmt = "wav"

        # Build tools for this item's dataset type
        tools = build_tools(
            task_type=item.task_type,
            dataset_name=item.source,
            corpus_dir=item.dataset_dir,
        )

        max_rounds = self.cfg.rollout.max_gen_round
        max_searches = self.cfg.rollout.text_search_limit  # total search cap
        search_count = 0
        seen_queries: set = set()
        end_reason = "rounds"
        # protocol skill: re-inject the remaining tool budget after each tool turn
        reinject = bool(self.cfg.hypergraph.protocol_skill and self.protocol.active and tools)

        rollout_system = "You are a helpful assistant."

        # Cache only in eval (temp=0 deterministic → identical calls must
        # return identical responses for reproducibility). In training the N
        # rollouts of a question share one prompt, so caching would collapse
        # them into identical samples and remove the GRPO variance.
        use_cache = (self.cfg.mode == "test")

        finish: List[Dict[str, Any]] = []      # finish_reason etc. per LLM call (logging only)
        for _ in range(max_rounds):
            # Generate next response
            meta: Dict[str, Any] = {}
            if image_b64_list or audio_bytes_to_send:
                response = await async_llm_call_with_images(
                    prompt=prompt,
                    image_b64_list=image_b64_list,
                    cfg=llm_cfg,
                    max_tokens=self.cfg.rollout.max_new_tokens,
                    system=rollout_system,
                    audio_bytes=audio_bytes_to_send,
                    audio_fmt=audio_fmt,
                    use_cache=use_cache,
                    meta=meta,
                )
                # Audio is in conversation history now; don't re-send on
                # subsequent search rounds.
                audio_bytes_to_send = None
            else:
                response = await async_llm_call(
                    prompt=prompt,
                    cfg=llm_cfg,
                    use_cache=use_cache,
                    max_tokens=self.cfg.rollout.max_new_tokens,
                    system=rollout_system,
                    meta=meta,
                )

            finish.append(meta)
            response_list.append(response)
            prompt += response

            # Check termination
            if len(prompt.split()) >= self.cfg.rollout.max_prompt_length:
                end_reason = "length"
                break
            if "<answer>" in response and "</answer>" in response:
                end_reason = "answer"
                break

            # Try each registered tool's tag
            action_taken = False
            if search_count >= max_searches:
                end_reason = "budget"
                break  # hit search limit → stop
            from ..strategies.prompting import parse_tool_call
            call = parse_tool_call(response, list(tools), lenient=self.cfg.hypergraph.tool_parse_lenient)
            for tag, tool in tools.items():
                if call is not None and call[0] == tag:
                    query = call[1]
                    if query in seen_queries:
                        end_reason = "repeat"
                        break  # repeated query → stop
                    seen_queries.add(query)
                    search_count += 1

                    # Execute the tool. For doc_search, restrict ranking to the
                    # PDF this question is about — corpus may have hundreds of
                    # unrelated docs that would otherwise dominate BM25.
                    no_text = None
                    if tag == "doc_search" and item.resource_paths and self.cfg.hypergraph.doc_own_only:
                        from ..strategies.prompting import doc_no_text_message
                        no_text = doc_no_text_message(tool, item.resource_paths)
                    if no_text is not None:
                        search_result = no_text
                    elif tag == "doc_search" and item.resource_paths:
                        search_result = await tool.search(
                            query=query, restrict_to=item.resource_paths,
                        )
                    else:
                        search_result = await tool.search(query=query)

                    # Re-retrieve skills mid-rollout: only during TRAINING, never
                    # at eval. In eval, the initial skill was already selected with
                    # full context; re-injecting a different skill after each search
                    # turn tends to disrupt the model's coherent reasoning arc.
                    skill_ctx = ""
                    if self.cfg.mode != "test":
                        try:
                            updated_skills = await self.skill_library.retrieve(
                                query=item.question + " " + query,
                                top_k=self.cfg.hypergraph.top_k,
                            )
                            if updated_skills:
                                skill_ctx = format_skills_context(updated_skills)
                        except Exception:
                            skill_ctx = ""

                    # Build continuation prompt with search results + refreshed skills
                    after = AFTER_SEARCH_PROMPT.format(
                        question=item.question,
                        search_results=f"[{tool.name} Results]\n{search_result}",
                    )
                    if reinject:
                        from ..strategies.protocol import reinject_text
                        rem = reinject_text(list(tools), search_count, max_searches)
                        skill_ctx = f"{skill_ctx}\n{rem}" if skill_ctx and skill_ctx != "(No relevant skills found)" else rem
                    if skill_ctx and skill_ctx != "(No relevant skills found)":
                        after += f"\n\n--- Updated Skills ---\n{skill_ctx}\n--- End ---\n"
                    prompt += after
                    action_taken = True
                    break

            if not action_taken:
                if end_reason != "repeat":
                    end_reason = "no_action"
                break  # no tool action → stop

        from ..strategies.protocol import classify_rollout, malformed_shape
        # protocol statistics: the rollout as the policy ended it (before any final-answer turn)
        fail = classify_rollout(response_list, list(tools), end_reason)
        shape = malformed_shape(response_list) if fail == "malformed_tool" else None
        final_turn = False
        if self.cfg.hypergraph.final_answer_turn and response_list:
            from ..strategies.prompting import has_closed_answer, final_turn_text
            if not has_closed_answer(response_list[-1]):
                # final_answer_turn: ask once for the final answer (no tool is executed afterwards)
                prompt += final_turn_text(end_reason, item.task_type,
                                          item.task_type != "math" and self._is_mc())
                meta = {}
                if image_b64_list or audio_bytes_to_send:
                    response = await async_llm_call_with_images(
                        prompt=prompt, image_b64_list=image_b64_list, cfg=llm_cfg,
                        max_tokens=self.cfg.rollout.max_new_tokens, system=rollout_system,
                        audio_bytes=audio_bytes_to_send, audio_fmt=audio_fmt,
                        use_cache=use_cache, meta=meta)
                else:
                    response = await async_llm_call(
                        prompt=prompt, cfg=llm_cfg, use_cache=use_cache,
                        max_tokens=self.cfg.rollout.max_new_tokens, system=rollout_system, meta=meta)
                finish.append(meta)
                response_list.append(response)
                final_turn = True
        full_response = "\n".join(response_list)
        out = {"full_response": full_response, "response_list": response_list,
               "end": end_reason, "fail": fail, "finish": finish, "shape": shape}
        if final_turn:
            out["final_turn"] = True
        return out

    async def _eval_rollout(self, item, prompt: str) -> Dict[str, Any]:
        """Evaluation rollout: question_presearch, then self-consistency (vote_k) or one
        _single_rollout; records of the located pages / pre-search / video timing."""
        hcfg = self.cfg.hypergraph
        rec_qp = None
        if hcfg.question_presearch:
            from ..agents.tools import build_tools
            from ..strategies.presearch import presearch
            tools = build_tools(task_type=item.task_type, dataset_name=item.source, corpus_dir=item.dataset_dir)
            prompt, rec_qp = await presearch(tools, item, prompt)
        if hcfg.vote_k > 0:
            import dataclasses
            from ..llm import SEED_SALT
            from ..strategies.voting import vote_rollouts, answer_kind, SEED_SALT as VOTE_SALT
            is_mc = item.task_type != "math" and self._is_mc()
            samp_cfg = dataclasses.replace(self._llm_cfg, temperature=hcfg.vote_temperature)
            salt = VOTE_SALT[bool(hcfg.vote_skip_refusals)]

            async def one(j):
                if j is None:
                    return await self._single_rollout(item, prompt)
                SEED_SALT.set(f"{salt}{j}")        # task-local: each gathered rollout is its own task
                return await self._single_rollout(item, prompt, llm_cfg=samp_cfg)
            out = await vote_rollouts(one, hcfg.vote_k, answer_kind(item.task_type, is_mc), item.question,
                                      skip_refusals=hcfg.vote_skip_refusals)
        else:
            out = await self._single_rollout(item, prompt)
        from ..strategies import locate_pages, video
        lp = locate_pages.lp_record(self, item)
        if lp is not None:
            out = dict(out)
            out["located_pages"] = lp
        if rec_qp is not None:
            out = dict(out)
            out["presearch"] = rec_qp
        if hcfg.time_grounding and item.video_path and not item.images:
            out = dict(out)
            out["video_timing"] = video.video_timing(self, item)
        return out

    # ------------------------------------------------------------------
    # Evaluation mode
    # ------------------------------------------------------------------

    async def evaluate(self, dataloader) -> Dict[str, float]:
        """Run evaluation with gated skill retrieval; the library is not updated."""
        all_f1: List[float] = []
        errors: List[str] = []
        n_injected: List[int] = []
        per_question: List[Dict[str, Any]] = []

        hcfg = self.cfg.hypergraph
        if hcfg.answer_form_line:
            await self._answer_forms()
            print(f"  [answer-form] descriptors for {len(self._ans_len)} skills", flush=True)
        if hcfg.hub_correction:
            self.skill_library.retriever.hub_mu = await self._hub_means()
            print(f"  [hub] corrected similarity for {len(self.skill_library.retriever.hub_mu)} skills", flush=True)

        eval_sem = asyncio.Semaphore(40)

        from ..agents.tools import build_tools, format_available_tools
        from ..prompts import format_skills_context

        async def _eval_one(item, skills):
            async with eval_sem:
                # Tool stack (knowledge-search/BM25/video-frames/audio) for this item;
                # the evolving skill library is consulted on top of it.
                tools = build_tools(
                    task_type=item.task_type,
                    dataset_name=item.source,
                    corpus_dir=item.dataset_dir,
                )
                tool_desc = format_available_tools(tools)
                skills_context = None
                if skills:
                    skills_context = await self._skill_context(item.question, skills, item.task_type)
                    line = self._form_line(skills, item.task_type)
                    if line:
                        skills_context = (skills_context or format_skills_context(skills)) + "\n" + line
                pctx = self._protocol_ctx(item)
                if pctx:
                    skills_context = (skills_context or (format_skills_context(skills) if skills else ""))
                    skills_context = f"{skills_context}\n{pctx}" if skills_context else pctx
                prompt = self._build_prompt(item, skills, tool_desc, skills_context)
                rec = {"idx": item._eval_idx,
                       "question": item.question[:200], "gold": item.gold_answers[:3],
                       "gold_all": list(item.gold_answers),
                       "skills": [getattr(s, "name", "") for s in skills],
                       "context": (skills_context or "")[:400],
                       "context_full": skills_context or ""}
                try:
                    rollout = await self._eval_rollout(item, prompt)
                except Exception as exc:
                    # scored 0.0, counted and logged
                    errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")
                    logger.warning("eval rollout failed: %s", exc)
                    rec.update(f1=0.0, error=str(exc)[:200]); per_question.append(rec)
                    return 0.0
                f1 = compute_f1_reward(
                    prediction=rollout["full_response"],
                    gold_answers=item.gold_answers,
                )
                rec.update(f1=f1, response_tail=rollout["full_response"][-300:],
                           rounds=len(rollout["response_list"]),
                           end=rollout.get("end"), fail=rollout.get("fail"),
                           response_full=rollout["full_response"],
                           response_list=rollout["response_list"],
                           finish=rollout.get("finish"), final_turn=rollout.get("final_turn", False))
                for k in ("vote", "located_pages", "presearch", "video_timing"):
                    if k in rollout:
                        rec[k] = rollout[k]
                per_question.append(rec)
                return f1

        _k = 0
        for batch in tqdm(dataloader, desc="Evaluating"):
            for item in batch:
                item._eval_idx = _k; _k += 1
            skills_per_query = await self.skill_library.retrieve_for_batch(
                queries=[item.question for item in batch],
                top_k=hcfg.test_top_k,
                min_similarity=hcfg.test_min_similarity,
                min_advantage=hcfg.test_min_advantage,
            )
            n_injected.extend(len(sk) for sk in skills_per_query)
            scores = await asyncio.gather(*[
                _eval_one(item, sk) for item, sk in zip(batch, skills_per_query)
            ])
            all_f1.extend(scores)

        # Bootstrap mean ± std over 7 resamples of 85% of the test set.
        # Uses the fixed per-question F1 list so the bootstrap is deterministic
        # given the same scores and seed; no extra LLM calls.
        bootstrap = _bootstrap_mean_std(all_f1, frac=0.85, n_runs=7, seed=42)

        return {
            "f1": float(np.mean(all_f1)) if all_f1 else 0.0,
            "f1_per_question": all_f1,
            "bootstrap_mean": bootstrap["mean"],
            "bootstrap_std":  bootstrap["std"],
            "n_samples": len(all_f1),
            "n_errors": len(errors),
            "errors": errors[:20],
            "skills_injected_per_question": n_injected,
            "frac_zero_skill": (sum(1 for n in n_injected if n == 0) / len(n_injected)) if n_injected else None,
            "protocol_active": self.protocol.active if hcfg.protocol_skill else [],
            "protocol_malformed_shapes": self.protocol.shapes,
            "finish_reason_counts": _finish_counts(per_question),
            "fail_counts": {k: sum(1 for r in per_question if r.get("fail") == k)
                            for k in ("malformed_tool", "over_budget", "tool_without_tools", "no_answer")},
            "per_question": sorted(per_question, key=lambda r: r["idx"]),
            **self.skill_library.stats(),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finish_counts(per_question: List[Dict[str, Any]]) -> Dict[str, int]:
    """finish_reason of the LAST LLM call of each question ("length" = truncated)."""
    out: Dict[str, int] = {}
    for r in per_question:
        fin = r.get("finish") or []
        k = str(fin[-1].get("finish_reason")) if fin else ("error" if r.get("error") else "none")
        out[k] = out.get(k, 0) + 1
    return out


def _bootstrap_mean_std(scores: List[float], frac: float = 0.85,
                        n_runs: int = 7, seed: int = 42) -> Dict[str, float]:
    """7× bootstrap over 85% of the per-question scores → mean ± std.

    Evaluation protocol: 7 random subsamples, each 85% of the test set.
    """
    if not scores:
        return {"mean": 0.0, "std": 0.0}
    rng = np.random.default_rng(seed)
    n_total = len(scores)
    k = max(1, int(round(n_total * frac)))
    arr = np.asarray(scores, dtype=float)
    subsample_means = []
    for _ in range(n_runs):
        idx = rng.choice(n_total, size=k, replace=False)
        subsample_means.append(float(arr[idx].mean()))
    return {
        "mean": float(np.mean(subsample_means)),
        "std":  float(np.std(subsample_means)),
    }
