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
import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ..config import HyperSkillConfig
from ..datasets.loaders import DataBatch
from ..prompts import build_skill_augmented_prompt, AFTER_SEARCH_PROMPT, build_initial_prompt
from ..hypergraph.skill_updater import SkillUpdater
from ..skill.library import SkillLibrary
from ..skill.models import SkillNode
from .rewards import (
    compute_combined_reward,
    compute_grpo_advantages,
    compute_grpo_advantages_global,
)


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


class HyperSkillGRPOTrainer:
    """
    Training-Free GRPO Trainer with omni-modal weighted hypergraph skill library.

    The trainer maintains:
      - A SkillLibrary (the evolving policy)
      - Statistics for global advantage normalization
      - Optional wandb logging
    """

    def __init__(self, cfg: HyperSkillConfig):
        self.cfg = cfg

        # Initialize skill library + CRUD updater
        self.skill_library = SkillLibrary(cfg.hypergraph, cfg.llm)
        self.skill_updater = SkillUpdater(self.skill_library.builder, cfg.llm)

        # Global running stats for advantage normalization
        self._global_reward_sum = 0.0
        self._global_reward_sq_sum = 0.0
        self._global_reward_count = 0

        # Wandb
        self._wandb = None
        if cfg.use_wandb:
            try:
                import wandb
                self._wandb = wandb
                wandb.init(
                    project=cfg.wandb_project,
                    name=cfg.exp_name,
                    config=_cfg_to_dict(cfg),
                )
            except ImportError:
                print("[Warning] wandb not installed; logging disabled.")

        self._llm_cfg = cfg.llm

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    async def train(self, dataloader, batch_callback=None) -> Dict[str, float]:
        """
        Run the full training-free GRPO loop over a dataloader.

        Args:
            batch_callback: optional callable(batch_idx, total_batches, metrics, elapsed)
                            called after each batch for real-time monitoring.

        Returns a dict of aggregate metrics.
        """
        total_batches = len(dataloader)
        total_f1 = 0.0
        total_format = 0.0
        batch_count = 0

        max_retries = 3
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            t0 = time.time()
            metrics = None
            for attempt in range(max_retries):
                try:
                    metrics = await self.train_batch(batch, batch_idx)
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
            total_format += metrics.get("avg_format", 0.0)
            batch_count += 1

            print(
                f"[Batch {batch_idx+1}/{total_batches}] "
                f"F1={metrics['avg_f1']:.4f}  Format={metrics['avg_format']:.4f}  "
                f"SkillInserted={metrics.get('skills_inserted', 0)}  "
                f"Elapsed={elapsed:.1f}s"
            )

            if batch_callback is not None:
                batch_callback(batch_idx, total_batches, metrics, elapsed)

            if self._wandb:
                self._wandb.log({
                    "batch": batch_idx,
                    "avg_f1": metrics["avg_f1"],
                    "avg_format": metrics["avg_format"],
                    "skills_inserted": metrics.get("skills_inserted", 0),
                    **self.skill_library.stats(),
                })

        # Save final library
        self.skill_library.save()

        return {
            "avg_f1": total_f1 / max(1, batch_count),
            "avg_format": total_format / max(1, batch_count),
            **self.skill_library.stats(),
        }

    async def train_batch(
        self, batch: List[DataBatch], batch_idx: int
    ) -> Dict[str, float]:
        """
        Process one batch:
          1. Retrieve skills for each question
          2. Generate N rollouts per question
          3. Compute rewards and advantages
          4. Evolve skill library
        """
        n = self.cfg.rollout.n

        all_rewards: List[float] = []
        all_f1: List[float] = []
        all_format: List[float] = []
        total_inserted = 0

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
            )
        except Exception as exc:
            logger.warning("Skill retrieval failed: %s — using empty skills", exc)
            skills_per_query = [[] for _ in queries]

        # -- Step 2: Build augmented prompts and generate rollouts -------
        from ..agents.tools import build_tools, format_available_tools
        rollout_prompts = []
        for item, skills in zip(expanded, skills_per_query):
            # Determine available tools for this item and format for prompt
            tools = build_tools(
                task_type=item.task_type,
                dataset_name=item.source,
                corpus_dir=item.dataset_dir,
                llm_cfg=self._llm_cfg,
            )
            tool_desc = format_available_tools(tools)
            prompt = build_skill_augmented_prompt(
                question=item.question,
                modality=self.cfg.data.modality,
                task_type=item.task_type,
                skills=skills,
                tool_desc=tool_desc,
            )
            rollout_prompts.append(prompt)

        rollouts = await self._generate_rollouts(expanded, rollout_prompts)

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
                response_list = roll.get("response_list", [full_response])
                task_type = group_items[0].task_type if group_items else "mmreasoning"

                reward = compute_combined_reward(
                    prediction=full_response,
                    gold_answers=gold_answers,
                    response_list=response_list,
                    task_type=task_type,
                )
                group_rewards.append(reward)
                group_trajs.append({"text": full_response, "reward": reward})
                all_f1.append(reward)

            all_rewards.extend(group_rewards)

            # GRPO advantages (kept for logging / future use)
            local_advantages = compute_grpo_advantages(group_rewards)
            self._update_global_stats(group_rewards)

            # Collect group for CRUD-based skill evolution
            updater_groups.append({
                "question": group_items[0].question if group_items else "",
                "gold_answer": ", ".join(gold_answers) if gold_answers else "",
                "trajectories": group_trajs,
            })

        # -- Step 4: CRUD-based skill evolution (TrainingFreeGRPO style) ----
        try:
            counts = await self.skill_updater.update(
                updater_groups, concurrency=4, num_skills=3,
            )
            total_inserted = counts.get("added", 0) + counts.get("updated", 0)
        except Exception as exc:
            logger.warning("SkillUpdater failed: %s", exc)
            total_inserted = 0

        # -- Step 5: Retrieval feedback → advantage update + co-occurrence edges
        self._retrieval_advantage_feedback(
            skills_per_query, all_rewards, n
        )

        # Format score: check all rollouts
        for roll in rollouts:
            response_list = roll.get("response_list", [roll.get("full_response", "")])
            task_type = roll.get("task_type", "mmreasoning")
            from .rewards import compute_format_reward
            all_format.append(compute_format_reward(response_list, task_type))

        return {
            "avg_f1": float(np.mean(all_f1)) if all_f1 else 0.0,
            "avg_format": float(np.mean(all_format)) if all_format else 0.0,
            "avg_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "skills_inserted": total_inserted,
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
        if builder is None:
            return

        num_questions = len(all_rewards) // n if n > 0 else 0
        if num_questions == 0:
            return

        # Batch-level stats for normalization
        batch_mean = sum(all_rewards) / len(all_rewards)
        batch_var = sum((r - batch_mean) ** 2 for r in all_rewards) / len(all_rewards)
        batch_std = max(batch_var ** 0.5, 1e-8)

        for q_idx in range(num_questions):
            idx = q_idx * n
            retrieved = skills_per_query[idx] if idx < len(skills_per_query) else []
            if not retrieved:
                continue

            # Average reward for this question's rollouts
            q_rewards = all_rewards[idx: idx + n]
            avg_reward = sum(q_rewards) / len(q_rewards)

            # Normalize to advantage signal ∈ [-1, 1]
            adv_signal = max(-1.0, min(1.0, (avg_reward - batch_mean) / batch_std))

            # EMA update for each retrieved skill
            skill_ids = []
            for skill in retrieved:
                sid = getattr(skill, "skill_id", None)
                if sid and builder.graph.has_node(sid):
                    builder._apply_advantage_update(sid, adv_signal)
                    skill_ids.append(sid)

            # Co-occurrence hyperedge: ≥2 skills + high reward
            if len(skill_ids) >= 2 and avg_reward >= 0.5:
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
                    builder.kv_hyperedges.set(edge_id, {
                        "edge_id": edge_id,
                        "description": f"Co-occurrence: {desc}",
                        "node_ids": skill_ids[:5],
                        "advantage": max(0.0, adv_signal),
                        "edge_type": "cooccurrence",
                    })
                    builder.graph.upsert_node(
                        edge_id, node_type="hyperedge",
                        description=f"Co-occurrence: {desc}",
                        advantage=max(0.0, adv_signal),
                    )
                    for sid in skill_ids[:5]:
                        builder.graph.upsert_edge(edge_id, sid, weight=0.1)

    # ------------------------------------------------------------------
    # Rollout generation
    # ------------------------------------------------------------------

    async def _generate_rollouts(
        self,
        items: List[DataBatch],
        prompts: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Generate agentic rollouts for all items concurrently.
        Handles multi-turn search loops (websearch tasks).
        Failed rollouts return an empty-response placeholder.
        """
        max_concurrent = min(len(items), 32)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _generate_one(item: DataBatch, prompt: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self._single_rollout(item, prompt)
                except Exception as exc:
                    logger.warning("Rollout failed: %s", exc)
                    return {
                        "full_response": "",
                        "response_list": [],
                        "task_type": item.task_type,
                    }

        results = await asyncio.gather(
            *[_generate_one(item, prompt) for item, prompt in zip(items, prompts)]
        )
        return list(results)

    async def _single_rollout(
        self,
        item: DataBatch,
        initial_prompt: str,
    ) -> Dict[str, Any]:
        """
        Execute one agentic rollout with unified multi-turn tool dispatch.

        Tools are determined by dataset type (see agents/tools.py).
        Skills are re-injected at each search turn for context-adaptive augmentation.
        """
        from ..llm import async_llm_call, async_llm_call_with_images
        from ..agents.tools import build_tools, format_available_tools
        from ..prompts import format_skills_context, AFTER_SEARCH_PROMPT
        import base64

        prompt = initial_prompt
        response_list: List[str] = []
        image_b64_list = [
            base64.b64encode(img).decode("utf-8") for img in (item.images or [])
        ]

        # For Omni: extract video keyframes as images
        if item.video_path and not image_b64_list:
            keyframes = _extract_keyframes(item.video_path, max_frames=3)
            image_b64_list.extend(keyframes)

        # Build tools for this item's dataset type
        tools = build_tools(
            task_type=item.task_type,
            dataset_name=item.source,
            corpus_dir=item.dataset_dir,
            llm_cfg=self._llm_cfg,
        )

        max_rounds = self.cfg.rollout.max_gen_round
        max_searches = self.cfg.rollout.text_search_limit  # total search cap
        search_count = 0
        seen_queries: set = set()

        for round_idx in range(max_rounds):
            # Generate next response
            if image_b64_list:
                response = await async_llm_call_with_images(
                    prompt=prompt,
                    image_b64_list=image_b64_list,
                    cfg=self._llm_cfg,
                    max_tokens=self.cfg.rollout.max_new_tokens,
                )
            else:
                response = await async_llm_call(
                    prompt=prompt,
                    cfg=self._llm_cfg,
                    use_cache=False,
                    max_tokens=self.cfg.rollout.max_new_tokens,
                )

            response_list.append(response)
            prompt += response

            # Check termination
            if len(prompt.split()) >= self.cfg.rollout.max_prompt_length:
                break
            if "<answer>" in response and "</answer>" in response:
                break

            # Try each registered tool's tag
            action_taken = False
            if search_count >= max_searches:
                break  # hit search limit → stop
            for tag, tool in tools.items():
                pattern = rf"<{tag}>(.*?)</{tag}>"
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    query = match.group(1).strip()
                    if query in seen_queries:
                        break  # repeated query → stop
                    seen_queries.add(query)
                    search_count += 1

                    # Execute the tool
                    search_result = await tool.search(query=query)

                    # Re-retrieve skills with updated context (question + search result)
                    try:
                        updated_skills = await self.skill_library.retrieve(
                            query=item.question + " " + query,
                            top_k=self.cfg.hypergraph.top_k,
                        )
                        skill_ctx = format_skills_context(updated_skills)
                    except Exception:
                        skill_ctx = ""

                    # Build continuation prompt with search results + refreshed skills
                    after = AFTER_SEARCH_PROMPT.format(
                        question=item.question,
                        search_results=f"[{tool.name} Results]\n{search_result}",
                    )
                    if skill_ctx and skill_ctx != "(No relevant skills found)":
                        after += f"\n\n--- Updated Skills ---\n{skill_ctx}\n--- End ---\n"
                    prompt += after
                    action_taken = True
                    break

            if not action_taken:
                break  # no tool action → stop

        full_response = "\n".join(response_list)
        return {
            "full_response": full_response,
            "response_list": response_list,
            "task_type": item.task_type,
            "extra_media": None,
        }

    # ------------------------------------------------------------------
    # Running statistics for global advantage normalization
    # ------------------------------------------------------------------

    def _update_global_stats(self, rewards: List[float]) -> None:
        for r in rewards:
            self._global_reward_sum += r
            self._global_reward_sq_sum += r * r
            self._global_reward_count += 1

    def _get_global_stats(self) -> Tuple[float, float]:
        if self._global_reward_count == 0:
            return 0.0, 1.0
        mean = self._global_reward_sum / self._global_reward_count
        var = (self._global_reward_sq_sum / self._global_reward_count) - mean ** 2
        std = max(1e-8, var ** 0.5)
        return mean, std

    # ------------------------------------------------------------------
    # Evaluation mode
    # ------------------------------------------------------------------

    async def evaluate(self, dataloader) -> Dict[str, float]:
        """Run evaluation without skill library updates."""
        all_f1: List[float] = []
        all_format: List[float] = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            queries = [item.question for item in batch]
            skills_per_query = await self.skill_library.retrieve_for_batch(
                queries=queries, top_k=self.cfg.hypergraph.top_k
            )

            for item, skills in zip(batch, skills_per_query):
                prompt = build_skill_augmented_prompt(
                    question=item.question,
                    modality=self.cfg.data.modality,
                    task_type=item.task_type,
                    skills=skills,
                )
                rollout = await self._single_rollout(item, prompt)
                f1 = compute_combined_reward(
                    prediction=rollout["full_response"],
                    gold_answers=item.gold_answers,
                    response_list=rollout["response_list"],
                    task_type=item.task_type,
                )
                all_f1.append(f1)

        return {
            "f1": float(np.mean(all_f1)) if all_f1 else 0.0,
            "n_samples": len(all_f1),
            **self.skill_library.stats(),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg_to_dict(cfg) -> Dict[str, Any]:
    """Recursively convert a config dataclass to a dict."""
    import dataclasses
    if dataclasses.is_dataclass(cfg):
        return {k: _cfg_to_dict(v) for k, v in dataclasses.asdict(cfg).items()}
    return cfg
