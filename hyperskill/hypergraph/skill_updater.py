"""
Contrastive Step-Level Skill Distillation with Counterfactual Verification.

Simplified 2-stage pipeline:
  Stage 1  contrastive_extract — Decompose trajectories into steps (regex),
           align good/bad, find divergence points, extract atomic skills
           + skill recipes. (1 LLM call per question)
  Stage 2  counterfactual_verify — For each candidate, test on failed
           trajectories: "would this skill have helped?" (1 LLM per skill)

No LLM-driven CRUD. Skills are directly inserted after verification.
Advantage evolution is handled externally by retrieval feedback (trainer.py).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

from ..config import LLMConfig
from ..llm import async_llm_call
from ..prompts import (
    SKILL_CONTRASTIVE_EXTRACT_SP,
    SKILL_CONTRASTIVE_EXTRACT_UP,
    SKILL_COUNTERFACTUAL_VERIFY_SP,
    SKILL_COUNTERFACTUAL_VERIFY_UP,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _decompose_trajectory(text: str) -> List[str]:
    """Parse a trajectory into reasoning steps (no LLM, regex only)."""
    m = re.search(r"<reason>(.*?)</reason>", text, re.DOTALL | re.IGNORECASE)
    body = m.group(1).strip() if m else text.strip()
    if not body:
        return [text.strip()] if text.strip() else []

    # Numbered steps: "Step 1:", "1.", "1)"
    steps = re.split(r"(?:^|\n)\s*(?:Step\s*\d+[:.)]|\d+[.)]\s)", body)
    steps = [s.strip() for s in steps if s.strip()]
    if len(steps) >= 2:
        return steps

    # Bullet points
    steps = re.split(r"\n\s*[-•*]\s+", body)
    steps = [s.strip() for s in steps if s.strip()]
    if len(steps) >= 2:
        return steps

    # Sentence splitting, group into chunks of 2
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", body)
    if len(sentences) >= 2:
        chunks = []
        for i in range(0, len(sentences), 2):
            chunk = " ".join(sentences[i:i + 2]).strip()
            if chunk:
                chunks.append(chunk)
        if len(chunks) >= 2:
            return chunks

    return [body]


def _parse_atomic_skills(text: str) -> List[Dict[str, str]]:
    """Extract SKILL lines from LLM contrastive output."""
    m = re.search(
        r"<Atomic Skills>\s*(.*?)\s*</Atomic Skills>",
        text, re.DOTALL | re.IGNORECASE,
    )
    body = m.group(1) if m else text
    skills = []
    for line in body.splitlines():
        line = line.strip()
        if line.upper().startswith("SKILL:"):
            parts = line[6:].strip().split("|", 1)
            name = parts[0].strip()
            desc = parts[1].strip() if len(parts) > 1 else name
            if name:
                skills.append({"name": name, "description": desc})
    return skills


def _parse_skill_recipes(text: str) -> List[Dict[str, Any]]:
    """Extract RECIPE lines from LLM contrastive output."""
    m = re.search(
        r"<Skill Recipes>\s*(.*?)\s*</Skill Recipes>",
        text, re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return []
    recipes = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if line.upper().startswith("RECIPE:"):
            parts = [p.strip() for p in line[7:].split("|")]
            if len(parts) >= 2:
                recipes.append({
                    "name": parts[0],
                    "skills": [s.strip() for s in parts[1].split("+")],
                    "condition": parts[2] if len(parts) > 2 else "",
                })
    return recipes


def make_skill_id(name: str) -> str:
    """Deterministic ID from skill name only (source-independent)."""
    raw = name.lower().strip()
    return "skill_" + hashlib.md5(raw.encode()).hexdigest()[:16]


def make_edge_id(node_ids: List[str]) -> str:
    raw = "::".join(sorted(node_ids))
    return "edge_" + hashlib.md5(raw.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════
# Main updater
# ═══════════════════════════════════════════════════════════════════════════

class SkillUpdater:
    """
    Simplified 2-stage skill distillation:
      1. Contrastive step-level extraction (decompose→align→diverge→skills+recipes)
      2. Counterfactual verification (filter non-causal skills)

    After verification, skills are directly inserted into the hypergraph.
    Advantage starts at 0.0 and evolves via retrieval feedback (external).
    """

    def __init__(self, builder, llm_cfg: LLMConfig):
        self.builder = builder
        self.llm_cfg = llm_cfg

    async def update(
        self,
        groups: List[Dict[str, Any]],
        concurrency: int = 4,
        num_skills: int = 3,
    ) -> Dict[str, int]:
        """
        Run extraction + verification + insert on one batch.

        Args:
            groups: list of {question, gold_answer, trajectories: [{text, reward}]}
        Returns:
            {"added": N, "verified": M, "recipes": R}
        """
        # Stage 1: Contrastive extraction
        extraction_results = await self._contrastive_extract(
            groups, concurrency, num_skills
        )

        all_skills: List[Dict[str, str]] = []
        all_recipes: List[Dict[str, Any]] = []
        for r in extraction_results:
            if r:
                all_skills.extend(r["skills"])
                all_recipes.extend(r["recipes"])

        no_var = getattr(self, "_last_no_variance", 0)
        n_groups = len(groups)
        n_with_results = sum(1 for r in extraction_results if r)
        print(f"  [SkillDistill] groups={n_groups}, with_variance_extracted={n_with_results}, "
              f"raw_skills={len(all_skills)}")

        if not all_skills:
            return {"added": 0, "verified": 0, "recipes": 0,
                    "groups_no_variance": no_var, "n_groups": n_groups}

        # Stage 2: Counterfactual verification
        verified = await self._counterfactual_verify(
            all_skills, groups, concurrency
        )

        if not verified:
            return {"added": 0, "verified": 0, "recipes": 0,
                    "groups_no_variance": no_var, "n_groups": n_groups}

        # Insert verified skills directly (no LLM CRUD)
        added = 0
        for skill in verified:
            ok = await self._insert_skill(skill["name"], skill["description"])
            if ok:
                added += 1

        # Create recipe hyperedges
        recipe_count = self._create_recipe_edges(all_recipes, verified)

        self.builder.save()

        print(
            f"  [SkillDistill] {len(all_skills)} extracted → "
            f"{len(verified)} verified → {added} added, "
            f"[recipe]{recipe_count} recipes"
        )
        return {"added": added, "verified": len(verified), "recipes": recipe_count,
                "groups_no_variance": no_var, "n_groups": n_groups}

    # ── Stage 1: Contrastive step-level extraction ────────────────────────

    async def _contrastive_extract(
        self,
        groups: List[Dict[str, Any]],
        concurrency: int,
        num_skills: int,
    ) -> List[Optional[Dict[str, Any]]]:
        """
        For each question with reward variance: decompose trajectories into
        steps, align good/bad, extract atomic skills + recipes.
        """
        semaphore = asyncio.Semaphore(concurrency)
        results: List[Optional[Dict]] = [None] * len(groups)

        skip_no_var = [0]
        skip_no_split = [0]
        empty_extract = [0]

        async def _process(idx: int, group: Dict[str, Any]):
            async with semaphore:
                trajs = group["trajectories"]
                rewards = [t["reward"] for t in trajs]
                # Need variance to compare
                if max(rewards) - min(rewards) < 0.15:  # need some signal
                    skip_no_var[0] += 1
                    return

                # Split by mean for even sizes, median for odd
                # (using strict > / <= avoids the n=2 edge case where max == median)
                pivot = sum(rewards) / len(rewards)
                good = [t for t in trajs if t["reward"] > pivot]
                bad = [t for t in trajs if t["reward"] < pivot]
                if not good or not bad:
                    # Fallback: split by max vs rest
                    max_r = max(rewards)
                    good = [t for t in trajs if t["reward"] >= max_r]
                    bad = [t for t in trajs if t["reward"] < max_r]
                if not good or not bad:
                    skip_no_split[0] += 1
                    return

                def _fmt(traj_list, label):
                    parts = []
                    for i, t in enumerate(traj_list[:3]):
                        steps = _decompose_trajectory(t["text"])
                        step_str = "\n".join(
                            f"  Step {j+1}: {s[:300]}"
                            for j, s in enumerate(steps)
                        )
                        parts.append(
                            f"{label} {i+1} (Reward {t['reward']:.2f}):\n{step_str}"
                        )
                    return "\n\n".join(parts)

                try:
                    response = await async_llm_call(
                        prompt=SKILL_CONTRASTIVE_EXTRACT_UP.format(
                            question=group["question"][:1000],
                            gold_answer=group.get("gold_answer", "[N/A]")[:500],
                            good_attempts=_fmt(good, "Good Attempt"),
                            bad_attempts=_fmt(bad, "Bad Attempt"),
                        ),
                        cfg=self.llm_cfg,
                        system=SKILL_CONTRASTIVE_EXTRACT_SP,
                        max_tokens=1500,
                        temperature=0.3,
                    )
                    skills = _parse_atomic_skills(response)
                    recipes = _parse_skill_recipes(response)
                    if skills:
                        results[idx] = {"skills": skills, "recipes": recipes}
                    else:
                        empty_extract[0] += 1
                        logger.debug("No skills parsed from response: %s", response[:200])
                except Exception as exc:
                    logger.warning("contrastive_extract failed: %s", exc)

        await asyncio.gather(*[_process(i, g) for i, g in enumerate(groups)])
        print(f"  [Extract debug] no_variance={skip_no_var[0]}, "
              f"no_split={skip_no_split[0]}, empty_extract={empty_extract[0]}")
        # Stash debug counts on self for the caller (lists can't take attrs)
        self._last_no_variance = skip_no_var[0]
        return results

    # ── Stage 2: Counterfactual verification ──────────────────────────────

    async def _counterfactual_verify(
        self,
        candidates: List[Dict[str, str]],
        groups: List[Dict[str, Any]],
        concurrency: int,
        max_checks: int = 1,  # was 2 — halve verify LLM calls
    ) -> List[Dict[str, str]]:
        """Test each candidate on failed trajectories: would it have helped?"""
        # Collect failures
        failures = []
        for g in groups:
            for t in g["trajectories"]:
                if t["reward"] < 0.3:
                    failures.append({
                        "question": g["question"],
                        "gold_answer": g.get("gold_answer", ""),
                        "text": t["text"],
                    })
        if not failures:
            return candidates  # no failures → keep all

        semaphore = asyncio.Semaphore(concurrency)
        verified = []
        lock = asyncio.Lock()

        async def _check(skill: Dict[str, str]):
            import random
            sampled = random.sample(failures, min(max_checks, len(failures)))
            yes_count = 0

            for fail in sampled:
                async with semaphore:
                    try:
                        resp = await async_llm_call(
                            prompt=SKILL_COUNTERFACTUAL_VERIFY_UP.format(
                                question=fail["question"][:500],
                                gold_answer=fail["gold_answer"][:200],
                                failed_trajectory=fail["text"][:1500],
                                skill=f"{skill['name']}: {skill['description']}",
                            ),
                            cfg=self.llm_cfg,
                            system=SKILL_COUNTERFACTUAL_VERIFY_SP,
                            max_tokens=100,
                            temperature=0.1,
                        )
                        if re.search(r"VERDICT:\s*YES", resp, re.IGNORECASE):
                            yes_count += 1
                    except Exception as exc:
                        logger.warning("verify failed: %s", exc)

            if yes_count >= max(1, len(sampled) // 2):
                async with lock:
                    verified.append(skill)

        await asyncio.gather(*[_check(s) for s in candidates])

        if len(verified) < len(candidates):
            print(
                f"  [Verify] {len(verified)}/{len(candidates)} passed "
                f"counterfactual check"
            )
        return verified

    # ── Insert helpers ────────────────────────────────────────────────────

    async def _insert_skill(self, name: str, description: str) -> bool:
        """Insert a new skill node. Returns True if actually inserted."""
        skill_id = make_skill_id(name)
        if self.builder.kv_skills.get(skill_id) is not None:
            return False  # already exists

        skill_dict = {
            "skill_id": skill_id,
            "name": name,
            "description": description,
            "modality": "text",
            "hop_type": "single",
            "advantage": 0.3,  # bootstrap positive: skill passed counterfactual verification
            "examples": [],
        }
        await self.builder._upsert_skills([skill_dict])
        return True

    def _create_recipe_edges(
        self,
        recipes: List[Dict[str, Any]],
        verified_skills: List[Dict[str, str]],
    ) -> int:
        """Create hyperedges for skill recipes (verified skill combinations)."""
        if not recipes:
            return 0

        # Build name→id lookup
        name_to_id = {}
        for s in verified_skills:
            sid = make_skill_id(s["name"])
            if self.builder.kv_skills.get(sid) is not None:
                name_to_id[s["name"].lower()] = sid

        count = 0
        vdb_batch = []  # (edge_id, desc) for batch embed at end
        for recipe in recipes:
            node_ids = []
            for skill_name in recipe.get("skills", []):
                for known, sid in name_to_id.items():
                    if skill_name.lower() in known or known in skill_name.lower():
                        node_ids.append(sid)
                        break

            if len(node_ids) < 2:
                continue

            edge_id = make_edge_id(node_ids)
            if self.builder.vdb_hyperedges.has(edge_id):
                continue  # already in VDB, skip

            desc = (
                f"{recipe['name']}: {recipe.get('condition', '')} "
                f"[{' + '.join(recipe.get('skills', []))}]"
            )

            self.builder.kv_hyperedges.set(edge_id, {
                "edge_id": edge_id,
                "description": desc,
                "node_ids": node_ids,
                "advantage": 0.0,
                "edge_type": "recipe",
            })
            self.builder.graph.upsert_node(
                edge_id, node_type="hyperedge",
                description=desc, advantage=0.0, edge_type="recipe",
            )
            for nid in node_ids:
                self.builder.graph.upsert_edge(
                    edge_id, nid, weight=0.1, relation="recipe_component",
                )
            vdb_batch.append((edge_id, desc))
            count += 1

        # Embed all new recipe hyperedges in one batch and write to VDB
        if vdb_batch:
            from ..llm import sync_embed_texts
            descs = [d for _, d in vdb_batch]
            try:
                vecs = sync_embed_texts(descs, self.llm_cfg)
                self.builder.vdb_hyperedges.upsert_batch(
                    [(eid, vecs[i]) for i, (eid, _) in enumerate(vdb_batch)]
                )
            except Exception as exc:
                logger.warning("Recipe hyperedge VDB embed failed: %s", exc)

        return count
