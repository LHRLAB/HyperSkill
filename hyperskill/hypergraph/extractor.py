"""
LLM-based skill and relation extractor for HyperSkill.

Given a reasoning trajectory (text), extracts:
  1. SkillNode candidates (named, generalizable skills)
  2. SkillHyperedge candidates (multi-way relationships between skills)

Uses the prompts from hyperskill.prompts and calls the LLM asynchronously.
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..config import LLMConfig
from ..llm import async_llm_call
from ..prompts import SKILL_EXTRACTION_PROMPT, SKILL_MERGE_PROMPT


# ---------------------------------------------------------------------------
# Data containers (lightweight, will be converted to SkillNode / SkillHyperedge)
# ---------------------------------------------------------------------------

RawSkill = Dict[str, Any]
RawHyperedge = Dict[str, Any]


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

async def extract_skills_from_trajectory(
    trajectory: str,
    source: str,
    advantage: float,
    modality: str,
    hop_type: str,
    llm_cfg: LLMConfig,
) -> Tuple[List[RawSkill], List[RawHyperedge]]:
    """
    Extract skills and hyperedges from a single reasoning trajectory.

    Returns:
        skills: list of raw skill dicts
        hyperedges: list of raw hyperedge dicts connecting related skills
    """
    prompt = SKILL_EXTRACTION_PROMPT.format(trajectory=trajectory)
    raw_response = await async_llm_call(
        prompt=prompt,
        cfg=llm_cfg,
        system="You are an expert at extracting reusable skills from reasoning trajectories. Output valid JSON only.",
        use_cache=False,
        max_tokens=1024,
        temperature=0.2,
    )

    skills = _parse_skill_json(raw_response)
    if not skills:
        # Surface parsing failures so they are visible in logs
        preview = raw_response.strip()[:120].replace("\n", " ")
        print(f"  [SkillExtract] 0 skills parsed from LLM output: {preview!r}")

    # Enrich each skill with metadata
    for skill in skills:
        skill["source"] = source
        skill["advantage"] = advantage
        skill.setdefault("modality", modality)
        skill.setdefault("hop_type", hop_type)
        skill["skill_id"] = _make_skill_id(skill.get("name", ""), source)

    # Build hyperedges: if multiple skills extracted from one trajectory,
    # create a hyperedge connecting them all (they co-occurred in a solution)
    hyperedges: List[RawHyperedge] = []
    if len(skills) >= 2:
        edge = {
            "edge_id": _make_edge_id([s["skill_id"] for s in skills]),
            "description": f"Co-occurring skills in a {hop_type}-hop {modality} trajectory",
            "node_ids": [s["skill_id"] for s in skills],
            "source": source,
            "advantage": advantage,
            "hop_type": hop_type,
            "modality": modality,
        }
        hyperedges.append(edge)

    return skills, hyperedges


async def batch_extract_skills(
    trajectories: List[Dict[str, Any]],
    llm_cfg: LLMConfig,
    max_concurrent: int = 4,
) -> Tuple[List[RawSkill], List[RawHyperedge]]:
    """
    Extract skills from multiple trajectories concurrently.

    Each trajectory dict must have keys:
      - text       : str  (the full trajectory text)
      - source     : str  (dataset source)
      - advantage  : float
      - modality   : str
      - hop_type   : str
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _extract_one(traj: Dict[str, Any]):
        async with semaphore:
            return await extract_skills_from_trajectory(
                trajectory=traj["text"],
                source=traj["source"],
                advantage=traj["advantage"],
                modality=traj.get("modality", "text"),
                hop_type=traj.get("hop_type", "single"),
                llm_cfg=llm_cfg,
            )

    results = await asyncio.gather(*[_extract_one(t) for t in trajectories])
    all_skills: List[RawSkill] = []
    all_edges: List[RawHyperedge] = []
    for skills, edges in results:
        all_skills.extend(skills)
        all_edges.extend(edges)
    return all_skills, all_edges


async def merge_similar_skills(
    skill_a: RawSkill,
    skill_b: RawSkill,
    llm_cfg: LLMConfig,
) -> RawSkill:
    """Ask the LLM to merge two similar skills into one."""
    prompt = SKILL_MERGE_PROMPT.format(
        skill_a=json.dumps(skill_a, ensure_ascii=False, indent=2),
        skill_b=json.dumps(skill_b, ensure_ascii=False, indent=2),
    )
    raw = await async_llm_call(
        prompt=prompt,
        cfg=llm_cfg,
        system="Merge these two skills into one. Output valid JSON only.",
        use_cache=False,
        max_tokens=512,
        temperature=0.1,
    )
    merged = _parse_single_skill_json(raw)
    # Retain best advantage
    merged["advantage"] = max(
        skill_a.get("advantage", 0.0),
        skill_b.get("advantage", 0.0),
    )
    merged["source"] = skill_a.get("source", "")
    merged["skill_id"] = _make_skill_id(merged.get("name", ""), merged["source"])
    return merged


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _parse_skill_json(text: str) -> List[RawSkill]:
    """
    Parse skills from LLM output.

    Primary format (pipe-separated lines, robust for all LLMs):
        SKILL: <name> | <description> | <modality> | <hop_type>

    Fallback: JSON array of skill objects (legacy).
    """
    text = text.strip()
    skills: List[RawSkill] = []

    # Primary: parse pipe-separated SKILL lines
    for line in text.splitlines():
        line = line.strip()
        if not line.upper().startswith("SKILL:"):
            continue
        body = line[6:].strip()  # drop "SKILL:"
        parts = [p.strip() for p in body.split("|")]
        if len(parts) < 2:
            continue
        name = parts[0]
        description = parts[1] if len(parts) > 1 else name
        modality = parts[2] if len(parts) > 2 else "text"
        hop_type = parts[3] if len(parts) > 3 else "single"
        if not name:
            continue
        skills.append({
            "name": name,
            "description": description,
            "modality": modality.lower() if modality.lower() in
                        {"text", "image", "audio", "video", "pdf", "mixed"} else "text",
            "hop_type": hop_type.lower() if hop_type.lower() in {"single", "multi"} else "single",
            "examples": [],
        })

    if skills:
        return skills

    # Fallback: JSON array
    text_stripped = re.sub(r"^```(?:json)?\s*", "", text)
    text_stripped = re.sub(r"\s*```$", "", text_stripped).strip()
    try:
        data = json.loads(text_stripped)
        if isinstance(data, list):
            return [s for s in data if isinstance(s, dict) and "name" in s]
        if isinstance(data, dict) and "skills" in data:
            return [s for s in data["skills"] if isinstance(s, dict) and "name" in s]
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text_stripped)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return [s for s in data if isinstance(s, dict) and "name" in s]
            except json.JSONDecodeError:
                pass

    return []


def _parse_single_skill_json(text: str) -> RawSkill:
    """Parse a single skill JSON object."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def _make_skill_id(name: str, source: str) -> str:
    raw = f"{name.lower().strip()}::{source}"
    return "skill_" + hashlib.md5(raw.encode()).hexdigest()[:16]


def _make_edge_id(node_ids: List[str]) -> str:
    raw = "::".join(sorted(node_ids))
    return "edge_" + hashlib.md5(raw.encode()).hexdigest()[:16]
