"""
HyperSkillAgent: An agentic reasoner that uses the omni-modal weighted
hypergraph skill library to augment its context at inference time.

The agent:
  1. Receives a question (and optional multimodal content)
  2. Retrieves relevant skills from the library
  3. Builds a skill-augmented prompt
  4. Executes a multi-turn agentic loop (with optional search tools)
  5. Returns the final answer
"""

from __future__ import annotations
import asyncio
import base64
import json
import re
from typing import Any, Dict, List, Optional

from ..config import HyperSkillConfig
from ..llm import async_llm_call, async_llm_call_with_images
from ..prompts import (
    build_skill_augmented_prompt,
    AFTER_SEARCH_PROMPT,
)
from ..skill.library import SkillLibrary
from ..skill.models import SkillNode


class HyperSkillAgent:
    """
    Skill-augmented reasoning agent.

    Intended for both evaluation and interactive use.
    """

    def __init__(self, cfg: HyperSkillConfig, skill_library: Optional[SkillLibrary] = None):
        self.cfg = cfg
        # Use a provided library or create a fresh one from config
        self.skill_library = skill_library or SkillLibrary(cfg.hypergraph, cfg.llm)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def answer(
        self,
        question: str,
        task_type: str = "mmreasoning",
        images: Optional[List[bytes]] = None,
        audio: Optional[bytes] = None,
        video_path: Optional[str] = None,
        top_k_skills: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Answer a question using the skill library.

        Returns:
            {
              "answer"       : str,
              "reasoning"    : str,
              "skills_used"  : List[SkillNode],
              "trajectory"   : List[str],
            }
        """
        images = images or []
        top_k = top_k_skills or self.cfg.hypergraph.top_k

        # -- 1. Retrieve relevant skills ----------------------------------
        primary_image = images[0] if images else None
        skills = await self.skill_library.retrieve(
            query=question,
            top_k=top_k,
            image_bytes=primary_image,
        )

        if verbose:
            print(f"[HyperSkillAgent] Retrieved {len(skills)} skills:")
            for s in skills:
                print(f"  - {s.name} (adv={s.advantage:.3f})")

        # -- 2. Build augmented prompt ------------------------------------
        prompt = build_skill_augmented_prompt(
            question=question,
            modality=self.cfg.data.modality,
            task_type=task_type,
            skills=skills,
        )

        # -- 3. Execute agentic loop --------------------------------------
        response_list: List[str] = []
        image_b64_list = [base64.b64encode(img).decode("utf-8") for img in images]
        seen_text_queries: set = set()
        image_search_count = 0

        current_prompt = prompt
        for round_idx in range(self.cfg.rollout.max_gen_round):
            if image_b64_list:
                response = await async_llm_call_with_images(
                    prompt=current_prompt,
                    image_b64_list=image_b64_list,
                    cfg=self.cfg.llm,
                    max_tokens=self.cfg.rollout.max_new_tokens,
                )
            else:
                response = await async_llm_call(
                    prompt=current_prompt,
                    cfg=self.cfg.llm,
                    use_cache=False,
                    max_tokens=self.cfg.rollout.max_new_tokens,
                )

            response_list.append(response)
            current_prompt += response

            if verbose:
                print(f"[Round {round_idx+1}] {response[:200]}...")

            # Terminal: direct answer
            if "<answer>" in response and "</answer>" in response:
                break

            # Text search
            ts_match = re.search(r"<text_search>(.*?)</text_search>", response, re.DOTALL)
            if ts_match and task_type == "websearch":
                query = ts_match.group(1).strip()
                if query in seen_text_queries:
                    break
                seen_text_queries.add(query)
                search_text = await self._search(query)
                after = AFTER_SEARCH_PROMPT.format(
                    question=question,
                    search_results=f"[Search Results]\n{search_text}",
                )
                current_prompt += after
                continue

            # Image search
            if "<search><img></search>" in response and task_type == "websearch":
                if image_search_count >= self.cfg.rollout.image_search_limit:
                    break
                image_search_count += 1
                continue

            # No recognised action → stop
            break

        # -- 4. Extract final answer ----------------------------------------
        full_text = "\n".join(response_list)
        answer_text = _extract_answer(full_text)
        reason_text = _extract_reasoning(full_text)

        return {
            "answer": answer_text,
            "reasoning": reason_text,
            "skills_used": skills,
            "trajectory": response_list,
            "full_text": full_text,
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def _search(self, query: str) -> str:
        """
        Web search for websearch tasks.
        Uses SerpAPI when api_key_serpapi.txt is present; falls back to
        skill library retrieval.
        """
        from ..search import serpapi_search, _resolve_serpapi_key
        serpapi_key = _resolve_serpapi_key()
        if serpapi_key:
            try:
                return await serpapi_search(query, serpapi_key)
            except Exception:
                pass
        try:
            skills = await self.skill_library.retrieve(query=query, top_k=10)
            if skills:
                return "\n".join(f"[Skill] {s.name}: {s.description}" for s in skills)
        except Exception:
            pass
        return f"[No search results for: {query}]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_answer(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # Fallback: last non-empty line
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line:
            return line
    return ""


def _extract_reasoning(text: str) -> str:
    matches = re.findall(r"<reason>(.*?)</reason>", text, re.DOTALL)
    if matches:
        return " ".join(m.strip() for m in matches)
    return ""
