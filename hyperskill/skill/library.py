"""
SkillLibrary: High-level interface for the omni-modal weighted hypergraph
skill store. Wraps HypergraphBuilder + HypergraphRetriever into a unified API
used by the trainer and agents.
"""

from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from ..config import HypergraphConfig, LLMConfig
from ..hypergraph.builder import HypergraphBuilder
from ..hypergraph.retriever import HypergraphRetriever
from .models import SkillNode, SkillHyperedge


class SkillLibrary:
    """
    Omni-modal weighted hypergraph skill library.

    Usage
    -----
    library = SkillLibrary(hcfg, llm_cfg)

    # Add experience from a training trajectory
    await library.evolve(trajectory, advantage, source, modality, hop_type)

    # Retrieve relevant skills for a new question
    skills = await library.retrieve(question, top_k=5)

    # Build skill-augmented prompt
    prompt = library.build_prompt(question, skills)
    """

    def __init__(self, cfg: HypergraphConfig, llm_cfg: LLMConfig):
        self.cfg = cfg
        self.llm_cfg = llm_cfg

        # Builder manages insert / update / prune
        self.builder = HypergraphBuilder(cfg, llm_cfg)

        # Retriever for querying
        self.retriever = HypergraphRetriever(
            cfg=cfg,
            llm_cfg=llm_cfg,
            kv_skills=self.builder.kv_skills,
            kv_hyperedges=self.builder.kv_hyperedges,
            graph=self.builder.graph,
            vdb_skills=self.builder.vdb_skills,
            vdb_hyperedges=self.builder.vdb_hyperedges,
        )

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    async def evolve(
        self,
        trajectory: str,
        advantage: float,
        source: str,
        modality: str = "text",
        hop_type: str = "single",
        extra_media: Optional[List[Tuple[str, bytes]]] = None,
    ) -> int:
        """
        Update the skill library from one trajectory experience.

        High-advantage trajectories contribute skills with more influence;
        low/negative-advantage trajectories down-weight existing skills.

        Returns number of new skills inserted.
        """
        n = await self.builder.insert_experience(
            trajectory=trajectory,
            advantage=advantage,
            source=source,
            modality=modality,
            hop_type=hop_type,
            extra_media=extra_media,
        )
        return n

    async def batch_evolve(
        self,
        experiences: List[Dict[str, Any]],
        max_concurrent: int = 4,
    ) -> int:
        """Batch evolve from multiple experiences."""
        return await self.builder.batch_insert_experiences(experiences, max_concurrent)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        image_bytes: Optional[bytes] = None,
        source_filter: Optional[str] = None,
        modality_filter: Optional[str] = None,
    ) -> List[SkillNode]:
        """
        Retrieve the most relevant skills for a query.
        Returns a list of SkillNode objects.
        """
        raw_results = await self.retriever.retrieve(
            query=query,
            top_k=top_k,
            image_bytes=image_bytes,
            source_filter=source_filter,
            modality_filter=modality_filter,
        )
        return [SkillNode.from_dict(r) for r in raw_results]

    async def retrieve_for_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        image_bytes_list: Optional[List[Optional[bytes]]] = None,
        min_similarity: Optional[float] = None,
    ) -> List[List[SkillNode]]:
        """Retrieve skills for a batch of queries."""
        raw_batch = await self.retriever.retrieve_for_batch(
            queries=queries,
            top_k=top_k,
            image_bytes_list=image_bytes_list,
            min_similarity=min_similarity,
        )
        return [[SkillNode.from_dict(r) for r in results] for results in raw_batch]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save all storage backends to disk."""
        self.builder.save()

    def load(self) -> None:
        """Reload storage from disk (all backends auto-load on init)."""
        pass  # Auto-loaded in HypergraphBuilder.__init__

    # ------------------------------------------------------------------
    # Info & statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return library statistics."""
        return self.builder.stats()

    def list_skills(
        self,
        modality: Optional[str] = None,
        source: Optional[str] = None,
        min_advantage: float = -1.0,
    ) -> List[SkillNode]:
        """List all skills with optional filters."""
        nodes = self.builder.graph.all_nodes(node_type="skill")
        results = []
        for node_id, node_attrs in nodes:
            adv = node_attrs.get("advantage", 0.0)
            if adv < min_advantage:
                continue
            mod = node_attrs.get("modality", "")
            src = node_attrs.get("source", "")
            if modality and mod != modality:
                continue
            if source and src != source:
                continue
            skill_data = self.builder.kv_skills.get(node_id)
            if skill_data:
                results.append(SkillNode.from_dict(skill_data))
        # Sort by advantage descending
        results.sort(key=lambda s: -s.advantage)
        return results

    def top_skills(self, n: int = 20) -> List[SkillNode]:
        """Return the top-N highest advantage skills."""
        return self.list_skills()[:n]
