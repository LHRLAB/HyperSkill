"""
SkillLibrary: High-level interface for the omni-modal weighted hypergraph
skill store. Wraps HypergraphBuilder + HypergraphRetriever into a unified API
used by the trainer and agents.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from ..config import HypergraphConfig, LLMConfig
from ..hypergraph.builder import HypergraphBuilder
from ..hypergraph.retriever import HypergraphRetriever
from .models import SkillNode


class SkillLibrary:
    """
    Omni-modal weighted hypergraph skill library.

    Usage
    -----
    library = SkillLibrary(hcfg, llm_cfg)

    # Retrieve relevant skills for a new question
    skills = await library.retrieve(question, top_k=5)
    """

    def __init__(self, cfg: HypergraphConfig, llm_cfg: LLMConfig):
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
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        image_bytes: Optional[bytes] = None,
    ) -> List[SkillNode]:
        """
        Retrieve the most relevant skills for a query.
        Returns a list of SkillNode objects.
        """
        raw_results = await self.retriever.retrieve(
            query=query,
            top_k=top_k,
            image_bytes=image_bytes,
        )
        return [SkillNode.from_dict(r) for r in raw_results]

    async def retrieve_for_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        image_bytes_list: Optional[List[Optional[bytes]]] = None,
        min_similarity: Optional[float] = None,
        min_advantage: float = 0.0,
    ) -> List[List[SkillNode]]:
        """Retrieve skills for a batch of queries.

        Applies an advantage floor: skills with negative learned advantage
        (consistently hurt rollouts) are filtered out.
        """
        raw_batch = await self.retriever.retrieve_for_batch(
            queries=queries,
            top_k=top_k,
            image_bytes_list=image_bytes_list,
            min_similarity=min_similarity,
            min_advantage=min_advantage,    # push the floor into the retriever so
                                            # RRF never wastes top-k on sub-threshold skills
        )
        return [[SkillNode.from_dict(r) for r in results] for results in raw_batch]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save all storage backends to disk."""
        self.builder.save()

    # ------------------------------------------------------------------
    # Info & statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return library statistics."""
        return self.builder.stats()
