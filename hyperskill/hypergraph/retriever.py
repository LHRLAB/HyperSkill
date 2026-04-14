"""
HypergraphRetriever: Retrieves relevant skills from the omni-modal weighted
hypergraph using Reciprocal Rank Fusion (RRF) over entity and hyperedge VDBs.

Also supports image-based queries via caption embedding.
"""

from __future__ import annotations
import base64
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import HypergraphConfig, LLMConfig
from ..llm import async_embed_texts, async_embed_image
from .storage import GraphStorage, JsonKVStorage, VectorStorage


class HypergraphRetriever:
    """
    Retrieves skill nodes and hyperedges relevant to a query.

    Ranking strategy:
      1. Embed query text (and image if provided)
      2. Search vdb_skills   → top-K skill ids
      3. Search vdb_hyperedges → top-K hyperedge ids
      4. Expand: for each hyperedge, add its connected skill nodes
      5. RRF fusion of all result lists, weighted by node advantage score
      6. Return top-N SkillNode objects
    """

    def __init__(
        self,
        cfg: HypergraphConfig,
        llm_cfg: LLMConfig,
        kv_skills: JsonKVStorage,
        kv_hyperedges: JsonKVStorage,
        graph: GraphStorage,
        vdb_skills: VectorStorage,
        vdb_hyperedges: VectorStorage,
    ):
        self.cfg = cfg
        self.llm_cfg = llm_cfg
        self.kv_skills = kv_skills
        self.kv_hyperedges = kv_hyperedges
        self.graph = graph
        self.vdb_skills = vdb_skills
        self.vdb_hyperedges = vdb_hyperedges

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        image_bytes: Optional[bytes] = None,
        source_filter: Optional[str] = None,
        modality_filter: Optional[str] = None,
        min_similarity: Optional[float] = None,
        min_advantage: float = -2.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant skills for a given query.

        Args:
            query          : text query string
            top_k          : number of results (defaults to cfg.top_k)
            image_bytes    : optional image for multimodal query
            source_filter  : restrict to a specific dataset source
            modality_filter: restrict to a specific modality
            min_similarity : cosine similarity floor (discard hits below this)

        Returns:
            List of skill dicts (from kv_skills), ranked by relevance.
        """
        top_k = top_k or self.cfg.top_k
        if len(self.vdb_skills) == 0:
            return []

        # -- 1. Encode query -----------------------------------------------
        text_vecs = await async_embed_texts([query], self.llm_cfg)
        query_vec = text_vecs[0]

        if image_bytes is not None:
            img_vec = await async_embed_image(image_bytes, self.llm_cfg)
            # Average text and image embeddings
            query_vec = (query_vec + img_vec) / 2.0
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec /= norm

        # -- 2. Search skill VDB -------------------------------------------
        # Fetch a WIDE candidate pool by similarity; downstream advantage/sim
        # filters + RRF re-ranking pick the best. A narrow pool can miss the
        # truly useful high-advantage skills because training-question embeddings
        # don't always sit in the top-k neighborhood of a new test question.
        # Fetch up to 100 candidates so high-advantage skills (which may sit
        # outside the top-k neighborhood by raw embedding similarity) still
        # surface after advantage filtering + RRF re-ranking.
        pool_k = max(top_k * 20, 100)
        skill_hits = self.vdb_skills.query(query_vec, top_k=pool_k)

        # -- 3. Search hyperedge VDB then expand ---------------------------
        edge_hits = self.vdb_hyperedges.query(query_vec, top_k=top_k)
        min_sim = min_similarity if min_similarity is not None else 0.0
        extra_hits: list = []
        for edge_id, edge_sim in edge_hits:
            if edge_sim < min_sim:          # 超边本身不相关，跳过
                continue
            edge_data = self.kv_hyperedges.get(edge_id, {})
            for nid in edge_data.get("node_ids", []):
                if not any(nid == h[0] for h in skill_hits):
                    extra_hits.append((nid, edge_sim * 0.8))
        all_hits = skill_hits + extra_hits

        # -- 4. Filter -------------------------------------------------------
        filtered = []
        for skill_id, sim in all_hits:
            if sim < min_sim:
                continue
            skill_data = self.kv_skills.get(skill_id)
            if skill_data is None:
                continue
            # Apply advantage floor BEFORE ranking so RRF doesn't waste its
            # top-1 slot on a negative-advantage skill that would later be
            # dropped by the library-level post-filter.
            if skill_data.get("advantage", 0.0) < min_advantage:
                continue
            if source_filter and skill_data.get("source") != source_filter:
                continue
            if modality_filter and skill_data.get("modality") != modality_filter:
                continue
            filtered.append((skill_id, sim, skill_data))

        # -- 5. Advantage-weighted RRF re-ranking --------------------------
        results = self._rrf_rank(filtered, top_k)
        return results

    async def retrieve_for_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        image_bytes_list: Optional[List[Optional[bytes]]] = None,
        min_similarity: Optional[float] = None,
        min_advantage: float = -2.0,
    ) -> List[List[Dict[str, Any]]]:
        """Retrieve skills for a batch of queries."""
        import asyncio
        tasks = []
        for i, query in enumerate(queries):
            img = image_bytes_list[i] if image_bytes_list else None
            tasks.append(self.retrieve(query, top_k=top_k, image_bytes=img,
                                       min_similarity=min_similarity,
                                       min_advantage=min_advantage))
        return await asyncio.gather(*tasks)

    # ------------------------------------------------------------------
    # RRF helpers
    # ------------------------------------------------------------------

    def _rrf_rank(
        self,
        hits: List[Tuple[str, float, Dict[str, Any]]],
        top_k: int,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion score combining:
          - embedding similarity rank
          - node advantage weight
        """
        if not hits:
            return []

        # Sort by similarity (rank 1 = highest)
        hits_sorted = sorted(hits, key=lambda x: -x[1])

        scores: Dict[str, float] = {}
        for rank, (skill_id, sim, skill_data) in enumerate(hits_sorted):
            # RRF contribution from rank
            rrf_score = 1.0 / (rrf_k + rank + 1)
            # Advantage boost (node advantage in [-1, 1])
            node = self.graph.get_node(skill_id)
            advantage = node.get("advantage", 0.0) if node else 0.0
            adv_boost = (advantage + 1.0) / 2.0  # map [-1,1] → [0,1]
            scores[skill_id] = rrf_score * (1.0 + adv_boost)

        # Sort by final score
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        results = []
        skill_map = {h[0]: h[2] for h in hits}
        for skill_id, _ in ranked:
            skill_data = skill_map.get(skill_id)
            if skill_data:
                results.append(skill_data)
        return results
