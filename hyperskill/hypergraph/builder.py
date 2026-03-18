"""
HypergraphBuilder: Constructs and evolves the omni-modal weighted hypergraph
from reasoning trajectory experiences.

Core operations:
  1. insert_experience(trajectory, advantage, ...)
     → Ingests modality-specific content, extracts skills, upserts into graph
  2. update_advantage(skill_id, new_advantage, configs)
     → Physics-inspired advantage update with driving force + KL restoring force
  3. prune(threshold)
     → Remove low-advantage nodes and their edges
"""

from __future__ import annotations
import asyncio
import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import numpy as np

from ..config import HypergraphConfig, LLMConfig
from ..llm import async_embed_texts
from .extractor import (
    RawSkill, RawHyperedge,
    extract_skills_from_trajectory,
)
from .ingestor import ingest_modality, TextChunk
from .storage import GraphStorage, JsonKVStorage, VectorStorage


class HypergraphBuilder:
    """
    Manages the lifecycle of the omni-modal weighted hypergraph skill library.
    """

    def __init__(self, cfg: HypergraphConfig, llm_cfg: LLMConfig):
        self.cfg = cfg
        self.llm_cfg = llm_cfg
        wd = cfg.working_dir
        os.makedirs(wd, exist_ok=True)

        # Storage backends
        self.kv_skills = JsonKVStorage(os.path.join(wd, "kv_skills.json"))
        self.kv_hyperedges = JsonKVStorage(os.path.join(wd, "kv_hyperedges.json"))
        self.graph = GraphStorage(os.path.join(wd, "hypergraph.graphml"))
        self.vdb_skills = VectorStorage(
            os.path.join(wd, "vdb_skills.json"), dim=llm_cfg.embedding_dim
        )
        self.vdb_hyperedges = VectorStorage(
            os.path.join(wd, "vdb_hyperedges.json"), dim=llm_cfg.embedding_dim
        )

        self._update_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def insert_experience(
        self,
        trajectory: str,
        advantage: float,
        source: str,
        modality: str = "text",
        hop_type: str = "single",
        extra_media: Optional[List[Tuple[str, bytes]]] = None,
    ) -> int:
        """
        Insert one reasoning trajectory into the hypergraph.

        Args:
            trajectory  : full text of the reasoning trajectory
            advantage   : GRPO advantage score for this trajectory
            source      : dataset name
            modality    : text | image | audio | video | pdf | mixed
            hop_type    : single | multi
            extra_media : list of (modality, bytes) for extra multimodal content

        Returns:
            number of new skill nodes inserted
        """
        # 1. Extract skills and hyperedges from the trajectory text
        raw_skills, raw_edges = await extract_skills_from_trajectory(
            trajectory=trajectory,
            source=source,
            advantage=advantage,
            modality=modality,
            hop_type=hop_type,
            llm_cfg=self.llm_cfg,
        )

        # 2. If extra media provided, ingest and extend the trajectory representation
        if extra_media:
            media_chunks: List[TextChunk] = []
            for mod, raw_bytes in extra_media:
                chunks = await ingest_modality(
                    modality=mod,
                    data=raw_bytes,
                    source=source,
                    advantage=advantage,
                    llm_cfg=self.llm_cfg,
                    chunk_size=self.cfg.chunk_size,
                    overlap=self.cfg.chunk_overlap,
                )
                media_chunks.extend(chunks)
            # Add captions/transcriptions as additional text context
            for chunk in media_chunks:
                extra_skills, extra_edges = await extract_skills_from_trajectory(
                    trajectory=chunk.content,
                    source=source,
                    advantage=advantage,
                    modality=chunk.modality,
                    hop_type=hop_type,
                    llm_cfg=self.llm_cfg,
                )
                raw_skills.extend(extra_skills)
                raw_edges.extend(extra_edges)

        # 3. Dedup / merge with existing skills then upsert
        inserted = await self._upsert_skills(raw_skills)

        # 4. Upsert hyperedges
        await self._upsert_hyperedges(raw_edges)

        # 5. Periodically prune
        self._update_count += 1
        if self._update_count % self.cfg.prune_interval == 0:
            self.prune(self.cfg.prune_min_advantage)

        return inserted

    async def batch_insert_experiences(
        self,
        experiences: List[Dict[str, Any]],
        max_concurrent: int = 4,
    ) -> int:
        """
        KnowledgeClaw-style batched pipeline (faster than per-trajectory inserts):

        1. Parallel skill extraction across all positive-advantage trajectories
        2. Collect ALL raw_skills and raw_edges from every extraction result
        3. Single _upsert_skills call → one batched embed API call for all new skills
        4. Single _upsert_hyperedges call

        Negative-advantage trajectories are skipped for extraction (no new skill
        knowledge from failed attempts); only positive ones generate new nodes.
        """
        if not experiences:
            return 0

        # Deduplicate by trajectory text to avoid redundant LLM extraction calls.
        # All rollouts from the same question share a similar trajectory structure,
        # so keep at most one per unique (source, advantage-sign) pair.
        # We still extract from all sign classes so low-advantage trajectories can
        # down-weight or skip nodes, and zero-advantage groups bootstrap the library.
        seen: set = set()
        deduped = []
        for e in experiences:
            key = (e.get("source", ""), round(e.get("advantage", 0.0), 2))
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        exps_to_extract = deduped if deduped else experiences

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _extract_one(exp: Dict[str, Any]):
            async with semaphore:
                try:
                    return await extract_skills_from_trajectory(
                        trajectory=exp["trajectory"],
                        source=exp.get("source", ""),
                        advantage=exp["advantage"],
                        modality=exp.get("modality", "text"),
                        hop_type=exp.get("hop_type", "single"),
                        llm_cfg=self.llm_cfg,
                    )
                except Exception as exc:
                    print(f"  [SkillLib] extraction error: {exc}")
                    return [], []

        results = await asyncio.gather(*[_extract_one(e) for e in exps_to_extract])

        # Collect all skills and edges from all trajectories
        all_skills: List[RawSkill] = []
        all_edges: List[RawHyperedge] = []
        for r in results:
            if isinstance(r, tuple) and len(r) == 2:
                skills, edges = r
                all_skills.extend(skills)
                all_edges.extend(edges)

        # Single batch upsert — one embed API call for all new skills
        print(f"  [SkillLib] extracted {len(all_skills)} skills, {len(all_edges)} edges from {len(exps_to_extract)} trajectories")
        inserted = await self._upsert_skills(all_skills)
        await self._upsert_hyperedges(all_edges)

        self._update_count += len(exps_to_extract)
        if self._update_count % self.cfg.prune_interval == 0:
            self.prune(self.cfg.prune_min_advantage)

        return inserted

    def prune(self, min_advantage: float = -0.5) -> int:
        """Remove all skill nodes with advantage < min_advantage."""
        to_remove = self.graph.get_nodes_below_advantage(min_advantage)
        count = 0
        for node_id in to_remove:
            self.graph.delete_node(node_id)
            self.kv_skills.delete(node_id)
            self.vdb_skills.delete(node_id)
            count += 1
        if count > 0:
            self.save()
        return count

    def save(self) -> None:
        """Persist all storage backends to disk."""
        self.kv_skills.save()
        self.kv_hyperedges.save()
        self.graph.save()
        self.vdb_skills.save()
        self.vdb_hyperedges.save()

    def stats(self) -> Dict[str, Any]:
        graph_stats = self.graph.stats()
        return {
            **graph_stats,
            "kv_skills": len(self.kv_skills),
            "kv_hyperedges": len(self.kv_hyperedges),
            "vdb_skills": len(self.vdb_skills),
            "vdb_hyperedges": len(self.vdb_hyperedges),
            "total_updates": self._update_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _upsert_skills(self, raw_skills: List[RawSkill]) -> int:
        """
        Insert/update skills with two-pass approach (from KnowledgeClaw pattern):

        Pass 1 — exact-ID check (no API call):
          Skills whose deterministic ID already exists in kv_skills get an
          advantage update only — no embedding needed.

        Pass 2 — embed genuinely new skills in one batched API call, then
          check for semantic near-duplicates and merge if needed.
          All VDB writes are flushed in a single upsert_batch call.
        """
        if not raw_skills:
            return 0

        # Assign deterministic IDs and split into existing vs new
        for skill in raw_skills:
            if "skill_id" not in skill:
                skill["skill_id"] = _make_skill_id(skill["name"], skill["source"])

        exact_existing = [s for s in raw_skills if self.kv_skills.get(s["skill_id"]) is not None]
        new_skills     = [s for s in raw_skills if self.kv_skills.get(s["skill_id"]) is None]

        # Pass 1: advantage-only update for exact duplicates (no embed)
        for skill in exact_existing:
            self._apply_advantage_update(skill["skill_id"], skill["advantage"])

        if not new_skills:
            return 0

        # Pass 2: embed all genuinely new skills in one batched API call
        descs = [s.get("description", s.get("name", "")) for s in new_skills]
        vecs = await async_embed_texts(descs, self.llm_cfg)

        inserted = 0
        vdb_batch: List[tuple] = []   # (id, vec) collected for single upsert_batch

        for skill, vec in zip(new_skills, vecs):
            sid = skill["skill_id"]

            # Semantic dedup: if near-duplicate exists, just update its advantage
            # (skip expensive merge LLM call during training)
            similar = self.vdb_skills.query(vec, top_k=1)
            if similar and similar[0][1] >= self.cfg.skill_dedup_threshold:
                existing_id = similar[0][0]
                self._apply_advantage_update(existing_id, skill["advantage"])
                continue

            # Genuinely new skill
            adv = max(-1.0, min(1.0, float(skill.get("advantage", 0.0))))
            self.kv_skills.set(sid, {**skill, "skill_id": sid})
            vdb_batch.append((sid, vec))
            self.graph.upsert_node(
                sid,
                node_type="skill",
                advantage=adv,
                source=skill.get("source", ""),
                modality=skill.get("modality", "text"),
                name=skill.get("name", ""),
            )
            inserted += 1

        # Single batch write to VDB — one np.vstack for all new/updated vectors
        if vdb_batch:
            self.vdb_skills.upsert_batch(vdb_batch)

        return inserted

    async def _upsert_hyperedges(self, raw_edges: List[RawHyperedge]) -> None:
        """
        Upsert hyperedge nodes: embed only edges not already in the VDB,
        then flush all VDB writes in a single upsert_batch call.
        """
        if not raw_edges:
            return

        # Assign IDs
        for edge in raw_edges:
            if "edge_id" not in edge:
                edge["edge_id"] = _make_edge_id(edge.get("node_ids", []))

        # Only embed edges not already in VDB
        new_edges  = [e for e in raw_edges if not self.vdb_hyperedges.has(e["edge_id"])]
        seen_edges = [e for e in raw_edges if     self.vdb_hyperedges.has(e["edge_id"])]

        # Embed new hyperedges in one batch
        vecs_map: dict = {}
        if new_edges:
            descs = [e.get("description", "") for e in new_edges]
            vecs  = await async_embed_texts(descs, self.llm_cfg)
            vecs_map = {e["edge_id"]: v for e, v in zip(new_edges, vecs)}

        vdb_batch: List[tuple] = []

        for edge in raw_edges:
            edge_id = edge["edge_id"]
            existing = self.kv_hyperedges.get(edge_id, {})
            q_snapshot = existing.get("advantage", 0.0)
            # q_current may have been updated within this batch
            q_current = self.graph.get_node(edge_id)
            q_current = q_current.get("advantage", q_snapshot) if q_current else q_snapshot
            new_adv = self._compute_new_advantage(
                q_current, float(edge.get("advantage", 0.0)), q_snapshot
            )
            edge["advantage"] = new_adv
            self.kv_hyperedges.set(edge_id, {**edge, "edge_id": edge_id})

            if edge_id in vecs_map:
                vdb_batch.append((edge_id, vecs_map[edge_id]))

            self.graph.upsert_node(
                edge_id,
                node_type="hyperedge",
                advantage=new_adv,
                source=edge.get("source", ""),
                modality=edge.get("modality", "text"),
                description=edge.get("description", ""),
            )
            for node_id in edge.get("node_ids", []):
                if self.graph.has_node(node_id):
                    self.graph.upsert_edge(node_id, edge_id, weight=abs(new_adv))
                    self.graph.upsert_edge(edge_id, node_id, weight=abs(new_adv))

        # Single batch VDB write
        if vdb_batch:
            self.vdb_hyperedges.upsert_batch(vdb_batch)

    def _apply_advantage_update(self, node_id: str, new_advantage: float) -> None:
        """
        EMA advantage update for skill nodes.

        q_new = (1 - η) · q_old + η · new_advantage
        Clamped to [-1, 1].

        Args:
            node_id:       skill or hyperedge node
            new_advantage: GRPO advantage signal from the current rollout
        """
        node = self.graph.get_node(node_id)
        if node is None:
            return
        q_old = float(node.get("advantage", 0.0))
        q_new = (1.0 - self.cfg.eta) * q_old + self.cfg.eta * new_advantage
        q_new = max(-1.0, min(1.0, q_new))
        self.graph.upsert_node(node_id, advantage=q_new)
        # Also sync to KV storage
        kv_node = self.kv_skills.get(node_id)
        if kv_node and isinstance(kv_node, dict):
            kv_node["advantage"] = q_new
            self.kv_skills.set(node_id, kv_node)

    def _compute_new_advantage(
        self, q_current: float, current_adv: float, q_snapshot: float = 0.0
    ) -> float:
        """EMA advantage update (shared helper for hyperedges)."""
        q_new = (1.0 - self.cfg.eta) * q_current + self.cfg.eta * current_adv
        return max(-1.0, min(1.0, q_new))

    def _get_node_advantage(self, node_id: str) -> float:
        node = self.graph.get_node(node_id)
        if node:
            return node.get("advantage", 0.0)
        return 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill_id(name: str, source: str) -> str:
    raw = f"{name.lower().strip()}::{source}"
    return "skill_" + hashlib.md5(raw.encode()).hexdigest()[:16]


def _make_edge_id(node_ids: List[str]) -> str:
    raw = "::".join(sorted(node_ids))
    return "edge_" + hashlib.md5(raw.encode()).hexdigest()[:16]
