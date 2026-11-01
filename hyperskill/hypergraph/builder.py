"""
HypergraphBuilder: Stores and evolves the omni-modal weighted hypergraph
skill library.

Core operations:
  1. _upsert_skills(skills)
     → Embeds verified skills and inserts them into the graph / VDB
  2. _apply_advantage_update(skill_id, advantage_signal)
     → EMA update of the node advantage q_i (Eq. 5)
  3. prune(threshold)
     → Remove low-advantage nodes and their edges
"""

from __future__ import annotations
import os
from typing import Any, Dict, List

from ..config import HypergraphConfig, LLMConfig
from ..llm import async_embed_texts
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
        self.vdb_skills = VectorStorage(os.path.join(wd, "vdb_skills.json"))
        self.vdb_hyperedges = VectorStorage(os.path.join(wd, "vdb_hyperedges.json"))

        self._update_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    async def _upsert_skills(self, new_skills: List[Dict[str, Any]]) -> None:
        """
        Embed new skills (the caller has checked that their IDs are not in
        kv_skills) in one batched API call, merge semantic near-duplicates into
        the existing skill, and flush all VDB writes in a single upsert_batch.
        """
        descs = [s.get("description", s.get("name", "")) for s in new_skills]
        vecs = await async_embed_texts(descs, self.llm_cfg)

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

        # Single batch write to VDB — one np.vstack for all new/updated vectors
        if vdb_batch:
            self.vdb_skills.upsert_batch(vdb_batch)

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
        self, q_current: float, current_adv: float
    ) -> float:
        """EMA advantage update (shared helper for hyperedges)."""
        q_new = (1.0 - self.cfg.eta) * q_current + self.cfg.eta * current_adv
        return max(-1.0, min(1.0, q_new))
