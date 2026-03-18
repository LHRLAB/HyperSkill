"""
Storage backends for the omni-modal weighted hypergraph.

Three backends:
  - JsonKVStorage   : persistent JSON key-value store (skills, hyperedges metadata)
  - VectorStorage   : nano_vectordb wrapper for embedding similarity search
  - GraphStorage    : NetworkX directed graph persisted as GraphML

All paths are relative to a working directory configured in HypergraphConfig.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# JSON KV Storage
# ---------------------------------------------------------------------------

class JsonKVStorage:
    """Simple JSON-file-backed key-value store."""

    def __init__(self, path: str):
        self._path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self._data: Dict[str, Any] = json.load(f)
        else:
            self._data = {}

    # -- CRUD ---------------------------------------------------------------

    def get(self, key: str, default=None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def all_keys(self) -> List[str]:
        return list(self._data.keys())

    def all_values(self) -> List[Any]:
        return list(self._data.values())

    def items(self):
        return self._data.items()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    # -- Persistence --------------------------------------------------------

    def save(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    # -- Filtered queries ---------------------------------------------------

    def filter_by_field(self, field: str, value: Any) -> List[Dict[str, Any]]:
        return [v for v in self._data.values() if isinstance(v, dict) and v.get(field) == value]

    def filter_by_source(self, source: str) -> List[Dict[str, Any]]:
        return self.filter_by_field("source", source)


# ---------------------------------------------------------------------------
# Vector Storage
# ---------------------------------------------------------------------------

class VectorStorage:
    """
    In-memory numpy-backed vector store with cosine similarity search.

    Performance improvements over naive list-based store:
      - O(1) key→index lookup via _id_to_idx dict (vs O(N) list.index())
      - upsert_batch(): single np.vstack for all new items (vs one per insert)
      - has(): O(1) existence check to skip embedding for known IDs
    """

    def __init__(self, path: str, dim: int = 1536):
        self._path = path
        self._dim = dim
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}   # O(1) lookup
        self._vecs: Optional[np.ndarray] = None  # (N, dim), L2-normalised
        if os.path.exists(path):
            self._load()

    # -- Existence check ----------------------------------------------------

    def has(self, key: str) -> bool:
        return key in self._id_to_idx

    # -- Upsert / Query -----------------------------------------------------

    def upsert(self, key: str, vec: np.ndarray) -> None:
        """Single-item upsert. Delegates to upsert_batch for consistency."""
        self.upsert_batch([(key, vec)])

    def upsert_batch(self, items: List[Tuple[str, np.ndarray]]) -> None:
        """
        Batch upsert — single allocation for all new items.

        items: list of (key, vec) pairs.
        Existing keys are updated in-place; new keys are appended with one
        np.vstack call, regardless of batch size.
        """
        if not items:
            return

        updates: List[Tuple[int, np.ndarray]] = []  # (idx, normed_vec)
        new_keys: List[str] = []
        new_vecs: List[np.ndarray] = []

        seen: set = set()
        for key, vec in items:
            if key in seen:
                continue
            seen.add(key)
            v = np.asarray(vec, dtype=np.float32).flatten()
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            if key in self._id_to_idx:
                updates.append((self._id_to_idx[key], v))
            else:
                new_keys.append(key)
                new_vecs.append(v)

        # Apply in-place updates
        if updates and self._vecs is not None:
            for idx, v in updates:
                self._vecs[idx] = v

        # Append new items with a single allocation
        if new_keys:
            new_mat = np.stack(new_vecs, axis=0)  # (M, dim)
            start = len(self._ids)
            if self._vecs is None:
                self._vecs = new_mat
            else:
                self._vecs = np.vstack([self._vecs, new_mat])
            for i, key in enumerate(new_keys):
                self._id_to_idx[key] = start + i
            self._ids.extend(new_keys)

    def query(
        self,
        vec: np.ndarray,
        top_k: int = 10,
        exclude_keys: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Return top_k (key, cosine_similarity) pairs, sorted descending."""
        if self._vecs is None or not self._ids:
            return []
        v = np.asarray(vec, dtype=np.float32).flatten()
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        sims = self._vecs @ v  # (N,)
        exclude_set = set(exclude_keys or [])
        results = [
            (self._ids[i], float(sims[i]))
            for i in range(len(self._ids))
            if self._ids[i] not in exclude_set
        ]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def delete(self, key: str) -> None:
        if key not in self._id_to_idx:
            return
        idx = self._id_to_idx.pop(key)
        self._ids.pop(idx)
        if self._vecs is not None:
            self._vecs = np.delete(self._vecs, idx, axis=0)
            if self._vecs.shape[0] == 0:
                self._vecs = None
        # Rebuild index map after deletion (indices shifted)
        self._id_to_idx = {k: i for i, k in enumerate(self._ids)}

    def __len__(self) -> int:
        return len(self._ids)

    # -- Persistence --------------------------------------------------------

    def save(self) -> None:
        data = {
            "ids": self._ids,
            "vecs": self._vecs.tolist() if self._vecs is not None else [],
        }
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _load(self) -> None:
        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._ids = data.get("ids", [])
        self._id_to_idx = {k: i for i, k in enumerate(self._ids)}
        raw = data.get("vecs", [])
        if raw:
            self._vecs = np.array(raw, dtype=np.float32)
        else:
            self._vecs = None


# ---------------------------------------------------------------------------
# Graph Storage
# ---------------------------------------------------------------------------

class GraphStorage:
    """
    NetworkX DiGraph backed by GraphML for omni-modal weighted hypergraph.

    Node attributes (required):
      - node_type  : "skill" | "hyperedge"
      - advantage  : float   (cumulative advantage weight, clamped to [-1, 1])
      - source     : str     (dataset source)
      - modality   : str     (text | image | audio | video | pdf | mixed)

    Edge attributes:
      - weight     : float   (advantage-weighted co-occurrence)
      - source     : str
    """

    def __init__(self, path: str):
        self._path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            self._graph: nx.DiGraph = nx.read_graphml(path)
        else:
            self._graph = nx.DiGraph()

    # -- Node operations ----------------------------------------------------

    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if not self._graph.has_node(node_id):
            return None
        return dict(self._graph.nodes[node_id])

    def upsert_node(self, node_id: str, **attrs) -> None:
        if self._graph.has_node(node_id):
            self._graph.nodes[node_id].update(attrs)
        else:
            self._graph.add_node(node_id, **attrs)

    def delete_node(self, node_id: str) -> None:
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)

    def all_nodes(
        self, node_type: Optional[str] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        if node_type is None:
            return [(n, dict(d)) for n, d in self._graph.nodes(data=True)]
        return [
            (n, dict(d))
            for n, d in self._graph.nodes(data=True)
            if d.get("node_type") == node_type
        ]

    # -- Edge operations ----------------------------------------------------

    def has_edge(self, src: str, dst: str) -> bool:
        return self._graph.has_edge(src, dst)

    def upsert_edge(self, src: str, dst: str, **attrs) -> None:
        if self._graph.has_edge(src, dst):
            self._graph.edges[src, dst].update(attrs)
        else:
            self._graph.add_edge(src, dst, **attrs)

    def get_neighbors(self, node_id: str) -> List[str]:
        if not self._graph.has_node(node_id):
            return []
        return list(self._graph.successors(node_id))

    def node_degree(self, node_id: str) -> int:
        if not self._graph.has_node(node_id):
            return 0
        return self._graph.degree(node_id)

    # -- Bulk queries -------------------------------------------------------

    def get_nodes_below_advantage(self, threshold: float) -> List[str]:
        """Return node ids whose advantage < threshold."""
        return [
            n
            for n, d in self._graph.nodes(data=True)
            if d.get("advantage", 0.0) < threshold
        ]

    # -- Stats --------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        nodes = self._graph.number_of_nodes()
        edges = self._graph.number_of_edges()
        skills = sum(1 for _, d in self._graph.nodes(data=True) if d.get("node_type") == "skill")
        hyperedges = sum(1 for _, d in self._graph.nodes(data=True) if d.get("node_type") == "hyperedge")
        return {
            "total_nodes": nodes,
            "skill_nodes": skills,
            "hyperedge_nodes": hyperedges,
            "total_edges": edges,
        }

    # -- Persistence --------------------------------------------------------

    def save(self) -> None:
        nx.write_graphml(self._graph, self._path)

    def load(self) -> None:
        if os.path.exists(self._path):
            self._graph = nx.read_graphml(self._path)
