"""
Data models for skills and hyperedges in the HyperSkill library.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class SkillNode:
    """
    Represents a single, generalizable skill stored in the hypergraph.

    A skill is:
      - Named: has a human-readable name and description
      - Modal: associated with a specific input modality
      - Weighted: has a cumulative advantage weight ∈ [-1, 1]
      - Exemplified: carries example usage strings
    """
    skill_id: str
    name: str
    description: str
    modality: str               # text | image | audio | video | pdf | mixed
    source: str                 # dataset or experience source
    advantage: float = 0.0      # cumulative advantage weight ∈ [-1, 1]
    hop_type: str = "single"    # single | multi
    examples: List[str] = field(default_factory=list)
    media_refs: List[str] = field(default_factory=list)  # base64 or file refs
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "modality": self.modality,
            "source": self.source,
            "advantage": self.advantage,
            "hop_type": self.hop_type,
            "examples": self.examples,
            "media_refs": self.media_refs,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SkillNode":
        return cls(
            skill_id=d.get("skill_id", ""),
            name=d.get("name", ""),
            description=d.get("description", ""),
            modality=d.get("modality", "text"),
            source=d.get("source", ""),
            advantage=d.get("advantage", 0.0),
            hop_type=d.get("hop_type", "single"),
            examples=d.get("examples", []),
            media_refs=d.get("media_refs", []),
            tags=d.get("tags", []),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
        )

    def __repr__(self) -> str:
        return (
            f"SkillNode(id={self.skill_id[:8]}, name={self.name!r}, "
            f"modality={self.modality}, adv={self.advantage:.3f})"
        )


@dataclass
class SkillHyperedge:
    """
    A hyperedge connecting multiple SkillNodes.

    Represents a multi-way relationship such as:
      - A chain of reasoning steps (sequential)
      - A set of co-occurring skills (co-occurrence)
      - A causal dependency between skills (causal)
    """
    edge_id: str
    description: str            # what reasoning procedure this represents
    node_ids: List[str]         # connected SkillNode IDs
    source: str
    advantage: float = 0.0
    hop_type: str = "single"
    modality: str = "text"
    edge_type: str = "cooccurrence"  # cooccurrence | sequential | causal
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "description": self.description,
            "node_ids": self.node_ids,
            "source": self.source,
            "advantage": self.advantage,
            "hop_type": self.hop_type,
            "modality": self.modality,
            "edge_type": self.edge_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SkillHyperedge":
        return cls(
            edge_id=d.get("edge_id", ""),
            description=d.get("description", ""),
            node_ids=d.get("node_ids", []),
            source=d.get("source", ""),
            advantage=d.get("advantage", 0.0),
            hop_type=d.get("hop_type", "single"),
            modality=d.get("modality", "text"),
            edge_type=d.get("edge_type", "cooccurrence"),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
        )

    def __repr__(self) -> str:
        return (
            f"SkillHyperedge(id={self.edge_id[:8]}, n_nodes={len(self.node_ids)}, "
            f"adv={self.advantage:.3f})"
        )
