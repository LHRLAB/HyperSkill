"""
Data model for skills in the HyperSkill library.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
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

