"""
Data model for skills in the HyperSkill library.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SkillNode:
    """A retrieved skill: a named, generalizable strategy stored in the hypergraph."""
    skill_id: str
    name: str
    description: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SkillNode":
        return cls(
            skill_id=d.get("skill_id", ""),
            name=d.get("name", ""),
            description=d.get("description", ""),
        )
