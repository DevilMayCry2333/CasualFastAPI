"""SelfModel — the agent's model of itself.

Maintains identity, beliefs, goals, drives, and personality.
This is ONLY about "me", not about the world or others.
Identity should emerge from experience, not from prompt engineering.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from runtime_kernel.runtime.drive import DriveModel
from runtime_kernel.runtime.models import DEFAULT_IDENTITY_ANCHOR


class SelfModel:
    """The agent's self-model.

    Fields:
        identity: Core identity descriptor.
        beliefs: Dict of belief statements -> confidence.
        goals: Current and pending goals.
        drives: Curiosity/boredom/belonging state.
        confidence: Overall self-confidence (0.0-1.0).
        recent_changes: Recent deltas to self-model.
        personality: Stable personality traits.
        identity_anchor: The structured identity anchor from reflection.
    """

    def __init__(self) -> None:
        self._identity: str = "unknown"
        self._beliefs: dict[str, float] = {}
        self._goals: list[dict] = []
        self._drives: dict[str, float] = DriveModel.initial()
        self._confidence: float = 0.0
        self._recent_changes: list[str] = []
        self._personality: dict[str, float] = {}
        self._identity_anchor: dict = dict(DEFAULT_IDENTITY_ANCHOR)

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def beliefs(self) -> dict[str, float]:
        return dict(self._beliefs)

    @property
    def goals(self) -> list[dict]:
        return list(self._goals)

    @property
    def drives(self) -> dict[str, float]:
        return dict(self._drives)

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def recent_changes(self) -> list[str]:
        return list(self._recent_changes)

    @property
    def personality(self) -> dict[str, float]:
        return dict(self._personality)

    @property
    def identity_anchor(self) -> dict:
        return dict(self._identity_anchor)

    def set_identity(self, identity: str) -> None:
        self._identity = identity

    def set_belief(self, statement: str, confidence: float = 0.5) -> None:
        self._beliefs[statement] = max(0.0, min(1.0, confidence))

    def set_goals(self, goals: list[dict]) -> None:
        self._goals = list(goals)

    def set_drives(self, drives: dict[str, float]) -> None:
        self._drives = dict(drives)

    def set_confidence(self, confidence: float) -> None:
        self._confidence = max(0.0, min(1.0, confidence))

    def add_change(self, change: str) -> None:
        self._recent_changes.append(change)
        if len(self._recent_changes) > 20:
            self._recent_changes = self._recent_changes[-20:]

    def set_identity_anchor(self, anchor: dict) -> None:
        self._identity_anchor = dict(anchor)
        if anchor.get("identity"):
            self._identity = anchor["identity"]

    def format_for_prompt(self) -> str:
        parts = ["【自我模型】"]
        parts.append(f"  身份: {self._identity}")
        if self._beliefs:
            parts.append("  信念:")
            for stmt, conf in sorted(self._beliefs.items(), key=lambda x: -x[1])[:5]:
                parts.append(f"    · {stmt[:60]} (conf={conf:.2f})")
        if self._goals:
            top = self._goals[0]
            parts.append(f"  当前目标: {top.get('goal', top.get('thought', '?'))[:60]}")
        parts.append(f"  自信度: {self._confidence:.2f}")
        if self._recent_changes:
            parts.append("  最近变化:")
            for c in self._recent_changes[-3:]:
                parts.append(f"    · {c[:60]}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "identity": self._identity,
            "beliefs": dict(self._beliefs),
            "goals": list(self._goals),
            "drives": dict(self._drives),
            "confidence": self._confidence,
            "recent_changes": list(self._recent_changes),
            "personality": dict(self._personality),
            "identity_anchor": dict(self._identity_anchor),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SelfModel":
        m = cls()
        m._identity = d.get("identity", "unknown")
        m._beliefs = dict(d.get("beliefs", {}))
        m._goals = list(d.get("goals", []))
        m._drives = dict(d.get("drives", DriveModel.initial()))
        m._confidence = float(d.get("confidence", 0.0))
        m._recent_changes = list(d.get("recent_changes", []))
        m._personality = dict(d.get("personality", {}))
        m._identity_anchor = dict(d.get("identity_anchor", DEFAULT_IDENTITY_ANCHOR))
        return m
