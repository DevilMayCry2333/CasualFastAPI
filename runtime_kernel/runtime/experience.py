"""
experience — Experience: the atomic unit of an agent's life.

An Experience captures not just what the state was, but what the agent
perceived, did, observed, and what it meant. Meaning is initially empty
and populated later by Reflection.

Identity is the accumulated sediment of Experience.

Key insight: An agent IS its sequence of Experiences, not its state.
State is the working memory; Experience is the life lived.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from runtime_kernel.runtime.models import (
    IDENTITY_MATURITY_MIN,
    IDENTITY_MATURITY_MAX,
    IDENTITY_MATURITY_ROUND_WEIGHT,
    IDENTITY_MATURITY_EXPERIENCE_WEIGHT,
    IDENTITY_MATURITY_REFLECTION_WEIGHT,
    IDENTITY_MATURITY_ROUND_DENOM,
    IDENTITY_MATURITY_EXPERIENCE_DENOM,
    IDENTITY_MATURITY_REFLECTION_DENOM,
)


@dataclass
class Experience:
    """One atomic unit of an agent's existence.

    Unlike a raw state transition, an Experience captures:
      - What the agent perceived (environment snapshot)
      - What the agent did (action)
      - What happened as a result (observation / world feedback)
      - What it meant (initially empty — filled by Reflection)

    Fields:
        round: Round number when this occurred.
        session_id: Owning session.
        perception: The environment context the agent observed.
        action: What the agent did ("look", "move garden", etc.).
        observation: World feedback / result of the action.
        meaning: What this experience meant to the agent (initially "").
        cause: "init" | "self" | "human" | "environment" | "reflect".
        state_before: State snapshot before the action.
        state_after: State snapshot after the action.
        room: Where this happened.
        timestamp: When it happened.
    """

    round: int
    session_id: str
    perception: str = ""
    action: str = ""
    observation: str = ""
    meaning: str = ""
    cause: str = "self"
    state_before: dict = field(default_factory=dict)
    state_after: dict = field(default_factory=dict)
    room: str = ""
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Experience":
        return cls(
            round=d.get("round", 0),
            session_id=d.get("session_id", ""),
            perception=d.get("perception", ""),
            action=d.get("action", ""),
            observation=d.get("observation", ""),
            meaning=d.get("meaning", ""),
            cause=d.get("cause", "self"),
            state_before=d.get("state_before", {}),
            state_after=d.get("state_after", {}),
            room=d.get("room", ""),
            timestamp=d.get("timestamp", 0.0),
        )

    def to_short_text(self) -> str:
        """Compact one-line representation for prompt injection."""
        action_part = f" [{self.action}]" if self.action else ""
        meaning_part = f" → {self.meaning[:60]}" if self.meaning else ""
        return (
            f"R{self.round}{action_part}: "
            f"{self.perception[:80] or '—'}"
            f"{meaning_part}"
        )


@dataclass
class IdentityDelta:
    """A small change to identity, produced by Reflection.

    Instead of rewriting the entire identity anchor, Reflection produces
    deltas that accumulate over time. Identity is the sum of its deltas.

    Fields:
        round: When this delta was generated.
        session_id: Owning session.
        change: What changed ("开始喜欢探索未知房间").
        because: Why it changed ("连续进入三个不同房间后获得更多发现").
        affected_field: Which identity anchor field this modifies.
        strength: How strongly this changes identity (0.0-1.0).
    """

    round: int
    session_id: str
    change: str = ""
    because: str = ""
    affected_field: str = "identity"
    strength: float = 0.5
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "IdentityDelta":
        return cls(
            round=d.get("round", 0),
            session_id=d.get("session_id", ""),
            change=d.get("change", ""),
            because=d.get("because", ""),
            affected_field=d.get("affected_field", "identity"),
            strength=float(d.get("strength", 0.5)),
            timestamp=d.get("timestamp", 0.0),
        )


def compute_identity_maturity(
    round_count: int,
    experience_count: int,
    reflection_count: int,
) -> float:
    """Compute identity maturity from life statistics.

    Maturity is a weighted combination of:
      - Rounds lived (25%): time-based seasoning
      - Experiences accumulated (35%): experiential depth
      - Reflections performed (40%): self-awareness

    Each component saturates at a denominator, then is clipped to [0, 1].

    Args:
        round_count: Total rounds the agent has lived.
        experience_count: Total experiences accumulated.
        reflection_count: Total reflections performed.

    Returns:
        Float between 0.0 (newborn) and 1.0 (stable personality).
    """
    round_component = min(1.0, round_count / IDENTITY_MATURITY_ROUND_DENOM)
    exp_component = min(1.0, experience_count / IDENTITY_MATURITY_EXPERIENCE_DENOM)
    refl_component = min(1.0, reflection_count / IDENTITY_MATURITY_REFLECTION_DENOM)

    maturity = (
        round_component * IDENTITY_MATURITY_ROUND_WEIGHT
        + exp_component * IDENTITY_MATURITY_EXPERIENCE_WEIGHT
        + refl_component * IDENTITY_MATURITY_REFLECTION_WEIGHT
    )

    return max(IDENTITY_MATURITY_MIN, min(IDENTITY_MATURITY_MAX, maturity))
