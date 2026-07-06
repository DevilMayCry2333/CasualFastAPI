"""
models — Self-Driven Scientific Agent data structures.

Goal: A research goal generated from Fold/uncertainty/failure.
ExperimentEntry: An experiment in the queue, tracking full lifecycle.
SDSAResult: Result of one SDSA cycle.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ResearchGoal:
    """A research goal for the autonomous scientific cycle.

    Generated from Fold analysis, uncertainties, or failures.
    """
    statement: str
    reason: str = ""
    priority: float = 0.5
    information_gain_estimate: float = 0.5
    id: str = ""
    created_at: float = 0.0
    status: str = "open"  # open, in_progress, completed, abandoned

    def __post_init__(self) -> None:
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "reason": self.reason,
            "priority": round(self.priority, 2),
            "information_gain": round(self.information_gain_estimate, 2),
            "status": self.status,
        }


@dataclass
class ExperimentEntry:
    """An experiment in the queue, tracking full lifecycle."""
    goal_id: str
    hypothesis: str
    variants: list[dict] = field(default_factory=list)
    cost_estimate: int = 0       # estimated ms
    expected_information_gain: float = 0.5
    id: str = ""
    status: str = "queued"  # queued, running, completed, failed
    results: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = uuid.uuid4().hex[:12]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "hypothesis": self.hypothesis,
            "variants": self.variants,
            "cost_estimate_ms": self.cost_estimate,
            "expected_information_gain": round(self.expected_information_gain, 2),
            "status": self.status,
            "results": self.results,
        }


@dataclass
class SDSACycleResult:
    """Result of one SDSA daemon cycle."""
    cycle: int = 0
    goal: Optional[ResearchGoal] = None
    experiments_run: int = 0
    actions_executed: int = 0
    causal_updates: int = 0
    world_model_updates: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "goal": self.goal.to_dict() if self.goal else None,
            "experiments_run": self.experiments_run,
            "actions_executed": self.actions_executed,
            "causal_updates": self.causal_updates,
            "world_model_updates": self.world_model_updates,
            "errors": self.errors,
        }
