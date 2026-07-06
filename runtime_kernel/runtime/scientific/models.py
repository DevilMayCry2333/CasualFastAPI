"""
models — Core data structures for the Autonomous Scientific Agent.

All scientific modules use these data types. They are pure data —
no LLM logic, no engine dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ScientificQuestion:
    """A question the agent generates about its world or knowledge.

    Types:
        exploration: "What happens if I do X?"
        debugging:   "Why did this action fail?"
        causal:      "Does X cause Y?"
    """
    question: str
    context: str = ""
    q_type: str = "exploration"  # exploration, debugging, causal
    priority: float = 0.5
    timestamp: float = 0.0
    answered: bool = False

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "context": self.context,
            "type": self.q_type,
            "priority": self.priority,
            "answered": self.answered,
        }


@dataclass
class Hypothesis:
    """A testable hypothesis derived from a question."""
    statement: str
    predicted_outcome: str = ""
    confidence: float = 0.5
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    status: str = "proposed"  # proposed, testing, supported, refuted

    def to_dict(self) -> dict:
        return {
            "statement": self.statement,
            "predicted_outcome": self.predicted_outcome,
            "confidence": self.confidence,
            "status": self.status,
        }


@dataclass
class ExperimentStep:
    """A single step within an experiment."""
    capability: str
    operation: str
    parameters: dict = field(default_factory=dict)
    expected: str = ""

    def to_dict(self) -> dict:
        return {
            "capability": self.capability,
            "operation": self.operation,
            "parameters": self.parameters,
            "expected": self.expected,
        }


@dataclass
class ExperimentResult:
    """Result of executing one experiment step."""
    step: int = 0
    capability: str = ""
    operation: str = ""
    success: bool = False
    observation: str = ""
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "step": self.step, "capability": self.capability,
            "operation": self.operation, "success": self.success,
            "observation": self.observation, "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class CausalEdge:
    """A directed causal relationship between two concepts."""
    source: str
    target: str
    strength: float = 0.5
    evidence_count: int = 0
    mechanism: str = ""     # how this causal link works

    def to_dict(self) -> dict:
        return {
            "source": self.source, "target": self.target,
            "strength": round(self.strength, 2),
            "evidence_count": self.evidence_count,
            "mechanism": self.mechanism,
        }


@dataclass
class CycleSummary:
    """Summary of one complete scientific cycle."""
    cycle: int = 0
    question: Optional[ScientificQuestion] = None
    hypotheses: list[Hypothesis] = field(default_factory=list)
    results: list[ExperimentResult] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    theory_delta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "question": self.question.to_dict() if self.question else None,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "results": [r.to_dict() for r in self.results],
            "insights": self.insights,
            "theory_delta": self.theory_delta,
        }
