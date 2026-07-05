"""KnowledgeModel — the agent's structured knowledge.

Maintains facts, hypotheses, evidence, contradictions,
open questions, and experiments. This is the agent's
epistemic model — what it knows and how confident it is.

Knowledge evolves through the evidence → hypothesis → belief pipeline,
never through direct prompt injection.
"""

from __future__ import annotations

from typing import Any, Optional

from runtime_kernel.runtime.hypothesis import Hypothesis, HypothesisManager
from runtime_kernel.runtime.evidence import Evidence, EvidenceManager


class KnowledgeModel:
    """The agent's knowledge model.

    Wraps HypothesisManager and EvidenceManager, adding
    high-level knowledge structure: facts, contradictions,
    open questions, and experiments.
    """

    def __init__(
        self,
        hypothesis_manager: Optional[HypothesisManager] = None,
        evidence_manager: Optional[EvidenceManager] = None,
    ) -> None:
        self._hypotheses: HypothesisManager = hypothesis_manager or HypothesisManager()
        self._evidence: EvidenceManager = evidence_manager or EvidenceManager()
        self._facts: dict[str, float] = {}  # statement -> confidence
        self._open_questions: list[str] = []
        self._experiments: list[dict] = []

    @property
    def hypotheses(self) -> HypothesisManager:
        return self._hypotheses

    @property
    def evidence(self) -> EvidenceManager:
        return self._evidence

    @property
    def facts(self) -> dict[str, float]:
        return dict(self._facts)

    @property
    def open_questions(self) -> list[str]:
        return list(self._open_questions)

    @property
    def experiments(self) -> list[dict]:
        return list(self._experiments)

    def add_fact(self, statement: str, confidence: float = 0.5) -> None:
        self._facts[statement] = max(0.0, min(1.0, confidence))

    def set_open_questions(self, questions: list[str]) -> None:
        self._open_questions = list(questions)

    def add_experiment(self, experiment: dict) -> None:
        self._experiments.append(experiment)
        if len(self._experiments) > 10:
            self._experiments = self._experiments[-10:]

    def get_contradictions(self) -> list[tuple[Hypothesis, Hypothesis]]:
        return self._hypotheses.get_contradictions()

    def get_uncertain_areas(self) -> list[dict]:
        return self._hypotheses.get_uncertain_areas()

    def get_belief_ready(self) -> list[Hypothesis]:
        return self._hypotheses.get_ready_for_belief()

    def format_for_prompt(self) -> str:
        parts = ["【知识模型】"]

        if self._facts:
            parts.append("  已知事实:")
            for stmt, conf in sorted(self._facts.items(), key=lambda x: -x[1])[:5]:
                parts.append(f"    ✓ {stmt[:60]} (conf={conf:.2f})")

        hyp_text = self._hypotheses.format_for_prompt(max_hypotheses=3)
        if hyp_text:
            parts.append(hyp_text)

        ev_text = self._evidence.format_for_prompt(max_evidence=3)
        if ev_text:
            parts.append(ev_text)

        if self._open_questions:
            parts.append("  未解决问题:")
            for q in self._open_questions[:3]:
                parts.append(f"    ❓ {q[:60]}")

        contradictions = self.get_contradictions()
        if contradictions:
            parts.append("  矛盾:")
            for a, b in contradictions[:2]:
                parts.append(f"    ⚡ {a.statement[:40]} ⇔ {b.statement[:40]}")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "hypotheses": self._hypotheses.to_dict(),
            "evidence": self._evidence.to_dict(),
            "facts": dict(self._facts),
            "open_questions": list(self._open_questions),
            "experiments": list(self._experiments),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeModel":
        m = cls(
            hypothesis_manager=HypothesisManager.from_dict(d.get("hypotheses", [])),
            evidence_manager=EvidenceManager.from_dict(d.get("evidence", [])),
        )
        m._facts = dict(d.get("facts", {}))
        m._open_questions = list(d.get("open_questions", []))
        m._experiments = list(d.get("experiments", []))
        return m
