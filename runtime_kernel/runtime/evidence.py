"""
evidence — EvidenceManager: all observations become Evidence before
influencing Belief.

Design principle:
    Observation → Evidence (with metadata)
    Evidence → Hypothesis (via support/contradict)
    Hypothesis → Belief (via repeated support)

The LLM can propose evidence but cannot directly set belief.
Evidence is stored with:
    - source: "observation" | "human" | "deduction" | "world_event"
    - statement: what was observed
    - confidence: how reliable the observation is (0.0-1.0)
    - domain: which aspect of the world this relates to
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from runtime_kernel.runtime.models import (
    BELIEF_UPDATE_EVIDENCE_THRESHOLD,
    BELIEF_CONTRADICTION_RATIO,
)


class Evidence:
    """A single piece of evidence — the atomic unit of world knowledge.

    Fields:
        id: Unique identifier.
        statement: What was observed (e.g., "the seed in the garden sprouted").
        source: How obtained (observation, human, deduction, world_event).
        confidence: How reliable (0.0-1.0).
        domain: Which aspect of the world (environment, agent, object, etc.).
        round_num: When it was collected.
        supporting_hypotheses: IDs of hypotheses this evidence supports.
        contradicting_hypotheses: IDs of hypotheses this evidence contradicts.
        raw_context: Optional raw observation text.
    """

    __slots__ = (
        "id", "statement", "source", "confidence", "domain",
        "round_num", "supporting_hypotheses", "contradicting_hypotheses",
        "raw_context", "timestamp",
    )

    def __init__(
        self,
        statement: str,
        source: str = "observation",
        confidence: float = 0.5,
        domain: str = "",
        round_num: int = 0,
        raw_context: str = "",
        evidence_id: Optional[str] = None,
    ) -> None:
        self.id: str = evidence_id or uuid.uuid4().hex[:10]
        self.statement: str = statement
        self.source: str = source
        self.confidence: float = max(0.0, min(1.0, confidence))
        self.domain: str = domain
        self.round_num: int = round_num
        self.supporting_hypotheses: list[str] = []
        self.contradicting_hypotheses: list[str] = []
        self.raw_context: str = raw_context

    def link_to_hypothesis(self, hypothesis_id: str, supports: bool) -> None:
        """Link this evidence to a hypothesis.

        Args:
            hypothesis_id: The hypothesis ID.
            supports: True if this evidence supports, False if contradicts.
        """
        if supports:
            if hypothesis_id not in self.supporting_hypotheses:
                self.supporting_hypotheses.append(hypothesis_id)
        else:
            if hypothesis_id not in self.contradicting_hypotheses:
                self.contradicting_hypotheses.append(hypothesis_id)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "source": self.source,
            "confidence": self.confidence,
            "domain": self.domain,
            "round_num": self.round_num,
            "supporting_hypotheses": self.supporting_hypotheses,
            "contradicting_hypotheses": self.contradicting_hypotheses,
            "raw_context": self.raw_context,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Evidence":
        e = cls(
            statement=d.get("statement", ""),
            source=d.get("source", "observation"),
            confidence=float(d.get("confidence", 0.5)),
            domain=d.get("domain", ""),
            round_num=int(d.get("round_num", 0)),
            raw_context=d.get("raw_context", ""),
            evidence_id=d.get("id"),
        )
        e.supporting_hypotheses = list(d.get("supporting_hypotheses", []))
        e.contradicting_hypotheses = list(d.get("contradicting_hypotheses", []))
        return e

    def __repr__(self) -> str:
        return (
            f"Evidence(id={self.id!r}, source={self.source!r}, "
            f"conf={self.confidence:.2f}, "
            f"sup={len(self.supporting_hypotheses)}, "
            f"con={len(self.contradicting_hypotheses)})"
        )


class EvidenceManager:
    """Manages the evidence collection for one session.

    Provides the OBSERVE → EVIDENCE pipeline:
    1. Every observation becomes Evidence
    2. Evidence links to hypotheses (support/contradict)
    3. Accumulated evidence triggers belief updates

    Usage:
        em = EvidenceManager()
        ev = em.add_observation("the garden has seeds", "world_event",
                                 confidence=0.8, round_num=5)
        em.link_to_hypothesis(ev.id, hyp_id, supports=True)
        ready = em.get_evidence_for_belief_update()
    """

    def __init__(self) -> None:
        self._evidence: dict[str, Evidence] = {}

    @property
    def all_evidence(self) -> list[Evidence]:
        return list(self._evidence.values())

    def get(self, evidence_id: str) -> Optional[Evidence]:
        return self._evidence.get(evidence_id)

    def add_observation(
        self,
        statement: str,
        source: str = "observation",
        confidence: float = 0.5,
        domain: str = "",
        round_num: int = 0,
        raw_context: str = "",
    ) -> Evidence:
        """Record an observation as Evidence.

        Args:
            statement: What was observed.
            source: observation | human | deduction | world_event.
            confidence: 0.0-1.0 reliability.
            domain: World domain this evidence relates to.
            round_num: Current round.
            raw_context: Full raw observation text.

        Returns:
            The new Evidence object.
        """
        ev = Evidence(
            statement=statement,
            source=source,
            confidence=confidence,
            domain=domain,
            round_num=round_num,
            raw_context=raw_context,
        )
        self._evidence[ev.id] = ev
        return ev

    def add_deduction(
        self,
        statement: str,
        confidence: float = 0.4,
        domain: str = "",
        round_num: int = 0,
    ) -> Evidence:
        """Record a deduction (internally generated evidence).

        Deductions have lower default confidence than direct observations.
        """
        return self.add_observation(
            statement=statement,
            source="deduction",
            confidence=confidence,
            domain=domain,
            round_num=round_num,
        )

    def add_human_statement(
        self,
        statement: str,
        confidence: float = 0.7,
        domain: str = "",
        round_num: int = 0,
    ) -> Evidence:
        """Record evidence from a human statement.

        Human statements start with higher confidence.
        """
        return self.add_observation(
            statement=statement,
            source="human",
            confidence=confidence,
            domain=domain,
            round_num=round_num,
        )

    def link_to_hypothesis(
        self,
        evidence_id: str,
        hypothesis_id: str,
        supports: bool,
    ) -> bool:
        """Link an evidence item to a hypothesis.

        Args:
            evidence_id: The evidence ID.
            hypothesis_id: The hypothesis ID.
            supports: True if evidence supports, False if contradicts.

        Returns:
            True if both evidence and hypothesis exist.
        """
        ev = self._evidence.get(evidence_id)
        if not ev:
            return False
        ev.link_to_hypothesis(hypothesis_id, supports)
        return True

    def get_evidence_for_hypothesis(
        self,
        hypothesis_id: str,
    ) -> tuple[list[Evidence], list[Evidence]]:
        """Get all evidence for a hypothesis, split by support/contradict.

        Args:
            hypothesis_id: The hypothesis ID.

        Returns:
            (supporting_evidence, contradicting_evidence) lists.
        """
        supporting = []
        contradicting = []
        for ev in self._evidence.values():
            if hypothesis_id in ev.supporting_hypotheses:
                supporting.append(ev)
            if hypothesis_id in ev.contradicting_hypotheses:
                contradicting.append(ev)
        return supporting, contradicting

    def get_evidence_for_domain(self, domain: str) -> list[Evidence]:
        """Get all evidence for a given world domain."""
        return [
            ev for ev in self._evidence.values()
            if ev.domain == domain
        ]

    def _compute_belief_confidence(self, domain: str) -> float:
        """Compute overall confidence about a domain based on evidence.

        Uses weighted average of evidence confidence, preferring recent
        and human-source evidence.

        Args:
            domain: Domain to evaluate.

        Returns:
            Weighted average confidence (0.0-1.0).
        """
        domain_evidence = [
            ev for ev in self._evidence.values()
            if ev.domain == domain and ev.source == "world_event"
        ]
        if not domain_evidence:
            return 0.0

        weights = []
        confidences = []
        for ev in domain_evidence:
            w = ev.confidence
            if ev.source == "human":
                w *= 1.2
            weights.append(w)
            confidences.append(ev.confidence)

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        return sum(w * c for w, c in zip(weights, confidences)) / total_weight

    def get_domain_confidence(self) -> dict[str, float]:
        """Return confidence levels per domain.

        Returns:
            Dict of {domain: confidence}.
        """
        domains = set(ev.domain for ev in self._evidence.values())
        return {
            d: self._compute_belief_confidence(d)
            for d in domains if d
        }

    def format_for_prompt(self, max_evidence: int = 5) -> str:
        """Format recent evidence for prompt injection.

        Args:
            max_evidence: Max evidence items to show.

        Returns:
            Formatted string, or empty if none.
        """
        if not self._evidence:
            return ""

        # Sort by round desc, take most recent
        sorted_ev = sorted(
            self._evidence.values(),
            key=lambda e: e.round_num,
            reverse=True,
        )

        source_icons = {
            "observation": "👁",
            "human": "🗣",
            "deduction": "🧠",
            "world_event": "🌍",
        }

        lines = ["【当前证据】"]
        for ev in sorted_ev[:max_evidence]:
            icon = source_icons.get(ev.source, "📄")
            lines.append(
                f"  {icon} [{ev.domain or 'general'}] "
                f"{ev.statement[:80]} "
                f"(conf:{ev.confidence:.2f})"
            )

        # Summary
        lines.append(
            f"  — 共 {len(self._evidence)} 条证据, "
            f"{len([e for e in sorted_ev if e.source == 'human'])} 条来自人类"
        )
        return "\n".join(lines)

    def to_dict(self) -> list[dict]:
        return [ev.to_dict() for ev in self._evidence.values()]

    @classmethod
    def from_dict(cls, data: list[dict]) -> "EvidenceManager":
        em = cls()
        for item in data:
            ev = Evidence.from_dict(item)
            em._evidence[ev.id] = ev
        return em

    def __len__(self) -> int:
        return len(self._evidence)
