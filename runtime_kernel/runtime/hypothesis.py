"""
hypothesis — HypothesisManager: lifecycle of every hypothesis the agent holds.

Each hypothesis follows a lifecycle:
    proposed → testing → supported | contradicted → revised → discarded

The agent does NOT directly write belief. Belief emerges from hypotheses
that accumulate enough supporting evidence.

Key design principles:
- Multiple concurrent hypotheses are allowed (not just one belief)
- Each hypothesis has confidence, support_count, contradiction_count
- LLM can propose new hypotheses but cannot directly become belief
- Contradictions trigger revision or discard, not denial
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional


# Hypothesis lifecycle status constants
class HypothesisStatus:
    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    REVISED = "revised"
    DISCARDED = "discarded"


class Hypothesis:
    """A single hypothesis with lifecycle tracking.

    Fields:
        id: Unique identifier.
        statement: The hypothesis content (e.g., "this room has a hidden door").
        confidence: 0.0-1.0 — how confident the agent is.
        status: proposed | testing | supported | contradicted | revised | discarded
        support_count: How many evidence items support this.
        contradiction_count: How many evidence items contradict this.
        evidence_ids: IDs of evidence supporting or contradicting.
        created_round: When this hypothesis was first proposed.
        last_updated_round: When it was last modified.
        source: How it was generated (observation, deduction, human, llm).
    """

    __slots__ = (
        "id", "statement", "confidence", "status",
        "support_count", "contradiction_count",
        "evidence_ids", "created_round", "last_updated_round", "source",
        "domain",  # optional: what domain this applies to
    )

    def __init__(
        self,
        statement: str,
        source: str = "observation",
        created_round: int = 0,
        domain: str = "",
        hypothesis_id: Optional[str] = None,
    ) -> None:
        self.id: str = hypothesis_id or uuid.uuid4().hex[:10]
        self.statement: str = statement
        self.confidence: float = 0.1  # start low
        self.status: str = "proposed"
        self.support_count: int = 0
        self.contradiction_count: int = 0
        self.evidence_ids: list[str] = []
        self.created_round: int = created_round
        self.last_updated_round: int = created_round
        self.source: str = source
        self.domain: str = domain

    def add_support(self, evidence_id: str, round_num: int) -> None:
        """Add a supporting evidence item.

        Updates confidence and may promote status to 'supported'.
        """
        if evidence_id not in self.evidence_ids:
            self.evidence_ids.append(evidence_id)
            self.support_count += 1
            self.last_updated_round = round_num
            self._recompute_confidence()

            # Status promotion: proposed → testing at 1 support, testing → supported at 3+
            if self.status == "proposed" and self.support_count >= 1:
                self.status = "testing"
            if self.status == "testing" and self.support_count >= 3:
                self.status = "supported"

    def add_contradiction(self, evidence_id: str, round_num: int) -> None:
        """Add a contradicting evidence item.

        Decreases confidence and may demote status to 'contradicted' or 'discarded'.
        """
        if evidence_id not in self.evidence_ids:
            self.evidence_ids.append(evidence_id)
            self.contradiction_count += 1
            self.last_updated_round = round_num
            self._recompute_confidence()

            # If contradictions form a significant fraction of total evidence, mark contradicted
            total = self.support_count + self.contradiction_count
            if total > 0 and self.contradiction_count >= 2:
                contradiction_ratio = self.contradiction_count / total
                if contradiction_ratio >= 0.3:  # 30%+ contradictory evidence
                    self.status = "contradicted"

    def revise(self, new_statement: str, round_num: int) -> None:
        """Revise the hypothesis statement after contradiction.

        Resets counters but keeps the hypothesis alive.
        """
        self.statement = new_statement
        self.status = "revised"
        self.support_count = 0
        self.contradiction_count = 0
        self.evidence_ids = []
        self.confidence = 0.15  # slightly above initial after revision
        self.last_updated_round = round_num

    def discard(self, round_num: int) -> None:
        """Mark this hypothesis as discarded (dead end)."""
        self.status = "discarded"
        self.confidence = 0.0
        self.last_updated_round = round_num

    def _recompute_confidence(self) -> None:
        """Recompute confidence from support/contradiction balance.

        Formula: confidence = sigmoid(support - contradiction * 2)
        Scaled to 0.0-1.0 range.
        """
        net = self.support_count - self.contradiction_count * 2
        # Simple sigmoid-like: clamp between 0 and 1
        raw = 1.0 / (1.0 + pow(2.718, -net * 0.8))
        self.confidence = max(0.0, min(1.0, raw))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "confidence": self.confidence,
            "status": self.status,
            "support_count": self.support_count,
            "contradiction_count": self.contradiction_count,
            "evidence_ids": self.evidence_ids,
            "created_round": self.created_round,
            "last_updated_round": self.last_updated_round,
            "source": self.source,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        h = cls(
            statement=d.get("statement", ""),
            source=d.get("source", "observation"),
            created_round=d.get("created_round", 0),
            domain=d.get("domain", ""),
            hypothesis_id=d.get("id"),
        )
        h.confidence = float(d.get("confidence", 0.1))
        h.status = d.get("status", "proposed")
        h.support_count = int(d.get("support_count", 0))
        h.contradiction_count = int(d.get("contradiction_count", 0))
        h.evidence_ids = list(d.get("evidence_ids", []))
        h.last_updated_round = d.get("last_updated_round", h.created_round)
        return h

    def __repr__(self) -> str:
        return (
            f"Hypothesis(id={self.id!r}, status={self.status!r}, "
            f"confidence={self.confidence:.2f}, "
            f"support={self.support_count}, contra={self.contradiction_count})"
        )


class HypothesisManager:
    """Manages the lifecycle of all hypotheses for one session.

    Provides the PROPOSAL → SUPPORT → CONTRADICT → REVISE pipeline
    that replaces direct belief-writing by the LLM.

    Usage:
        hm = HypothesisManager()
        hm.propose("the garden has hidden seeds", "observation", round_num=5)
        hm.add_evidence_to_hypothesis(hyp_id, ev_id, supports=True, round_num=6)
        ready = hm.get_ready_for_belief()  # hypotheses ready to become belief
        stale = hm.get_stale()  # hypotheses needing revision or discard
    """

    def __init__(self) -> None:
        self._hypotheses: dict[str, Hypothesis] = {}

    @property
    def hypotheses(self) -> list[Hypothesis]:
        return list(self._hypotheses.values())

    @property
    def active_hypotheses(self) -> list[Hypothesis]:
        """Return hypotheses that are still under consideration."""
        return [
            h for h in self._hypotheses.values()
            if h.status not in ("discarded",)
        ]

    def get(self, hypothesis_id: str) -> Optional[Hypothesis]:
        return self._hypotheses.get(hypothesis_id)

    def propose(
        self,
        statement: str,
        source: str = "observation",
        round_num: int = 0,
        domain: str = "",
    ) -> Hypothesis:
        """Propose a new hypothesis and add it to the managed set.

        Args:
            statement: The hypothesis text.
            source: How it was generated (observation, deduction, human, llm).
            round_num: Current round number.
            domain: Optional domain tag.

        Returns:
            The new Hypothesis object.
        """
        h = Hypothesis(
            statement=statement,
            source=source,
            created_round=round_num,
            domain=domain,
        )
        self._hypotheses[h.id] = h
        return h

    def add_evidence(
        self,
        hypothesis_id: str,
        evidence_id: str,
        supports: bool,
        round_num: int,
    ) -> bool:
        """Add evidence to a hypothesis.

        Args:
            hypothesis_id: Target hypothesis.
            evidence_id: Evidence item ID.
            supports: True if evidence supports, False if contradicts.
            round_num: Current round number.

        Returns:
            True if the hypothesis was found and updated.
        """
        h = self._hypotheses.get(hypothesis_id)
        if not h:
            return False

        if supports:
            h.add_support(evidence_id, round_num)
        else:
            h.add_contradiction(evidence_id, round_num)
        return True

    def revise(
        self,
        hypothesis_id: str,
        new_statement: str,
        round_num: int,
    ) -> bool:
        """Revise a hypothesis after contradiction.

        Args:
            hypothesis_id: Target hypothesis.
            new_statement: Revised statement.
            round_num: Current round number.

        Returns:
            True if found and revised.
        """
        h = self._hypotheses.get(hypothesis_id)
        if not h:
            return False
        h.revise(new_statement, round_num)
        return True

    def discard(self, hypothesis_id: str, round_num: int) -> bool:
        """Discard a hypothesis permanently.

        Args:
            hypothesis_id: Target hypothesis.
            round_num: Current round number.

        Returns:
            True if found and discarded.
        """
        h = self._hypotheses.get(hypothesis_id)
        if not h:
            return False
        h.discard(round_num)
        return True

    def get_ready_for_belief(self) -> list[Hypothesis]:
        """Return hypotheses that have enough support to influence belief.

        A hypothesis is 'belief-ready' when:
        - Status is 'supported'
        - Confidence > 0.6
        - At least 3 supporting evidence items
        - Contradiction ratio < 0.5
        """
        ready = []
        for h in self._hypotheses.values():
            if h.status != "supported":
                continue
            if h.confidence < 0.6:
                continue
            if h.support_count < 3:
                continue
            total = h.support_count + h.contradiction_count
            if total > 0 and (h.contradiction_count / total) >= 0.5:
                continue
            ready.append(h)
        return ready

    def get_contradictions(self) -> list[tuple[Hypothesis, Hypothesis]]:
        """Find pairs of active hypotheses that contradict each other.

        Simple heuristic: two hypotheses in the same domain with
        opposite-supporting evidence patterns.

        Returns:
            List of (hyp_a, hyp_b) contradictory pairs.
        """
        active = self.active_hypotheses
        contradictions = []
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                # Same domain, both have evidence, but opposite trends
                if a.domain and b.domain and a.domain == b.domain:
                    if (
                        a.support_count > 0 and b.contradiction_count > 0
                    ) or (
                        a.contradiction_count > 0 and b.support_count > 0
                    ):
                        contradictions.append((a, b))
        return contradictions

    def get_uncertain_areas(self) -> list[dict]:
        """Return areas where the agent has low confidence.

        Used to populate the uncertainties list in state.

        Returns:
            List of {domain, hypothesis_count, avg_confidence}.
        """
        if not self._hypotheses:
            return []

        domains: dict[str, list[float]] = {}
        for h in self._hypotheses.values():
            if h.status == "discarded":
                continue
            dom = h.domain or "general"
            if dom not in domains:
                domains[dom] = []
            domains[dom].append(h.confidence)

        areas = []
        for domain, confs in domains.items():
            avg_conf = sum(confs) / len(confs)
            if avg_conf < 0.6:
                areas.append({
                    "domain": domain,
                    "hypothesis_count": len(confs),
                    "avg_confidence": round(avg_conf, 2),
                })
        return sorted(areas, key=lambda x: x["avg_confidence"])

    def get_stale(self, current_round: int, max_age: int = 50) -> list[Hypothesis]:
        """Return hypotheses that haven't been updated in max_age rounds.

        These need the agent's attention — either revise or discard.
        """
        stale = []
        for h in self._hypotheses.values():
            if h.status in ("discarded", "supported"):
                continue
            age = current_round - h.last_updated_round
            if age >= max_age:
                stale.append(h)
        return stale

    def format_for_prompt(self, max_hypotheses: int = 3) -> str:
        """Format active hypotheses for prompt injection.

        Args:
            max_hypotheses: Max hypotheses to show (sorted by confidence).

        Returns:
            Formatted string, or empty if none.
        """
        active = sorted(
            self.active_hypotheses,
            key=lambda h: h.confidence,
            reverse=True,
        )
        if not active:
            return ""

        lines = ["【当前假设】"]
        for i, h in enumerate(active[:max_hypotheses], 1):
            status_icon = {
                "proposed": "🆕",
                "testing": "🔬",
                "supported": "✅",
                "contradicted": "❌",
                "revised": "🔄",
            }.get(h.status, "❓")
            lines.append(
                f"  [{i}] {status_icon} {h.statement[:80]} "
                f"(conf:{h.confidence:.2f} "
                f"支持:{h.support_count} 反对:{h.contradiction_count})"
            )
        return "\n".join(lines)

    def format_contradictions_for_prompt(self) -> str:
        """Format contradictions for prompt injection."""
        contradictions = self.get_contradictions()
        if not contradictions:
            return ""

        lines = ["【矛盾检测】"]
        for a, b in contradictions:
            lines.append(f"  ⚡ {a.statement[:60]} ⇔ {b.statement[:60]}")
        return "\n".join(lines)

    def to_dict(self) -> list[dict]:
        return [h.to_dict() for h in self._hypotheses.values()]

    @classmethod
    def from_dict(cls, data: list[dict]) -> "HypothesisManager":
        hm = cls()
        for item in data:
            h = Hypothesis.from_dict(item)
            hm._hypotheses[h.id] = h
        return hm

    def __len__(self) -> int:
        return len(self._hypotheses)
