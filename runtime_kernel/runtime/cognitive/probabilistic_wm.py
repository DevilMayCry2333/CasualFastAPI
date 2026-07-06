"""
probabilistic_wm — Probabilistic World Model for the Cognitive Layer.

Unlike the deterministic state.world_model, this tracks:
    - belief distributions (rather than point estimates)
    - uncertainty per concept
    - causal edge confidence
    - tool effectiveness probabilities

All updates are driven by Fold evidence.
"""

from __future__ import annotations

from typing import Any


class ProbabilisticWorldModel:
    """Probabilistic belief tracker.

    Each belief is tracked as (mean, uncertainty, samples).
    Uncertainty decreases with evidence. Confidence is mean * (1 - uncertainty).

    Example:
        beliefs = {
            "Search → success": {"mean": 0.8, "uncertainty": 0.2, "samples": 5},
            "Browser → success": {"mean": 0.6, "uncertainty": 0.4, "samples": 2},
        }
    """

    def __init__(self) -> None:
        self._beliefs: dict[str, dict] = {}

    # ── Belief tracking ──

    def observe(self, concept: str, outcome: float, weight: float = 1.0) -> None:
        """Update a belief with a new observation.

        Args:
            concept: The concept being observed (e.g. "Search → success").
            outcome: 1.0 for success, 0.0 for failure.
            weight: How much this observation counts (default 1.0).
        """
        if concept not in self._beliefs:
            self._beliefs[concept] = {
                "mean": outcome,
                "uncertainty": 1.0,
                "samples": 1,
                "successes": 1 if outcome > 0.5 else 0,
                "failures": 0 if outcome > 0.5 else 1,
            }
            return

        b = self._beliefs[concept]
        # Online mean update with weighted contribution
        old_total = b["samples"]
        new_total = old_total + weight
        b["mean"] = (b["mean"] * old_total + outcome * weight) / new_total
        b["samples"] = int(new_total)
        if outcome > 0.5:
            b["successes"] += 1
        else:
            b["failures"] += 1

        # Uncertainty decreases with more samples
        b["uncertainty"] = max(0.05, 1.0 / (1.0 + b["samples"] * 0.3))

    def get_confidence(self, concept: str) -> float | None:
        """Get confidence (mean * (1 - uncertainty)) for a concept.

        Returns None if concept not tracked.
        """
        b = self._beliefs.get(concept)
        if not b:
            return None
        return round(b["mean"] * (1.0 - b["uncertainty"]), 2)

    def get_all_beliefs(self) -> dict:
        """Return all beliefs (read-only copy)."""
        return {k: dict(v) for k, v in self._beliefs.items()}

    # ── Concept query ──

    def concepts_with_high_uncertainty(self, threshold: float = 0.5) -> list[str]:
        """Return concepts with uncertainty above threshold."""
        return [
            k for k, v in self._beliefs.items()
            if v["uncertainty"] > threshold
        ]

    def concepts_with_low_confidence(self, threshold: float = 0.3) -> list[str]:
        """Return concepts with confidence below threshold."""
        return [
            k for k in self._beliefs
            if (self.get_confidence(k) or 0) < threshold
        ]

    # ── Tool effectiveness ──

    def get_tool_effectiveness(self) -> dict:
        """Return a simplified view of tool success probabilities."""
        result = {}
        for k, v in self._beliefs.items():
            if "→ success" in k or " → " in k:
                conf = self.get_confidence(k)
                if conf is not None:
                    result[k] = conf
        return result

    # ── Serialization ──

    def to_dict(self) -> dict:
        return {
            "beliefs": self.get_all_beliefs(),
            "high_uncertainty": self.concepts_with_high_uncertainty(),
            "tool_effectiveness": self.get_tool_effectiveness(),
        }

    def format_for_prompt(self, max_beliefs: int = 8) -> str:
        """Format probabilistic beliefs for LLM prompt."""
        if not self._beliefs:
            return ""
        lines = ["【概率世界模型】"]
        # Sort by uncertainty descending (show most uncertain first)
        sorted_b = sorted(
            self._beliefs.items(),
            key=lambda x: x[1]["uncertainty"],
            reverse=True,
        )
        for concept, b in sorted_b[:max_beliefs]:
            conf = self.get_confidence(concept)
            unc = b["uncertainty"]
            samples = b["samples"]
            lines.append(
                f"  {concept}: conf={conf:.2f}, "
                f"uncertainty={unc:.2f}, "
                f"samples={samples}"
            )
        return "\n".join(lines)
