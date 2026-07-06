"""
experiment_scheduler — Experiment scheduling with A/B testing support.

Schedules and tracks experiments (controlled comparisons).
Experiments can have multiple variants compared against each other.

Supports:
    - A/B testing (two variants)
    - Multi-variant (A/B/C...)
    - Cost estimation before execution
    - Failure logging
    - Result comparison
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Optional


class Experiment:
    """A single experiment with variants."""

    def __init__(
        self,
        hypothesis_id: str,
        variants: list[dict],
        expected_outcomes: Optional[list[str]] = None,
    ):
        self.id: str = uuid.uuid4().hex[:12]
        self.hypothesis_id: str = hypothesis_id
        self.variants: list[dict] = variants
        self.expected_outcomes: list[str] = expected_outcomes or []
        self.status: str = "designed"  # designed, running, completed, failed
        self.results: list[dict] = []
        self.created_at: float = time.time()
        self.completed_at: Optional[float] = None
        self.conclusion: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "hypothesis_id": self.hypothesis_id,
            "variants": self.variants,
            "expected_outcomes": self.expected_outcomes,
            "status": self.status,
            "results": self.results,
            "conclusion": self.conclusion,
        }


class ExperimentScheduler:
    """Manages experiment lifecycle: design → schedule → track → analyze.

    Does NOT execute actions — that's the Core Layer's job.
    The scheduler only tracks what experiments exist and compares results.
    """

    def __init__(self) -> None:
        self._experiments: dict[str, Experiment] = {}
        self._completed_count: int = 0

    # ── Design ──

    def create_experiment(
        self,
        hypothesis_id: str,
        variants: list[dict],
        expected_outcomes: Optional[list[str]] = None,
    ) -> Experiment:
        """Create a new experiment.

        Args:
            hypothesis_id: Link to the hypothesis being tested.
            variants: List of action variant dicts, each with:
                {"name": "A", "action": {"capability": ..., "operation": ..., "parameters": ...}}
            expected_outcomes: Optional list of expected results.

        Returns the created Experiment.
        """
        exp = Experiment(hypothesis_id, variants, expected_outcomes)
        self._experiments[exp.id] = exp
        return exp

    # ── Track ──

    def start_experiment(self, exp_id: str) -> None:
        """Mark an experiment as running."""
        exp = self._experiments.get(exp_id)
        if exp:
            exp.status = "running"

    def record_result(self, exp_id: str, variant_name: str, observation: dict) -> None:
        """Record a single variant result."""
        exp = self._experiments.get(exp_id)
        if not exp:
            return
        exp.results.append({
            "variant": variant_name,
            "observation": observation,
            "timestamp": time.time(),
        })

    def complete_experiment(self, exp_id: str, conclusion: str = "") -> None:
        """Mark an experiment as completed."""
        exp = self._experiments.get(exp_id)
        if not exp:
            return
        exp.status = "completed"
        exp.completed_at = time.time()
        exp.conclusion = conclusion
        self._completed_count += 1

    # ── Cost estimation ──

    def estimate_cost(self, variants: list[dict]) -> dict:
        """Estimate cost of running an experiment.

        Pure estimation — no execution.
        Returns dict with estimated steps, time, and tool calls.
        """
        total_steps = sum(
            len(v.get("action", {}).get("parameters", {}))
            for v in variants
        ) or len(variants)
        return {
            "estimated_steps": total_steps,
            "estimated_time_ms": total_steps * 1000,  # rough estimate
            "variant_count": len(variants),
            "note": "Pre-execution estimate only",
        }

    # ── Result comparison ──

    def compare_variants(self, exp_id: str) -> dict:
        """Compare results across variants.

        Returns which variant performed best based on success rate.
        """
        exp = self._experiments.get(exp_id)
        if not exp or not exp.results:
            return {"error": "No results to compare"}

        variant_scores: dict[str, list[bool]] = {}
        for r in exp.results:
            v = r.get("variant", "?")
            obs = r.get("observation", {})
            success = obs.get("success", False) if isinstance(obs, dict) else False
            if v not in variant_scores:
                variant_scores[v] = []
            variant_scores[v].append(success)

        best_variant = max(
            variant_scores,
            key=lambda v: sum(variant_scores[v]) / max(len(variant_scores[v]), 1),
        )

        return {
            "experiment_id": exp_id,
            "variant_scores": {
                v: {
                    "success_rate": round(sum(scores) / max(len(scores), 1), 2),
                    "total": len(scores),
                    "successes": sum(scores),
                }
                for v, scores in variant_scores.items()
            },
            "best_variant": best_variant,
        }

    # ── Status ──

    def list_experiments(self, limit: int = 20) -> list[dict]:
        """List recent experiments."""
        all_exp = sorted(
            self._experiments.values(),
            key=lambda e: e.created_at,
            reverse=True,
        )
        return [e.to_dict() for e in all_exp[:limit]]

    def get_stats(self) -> dict:
        """Return scheduler statistics."""
        total = len(self._experiments)
        running = sum(1 for e in self._experiments.values() if e.status == "running")
        completed = sum(1 for e in self._experiments.values() if e.status == "completed")
        return {
            "total": total,
            "running": running,
            "completed": completed,
            "designed": total - running - completed,
        }
