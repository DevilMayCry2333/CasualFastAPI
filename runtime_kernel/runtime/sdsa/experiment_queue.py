"""
experiment_queue — Priority-sorted queue of autonomous experiments.

The queue is the system's "todo list" of experiments to run.
Entries are sorted by expected_information_gain (highest first).

Supports:
    - Priority sorting
    - Deduplication (same goal_id + hypothesis)
    - Cost-aware budgeting
    - Status tracking (queued → running → completed)
"""

from __future__ import annotations

from typing import Any, Optional

from runtime_kernel.runtime.sdsa.models import ExperimentEntry


class ExperimentQueue:
    """Priority queue of experiments for the autonomous loop.

    The daemon loop pops experiments from this queue, executes them
    via the Core Layer, and records results.

    Usage:
        queue = ExperimentQueue()
        entry = queue.enqueue(goal_id="g1", hypothesis="X causes Y", variants=[...])
        next_exp = queue.dequeue()  # highest priority first
        queue.mark_completed(entry.id, results)
    """

    def __init__(self) -> None:
        self._entries: list[ExperimentEntry] = []
        self._completed: list[ExperimentEntry] = []

    # ── Enqueue ──

    def enqueue(
        self,
        goal_id: str,
        hypothesis: str,
        variants: list[dict],
        cost_estimate: int = 0,
        expected_information_gain: float = 0.5,
    ) -> ExperimentEntry:
        """Add an experiment to the queue.

        Deduplicates: if the same goal_id + hypothesis already exists
        as queued, returns the existing entry instead.

        Args:
            goal_id: The research goal this experiment tests.
            hypothesis: The hypothesis being tested.
            variants: Action variant dicts with name, action.
            cost_estimate: Estimated execution time in ms.
            expected_information_gain: 0.0-1.0.

        Returns the (new or existing) ExperimentEntry.
        """
        # Dedup
        existing = self._find_duplicate(goal_id, hypothesis)
        if existing:
            return existing

        entry = ExperimentEntry(
            goal_id=goal_id,
            hypothesis=hypothesis,
            variants=variants,
            cost_estimate=cost_estimate,
            expected_information_gain=expected_information_gain,
        )
        self._entries.append(entry)
        self._sort()
        return entry

    def _find_duplicate(self, goal_id: str, hypothesis: str) -> Optional[ExperimentEntry]:
        """Check if an equivalent experiment is already queued."""
        h_short = hypothesis[:80]
        for e in self._entries:
            if e.status == "queued" and e.goal_id == goal_id and e.hypothesis[:80] == h_short:
                return e
        return None

    # ── Dequeue ──

    def dequeue(self) -> Optional[ExperimentEntry]:
        """Get the highest-priority queued experiment.

        Returns the entry with highest expected_information_gain,
        or None if queue is empty.
        """
        for e in self._entries:
            if e.status == "queued":
                e.status = "running"
                return e
        return None

    def peek(self) -> Optional[ExperimentEntry]:
        """Peek at the next experiment without dequeuing."""
        for e in self._entries:
            if e.status == "queued":
                return e
        return None

    # ── Status updates ──

    def mark_completed(self, entry_id: str, results: list[dict], conclusion: str = "") -> None:
        """Mark an experiment as completed."""
        entry = self._get(entry_id)
        if not entry:
            return
        entry.status = "completed"
        entry.results = results
        self._entries.remove(entry)
        self._completed.append(entry)

    def mark_failed(self, entry_id: str, error: str) -> None:
        """Mark an experiment as failed."""
        entry = self._get(entry_id)
        if not entry:
            return
        entry.status = "failed"
        entry.results = [{"error": error}]
        self._entries.remove(entry)
        self._completed.append(entry)

    # ── Queries ──

    def queued_count(self) -> int:
        return sum(1 for e in self._entries if e.status == "queued")

    def running_count(self) -> int:
        return sum(1 for e in self._entries if e.status == "running")

    def list_queued(self, limit: int = 20) -> list[dict]:
        return [e.to_dict() for e in self._entries[:limit]]

    def list_completed(self, limit: int = 20) -> list[dict]:
        return [e.to_dict() for e in self._completed[-limit:]]

    def get_stats(self) -> dict:
        return {
            "queued": self.queued_count(),
            "running": self.running_count(),
            "completed": len(self._completed),
            "total": len(self._entries) + len(self._completed),
        }

    # ── Helpers ──

    def _sort(self) -> None:
        self._entries.sort(
            key=lambda e: e.expected_information_gain,
            reverse=True,
        )

    def _get(self, entry_id: str) -> Optional[ExperimentEntry]:
        for e in self._entries:
            if e.id == entry_id:
                return e
        for e in self._completed:
            if e.id == entry_id:
                return e
        return None
