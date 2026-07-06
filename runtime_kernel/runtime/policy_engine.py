"""
policy_engine — Causal Policy Evolution Loop.

Safe, traceable policy learning without self-modifying code.

How it works:
    Every completed step feeds outcome data into the PolicyEngine.
    The PolicyEngine analyzes patterns (success rates, failure modes,
    efficiency) and produces policy biases as structured DATA.

    These biases are injected into the Planner's prompt as context —
    the Planner sees them and may adjust its decisions accordingly.

    The LLM NEVER modifies its own prompt or system instructions.
    Policy biases are data, not code. The causal chain is preserved.

Architecture:
    OutcomeEvaluator.evaluate(step_result)
        ↓
    PolicyEngine.update(evaluations)
        ↓
    policy_biases (stored on session)
        ↓
    format_for_prompt() → injected as 【策略偏好】context
        ↓
    Planner sees biases and may adjust actions
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional


# ── Default biases ──

DEFAULT_POLICY_BIASES: dict[str, Any] = {
    "preferred_capabilities": [],
    "avoided_operations": [],
    "action_success_rates": {},
    "operation_failure_count": {},
    "total_actions": 0,
    "successful_actions": 0,
    "failed_actions": 0,
    "efficiency_score": 0.5,
    "total_elapsed_ms": 0,
    "consecutive_failures": 0,
}


# ── Outcome Evaluation ──


class OutcomeEvaluator:
    """Evaluates completed actions and produces structured outcome data.

    This is a pure-function evaluator. No LLM calls.
    It analyzes:
        - Did the action succeed?
        - How long did it take? (efficiency)
        - Was it a repeated failure?
        - Did it achieve its apparent goal?
    """

    @staticmethod
    def evaluate_action(
        capability: str,
        operation: str,
        success: bool,
        elapsed_ms: int = 0,
        error: str = "",
    ) -> dict:
        """Evaluate a single completed action.

        Returns a structured evaluation dict.
        """
        return {
            "capability": capability,
            "operation": operation,
            "success": success,
            "elapsed_ms": elapsed_ms,
            "error": error[:100] if error else "",
            "efficiency": OutcomeEvaluator._score_efficiency(success, elapsed_ms),
        }

    @staticmethod
    def evaluate_step(action_results: list[dict]) -> list[dict]:
        """Evaluate all actions in a completed step.

        Args:
            action_results: List of action result dicts with
                capability, operation, success, elapsed_ms, error.

        Returns list of evaluation dicts.
        """
        return [
            OutcomeEvaluator.evaluate_action(
                capability=a.get("capability", "?"),
                operation=a.get("operation", "?"),
                success=a.get("success", False),
                elapsed_ms=a.get("elapsed_ms", 0),
                error=a.get("error", ""),
            )
            for a in action_results
        ]

    @staticmethod
    def _score_efficiency(success: bool, elapsed_ms: int) -> float:
        """Score efficiency on a 0.0-1.0 scale.

        Fast successes → high score.
        Slow or failed → low score.
        """
        if not success:
            return 0.0
        if elapsed_ms <= 0:
            return 0.5
        # Under 500ms → 1.0, over 10s → 0.1
        score = max(0.1, min(1.0, 1.0 - (elapsed_ms / 10000)))
        return round(score, 2)


# ── Policy Engine ──


class PolicyEngine:
    """Causal policy evolution engine.

    Accumulates outcome evaluations and produces policy biases
    that influence future Planner decisions.

    This is NOT self-modifying code. This is data-driven preference
    learning. The biases are stored as structured data and injected
    into prompts as context for the Planner to use or ignore.
    """

    def __init__(self) -> None:
        self._biases: dict[str, Any] = dict(DEFAULT_POLICY_BIASES)

    # ── Public API ──

    def get_biases(self) -> dict:
        """Return current policy biases (as a copy)."""
        return dict(self._biases)

    def reset(self) -> None:
        """Reset biases to defaults."""
        self._biases = dict(DEFAULT_POLICY_BIASES)

    def update_from_step(self, action_results: list[dict]) -> dict:
        """Update policy biases based on a completed step's action results.

        Called after every step that had capability actions.
        This is the main learning entry point.

        Args:
            action_results: List of completed action result dicts with
                capability, operation, success, elapsed_ms, error.

        Returns the updated biases dict.
        """
        if not action_results:
            return self._biases

        # Evaluate each action
        evaluations = OutcomeEvaluator.evaluate_step(action_results)

        # Update counters
        for ev in evaluations:
            cap = ev["capability"]
            op = ev["operation"]
            success = ev["success"]

            # Total counts
            self._biases["total_actions"] += 1
            if success:
                self._biases["successful_actions"] += 1
                self._biases["consecutive_failures"] = 0
            else:
                self._biases["failed_actions"] += 1
                self._biases["consecutive_failures"] += 1

            # Per-capability success rate
            rates = self._biases["action_success_rates"]
            if cap not in rates:
                rates[cap] = {"total": 0, "success": 0}
            rates[cap]["total"] += 1
            if success:
                rates[cap]["success"] += 1

            # Operation failure count
            if not success:
                fail_count = self._biases["operation_failure_count"]
                fail_count[op] = fail_count.get(op, 0) + 1

            # Elapsed time
            ms = ev.get("elapsed_ms", 0)
            self._biases["total_elapsed_ms"] += ms

        # Derive preferred capabilities
        self._update_preferred_capabilities()

        # Derive avoided operations
        self._update_avoided_operations()

        # Update efficiency score
        self._update_efficiency()

        return self._biases

    # ── Internal update rules ──

    def _update_preferred_capabilities(self) -> None:
        """Update capability preferences based on success rates.

        A capability is "preferred" if its success rate > 0.6
        and it has been used at least 2 times.
        """
        rates = self._biases["action_success_rates"]
        preferred = []
        for cap, data in rates.items():
            total = data["total"]
            success = data["success"]
            if total >= 2 and (success / total) > 0.6:
                preferred.append(cap)
        # Sort by success rate descending
        preferred.sort(
            key=lambda c: (
                rates[c]["success"] / rates[c]["total"]
                if rates[c]["total"] > 0 else 0
            ),
            reverse=True,
        )
        self._biases["preferred_capabilities"] = preferred

    def _update_avoided_operations(self) -> None:
        """Update avoided operations list.

        An operation is "avoided" if it has failed more than once
        consecutively or has a failure rate > 0.5.
        """
        fail_count = self._biases["operation_failure_count"]
        avoided = [
            op for op, count in fail_count.items()
            if count >= 2
        ]
        self._biases["avoided_operations"] = avoided

    def _update_efficiency(self) -> None:
        """Update overall efficiency score.

        Based on success rate and average elapsed time.
        """
        total = self._biases["total_actions"]
        if total == 0:
            self._biases["efficiency_score"] = 0.5
            return

        success_rate = self._biases["successful_actions"] / total

        avg_ms = (
            self._biases["total_elapsed_ms"] / total
            if total > 0 else 0
        )
        time_score = max(0.1, min(1.0, 1.0 - (avg_ms / 10000)))

        # Blend success rate (weight 0.7) and time score (weight 0.3)
        efficiency = round(success_rate * 0.7 + time_score * 0.3, 2)
        self._biases["efficiency_score"] = efficiency

    # ── Prompt integration ──

    def format_for_prompt(self) -> str:
        """Format policy biases as a context string for Planner prompts.

        Returns a formatted string or empty string if no significant data.
        """
        b = self._biases
        if b["total_actions"] == 0:
            return ""

        parts: list[str] = ["【策略偏好】"]

        # Success rate
        total = b["total_actions"]
        success = b["successful_actions"]
        rate = (success / total * 100) if total > 0 else 0
        parts.append(f"  历史行动成功率: {rate:.0f}% ({success}/{total})")

        # Efficiency
        parts.append(f"  效率评分: {b['efficiency_score']:.2f}")

        # Preferred capabilities
        if b["preferred_capabilities"]:
            parts.append(
                f"  偏爱能力: {' > '.join(b['preferred_capabilities'])}"
            )

        # Avoided operations
        if b["avoided_operations"]:
            parts.append(
                f"  避免操作: {', '.join(b['avoided_operations'])}"
            )

        # Consecutive failures warning
        if b["consecutive_failures"] >= 2:
            parts.append(
                f"  ⚠ 连续 {b['consecutive_failures']} 次失败，建议更换策略"
            )

        parts.append(
            "  以上是基于历史数据的策略偏好。"
            "你可以参考它们来优化决策，但不必严格遵循。"
        )

        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Serialize biases for API display."""
        return dict(self._biases)
