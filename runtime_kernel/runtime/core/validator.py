"""
validator — ActionValidator: gates all action execution.

The three-layer execution rule:
    - Exploration Layer proposes actions (candidates)
    - Cognitive Layer selects preferred actions
    - **Stable Core Layer validates and executes** (the ONLY layer that runs MCP)

Flow:
    Candidate Action → Core Validator → Safety Check → ActionExecutor → Observation

No learning happens here. This is pure gatekeeping.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

from runtime_kernel.runtime.action import Action, Observation
from runtime_kernel.runtime.core.safety import SafetyRules


class ActionValidator:
    """Pre-execution action validation gate.

    Validates every action against:
        1. Capability exists
        2. Operation is supported
        3. Safety rules pass
        4. Parameters are valid

    Only after ALL checks pass does the action reach ActionExecutor.
    """

    def __init__(
        self,
        action_executor: Any,
        safety_rules: Optional[SafetyRules] = None,
    ) -> None:
        self._executor = action_executor
        self._safety = safety_rules or SafetyRules()
        self._rejected_count: int = 0
        self._approved_count: int = 0

    # ── Core validation method ──

    def validate_and_execute(
        self,
        action: Action,
        session_id: str = "",
    ) -> Observation:
        """Validate an action, then execute if approved.

        This is the ONLY entry point for executing actions.
        Cognitive and Exploration layers MUST go through here.

        Args:
            action: The action to validate and execute.
            session_id: Session ID for rate limiting.

        Returns:
            Observation. If rejected, Observation(success=False, error=reason).
        """
        # Step 1: Check capability exists
        if not self._executor.has_capability(action.capability):
            self._rejected_count += 1
            return Observation(
                success=False,
                error=f"Capability '{action.capability}' not available",
                metadata={"rejected_by": "core_validator", "reason": "unknown_capability"},
            )

        # Step 2: Check operation is supported
        ops = self._executor.get_operations(action.capability)
        op_names = {o.get("name") for o in ops}
        if action.operation not in op_names:
            self._rejected_count += 1
            return Observation(
                success=False,
                error=f"Operation '{action.operation}' not supported by {action.capability}",
                metadata={"rejected_by": "core_validator", "reason": "unknown_operation"},
            )

        # Step 3: Safety check
        safety_result = self._safety.validate_action(
            session_id=session_id,
            capability=action.capability,
            operation=action.operation,
            parameters=action.parameters,
        )
        if not safety_result["approved"]:
            self._rejected_count += 1
            return Observation(
                success=False,
                error=safety_result["reason"],
                metadata={
                    "rejected_by": "core_validator",
                    "reason": "safety",
                    "safety_detail": safety_result["reason"],
                },
            )

        # Step 4: Execute (only Core can reach here)
        self._approved_count += 1
        return self._executor.execute(action, session_id=session_id)

    def validate_only(self, action: Action) -> dict:
        """Validate without executing. Used by Cognitive Layer for planning.

        Returns validation result dict.
        """
        if not self._executor.has_capability(action.capability):
            return {"approved": False, "reason": f"Capability '{action.capability}' not available"}

        ops = self._executor.get_operations(action.capability)
        op_names = {o.get("name") for o in ops}
        if action.operation not in op_names:
            return {"approved": False, "reason": f"Operation '{action.operation}' not supported"}

        return {"approved": True, "reason": ""}

    # ── Status ──

    def get_stats(self) -> dict:
        """Return validator statistics."""
        return {
            "approved_count": self._approved_count,
            "rejected_count": self._rejected_count,
            "total": self._approved_count + self._rejected_count,
            "rejection_rate": round(
                self._rejected_count / max(1, self._approved_count + self._rejected_count),
                3,
            ),
        }

    def get_safety_rules(self) -> dict:
        """Return current safety rules (read-only)."""
        return self._safety.get_status()
