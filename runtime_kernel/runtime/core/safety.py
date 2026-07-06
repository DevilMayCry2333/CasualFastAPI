"""
safety — Stable Core safety constraints and blacklist.

No learning. No strategy. Pure constraint enforcement.

Rules:
    - Blacklisted operations are always rejected
    - Rate limits prevent abuse
    - Capability must be registered
    - Operation must be supported by the capability
"""

from __future__ import annotations

import sys
import time
from typing import Any


class SafetyRules:
    """Immutable safety rules. Cannot be modified at runtime.

    Attributes:
        blacklisted_operations: operations never allowed (e.g. "delete_file")
        max_action_rate: max actions per minute per session
        max_action_parameters_size: max size of action parameters dict
    """

    def __init__(
        self,
        blacklisted_operations: tuple[str, ...] = (),
        max_action_rate: int = 30,
        max_action_parameters_size: int = 10,
    ) -> None:
        self._blacklisted = set(blacklisted_operations)
        self._max_rate = max_action_rate
        self._max_params = max_action_parameters_size
        # Per-session rate tracking
        self._action_timestamps: dict[str, list[float]] = {}

    # ── Read-only checks ──

    def is_blacklisted(self, capability: str, operation: str) -> bool:
        """Check if an operation is blacklisted."""
        key = f"{capability}.{operation}"
        return key in self._blacklisted

    def check_rate_limit(self, session_id: str) -> bool:
        """Check if session is within rate limits.

        Returns True if action is allowed, False if rate limited.
        """
        now = time.time()
        window = 60.0  # 1 minute window

        timestamps = self._action_timestamps.get(session_id, [])
        # Prune old entries
        timestamps = [t for t in timestamps if now - t < window]
        self._action_timestamps[session_id] = timestamps

        if len(timestamps) >= self._max_rate:
            print(
                f"  [Core/Safety] ⚠ Rate limit hit for {session_id[:8]} "
                f"({len(timestamps)}/{self._max_rate} per min)",
                file=sys.stderr,
            )
            return False

        timestamps.append(now)
        return True

    def check_parameters_size(self, params: dict) -> bool:
        """Check if action parameters are within size limits."""
        return len(params) <= self._max_params

    # ── Validation summary ──

    def validate_action(
        self,
        session_id: str,
        capability: str,
        operation: str,
        parameters: dict,
    ) -> dict:
        """Run all validation checks. Returns a validation result dict.

        Returns:
            {"approved": True/False, "reason": str}
        """
        # Check blacklist
        if self.is_blacklisted(capability, operation):
            return {
                "approved": False,
                "reason": f"操作被安全规则禁止: {capability}.{operation}",
            }

        # Check rate limit
        if not self.check_rate_limit(session_id):
            return {
                "approved": False,
                "reason": "操作频率过高，请稍后再试",
            }

        # Check parameters size
        if not self.check_parameters_size(parameters):
            return {
                "approved": False,
                "reason": f"参数过多 (上限{self._max_params}个)",
            }

        return {"approved": True, "reason": ""}

    def get_status(self) -> dict:
        """Return safety rules status (read-only)."""
        return {
            "blacklisted_operations": sorted(self._blacklisted),
            "max_action_rate": self._max_rate,
            "max_parameters": self._max_params,
        }
