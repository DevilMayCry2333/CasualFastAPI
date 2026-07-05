"""
memory — Working memory management for the Agent.

Memory in this context refers to the agent's working memory:
the current State plus the recent history that informs its next step.

This module provides:
- Working memory access patterns
- History truncation policies
- Memory compaction strategies

The long-term compressed memory is managed by FoldManager.
"""

from __future__ import annotations

from typing import Any, Optional

from runtime_kernel.runtime.models import HistoryEntry
from runtime_kernel.runtime.state import State


class WorkingMemory:
    """Lightweight wrapper around the agent's working memory context.

    Provides convenient access to current state, recent history,
    and derived context for prompt building.
    """

    MAX_HISTORY_IN_PROMPT = 3  # max recent states to include in context

    def __init__(
        self,
        state: Optional[State] = None,
        history: Optional[list[HistoryEntry]] = None,
    ) -> None:
        self._state = state or State()
        self._history = list(history) if history else []

    @property
    def state(self) -> State:
        return self._state

    @state.setter
    def state(self, new_state: State) -> None:
        self._state = new_state

    @property
    def history(self) -> list[HistoryEntry]:
        return list(self._history)

    def append_history(self, entry: HistoryEntry) -> None:
        self._history.append(entry)

    def recent_states(self, n: int = MAX_HISTORY_IN_PROMPT) -> list[dict]:
        """Return the N most recent state dicts for context."""
        return [
            h.get("state", {})
            for h in self._history[-n:]
        ]

    def recent_transitions(self, n: int = 5) -> list[dict]:
        """Return recent transitions formatted for display."""
        return [
            {
                "round": h.get("round", 0),
                "cause": h.get("cause", "?"),
                "state": h.get("state", {}),
            }
            for h in self._history[-n:]
        ]

    def truncate_history(self, max_len: int = 1000) -> None:
        """Trim history to prevent unbounded growth."""
        if len(self._history) > max_len:
            self._history = self._history[-max_len:]

    def clear(self) -> None:
        """Reset working memory."""
        self._state = State()
        self._history = []

    def to_context(self) -> str:
        """Format working memory as a context string for prompt building."""
        parts = [f"Current state: {self._state.serialize_pretty()}"]
        if self._history:
            recent = self.recent_states(3)
            parts.append("Recent states:")
            for i, s in enumerate(recent):
                parts.append(f"  [{-(i + 1)}] {s}")
        return "\n".join(parts)
