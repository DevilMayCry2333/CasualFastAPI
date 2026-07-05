"""WorkingMemory — the agent's current cognitive focus.

Working memory is NOT chat history. It's what the agent is
currently thinking about: the active question, relevant evidence,
unresolved issues, and current reasoning context.

Working memory is limited and ephemeral. It's rebuilt each step
from attention output and current cognitive models.
"""

from __future__ import annotations

from typing import Any, Optional


class WorkingMemory:
    """The agent's current cognitive focus.

    Fields:
        current_focus: What the agent is actively thinking about.
        active_question: The primary question being explored.
        relevant_evidence: Evidence relevant to the current focus.
        unresolved: Unresolved issues or contradictions.
        recent_reasoning: Recent reasoning steps.
        context_tags: Tags describing the current cognitive context.
    """

    def __init__(self) -> None:
        self._current_focus: str = ""
        self._active_question: str = ""
        self._relevant_evidence: list[str] = []
        self._unresolved: list[str] = []
        self._recent_reasoning: list[str] = []
        self._context_tags: list[str] = []

    @property
    def current_focus(self) -> str:
        return self._current_focus

    @property
    def active_question(self) -> str:
        return self._active_question

    @property
    def relevant_evidence(self) -> list[str]:
        return list(self._relevant_evidence)

    @property
    def unresolved(self) -> list[str]:
        return list(self._unresolved)

    @property
    def recent_reasoning(self) -> list[str]:
        return list(self._recent_reasoning)

    def reset(self) -> None:
        """Clear working memory for a new step."""
        self._current_focus = ""
        self._active_question = ""
        self._relevant_evidence = []
        self._unresolved = []

    def focus_on(self, topic: str, question: str = "") -> None:
        """Set the current cognitive focus."""
        self._current_focus = topic
        if question:
            self._active_question = question

    def add_evidence(self, evidence: str) -> None:
        if evidence not in self._relevant_evidence:
            self._relevant_evidence.append(evidence)
            if len(self._relevant_evidence) > 5:
                self._relevant_evidence = self._relevant_evidence[-5:]

    def add_unresolved(self, issue: str) -> None:
        if issue not in self._unresolved:
            self._unresolved.append(issue)
            if len(self._unresolved) > 3:
                self._unresolved = self._unresolved[-3:]

    def add_reasoning(self, step: str) -> None:
        self._recent_reasoning.append(step)
        if len(self._recent_reasoning) > 5:
            self._recent_reasoning = self._recent_reasoning[-5:]

    def build_from_models(
        self,
        topic: str,
        belief: str,
        goal: str,
        open_questions: list[str],
        contradictions: list[tuple],
    ) -> None:
        """Build working memory from cognitive models and state.

        Called at the start of each step to set the cognitive focus.
        """
        self._current_focus = topic
        self._active_question = goal
        self._relevant_evidence = []

        # Convert open questions to unresolved items
        if open_questions:
            self._unresolved = list(open_questions[:3])

        # Contradictions are always unresolved
        for a, b in contradictions[:2]:
            issue = f"{a.statement[:40]} vs {b.statement[:40]}"
            if issue not in self._unresolved:
                self._unresolved.append(issue)

    def format_for_prompt(self) -> str:
        parts = ["【工作记忆】"]
        if self._current_focus:
            parts.append(f"  当前关注: {self._current_focus[:60]}")
        if self._active_question:
            parts.append(f"  正在探索: {self._active_question[:60]}")
        if self._relevant_evidence:
            parts.append("  相关证据:")
            for e in self._relevant_evidence:
                parts.append(f"    · {e[:60]}")
        if self._unresolved:
            parts.append("  未解决:")
            for u in self._unresolved:
                parts.append(f"    · {u[:60]}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "current_focus": self._current_focus,
            "active_question": self._active_question,
            "relevant_evidence": list(self._relevant_evidence),
            "unresolved": list(self._unresolved),
            "recent_reasoning": list(self._recent_reasoning),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkingMemory":
        wm = cls()
        wm._current_focus = d.get("current_focus", "")
        wm._active_question = d.get("active_question", "")
        wm._relevant_evidence = list(d.get("relevant_evidence", []))
        wm._unresolved = list(d.get("unresolved", []))
        wm._recent_reasoning = list(d.get("recent_reasoning", []))
        return wm
