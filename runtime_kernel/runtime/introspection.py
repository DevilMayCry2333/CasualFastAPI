"""
introspection — Meta-cognitive self-analysis.

Introspection reviews the recent state evolution history and produces
a natural language summary that captures:
  - Overall trajectory
  - Key transitions or turning points
  - Significance of the current state

The summary is injected into the prompt context so the agent can
adjust its exploration direction when it detects stagnation or loops.
"""

from __future__ import annotations

import json
from typing import Optional

from runtime_kernel.runtime.llm import LLMClient
from runtime_kernel.runtime.prompt import PromptBuilder
from runtime_kernel.runtime.session import AgentSession


INTROSPECTION_INTERVAL = 20  # rounds between introspection
INTROSPECTION_WINDOW = 20  # how many recent history entries to analyze
INTROSPECTION_INJECTION_WINDOW = 10  # only inject introspections newer than this


class Introspector:
    """Performs meta-cognitive analysis on session state evolution."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def should_introspect(self, session: AgentSession) -> bool:
        """Check whether introspection is due."""
        return (
            session.round > 0
            and session.round % INTROSPECTION_INTERVAL == 0
            and len(session.history) >= 3
        )

    def introspect(self, session: AgentSession) -> Optional[str]:
        """Run introspection on the session's recent history.

        Args:
            session: The session to analyze.

        Returns:
            The summary string, or None if introspection failed.
        """
        recent = session.history[-INTROSPECTION_WINDOW:]
        if len(recent) < 3:
            return None

        messages = PromptBuilder.build_introspection(recent)
        summary = self._llm.complete(
            messages,
            temperature=0.7,
            max_tokens=400,
        )
        return summary or None

    @staticmethod
    def should_inject(session: AgentSession) -> bool:
        """Check if the most recent introspection should be injected into prompt.

        Only inject if:
          - Introspections exist
          - The last one is within the injection window
          - It mentions stagnation or loops
        """
        if not session.introspections:
            return False

        last_intro = session.introspections[-1]
        intro_round = last_intro.get("round", 0)
        rounds_diff = session.round - intro_round

        if not (0 < rounds_diff < INTROSPECTION_INJECTION_WINDOW):
            return False

        summary = last_intro.get("summary", "")
        stagnation_keywords = [
            "泥潭", "停滞", "细节", "琐碎", "循环", "重复",
            "stuck", "detail", "loop", "cycle", "trivial",
        ]
        return any(kw in summary.lower() for kw in stagnation_keywords)

    @staticmethod
    def get_last_introspection(session: AgentSession) -> Optional[dict]:
        """Return the most recent introspection entry, if any."""
        if session.introspections:
            return session.introspections[-1]
        return None
