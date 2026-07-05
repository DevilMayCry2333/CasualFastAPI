"""TheoryOfMind — the agent's model of what other agents believe.

Not just "agent B exists", but "agent B believes X with confidence Y".
This is distinct from SocialModel (which tracks trust/cooperation).
TheoryOfMind tracks epistemic states: what does each agent know,
what do they believe, and how confident are they?

Theory of mind is built from communication and observation,
never from prompt injection.
"""

from __future__ import annotations

from typing import Any, Optional


class MentalState:
    """What the agent believes about another agent's beliefs."""

    def __init__(self, agent_id: str) -> None:
        self._agent_id: str = agent_id
        self._perceived_beliefs: dict[str, float] = {}
        self._perceived_goals: list[str] = []
        self._perceived_confidence: float = 0.0
        self._perceived_knowledge_domains: list[str] = []
        self._last_observed_round: int = 0

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def perceived_beliefs(self) -> dict[str, float]:
        return dict(self._perceived_beliefs)

    @property
    def perceived_goals(self) -> list[str]:
        return list(self._perceived_goals)

    @property
    def perceived_confidence(self) -> float:
        return self._perceived_confidence

    def observe_belief(self, belief: str, confidence: float = 0.5) -> None:
        """Infer that another agent holds a belief with some confidence."""
        self._perceived_beliefs[belief] = confidence
        if len(self._perceived_beliefs) > 10:
            oldest = min(self._perceived_beliefs, key=lambda k: self._perceived_beliefs[k])
            del self._perceived_beliefs[oldest]

    def observe_goal(self, goal: str) -> None:
        """Infer that another agent has a goal."""
        if goal not in self._perceived_goals:
            self._perceived_goals.append(goal)
            if len(self._perceived_goals) > 5:
                self._perceived_goals = self._perceived_goals[-5:]

    def update_from_message(self, msg_content: str, round_num: int) -> None:
        """Update mental state based on a received message."""
        self._last_observed_round = round_num
        # Extract potential belief statements from message
        if msg_content:
            self._perceived_confidence = min(1.0, self._perceived_confidence + 0.1)

    def format_for_prompt(self) -> str:
        parts = [f"    {self._agent_id[:8]}: "]
        if self._perceived_beliefs:
            top_belief = max(self._perceived_beliefs, key=self._perceived_beliefs.get)
            parts.append(f"相信「{top_belief[:40]}」(conf={self._perceived_beliefs[top_belief]:.2f})")
        else:
            parts.append("信念未知")
        return "".join(parts)

    def to_dict(self) -> dict:
        return {
            "agent_id": self._agent_id,
            "perceived_beliefs": dict(self._perceived_beliefs),
            "perceived_goals": list(self._perceived_goals),
            "perceived_confidence": self._perceived_confidence,
            "last_observed_round": self._last_observed_round,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MentalState":
        ms = cls(d.get("agent_id", ""))
        ms._perceived_beliefs = dict(d.get("perceived_beliefs", {}))
        ms._perceived_goals = list(d.get("perceived_goals", []))
        ms._perceived_confidence = float(d.get("perceived_confidence", 0.0))
        ms._last_observed_round = int(d.get("last_observed_round", 0))
        return ms


class TheoryOfMind:
    """The agent's theory of mind — its model of what others believe.

    Maps agent_id -> MentalState for each known agent.
    """

    def __init__(self) -> None:
        self._mental_states: dict[str, MentalState] = {}

    def get_state(self, agent_id: str) -> Optional[MentalState]:
        return self._mental_states.get(agent_id)

    def get_or_create(self, agent_id: str) -> MentalState:
        if agent_id not in self._mental_states:
            self._mental_states[agent_id] = MentalState(agent_id)
        return self._mental_states[agent_id]

    def infer_from_message(self, from_agent: str, content: str, round_num: int) -> None:
        """Infer mental state from an incoming message."""
        ms = self.get_or_create(from_agent)
        ms.update_from_message(content, round_num)

    def infer_from_action(self, agent_id: str, action: str, round_num: int) -> None:
        """Infer mental state from observing another agent's action."""
        ms = self.get_or_create(agent_id)
        ms._last_observed_round = round_num
        if action:
            ms.observe_goal(action[:60])

    def format_for_prompt(self) -> str:
        if not self._mental_states:
            return ""
        parts = ["【心智理论】"]
        for ms in self._mental_states.values():
            parts.append(ms.format_for_prompt())
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            aid: ms.to_dict() for aid, ms in self._mental_states.items()
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TheoryOfMind":
        tom = cls()
        for aid, ms_data in d.items():
            tom._mental_states[aid] = MentalState.from_dict(ms_data)
        return tom
