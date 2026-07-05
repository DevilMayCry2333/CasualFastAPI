"""SocialModel — the agent's model of other agents.

Maintains long-term understanding of each other agent:
trust, cooperation history, beliefs about their beliefs,
reliability, last interaction. Built from actual interactions,
not from prompt engineering.
"""

from __future__ import annotations

from typing import Any, Optional


class AgentSocialProfile:
    """The agent's understanding of one other agent."""

    def __init__(self, agent_id: str) -> None:
        self._agent_id: str = agent_id
        self._trust: float = 0.5
        self._cooperation: str = "unknown"
        self._last_interaction_round: int = 0
        self._belief_about_agent: str = ""
        self._reliability: float = 0.5
        self._interaction_count: int = 0
        self._message_history: list[str] = []

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def trust(self) -> float:
        return self._trust

    @property
    def cooperation(self) -> str:
        return self._cooperation

    @property
    def reliability(self) -> float:
        return self._reliability

    @property
    def interaction_count(self) -> int:
        return self._interaction_count

    def record_interaction(self, round_num: int, message: str = "", cooperative: bool = True) -> None:
        self._last_interaction_round = round_num
        self._interaction_count += 1
        if message:
            self._message_history.append(message)
            if len(self._message_history) > 10:
                self._message_history = self._message_history[-10:]
        # Trust adjusts based on cooperation
        if cooperative:
            self._trust = min(1.0, self._trust + 0.05)
        else:
            self._trust = max(0.0, self._trust - 0.1)
        self._reliability = (self._reliability + (1.0 if cooperative else 0.0)) / 2

    def set_belief_about(self, belief: str) -> None:
        self._belief_about_agent = belief

    def format_for_prompt(self) -> str:
        trust_bar = "█" * int(self._trust * 10) + "░" * (10 - int(self._trust * 10))
        return (f"    {self._agent_id[:8]}: 信任={trust_bar} ({self._trust:.2f}), "
                f"可靠={self._reliability:.2f}, "
                f"互动={self._interaction_count}次, "
                f"合作={self._cooperation}")

    def to_dict(self) -> dict:
        return {
            "agent_id": self._agent_id,
            "trust": round(self._trust, 2),
            "cooperation": self._cooperation,
            "last_interaction_round": self._last_interaction_round,
            "belief_about_agent": self._belief_about_agent,
            "reliability": round(self._reliability, 2),
            "interaction_count": self._interaction_count,
            "message_history": list(self._message_history),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentSocialProfile":
        p = cls(d.get("agent_id", ""))
        p._trust = float(d.get("trust", 0.5))
        p._cooperation = d.get("cooperation", "unknown")
        p._last_interaction_round = int(d.get("last_interaction_round", 0))
        p._belief_about_agent = d.get("belief_about_agent", "")
        p._reliability = float(d.get("reliability", 0.5))
        p._interaction_count = int(d.get("interaction_count", 0))
        p._message_history = list(d.get("message_history", []))
        return p


class SocialModel:
    """The agent's understanding of all other agents.

    Each peer agent gets an AgentSocialProfile that evolves over time
    based on actual interactions, not prompt descriptions.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, AgentSocialProfile] = {}

    @property
    def profiles(self) -> dict[str, AgentSocialProfile]:
        return dict(self._profiles)

    def get_profile(self, agent_id: str) -> Optional[AgentSocialProfile]:
        return self._profiles.get(agent_id)

    def get_or_create(self, agent_id: str) -> AgentSocialProfile:
        if agent_id not in self._profiles:
            self._profiles[agent_id] = AgentSocialProfile(agent_id)
        return self._profiles[agent_id]

    def record_message_received(self, from_agent: str, round_num: int, content: str) -> None:
        profile = self.get_or_create(from_agent)
        profile.record_interaction(round_num, message=content, cooperative=True)

    def record_message_sent(self, to_agent: str, round_num: int, content: str) -> None:
        profile = self.get_or_create(to_agent)
        profile.record_interaction(round_num, message=content, cooperative=True)

    def format_for_prompt(self) -> str:
        if not self._profiles:
            return ""
        parts = ["【社会模型】"]
        for pid, profile in self._profiles.items():
            parts.append(profile.format_for_prompt())
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            pid: p.to_dict() for pid, p in self._profiles.items()
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SocialModel":
        m = cls()
        for pid, pdata in d.items():
            m._profiles[pid] = AgentSocialProfile.from_dict(pdata)
        return m
