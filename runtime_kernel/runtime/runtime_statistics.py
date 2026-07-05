"""RuntimeStatistics — the system observing itself.

Collects metrics about agent behavior and system state each step.
These stats are used by the Evolution Engine to detect trends
and adjust Runtime Parameters.

NOT part of any Cognitive Model. Statistics belong to the Runtime,
not to any individual Agent.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Optional


class AgentStats:
    """Statistics for a single agent over a rolling window.

    Each field tracks a different dimension of agent behavior.
    """

    def __init__(self, window: int = 100) -> None:
        self._window = window

        # Action distribution (last N rounds)
        self._action_types: deque[str] = deque(maxlen=window)

        # Hypothesis tracking
        self._hypotheses_proposed: int = 0
        self._hypotheses_supported: int = 0
        self._hypotheses_contradicted: int = 0
        self._hypotheses_discarded: int = 0

        # Evidence tracking
        self._evidence_collected: int = 0

        # Communication tracking
        self._messages_sent: int = 0
        self._messages_received: int = 0

        # Belief tracking
        self._belief_changes: int = 0

        # World model tracking
        self._places_discovered: set[str] = set()
        self._objects_discovered: set[str] = set()
        self._agents_met: set[str] = set()

        # Round tracking
        self._total_rounds: int = 0
        self._last_active_round: int = 0

        # Attention metrics
        self._attended_event_types: deque[str] = deque(maxlen=window)

        # Action novelty tracking
        self._unique_actions: set[str] = set()
        self._total_actions: int = 0

    @property
    def window(self) -> int:
        return self._window

    # ── Step updates ──

    def record_action(self, action: str) -> None:
        """Record an action and classify its type."""
        if not action:
            return
        self._total_actions += 1
        self._unique_actions.add(action.lower().split()[0] if action else "")
        action_lower = action.lower()

        # Classify action type
        if any(w in action_lower for w in ("send_message", "send ", "say", "tell")):
            self._action_types.append("social")
        elif any(w in action_lower for w in ("look", "examine", "observe", "read", "search")):
            self._action_types.append("observe")
        elif any(w in action_lower for w in ("move", "go", "north", "south", "east", "west")):
            self._action_types.append("explore")
        elif any(w in action_lower for w in ("use", "take", "drop", "write", "note")):
            self._action_types.append("interact")
        elif any(w in action_lower for w in ("wait", "think", "reflect")):
            self._action_types.append("reflect")
        else:
            self._action_types.append("other")

    def record_hypothesis_event(self, event: str) -> None:
        if event == "proposed":
            self._hypotheses_proposed += 1
        elif event == "supported":
            self._hypotheses_supported += 1
        elif event == "contradicted":
            self._hypotheses_contradicted += 1
        elif event == "discarded":
            self._hypotheses_discarded += 1

    def record_evidence(self) -> None:
        self._evidence_collected += 1

    def record_message_sent(self) -> None:
        self._messages_sent += 1

    def record_message_received(self) -> None:
        self._messages_received += 1

    def record_belief_change(self) -> None:
        self._belief_changes += 1

    def record_place(self, place: str) -> None:
        if place:
            self._places_discovered.add(place)

    def record_object(self, obj: str) -> None:
        if obj:
            self._objects_discovered.add(obj)

    def record_agent_met(self, agent_id: str) -> None:
        if agent_id:
            self._agents_met.add(agent_id)

    def record_attended_event(self, event_type: str) -> None:
        self._attended_event_types.append(event_type)

    def set_round(self, round_num: int) -> None:
        self._total_rounds = max(self._total_rounds, round_num)
        self._last_active_round = round_num

    # ── Derived metrics ──

    @property
    def exploration_ratio(self) -> float:
        """Proportion of explore actions in recent window."""
        if not self._action_types:
            return 0.0
        explore = sum(1 for a in self._action_types if a == "explore")
        return explore / len(self._action_types)

    @property
    def social_ratio(self) -> float:
        if not self._action_types:
            return 0.0
        social = sum(1 for a in self._action_types if a == "social")
        return social / len(self._action_types)

    @property
    def observation_ratio(self) -> float:
        if not self._action_types:
            return 0.0
        obs = sum(1 for a in self._action_types if a == "observe")
        return obs / len(self._action_types)

    @property
    def interaction_ratio(self) -> float:
        if not self._action_types:
            return 0.0
        interact = sum(1 for a in self._action_types if a == "interact")
        return interact / len(self._action_types)

    @property
    def hypothesis_success_rate(self) -> float:
        total = self._hypotheses_supported + self._hypotheses_contradicted
        if total == 0:
            return 0.0
        return self._hypotheses_supported / total

    @property
    def evidence_efficiency(self) -> float:
        if self._total_rounds == 0:
            return 0.0
        return self._evidence_collected / self._total_rounds

    @property
    def communication_density(self) -> float:
        if self._total_rounds == 0:
            return 0.0
        return (self._messages_sent + self._messages_received) / self._total_rounds

    @property
    def belief_revision_rate(self) -> float:
        if self._total_rounds == 0:
            return 0.0
        return self._belief_changes / self._total_rounds

    @property
    def action_diversity(self) -> float:
        if self._total_actions == 0:
            return 0.0
        return len(self._unique_actions) / max(self._total_actions, 1)

    @property
    def world_growth(self) -> int:
        return len(self._places_discovered) + len(self._objects_discovered)

    @property
    def knowledge_growth(self) -> int:
        return self._hypotheses_proposed + self._evidence_collected

    @property
    def entropy(self) -> float:
        """Shannon entropy of action type distribution."""
        if not self._action_types:
            return 0.0
        n = len(self._action_types)
        counts = {}
        for a in self._action_types:
            counts[a] = counts.get(a, 0) + 1
        ent = 0.0
        for c in counts.values():
            p = c / n
            if p > 0:
                ent -= p * math.log2(p)
        return ent / math.log2(max(len(counts), 2))  # normalized 0-1

    def to_dict(self) -> dict:
        return {
            "total_rounds": self._total_rounds,
            "exploration_ratio": round(self.exploration_ratio, 3),
            "social_ratio": round(self.social_ratio, 3),
            "observation_ratio": round(self.observation_ratio, 3),
            "interaction_ratio": round(self.interaction_ratio, 3),
            "hypothesis_success_rate": round(self.hypothesis_success_rate, 3),
            "hypotheses_proposed": self._hypotheses_proposed,
            "evidence_efficiency": round(self.evidence_efficiency, 3),
            "communication_density": round(self.communication_density, 3),
            "messages_sent": self._messages_sent,
            "messages_received": self._messages_received,
            "belief_revision_rate": round(self.belief_revision_rate, 3),
            "belief_changes": self._belief_changes,
            "action_diversity": round(self.action_diversity, 3),
            "world_growth": self.world_growth,
            "knowledge_growth": self.knowledge_growth,
            "entropy": round(self.entropy, 3),
            "places_discovered": len(self._places_discovered),
            "agents_met": len(self._agents_met),
        }


class RuntimeStatistics:
    """Collects and aggregates statistics across all agents.

    This is the system's self-observation layer.
    """

    def __init__(self, window: int = 100) -> None:
        self._window = window
        self._agent_stats: dict[str, AgentStats] = {}
        self._snapshots: list[dict] = []  # historical snapshots for trend analysis
        self._snapshot_interval = 10  # save snapshot every N rounds
        self._last_snapshot_round: dict[str, int] = {}

    def get_agent_stats(self, agent_id: str) -> AgentStats:
        if agent_id not in self._agent_stats:
            self._agent_stats[agent_id] = AgentStats(window=self._window)
        return self._agent_stats[agent_id]

    @property
    def all_agent_stats(self) -> dict[str, AgentStats]:
        return dict(self._agent_stats)

    def snapshot_if_needed(self, agent_id: str, round_num: int) -> None:
        """Save a statistics snapshot periodically for trend analysis."""
        last = self._last_snapshot_round.get(agent_id, 0)
        if round_num - last >= self._snapshot_interval:
            stats = self.get_agent_stats(agent_id)
            self._snapshots.append({
                "agent_id": agent_id,
                "round": round_num,
                "time": time.time(),
                "stats": stats.to_dict(),
            })
            self._last_snapshot_round[agent_id] = round_num
            # Keep only recent snapshots
            if len(self._snapshots) > 100:
                self._snapshots = self._snapshots[-100:]

    def get_snapshots(self, agent_id: str, n: int = 10) -> list[dict]:
        """Get recent snapshots for an agent."""
        return [
            s for s in self._snapshots[-n*2:]
            if s["agent_id"] == agent_id
        ][-n:]

    def get_trend(self, agent_id: str, metric: str, n: int = 5) -> list[float]:
        """Get the trend of a metric over recent snapshots."""
        snaps = self.get_snapshots(agent_id, n=n)
        values = []
        for s in snaps:
            val = s.get("stats", {}).get(metric)
            if val is not None:
                values.append(val)
        return values

    def get_global_stats(self) -> dict:
        """Aggregate stats across all agents."""
        if not self._agent_stats:
            return {}
        total = len(self._agent_stats)
        agg = {
            "agent_count": total,
            "total_rounds": sum(s._total_rounds for s in self._agent_stats.values()),
            "avg_exploration": 0.0,
            "avg_social": 0.0,
            "avg_observation": 0.0,
            "avg_entropy": 0.0,
            "avg_hypothesis_success": 0.0,
            "avg_communication_density": 0.0,
            "total_messages": 0,
            "total_hypotheses": 0,
            "total_evidence": 0,
        }
        for s in self._agent_stats.values():
            agg["avg_exploration"] += s.exploration_ratio
            agg["avg_social"] += s.social_ratio
            agg["avg_observation"] += s.observation_ratio
            agg["avg_entropy"] += s.entropy
            agg["avg_hypothesis_success"] += s.hypothesis_success_rate
            agg["avg_communication_density"] += s.communication_density
            agg["total_messages"] += s._messages_sent + s._messages_received
            agg["total_hypotheses"] += s._hypotheses_proposed
            agg["total_evidence"] += s._evidence_collected
        for key in ("avg_exploration", "avg_social", "avg_observation",
                     "avg_entropy", "avg_hypothesis_success", "avg_communication_density"):
            agg[key] = round(agg[key] / total, 3)
        return agg

    def to_dict(self) -> dict:
        return {
            "agents": {
                aid: stats.to_dict()
                for aid, stats in self._agent_stats.items()
            },
            "global": self.get_global_stats(),
            "snapshot_count": len(self._snapshots),
        }
