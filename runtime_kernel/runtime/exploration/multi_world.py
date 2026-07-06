"""
multi_world — Multi-World Simulation for Exploration Layer.

Simulates multiple parallel "worlds" with different strategy biases.
Compares expected outcomes across worlds to find optimal strategies.

Does NOT execute actions — only simulates expected outcomes based on
historical success rates from the ProbabilisticWorldModel.
"""

from __future__ import annotations

import random
from typing import Any, Optional


# ── World definitions ──

WORLD_TEMPLATES: dict[str, dict] = {
    "search_centric": {
        "label": "以搜索为中心",
        "description": "优先使用 Search 能力",
        "bias": {"Search": 1.0, "Human": 0.3},
    },
    "human_centric": {
        "label": "以人类交互为中心",
        "description": "优先向人类提问",
        "bias": {"Search": 0.3, "Human": 1.0},
    },
    "balanced": {
        "label": "均衡策略",
        "description": "Search 和 Human 均衡使用",
        "bias": {"Search": 0.7, "Human": 0.7},
    },
}


class WorldSimulation:
    """Simulation of one world with a specific strategy bias."""

    def __init__(self, name: str, config: dict):
        self.name: str = name
        self.label: str = config.get("label", name)
        self.description: str = config.get("description", "")
        self.bias: dict[str, float] = config.get("bias", {})
        self.expected_reward: float = 0.0
        self.uncertainty_reduction: float = 0.0
        self.success_probability: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "bias": self.bias,
            "expected_reward": round(self.expected_reward, 3),
            "uncertainty_reduction": round(self.uncertainty_reduction, 3),
            "success_probability": round(self.success_probability, 3),
        }


class MultiWorldSimulator:
    """Simulates parallel worlds with different strategy biases.

    Uses historical data from ProbabilisticWorldModel to estimate
    expected outcomes for each world configuration.

    Flow:
        Exploration proposes worlds → Cognitive evaluates →
        Core selects best world for execution
    """

    def __init__(self) -> None:
        self._worlds: dict[str, WorldSimulation] = {}
        self._simulation_count: int = 0

    def simulate(
        self,
        tool_effectiveness: dict[str, float],
        uncertainties: list[str],
        action_history: list[dict],
    ) -> list[WorldSimulation]:
        """Run multi-world simulation based on historical data.

        Args:
            tool_effectiveness: Dict of "capability → operation" → success_probability.
            uncertainties: Current high-uncertainty concepts.
            action_history: Recent action results.

        Returns list of WorldSimulation results, sorted by expected reward.
        """
        self._simulation_count += 1
        worlds: list[WorldSimulation] = []

        for name, config in WORLD_TEMPLATES.items():
            sim = WorldSimulation(name, config)
            self._simulate_world(sim, tool_effectiveness, uncertainties, action_history)
            worlds.append(sim)

        # Sort by expected reward descending
        worlds.sort(key=lambda w: w.expected_reward, reverse=True)
        self._worlds = {w.name: w for w in worlds}
        return worlds

    def _simulate_world(
        self,
        sim: WorldSimulation,
        tool_effectiveness: dict[str, float],
        uncertainties: list[str],
        action_history: list[dict],
    ) -> None:
        """Simulate one world's expected outcomes."""
        # Expected reward: weighted sum of tool effectiveness by bias
        total_reward = 0.0
        weight_sum = 0.0

        for cap, bias in sim.bias.items():
            # Find matching tool effectiveness entries
            matching = [
                v for k, v in tool_effectiveness.items()
                if k.startswith(cap)
            ]
            if matching:
                avg_eff = sum(matching) / len(matching)
                total_reward += avg_eff * bias
                weight_sum += bias

        sim.expected_reward = total_reward / max(weight_sum, 0.1)

        # Uncertainty reduction: worlds with bias toward uncertain areas
        uncertainty_relevant = sum(
            1 for u in uncertainties
            if any(cap in u.lower() for cap in sim.bias)
        )
        sim.uncertainty_reduction = min(
            1.0, uncertainty_relevant / max(len(uncertainties), 1)
        )

        # Success probability: blend reward and uncertainty reduction
        sim.success_probability = (
            sim.expected_reward * 0.6 + sim.uncertainty_reduction * 0.4
        )

    def get_best_world(self) -> Optional[WorldSimulation]:
        """Return the highest-ranked world from last simulation."""
        if not self._worlds:
            return None
        return max(self._worlds.values(), key=lambda w: w.expected_reward)

    def get_worlds(self) -> list[WorldSimulation]:
        """Return all simulated worlds."""
        return list(self._worlds.values())

    def format_for_prompt(self) -> str:
        """Format worlds for LLM prompt."""
        if not self._worlds:
            return ""
        lines = ["【多世界模拟】"]
        for w in sorted(self._worlds.values(), key=lambda x: x.expected_reward, reverse=True):
            lines.append(
                f"  {w.label}: reward={w.expected_reward:.2f}, "
                f"uncertainty_reduction={w.uncertainty_reduction:.2f}, "
                f"success_prob={w.success_probability:.2f}"
            )
        return "\n".join(lines)
