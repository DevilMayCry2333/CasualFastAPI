"""Evolution Engine — observes the runtime and adjusts parameters.

THIS IS NOT A COGNITIVE MODEL.

It does not modify Agent beliefs, goals, hypotheses, or reasoning.
It only adjusts Runtime Parameters that control the environment.

The Evolution Engine runs every N rounds and:
  1. Collects Runtime Statistics
  2. Computes Evolution Signals (trends)
  3. Generates an Evolution Report
  4. Adjusts Runtime Parameters

Three-layer separation:
  - Runtime is the ecosystem
  - LLM is the lifeform
  - Evolution Engine is the ecosystem manager
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional


# ── Default Runtime Parameters ──
# These are ALL evolvable. Starting defaults.
DEFAULT_RUNTIME_PARAMS: dict[str, Any] = {
    # Attention
    "attention_curiosity_weight": 0.25,
    "attention_importance_weight": 0.25,
    "attention_novelty_weight": 0.20,
    "attention_relationship_weight": 0.15,
    "attention_uncertainty_weight": 0.15,
    "attention_max_events": 3,

    # Drive
    "curiosity_decay": 0.97,
    "curiosity_baseline": 0.3,
    "boredom_increment": 0.05,
    "boredom_decrement": 0.03,
    "belonging_decay_rate": 0.01,
    "belonging_boost": 0.3,

    # Hypothesis
    "hypothesis_min_evidence_for_confirm": 3,
    "hypothesis_max_active": 5,
    "hypothesis_contradiction_threshold": 0.3,

    # Belief
    "belief_update_evidence_threshold": 3,
    "belief_force_threshold": 0.6,

    # Communication
    "message_probability": 0.3,
    "mailbox_max_size": 20,

    # Working Memory
    "working_memory_max_evidence": 5,
    "working_memory_max_unresolved": 3,

    # Goal
    "drive_threshold": 0.45,

    # Reflection
    "identity_reflection_interval": 5,
    "introspection_interval": 20,

    # General
    "memory_retrieval_top_k": 3,
    "world_event_broadcast_interval": 3,
    "auto_save_interval": 10,
}

# Parameter bounds (prevent extreme values)
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "attention_curiosity_weight": (0.05, 0.5),
    "attention_importance_weight": (0.05, 0.5),
    "attention_novelty_weight": (0.05, 0.5),
    "attention_relationship_weight": (0.05, 0.4),
    "attention_uncertainty_weight": (0.05, 0.4),
    "curiosity_decay": (0.9, 0.999),
    "curiosity_baseline": (0.1, 0.6),
    "boredom_increment": (0.01, 0.15),
    "boredom_decrement": (0.01, 0.1),
    "hypothesis_contradiction_threshold": (0.1, 0.5),
    "belief_update_evidence_threshold": (1, 8),
    "belief_force_threshold": (0.3, 0.9),
    "message_probability": (0.05, 0.8),
    "drive_threshold": (0.2, 0.7),
}

# Parameters that should NOT be auto-evolved (safety)
PROTECTED_PARAMS = {
    "hypothesis_max_active", "mailbox_max_size",
    "identity_reflection_interval", "introspection_interval",
    "auto_save_interval",
}


class RuntimeParameters:
    """Evolvable runtime parameters.

    All parameters have defaults and bounds.
    The Evolution Engine can slowly adjust them.
    """

    def __init__(self, initial: Optional[dict] = None) -> None:
        self._params: dict[str, Any] = dict(DEFAULT_RUNTIME_PARAMS)
        if initial:
            self._params.update(initial)
        self._history: list[dict] = []  # track parameter changes

    def get(self, key: str, default: Any = None) -> Any:
        return self._params.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a parameter with bounds checking."""
        if key in PROTECTED_PARAMS:
            return  # cannot change protected params
        if key in PARAM_BOUNDS:
            lo, hi = PARAM_BOUNDS[key]
            value = max(lo, min(hi, float(value)))
        self._params[key] = value
        self._history.append({
            "key": key,
            "value": value,
            "time": time.time(),
        })
        if len(self._history) > 100:
            self._history = self._history[-100:]

    def adjust(self, key: str, delta: float) -> None:
        """Adjust a parameter by a delta (positive or negative)."""
        current = self._params.get(key, 0.0)
        if isinstance(current, (int, float)):
            self.set(key, current + delta)

    @property
    def all_params(self) -> dict:
        return dict(self._params)

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def to_dict(self) -> dict:
        return {
            "params": dict(self._params),
            "change_count": len(self._history),
            "last_changes": self._history[-10:] if self._history else [],
        }


# ── Evolution Signals ──

class EvolutionSignals:
    """Trend signals computed from RuntimeStatistics.

    These are NOT metrics. They are TRENDS — whether a metric
    is increasing, decreasing, or stable.
    """

    def __init__(self) -> None:
        self.novelty: float = 0.5       # Are agents finding new things?
        self.entropy: float = 0.5        # Is behavior diverse?
        self.stability: float = 0.5      # Is the system stable?
        self.exploration: float = 0.5    # Are agents exploring?
        self.cooperation: float = 0.5    # Are agents cooperating?
        self.conflict: float = 0.0       # Are agents conflicting?
        self.isolation: float = 0.5      # Are agents isolated?
        self.knowledge_growth: float = 0.5  # Is knowledge growing?
        self.world_compression: float = 0.5  # Is world model comprehensive?
        self.attention_diversity: float = 0.5  # Is attention spread out?
        self.communication_density: float = 0.5  # Are agents talking?
        self.hypothesis_cycling: float = 0.0  # Are hypotheses repeating?

    def compute_from_stats(self, stats: dict, prev_stats: Optional[dict] = None) -> None:
        """Compute signals from current and previous statistics."""
        ag = stats.get("global", {})

        # Exploration — how much are agents exploring vs. repeating
        self.exploration = ag.get("avg_exploration", 0.5)

        # Entropy — how diverse is behavior
        self.entropy = ag.get("avg_entropy", 0.5)

        # Communication density
        self.communication_density = ag.get("avg_communication_density", 0.5)

        # Knowledge growth — are new hypotheses/evidence being created?
        total_knowledge = ag.get("total_hypotheses", 0) + ag.get("total_evidence", 0)
        self.knowledge_growth = min(1.0, total_knowledge / 50)

        # Hypothesis cycling — high proposal + low success = cycling
        success = ag.get("avg_hypothesis_success", 0.5)
        self.hypothesis_cycling = max(0.0, 1.0 - success * 2)

        # Cooperation (from social ratio)
        social = ag.get("avg_social", 0.0)
        if social > 0.3:
            self.cooperation = min(1.0, social * 2)
            self.isolation = max(0.0, 1.0 - social * 3)
        else:
            self.cooperation = social
            self.isolation = min(1.0, 1.0 - social)

        # Stability — low change + moderate entropy = stable
        self.stability = 1.0 - self.entropy * 0.5

        # Novelty — high exploration + high knowledge growth
        self.novelty = (self.exploration + self.knowledge_growth) / 2

        # World compression — have agents covered the world?
        total_discovered = sum(
            s.get("world_growth", 0)
            for s in stats.get("agents", {}).values()
        )
        self.world_compression = min(1.0, total_discovered / 20)

        # Attention diversity
        self.attention_diversity = self.entropy

    def to_dict(self) -> dict:
        return {
            "novelty": round(self.novelty, 3),
            "entropy": round(self.entropy, 3),
            "stability": round(self.stability, 3),
            "exploration": round(self.exploration, 3),
            "cooperation": round(self.cooperation, 3),
            "conflict": round(self.conflict, 3),
            "isolation": round(self.isolation, 3),
            "knowledge_growth": round(self.knowledge_growth, 3),
            "world_compression": round(self.world_compression, 3),
            "attention_diversity": round(self.attention_diversity, 3),
            "communication_density": round(self.communication_density, 3),
            "hypothesis_cycling": round(self.hypothesis_cycling, 3),
        }

    def format_report(self) -> str:
        """Format a human-readable evolution report."""
        lines = ["【演化报告】"]
        lines.append(f"  新颖性: {self.novelty:.2f} — "
                     f"{'正在发现新事物' if self.novelty > 0.5 else '趋于重复'}")
        lines.append(f"  熵: {self.entropy:.2f} — "
                     f"{'行为多样' if self.entropy > 0.5 else '行为收敛'}")
        lines.append(f"  稳定性: {self.stability:.2f} — "
                     f"{'系统稳定' if self.stability > 0.5 else '波动较大'}")
        lines.append(f"  探索: {self.exploration:.2f} — "
                     f"{'积极探索' if self.exploration > 0.3 else '探索不足'}")
        lines.append(f"  知识增长: {self.knowledge_growth:.2f} — "
                     f"{'知识在积累' if self.knowledge_growth > 0.3 else '知识停滞'}")
        lines.append(f"  通信密度: {self.communication_density:.2f} — "
                     f"{'社交活跃' if self.communication_density > 0.2 else '社交稀少'}")
        lines.append(f"  假设循环: {self.hypothesis_cycling:.2f} — "
                     f"{'假设在收敛' if self.hypothesis_cycling < 0.3 else '假设在空转'}")
        return "\n".join(lines)


# ── Evolution Engine ──

class EvolutionEngine:
    """Periodically analyzes Runtime Statistics and adjusts parameters.

    Runs every EVOLUTION_INTERVAL rounds.
    Only modifies Runtime Parameters. Never modifies Agent internals.
    """

    def __init__(
        self,
        params: Optional[RuntimeParameters] = None,
        interval: int = 50,
    ) -> None:
        self._params: RuntimeParameters = params or RuntimeParameters()
        self._interval: int = interval
        self._last_run_round: int = 0
        self._signals_history: list[dict] = []
        self._reports: list[dict] = []

    @property
    def params(self) -> RuntimeParameters:
        return self._params

    @property
    def interval(self) -> int:
        return self._interval

    @property
    def reports(self) -> list[dict]:
        return list(self._reports)

    @property
    def last_run_round(self) -> int:
        return self._last_run_round

    def should_run(self, round_num: int) -> bool:
        """Check if the evolution engine should run this round."""
        if round_num < self._interval:
            return False  # Let the system stabilize first
        return (round_num - self._last_run_round) >= self._interval

    def run(
        self,
        round_num: int,
        stats: "RuntimeStatistics",  # type: ignore  # noqa: F821
    ) -> dict:
        """Execute one evolution cycle.

        Args:
            round_num: Current system round.
            stats: RuntimeStatistics instance.

        Returns:
            Evolution report dict.
        """
        self._last_run_round = round_num

        # 1. Collect statistics
        global_stats = stats.get_global_stats()
        agent_stats_list = [
            agent.to_dict() for agent in stats._agent_stats.values()
        ]

        # 2. Compute evolution signals
        prev_signals = self._signals_history[-1] if self._signals_history else None
        signals = EvolutionSignals()
        signals.compute_from_stats(
            {"global": global_stats, "agents": {}},
            prev_signals,
        )

        # 3. Generate evolution report
        report_text = signals.format_report()

        # 4. Adjust runtime parameters based on signals
        adjustments = self._compute_adjustments(signals)
        for key, delta in adjustments:
            self._params.adjust(key, delta)

        # 5. Record
        report = {
            "round": round_num,
            "time": time.time(),
            "signals": signals.to_dict(),
            "params": self._params.all_params,
            "adjustments": adjustments,
            "report": report_text,
        }
        self._reports.append(report)
        self._signals_history.append(signals.to_dict())

        if len(self._reports) > 20:
            self._reports = self._reports[-20:]
        if len(self._signals_history) > 50:
            self._signals_history = self._signals_history[-50:]

        return report

    def _compute_adjustments(self, signals: EvolutionSignals) -> list[tuple[str, float]]:
        """Compute parameter adjustments based on evolution signals.

        Each adjustment is (param_name, delta). The Evolution Engine
        makes small, gradual changes — never sudden jumps.
        """
        adjustments: list[tuple[str, float]] = []

        # If exploration is low, increase curiosity and novelty weight
        if signals.exploration < 0.25:
            adjustments.append(("attention_novelty_weight", 0.02))
            adjustments.append(("curiosity_baseline", 0.01))
        elif signals.exploration > 0.6:
            adjustments.append(("attention_novelty_weight", -0.01))
            adjustments.append(("curiosity_baseline", -0.005))

        # If hypothesis cycling (lots of proposals with low success), raise threshold
        if signals.hypothesis_cycling > 0.5:
            adjustments.append(("hypothesis_contradiction_threshold", 0.02))
            adjustments.append(("belief_force_threshold", 0.02))
        elif signals.hypothesis_cycling < 0.2:
            adjustments.append(("hypothesis_contradiction_threshold", -0.01))
            adjustments.append(("belief_force_threshold", -0.01))

        # If knowledge growth is stagnating, increase curiosity
        if signals.knowledge_growth < 0.2:
            adjustments.append(("curiosity_decay", -0.005))  # slower decay = more curious
        elif signals.knowledge_growth > 0.6:
            adjustments.append(("curiosity_decay", 0.003))   # faster decay = settle

        # If social isolation, adjust message probability
        if signals.isolation > 0.7 and signals.communication_density < 0.1:
            adjustments.append(("message_probability", 0.02))
        elif signals.cooperation > 0.6:
            adjustments.append(("message_probability", -0.01))

        # If entropy is low (behavior too repetitive), increase boredom
        if signals.entropy < 0.3:
            adjustments.append(("boredom_increment", 0.005))
        elif signals.entropy > 0.7:
            adjustments.append(("boredom_increment", -0.003))

        # If attention diversity is low, adjust weights
        if signals.attention_diversity < 0.3:
            adjustments.append(("attention_novelty_weight", 0.01))
        elif signals.attention_diversity > 0.7:
            adjustments.append(("attention_novelty_weight", -0.005))

        # If world is mostly explored, reduce exploration drive
        if signals.world_compression > 0.8:
            adjustments.append(("boredom_increment", -0.002))

        return adjustments

    def to_dict(self) -> dict:
        return {
            "interval": self._interval,
            "last_run_round": self._last_run_round,
            "params": self._params.to_dict(),
            "reports": self._reports[-5:] if self._reports else [],
            "signals_history": self._signals_history[-10:],
        }
