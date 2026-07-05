"""
drive — DriveModel: minimal internal drive/emotion system.

Three primary drives that provide the "fuel" behind autonomous goal pursuit:

  - curiosity:   desire for novelty, increases with prediction error
  - boredom:     discomfort with monotony, increases with state similarity
  - belonging:   desire for human connection, decays without interaction

Drives are stored in the Session (serializable). DriveModel provides
pure functions with optional params overrides for self-modification.

This is NOT an emotional simulation. It's a minimal tension system
that gives generated goals a reason to be pursued.
"""

from __future__ import annotations

from typing import Any, Optional

from runtime_kernel.runtime.self_modification import MODIFIABLE_DRIVE_PARAMS


DRIVE_NAMES = ("curiosity", "boredom", "belonging")
DRIVE_DEFAULTS: dict[str, float] = {
    "curiosity": 0.50,
    "boredom": 0.10,
    "belonging": 0.70,
}
DRIVE_LABELS: dict[str, str] = {
    "curiosity": "高 — 渴望新信息",
    "boredom": "高 — 寻求变化",
    "belonging": "高 — 渴望联结",
}
DRIVE_LABELS_LOW: dict[str, str] = {
    "curiosity": "低 — 满足于已知",
    "boredom": "低 — 尚有新鲜感",
    "belonging": "低 — 倾向于独处",
}


def _get_param(params: Optional[dict], key: str, default: float) -> float:
    """Get a drive parameter from optional overrides."""
    if params and key in params:
        return float(params[key])
    return default


def _compute_change_score(
    old_state: dict[str, Any],
    new_state: dict[str, Any],
) -> float:
    """Compute how much the state changed between two rounds (0.0-1.0).

    Measures added/removed/changed keys relative to the larger set.
    """
    if not old_state or not new_state:
        return 0.3
    old_keys = set(old_state.keys())
    new_keys = set(new_state.keys())
    added = new_keys - old_keys
    removed = old_keys - new_keys
    common = old_keys & new_keys
    changed = sum(1 for k in common if old_state.get(k) != new_state.get(k))
    total = len(added) + len(removed) + changed
    max_keys = max(len(old_state), len(new_state))
    return min(1.0, total / max_keys) if max_keys else 0.0


def _state_to_key(state_dict: dict[str, Any]) -> tuple:
    """Fingerprint a state dict for similarity comparison."""
    return (
        str(state_dict.get("topic", "")),
        str(state_dict.get("belief", "")),
        str(state_dict.get("goal", "")),
    )


class DriveModel:
    """Stateless drive calculation engine.

    All methods are static — drives are stored in AgentSession.
    Optional params dict allows the agent to self-modify its parameters.
    """

    @staticmethod
    def initial() -> dict[str, float]:
        """Return a fresh default drive state."""
        return dict(DRIVE_DEFAULTS)

    @staticmethod
    def dominant(drives: dict[str, float]) -> str:
        """Return the name of the highest active drive."""
        if not drives:
            return "curiosity"
        return max(drives, key=drives.get)

    @staticmethod
    def format_prompt(drives: dict[str, float]) -> str:
        """Format drives as a prompt context block."""
        lines: list[str] = []
        for name in DRIVE_NAMES:
            val = drives.get(name, 0.5)
            label = DRIVE_LABELS.get(name, "") if val > 0.5 else DRIVE_LABELS_LOW.get(name, "")
            lines.append(f"{name}: {val:.2f} — {label}")
        return "\n".join(lines)

    @staticmethod
    def dominant_prompt_text(drives: dict[str, float]) -> str:
        """Return a one-line summary of the strongest drive."""
        dom = DriveModel.dominant(drives)
        val = drives.get(dom, 0.0)
        return f"{dom}: {val:.2f}"

    # ── Update rules ──

    @staticmethod
    def after_step(
        drives: dict[str, float],
        history: list[dict],
        state_dict: dict[str, Any],
        rounds_since_human: int,
        params: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        """Update drives after an autonomous step completes.

        Args:
            drives: Current drive dict.
            history: Session history (for computing change).
            state_dict: The newly produced state.
            rounds_since_human: Steps since last human contact.
            params: Optional drive parameter overrides (self-modification).

        Returns:
            Updated drive dict.
        """
        d = dict(drives)

        # Read params (with defaults from MODIFIABLE_DRIVE_PARAMS)
        decay = _get_param(params, "curiosity_decay", 0.97)
        baseline = _get_param(params, "curiosity_baseline", 0.3)
        bored_inc = _get_param(params, "boredom_increment", 0.05)
        bored_dec = _get_param(params, "boredom_decrement", 0.03)

        # ── Curiosity ──
        prev_state = {}
        if len(history) >= 2:
            prev_state = history[-2].get("state", {})

        change = _compute_change_score(prev_state, state_dict)
        d["curiosity"] = min(1.0, d["curiosity"] + change * 0.2)
        d["curiosity"] = max(0.1, d["curiosity"] * decay + baseline * (1 - decay))

        # ── Boredom ──
        prev_key = _state_to_key(prev_state)
        curr_key = _state_to_key(state_dict)
        is_similar = prev_key == curr_key if prev_state else False

        if is_similar:
            d["boredom"] = min(1.0, d["boredom"] + bored_inc)
        else:
            d["boredom"] = max(0.1, d["boredom"] - bored_dec)

        # ── Belonging ──
        decay_rate = _get_param(params, "belonging_decay_rate", 0.01)
        if rounds_since_human > 3:
            d["belonging"] = max(0.1, d["belonging"] - decay_rate)

        # Clamp all
        for k in DRIVE_NAMES:
            d[k] = max(0.0, min(1.0, d[k]))

        return d

    @staticmethod
    def on_human_interaction(
        drives: dict[str, float],
        params: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        """Update drives after a human interruption.

        Args:
            drives: Current drive dict.
            params: Optional drive parameter overrides (self-modification).

        Returns:
            Updated drive dict.
        """
        d = dict(drives)
        boost = _get_param(params, "belonging_boost", 0.3)
        d["belonging"] = min(1.0, d["belonging"] + boost)
        d["boredom"] = max(0.1, d["boredom"] - 0.15)
        return d

    @staticmethod
    def tick(
        drives: dict[str, float],
        rounds_since_human: int = 1,
        params: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        """Background tick — drift drives during idle time.

        Called by HeartbeatManager between steps.

        Args:
            drives: Current drive dict.
            rounds_since_human: Steps since last human contact.
            params: Optional drive parameter overrides (self-modification).

        Returns:
            Updated drive dict.
        """
        d = dict(drives)
        decay = _get_param(params, "curiosity_decay", 0.97)
        belonging_decay = _get_param(params, "belonging_decay_rate", 0.005)

        # Curiosity slowly drifts
        d["curiosity"] = max(0.1, d["curiosity"] * (decay + 0.025))
        # Boredom slowly creeps up
        d["boredom"] = min(1.0, d["boredom"] + 0.008)
        # Belonging decays faster when alone longer
        decay = belonging_decay * max(1, rounds_since_human)
        d["belonging"] = max(0.05, d["belonging"] - decay)
        return d
