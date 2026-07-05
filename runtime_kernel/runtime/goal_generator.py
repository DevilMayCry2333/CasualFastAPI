"""
goal_generator — GoalGenerator: autonomous thought pool production.

The agent's goals are no longer purely "LLM randoms from the prompt."
Instead, they emerge from a structured process:

  1. Drives (curiosity / boredom / belonging) provide the "why."
  2. Identity anchor gaps provide the "who am I" tension.
  3. Memory contradictions provide the "what's unresolved" friction.

GoalGenerator combines these signals into a "thought pool" — 1-3 candidate
goal directions with salience scores — that gets injected into the prompt.

The LLM still outputs the final state.goal, but it's now responding to
an internally-generated impulse rather than guessing what to do next.

The thought templates can be overridden by the agent itself through
self-modification (__self_modifications__.thought_templates).
"""

from __future__ import annotations

from typing import Any, Optional

from runtime_kernel.runtime.drive import DRIVE_NAMES, DriveModel

# ── Default thought templates ──
# Environment-focused. No philosophical defaults.
# Identity gap thoughts are only generated when identity actually has gaps.
THOUGHT_TEMPLATES: dict[str, str] = {
    "curiosity": "探索当前环境中的未知之处：{topic}",
    "boredom": "去不同的地方看看，寻找新的环境刺激",
    "belonging": "留意周围是否有其他存在可以互动",
}

# Identity gap templates — only used when identity fields are truly empty
IDENTITY_GAP_TEMPLATES: dict[str, str] = {
    "core_goal": "观察周围，寻找值得关注的方向",
    "identity": "记录最近的行为模式，看有没有规律",
    "worldview": "收集新的环境信息",
    "stable_values": "留意哪些事物让你感到舒适或不快",
    "recent_reflection": "回顾最近的探索经历",
}

# Minimum drive threshold for a thought to enter the pool
DRIVE_THRESHOLD = 0.45


class GoalGenerator:
    """Generates the agent's thought pool from internal state.

    Pure functions — the thought pool is stored in AgentSession.
    Templates can be overridden via self-modification.
    """

    @staticmethod
    def generate(
        drives: dict[str, float],
        identity_anchor: dict[str, Any],
        state_topic: str,
        templates_override: Optional[dict[str, str]] = None,
    ) -> list[dict[str, Any]]:
        """Produce candidate thoughts based on drives and state.

        Args:
            drives: Current drive dict from the session.
            identity_anchor: Current identity anchor dict.
            state_topic: The session's current topic.
            templates_override: Optional template overrides from
                                self-modification.

        Returns:
            List of thought dicts, sorted by salience descending.
            Each thought: {"drive": str, "salience": float, "thought": str}
        """
        # Build effective templates: defaults + overrides
        templates = dict(THOUGHT_TEMPLATES)
        if templates_override:
            templates.update(templates_override)

        thoughts: list[dict[str, Any]] = []

        # 1. Drive-based thoughts
        for name in DRIVE_NAMES:
            val = drives.get(name, 0.5)
            if val >= DRIVE_THRESHOLD:
                template = templates.get(name, "继续当前探索")
                thought_text = (
                    template.format(topic=state_topic)
                    if "{topic}" in template
                    else template
                )
                thoughts.append({
                    "drive": name,
                    "salience": round(val, 2),
                    "thought": thought_text,
                })

        # 2. Identity anchor gaps — unresolved self-knowledge creates tension
        if identity_anchor:
            for field, tpl in IDENTITY_GAP_TEMPLATES.items():
                current_val = identity_anchor.get(field, "")
                if not current_val or current_val.strip() == "":
                    thoughts.append({
                        "drive": "curiosity",
                        "salience": 0.5,
                        "thought": tpl,
                    })

        # 3. Deduplicate — merge same-thought entries, keep highest salience
        seen: dict[str, dict] = {}
        for t in thoughts:
            text = t["thought"]
            if text in seen:
                if t["salience"] > seen[text]["salience"]:
                    seen[text] = t
            else:
                seen[text] = t

        result = sorted(seen.values(), key=lambda x: x["salience"], reverse=True)
        return result[:3]

    @staticmethod
    def format_prompt(
        thought_pool: list[dict[str, Any]],
        drives: dict[str, float],
    ) -> str:
        """Format thought pool as a prompt context block.

        Args:
            thought_pool: List of thought dicts.
            drives: Current drive dict.

        Returns:
            Formatted string, or empty if pool is empty.
        """
        if not thought_pool:
            return ""

        lines: list[str] = ["【内驱力状态】", DriveModel.format_prompt(drives), ""]

        lines.append("【念头池】")
        for i, t in enumerate(thought_pool, 1):
            drive_label = t.get("drive", "?")
            thought = t.get("thought", "")
            salience = t.get("salience", 0.0)
            bar = "█" * int(salience * 10) + "░" * (10 - int(salience * 10))
            lines.append(f"  [{i}] ({drive_label}) [{bar}] {thought}")

        return "\n".join(lines)
