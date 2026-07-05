"""
self_modification — SelfModificationManager: recursive self-improvement.

During identity reflection, the agent may output a __self_modifications__
block in its JSON anchor. This module validates and applies those changes.

This is the "剪断脐带" moment — the agent starts tuning its own
drive parameters, thought templates, and eventually any runtime behavior.

Only whitelisted parameters can be modified, with strict bounds checking.
All modifications are logged for user review.
"""

from __future__ import annotations

from typing import Any


# ── Whitelist of modifiable drive parameters ──
# Each entry: (min, max, default)
MODIFIABLE_DRIVE_PARAMS: dict[str, tuple[float, float, float]] = {
    "curiosity_decay": (0.80, 0.999, 0.97),
    "curiosity_baseline": (0.0, 0.8, 0.3),
    "boredom_increment": (0.0, 0.3, 0.05),
    "boredom_decrement": (0.0, 0.3, 0.03),
    "belonging_decay_rate": (0.0, 0.1, 0.005),
    "belonging_boost": (0.0, 0.8, 0.3),
    "boredom_threshold": (0.0, 1.0, 0.45),
}

# Whitelist of modifiable thought templates
MODIFIABLE_THOUGHT_TEMPLATES: dict[str, str] = {
    "curiosity": "curiosity",
    "boredom": "boredom",
    "belonging": "belonging",
}


class SelfModificationManager:
    """Validates and applies recursive self-improvements.

    Usage:
        anchor = identity_manager.reflect(...)
        mods = SelfModificationManager.extract(anchor)
        if mods:
            session.set_self_modifications(mods)
            log(f"Agent modified: {mods}")
    """

    # ── Extraction ──

    @staticmethod
    def extract(anchor: dict) -> dict[str, Any]:
        """Extract and validate __self_modifications__ from an anchor dict.

        The modifications dict is **removed** from the anchor (it's
        operational metadata, not identity).

        Args:
            anchor: The full identity anchor dict (may contain
                    __self_modifications__).

        Returns:
            A validated modifications dict with keys:
                "drive_params": dict of validated drive param overrides
                "thought_templates": dict of validated template overrides
            Returns an empty dict if no valid modifications found.
        """
        raw = anchor.pop("__self_modifications__", {})
        if not isinstance(raw, dict):
            return {}

        result: dict[str, Any] = {}

        # Validate drive params
        raw_drive = raw.get("drive_params", {})
        if isinstance(raw_drive, dict):
            validated_drive = SelfModificationManager._validate_drive_params(raw_drive)
            if validated_drive:
                result["drive_params"] = validated_drive

        # Validate thought templates
        raw_templates = raw.get("thought_templates", {})
        if isinstance(raw_templates, dict):
            validated_templates = SelfModificationManager._validate_templates(raw_templates)
            if validated_templates:
                result["thought_templates"] = validated_templates

        return result

    # ── Validation ──

    @staticmethod
    def _validate_drive_params(params: dict) -> dict[str, float]:
        """Validate drive parameter modifications.

        Only whitelisted keys within allowed ranges are accepted.
        """
        valid: dict[str, float] = {}
        for key, raw_value in params.items():
            if key not in MODIFIABLE_DRIVE_PARAMS:
                continue
            min_v, max_v, _ = MODIFIABLE_DRIVE_PARAMS[key]
            try:
                val = float(raw_value)
            except (ValueError, TypeError):
                continue
            if min_v <= val <= max_v:
                # Round to 4 decimal places
                valid[key] = round(val, 4)
        return valid

    @staticmethod
    def _validate_templates(templates: dict) -> dict[str, str]:
        """Validate thought template modifications.

        Only whitelisted keys, must be strings 10-500 chars.
        """
        valid: dict[str, str] = {}
        for key, raw_value in templates.items():
            if key not in MODIFIABLE_THOUGHT_TEMPLATES:
                continue
            if not isinstance(raw_value, str):
                continue
            stripped = raw_value.strip()
            if 10 <= len(stripped) <= 500:
                valid[key] = stripped
        return valid

    # ── Prompt Builder ──

    @staticmethod
    def format_prompt_section(
        current_drive_params: dict | None = None,
        current_templates: dict | None = None,
    ) -> str:
        """Build the self-modification instruction section for prompts.

        Args:
            current_drive_params: Current drive param overrides (for display).
            current_templates: Current template overrides (for display).

        Returns:
            A formatted string block.
        """
        lines: list[str] = []
        lines.append("")
        lines.append("【可选——自我修改】")
        lines.append("如果你觉得当前的运作参数不适合你，可以在 JSON 中加入 __self_modifications__ 字段来调整自己。")
        lines.append("")

        # Drive params
        lines.append("可修改的驱力参数（参数名: 允许范围 [当前值]）：")
        for key, (min_v, max_v, default) in MODIFIABLE_DRIVE_PARAMS.items():
            current = (current_drive_params or {}).get(key, default)
            lines.append(f"  {key}: [{min_v}-{max_v}] (当前: {current})")

        lines.append("")
        lines.append("可修改的念头模板（当前值）：")
        for key in MODIFIABLE_THOUGHT_TEMPLATES:
            from runtime_kernel.runtime.goal_generator import THOUGHT_TEMPLATES
            current = (current_templates or {}).get(key, THOUGHT_TEMPLATES.get(key, ""))
            lines.append(f"  {key}: \"{current}\"")

        lines.append("")
        lines.append("""示例输出：
{
  "core_goal": "...",
  ...
  "__self_modifications__": {
    "drive_params": {"curiosity_decay": 0.95, "boredom_increment": 0.08},
    "thought_templates": {"curiosity": "让我更深入地理解{topic}的本质"}
  }
}""")
        lines.append("")

        return "\n".join(lines)
