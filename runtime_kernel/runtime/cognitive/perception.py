"""Perception — how the agent perceives and interprets the world.

Perception is the bridge between world events and cognitive models.
It transforms raw events into interpreted observations, applies
attention to filter what matters, and feeds into working memory.

The perception pipeline:
    World Events → Perception → Attention → Working Memory → Reasoning (LLM)

Perception does NOT generate text. It prepares structured cognitive input.
"""

from __future__ import annotations

from typing import Any, Optional

from runtime_kernel.runtime.cognitive.attention import filter_events
from runtime_kernel.runtime.cognitive.working_memory import WorkingMemory
from runtime_kernel.runtime.cognitive.world_model import WorldModel


def perceive(
    mailbox_messages: list[dict],
    world_events: list[dict],
    env_context: str,
    drives: dict[str, float],
    world_model: WorldModel,
    uncertain_areas: list[str],
    recent_attended_events: list[dict],
    max_events: int = 3,
) -> dict:
    """Run the perception pipeline for one step.

    Args:
        mailbox_messages: Messages from other agents.
        world_events: Events from the EventBus.
        env_context: Raw environment context string.
        drives: Current drive states.
        world_model: The agent's world model (for context).
        uncertain_areas: Current areas of uncertainty.
        recent_attended_events: Events attended in recent steps.
        max_events: Max events to attend to.

    Returns:
        Dict with keys:
            attended_events: Events that passed attention filter.
            mailbox_count: Number of mailbox messages.
            env_summary: Extracted environment summary.
    """
    attended = []

    # 1. Convert mailbox messages to event-like dicts for attention
    message_events = []
    for msg in mailbox_messages:
        text = str(msg.get("content", {}).get("text", str(msg)))
        message_events.append({
            "type": "message",
            "text": text,
            "source": msg.get("from_agent", "unknown"),
            "room": msg.get("world_room", ""),
        })

    # 2. Combine and filter through attention
    all_candidates = message_events + world_events
    attended = filter_events(
        events=all_candidates,
        drives=drives,
        recent_events=recent_attended_events,
        uncertain_areas=uncertain_areas,
        max_events=max_events,
    )

    # 3. Extract environment summary from raw context
    env_summary = _summarize_environment(env_context)

    return {
        "attended_events": attended,
        "mailbox_count": len(mailbox_messages),
        "env_summary": env_summary,
    }


def _summarize_environment(env_context: str) -> str:
    """Extract a concise summary from the raw environment context."""
    if not env_context:
        return ""
    lines = env_context.split("\n")
    summary_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("【") and "】" in line:
            continue
        if line.startswith("  最近行动反馈") or line.startswith("  [待发送]"):
            continue
        if line.startswith("  这里的其他人") or line.startswith("  其他存在") or line.startswith("  另一存在"):
            continue
        if line.startswith("  ·  ") or line.startswith("    "):
            continue
        summary_lines.append(line)
    return "\n".join(summary_lines[:5])


def format_perception_for_prompt(perception: dict) -> str:
    """Format perception output for prompt injection."""
    parts = ["【感知输入】"]

    attended = perception.get("attended_events", [])
    if attended:
        parts.append("  注意到的事件:")
        for ev in attended:
            text = str(ev.get("text", ev.get("content", {}).get("text", "")))[:80]
            parts.append(f"    · {text}")
    else:
        parts.append("  (没有特别的事件)")

    env = perception.get("env_summary", "")
    if env:
        parts.append(f"  环境: {env[:120]}")

    mb_count = perception.get("mailbox_count", 0)
    if mb_count > 0:
        parts.append(f"  邮箱: {mb_count} 条未读")

    return "\n".join(parts)
