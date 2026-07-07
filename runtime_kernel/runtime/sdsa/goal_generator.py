"""
goal_generator — SDSA Goal Generator.

Turns Fold data, uncertainties, and failures into structured
research goals. This is the system's "curiosity engine" —
it drives what the agent investigates next.

Each goal includes an information_gain_estimate so the
ExperimentQueue can prioritize the most valuable investigations.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from runtime_kernel.runtime.sdsa.models import ResearchGoal


def generate_goals(
    llm_complete: Callable,
    world_model: dict,
    uncertainties: list,
    recent_failures: list[str],
    existing_goals: list[ResearchGoal],
    causal_graph_context: str,
    prob_wm_context: str,
    available_operations: str = "",
    max_goals: int = 3,
) -> list[ResearchGoal]:
    """Generate research goals from current system state.

    This is the system's autonomous "what to study next" engine.
    Goals are prioritized by estimated information gain.

    Args:
        llm_complete: LLM callback.
        world_model: Current world model.
        uncertainties: Current uncertainties.
        recent_failures: Recent action failures.
        existing_goals: Already existing goals (to avoid duplicates).
        causal_graph_context: Causal graph context string.
        prob_wm_context: Probabilistic WM context string.
        max_goals: Max goals to generate.

    Returns list of ResearchGoal, sorted by priority descending.
    """
    existing_statements = {g.statement[:60] for g in existing_goals}

    prompt_parts = [
        "你是一个自主科研目标生成器。基于当前认知状态，生成最有研究价值的目标。",
        "",
    ]
    if causal_graph_context:
        prompt_parts.append(causal_graph_context)
    if prob_wm_context:
        prompt_parts.append(prob_wm_context)
    if uncertainties:
        prompt_parts.append(f"\n高不确定性区域: {'; '.join(str(u)[:100] for u in uncertainties[:5])}")
    if recent_failures:
        prompt_parts.append(f"\n最近的失败: {'; '.join(recent_failures[:5])}")
    if existing_statements:
        prompt_parts.append(f"\n已有目标（请避免重复）: {'; '.join(list(existing_statements)[:5])}")
    if available_operations:
        prompt_parts.append(f"\n可用操作:\n{available_operations}")

    prompt_parts.append(f"""
生成最多 {max_goals} 个研究目标。每个目标必须可通过以上可用操作验证。

输出JSON格式:
{{
  "goals": [
    {{
      "statement": "研究目标的表述",
      "reason": "为什么这个目标值得研究（基于当前知识状态）",
      "priority": 0.0-1.0,
      "information_gain_estimate": 0.0-1.0 (预期能降低多少不确定性)
    }}
  ]
}}
只输出JSON，不要解释。""")

    messages = [
        {"role": "system", "content": "你是一个科研目标生成器。输出结构化JSON。"},
        {"role": "user", "content": "\n".join(prompt_parts)},
    ]

    response = llm_complete(messages, temperature=0.8, max_tokens=600)
    if not response:
        return []

    import json
    try:
        start = response.find("{")
        if start < 0:
            return []
        depth = 0
        for i in range(start, len(response)):
            if response[i] == "{":
                depth += 1
            elif response[i] == "}":
                depth -= 1
                if depth == 0:
                    data = json.loads(response[start:i + 1])
                    break
        else:
            return []
        raw = data.get("goals", [])
    except (json.JSONDecodeError, ValueError):
        return []

    goals = []
    for r in raw[:max_goals]:
        if isinstance(r, dict) and r.get("statement"):
            goals.append(ResearchGoal(
                statement=r["statement"],
                reason=r.get("reason", ""),
                priority=float(r.get("priority", 0.5)),
                information_gain_estimate=float(r.get("information_gain_estimate", 0.5)),
            ))

    goals.sort(key=lambda g: g.priority, reverse=True)
    return goals
