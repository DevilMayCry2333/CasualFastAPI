"""
question_generator — Automatically generates scientific questions
from the agent's current World Model, Fold history, and uncertainties.

Questions are categorized as:
    exploration:  "What happens if I try X?"
    debugging:    "Why did this action fail?"
    causal:       "What causes this effect?"
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from runtime_kernel.runtime.scientific.models import ScientificQuestion


def generate_questions(
    llm_complete: Callable,
    world_model: dict,
    uncertainties: list,
    open_questions: list[str],
    recent_failures: list[str],
    max_questions: int = 3,
) -> list[ScientificQuestion]:
    """Generate scientific questions from current agent state.

    Uses the LLM to produce structured questions. No state mutation.

    Args:
        llm_complete: Function(messages) → str, wrapping the LLM call.
        world_model: Current world model dict.
        uncertainties: List of uncertainty descriptions.
        open_questions: Existing open questions.
        recent_failures: Recent action/tool failures.
        max_questions: Max questions to generate.

    Returns list of ScientificQuestion objects.
    """
    prompt_lines = [
        "你是一个自主科学智能体的问题生成器。",
        "基于当前世界模型和知识状态，提出你最想回答的科学问题。",
        "",
        "当前世界模型:",
        str(world_model)[:300],
    ]
    if uncertainties:
        prompt_lines.append(f"\n当前不确定性: {'; '.join(str(u)[:100] for u in uncertainties[:3])}")
    if open_questions:
        prompt_lines.append(f"\n已有未回答的问题: {'; '.join(open_questions[:3])}")
    if recent_failures:
        prompt_lines.append(f"\n最近的失败: {'; '.join(recent_failures[:3])}")

    prompt_lines.append(f"""
输出最多 {max_questions} 个问题，JSON格式:
{{
  "questions": [
    {{
      "question": "问题的具体表述",
      "context": "什么背景下提出这个问题",
      "type": "exploration | debugging | causal",
      "priority": 0.0-1.0  (0=低优先, 1=高优先)
    }}
  ]
}}
只输出JSON，不要解释。""")

    messages = [
        {"role": "system", "content": "你是一个科学问题生成器。输出结构化JSON。"},
        {"role": "user", "content": "\n".join(prompt_lines)},
    ]

    response = llm_complete(messages, temperature=0.8, max_tokens=500)
    if not response:
        return []

    # Parse JSON from response
    import json
    state_part = response
    if "{" in state_part:
        start = state_part.find("{")
        depth = 0
        for i in range(start, len(state_part)):
            if state_part[i] == "{": depth += 1
            elif state_part[i] == "}":
                depth -= 1
                if depth == 0:
                    state_part = state_part[start:i+1]
                    break

    try:
        data = json.loads(state_part)
        raw = data.get("questions", [])
    except (json.JSONDecodeError, ValueError):
        # Fallback: try to extract any JSON
        import re
        matches = re.findall(r'"question"\s*:\s*"([^"]+)"', response)
        raw = [{"question": m, "type": "exploration", "priority": 0.5} for m in matches]

    questions = []
    for r in raw[:max_questions]:
        if isinstance(r, dict) and r.get("question"):
            questions.append(ScientificQuestion(
                question=r["question"],
                context=r.get("context", ""),
                q_type=r.get("type", "exploration"),
                priority=float(r.get("priority", 0.5)),
            ))
    return questions
