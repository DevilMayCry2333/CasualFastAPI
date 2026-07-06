"""
hypothesis_layer — Converts scientific questions into testable hypotheses.

Each hypothesis includes a predicted outcome that can be verified
through experiment (action execution).
"""

from __future__ import annotations

from typing import Any, Callable

from runtime_kernel.runtime.scientific.models import Hypothesis, ScientificQuestion


def form_hypotheses(
    llm_complete: Callable,
    question: ScientificQuestion,
    max_hypotheses: int = 2,
) -> list[Hypothesis]:
    """Convert a scientific question into testable hypotheses.

    Each hypothesis must have a predicted outcome that can be
    verified by executing capability actions.

    Args:
        llm_complete: Function(messages) → str.
        question: The question to form hypotheses for.
        max_hypotheses: Max hypotheses to generate.

    Returns list of Hypothesis objects.
    """
    prompt = f"""你是一个假设生成器。基于以下问题，提出最多 {max_hypotheses} 个可验证假设。

问题: {question.question}
背景: {question.context}
类型: {question.q_type}

每个假设必须包含一个"可被行动验证的预测结果"。

输出JSON格式:
{{
  "hypotheses": [
    {{
      "statement": "假设的具体表述",
      "predicted_outcome": "如果假设成立，应该观察到的结果（可被工具验证）",
      "confidence": 0.0-1.0
    }}
  ]
}}
只输出JSON，不要解释。"""

    messages = [
        {"role": "system", "content": "你是一个科学假设生成器。输出结构化JSON。"},
        {"role": "user", "content": prompt},
    ]

    response = llm_complete(messages, temperature=0.7, max_tokens=500)
    if not response:
        return []

    import json
    try:
        # Extract JSON block
        start = response.find("{")
        if start < 0:
            return []
        depth = 0
        for i in range(start, len(response)):
            if response[i] == "{": depth += 1
            elif response[i] == "}":
                depth -= 1
                if depth == 0:
                    data = json.loads(response[start:i+1])
                    break
        else:
            return []

        raw = data.get("hypotheses", [])
    except (json.JSONDecodeError, ValueError):
        return []

    hypotheses = []
    for r in raw[:max_hypotheses]:
        if isinstance(r, dict) and r.get("statement"):
            hypotheses.append(Hypothesis(
                statement=r["statement"],
                predicted_outcome=r.get("predicted_outcome", ""),
                confidence=float(r.get("confidence", 0.5)),
            ))
    return hypotheses
