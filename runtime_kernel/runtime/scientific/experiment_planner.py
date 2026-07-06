"""
experiment_planner — Designs experiments (sequences of capability actions)
to test hypotheses.

An experiment is a sequence of Action-like steps that the agent
executes via the Action System (Search, Human, MCP, etc.).

Supports:
    - Single-step experiments (one action)
    - Multi-step experiments (sequential actions)
    - Controlled comparisons (A/B-like patterns)
"""

from __future__ import annotations

from typing import Any, Callable

from runtime_kernel.runtime.scientific.models import (
    ExperimentStep,
    ExperimentResult,
    Hypothesis,
)


def design_experiment(
    llm_complete: Callable,
    hypothesis: Hypothesis,
    available_operations: list[dict],
    capabilities_context: str = "",
) -> list[ExperimentStep]:
    """Design an experiment to test a hypothesis.

    Returns a sequence of ExperimentSteps (capability actions).
    Empty list if no experiment can be designed.

    Args:
        llm_complete: Function(messages) → str.
        hypothesis: The hypothesis to test.
        available_operations: List of available operation dicts.
        capabilities_context: Current capabilities prompt context.

    Returns list of ExperimentStep objects.
    """
    ops_section = ""
    if available_operations:
        ops_list = []
        for op in available_operations[:15]:
            name = op.get("name", "?")
            desc = op.get("description", "")[:60]
            params = op.get("parameters", {})
            props = params.get("properties", {})
            param_str = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items() if k != "name")
            ops_list.append(f"  {name}({param_str}): {desc}")
        ops_section = "\n".join(ops_list)

    prompt = f"""你是一个实验设计师。为以下假设设计实验步骤。

假设: {hypothesis.statement}
预测: {hypothesis.predicted_outcome}
置信度: {hypothesis.confidence}

{capabilities_context if capabilities_context else '可用工具:' + ops_section}

设计一组实验步骤来验证这个假设。
每步使用一个可用能力，实验目标是收集证据来支持或反驳假设。

输出JSON格式:
{{
  "experiment_steps": [
    {{
      "capability": "Search | Human | ...",
      "operation": "web_search | ask | ...",
      "parameters": {{...}},
      "expected": "预期这个步骤会观察到什么结果"
    }}
  ]
}}
如果假设在当前能力范围内不可验证，输出 {{"experiment_steps": []}}。
只输出JSON，不要解释。"""

    messages = [
        {"role": "system", "content": "你是一个实验设计师。输出结构化JSON。"},
        {"role": "user", "content": prompt},
    ]

    response = llm_complete(messages, temperature=0.6, max_tokens=800)
    if not response:
        return []

    import json
    try:
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
        raw = data.get("experiment_steps", [])
    except (json.JSONDecodeError, ValueError):
        return []

    steps = []
    for r in raw:
        if isinstance(r, dict) and r.get("capability") and r.get("operation"):
            steps.append(ExperimentStep(
                capability=r["capability"],
                operation=r["operation"],
                parameters=r.get("parameters", {}),
                expected=r.get("expected", ""),
            ))
    return steps
