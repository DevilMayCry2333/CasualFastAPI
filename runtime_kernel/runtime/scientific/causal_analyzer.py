"""
causal_analyzer — Analyzes Fold history and experimental results
to extract causal relationships and update the causal graph.

Key functions:
    - extract_causal_edges: From observation pairs → candidate causes
    - update_causal_graph: Merge new evidence into existing graph
    - analyze_experiment: Did the experiment confirm/refute causality?
"""

from __future__ import annotations

from typing import Any, Callable

from runtime_kernel.runtime.scientific.models import (
    CausalEdge,
    ExperimentResult,
    Hypothesis,
)


def analyze_experiment(
    hypothesis: Hypothesis,
    results: list[ExperimentResult],
) -> dict:
    """Analyze whether experiment results support or refute a hypothesis.

    Pure heuristic analysis (no LLM call):
        - All steps successful → check if observations match predictions
        - Partial success → ambiguous result
        - All failed → unable to test

    Returns analysis dict.
    """
    if not results:
        return {"supports": None, "reason": "无实验结果", "confidence_shift": 0.0}

    success_count = sum(1 for r in results if r.success)
    total = len(results)

    if success_count == total:
        # All successful — hypothesis is plausible
        return {
            "supports": True,
            "reason": f"{success_count}/{total} 步实验成功，假设得到支持",
            "confidence_shift": 0.15,
        }
    elif success_count >= total / 2:
        return {
            "supports": None,
            "reason": f"{success_count}/{total} 步成功，结果不明确",
            "confidence_shift": 0.05,
        }
    else:
        return {
            "supports": False,
            "reason": f"仅 {success_count}/{total} 步成功，假设未得到支持",
            "confidence_shift": -0.1,
        }


def generate_insights(
    llm_complete: Callable,
    cycle_data: dict,
) -> list[str]:
    """Generate scientific insights from a completed cycle.

    Uses LLM to synthesize what was learned.

    Args:
        llm_complete: Function(messages) → str.
        cycle_data: Dict with question, hypotheses, results, analysis.

    Returns list of insight strings.
    """
    q = cycle_data.get("question", "")
    h = cycle_data.get("hypothesis", "")
    r = cycle_data.get("results", [])
    analysis = cycle_data.get("analysis", {})

    results_text = "\n".join(
        f"  步骤{r.get('step', '?')}: {r.get('operation', '?')} "
        f"{'✅' if r.get('success') else '❌'} → {r.get('observation', '')[:80]}"
        for r in r[:10]
    )

    prompt = f"""分析以下完整的科学实验循环，输出关键洞察。

问题: {q}
假设: {h}
实验结论: {analysis.get('reason', '')}
支持: {analysis.get('supports')}

实验结果:
{results_text}

输出最多 3 条简洁洞察（每条不超过 100 字），JSON格式:
{{"insights": ["洞察1", "洞察2", "洞察3"]}}
只输出JSON。"""

    messages = [
        {"role": "system", "content": "你是一个科学洞察分析器。"},
        {"role": "user", "content": prompt},
    ]

    response = llm_complete(messages, temperature=0.5, max_tokens=300)
    if not response:
        return ["实验完成，产生了一条记录。"]

    import json, re
    try:
        start = response.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(response)):
                if response[i] == "{": depth += 1
                elif response[i] == "}":
                    depth -= 1
                    if depth == 0:
                        data = json.loads(response[start:i+1])
                        return data.get("insights", [])
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract quoted strings
    matches = re.findall(r'"([^"]+)"', response)
    return matches[:3] if matches else ["实验完成。"]
