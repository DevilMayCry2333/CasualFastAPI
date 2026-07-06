"""
hypothesis_generator — Stochastic hypothesis generation for Exploration Layer.

Generates diverse, high-variance hypotheses that the Cognitive Layer
can evaluate. Does NOT execute any actions — only produces candidates.
"""

from __future__ import annotations

import random
from typing import Any, Callable


class StochasticHypothesisGenerator:
    """Generates exploratory hypotheses with controlled randomness.

    The Exploration Layer injects stochastic noise to avoid local optima.
    Outputs are candidate hypotheses for the Cognitive Layer to evaluate.
    """

    def __init__(self, temperature: float = 0.8) -> None:
        self._temperature = temperature  # 0.0 = deterministic, 1.0 = max random
        self._generated_count: int = 0

    def generate_candidates(
        self,
        llm_complete: Callable,
        world_model: dict,
        prob_wm_context: str,
        causal_graph_context: str,
        uncertainties: list,
        count: int = 3,
    ) -> list[dict]:
        """Generate diverse candidate hypotheses with controlled randomness.

        Each candidate includes a statement and an action proposal.
        These are candidates — NOT executed. The Cognitive Layer evaluates them.

        Args:
            llm_complete: LLM callback.
            world_model: Current world model.
            prob_wm_context: Probabilistic world model context string.
            causal_graph_context: Causal graph context string.
            uncertainties: Current uncertainties.
            count: Number of candidates to generate.

        Returns list of candidate dicts.
        """
        self._generated_count += 1

        randomness_hint = (
            "highly creative, unexpected connections"
            if self._temperature > 0.7
            else "balanced between novelty and plausibility"
            if self._temperature > 0.4
            else "cautious, incremental improvements"
        )

        prompt_parts = [
            f"你是一个探索性假设生成器（温度={self._temperature}）。",
            f"风格: {randomness_hint}",
            "",
        ]
        if prob_wm_context:
            prompt_parts.append(prob_wm_context)
        if causal_graph_context:
            prompt_parts.append(causal_graph_context)
        if uncertainties:
            prompt_parts.append(f"\n当前高不确定性区域: {'; '.join(str(u)[:80] for u in uncertainties[:5])}")

        prompt_parts.append(f"""
生成 {count} 个多样化的探索性假设。
每个假设包含一个可行的行动提案（基于 Search / Human 能力）。

输出JSON格式:
{{
  "candidates": [
    {{
      "hypothesis": "假设陈述",
      "rationale": "为什么这个假设值得探索",
      "suggested_action": {{
        "capability": "Search 或 Human",
        "operation": "具体操作",
        "parameters": {{...}}
      }},
      "expected_information_gain": 0.0-1.0,
      "novelty_score": 0.0-1.0
    }}
  ]
}}
只输出JSON，不要解释。""")

        messages = [
            {"role": "system", "content": "你是一个探索性假设生成器。输出结构化JSON。"},
            {"role": "user", "content": "\n".join(prompt_parts)},
        ]

        response = llm_complete(messages, temperature=0.9, max_tokens=600)
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
                        data = json.loads(response[start:i+1])
                        break
            else:
                return []
            return data.get("candidates", [])[:count]
        except (json.JSONDecodeError, ValueError):
            return []

    def set_temperature(self, t: float) -> None:
        """Adjust exploration temperature."""
        self._temperature = max(0.0, min(1.0, t))
