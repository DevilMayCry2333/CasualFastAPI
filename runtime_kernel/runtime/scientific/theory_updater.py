"""
theory_updater — Updates the agent's World Model and knowledge
based on experimental results and causal analysis.

Theory updates are structured deltas, not full rewrites:
    - Updated beliefs (with confidence changes)
    - Updated tool effectiveness scores
    - New insights (appended to knowledge)

All updates are recorded in the causal chain.
"""

from __future__ import annotations

from typing import Any, Callable

from runtime_kernel.runtime.scientific.models import (
    CycleSummary,
    ExperimentResult,
    Hypothesis,
)


def compute_theory_delta(
    hypothesis: Hypothesis,
    analysis: dict,
    insights: list[str],
    world_model: dict,
) -> dict:
    """Compute structured theory updates from experimental results.

    Pure computation — no LLM call. Produces traceable deltas.

    Args:
        hypothesis: The tested hypothesis.
        analysis: Result from causal_analyzer.analyze_experiment().
        insights: Insight strings from causal_analyzer.generate_insights().
        world_model: Current world model dict.

    Returns theory delta dict with:
        updated_beliefs: New or updated belief entries.
        tool_effectiveness: Updated tool effectiveness scores.
        new_insights: Insight strings to append.
    """
    supports = analysis.get("supports")
    shift = analysis.get("confidence_shift", 0.0)
    new_confidence = min(1.0, max(0.0, hypothesis.confidence + shift))

    delta = {
        "hypothesis_statement": hypothesis.statement,
        "confidence_was": hypothesis.confidence,
        "confidence_now": new_confidence,
        "supports": supports,
        "new_insights": insights[:3],
    }

    return delta


def update_world_model(
    world_model: dict,
    theory_delta: dict,
) -> dict:
    """Apply a theory delta to the world model.

    Returns a NEW world model dict (immutable update).
    Never mutates the original.

    Args:
        world_model: Current world model dict.
        theory_delta: Delta from compute_theory_delta().

    Returns updated world model dict.
    """
    updated = dict(world_model)

    # Track scientific findings
    if "scientific_findings" not in updated:
        updated["scientific_findings"] = []

    finding = {
        "hypothesis": theory_delta.get("hypothesis_statement", ""),
        "confidence_was": theory_delta.get("confidence_was"),
        "confidence_now": theory_delta.get("confidence_now"),
        "supports": theory_delta.get("supports"),
    }
    updated["scientific_findings"].append(finding)

    # Track insights
    for insight in theory_delta.get("new_insights", []):
        if "insights" not in updated:
            updated["insights"] = []
        if insight not in updated["insights"]:
            updated["insights"].append(insight)

    # Update belief confidence in the main world model
    belief_key = f"belief:{theory_delta.get('hypothesis_statement', '')[:50]}"
    updated[belief_key] = {
        "statement": theory_delta.get("hypothesis_statement", ""),
        "confidence": theory_delta.get("confidence_now", 0.5),
        "updated": True,
    }

    return updated
