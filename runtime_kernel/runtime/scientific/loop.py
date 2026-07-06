"""
loop — ScientificLoop: orchestrates one complete scientific cycle.

The cycle:
    Question → Hypothesis → Experiment Plan → Execute → Analyze → Theory Update

This is called periodically by the RuntimeEngine after autonomous steps.
It uses the LLM for generation but all outputs are structured data.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Optional

from runtime_kernel.runtime.action import Action
from runtime_kernel.runtime.scientific.models import (
    CycleSummary,
    ExperimentResult,
    ExperimentStep,
    Hypothesis,
    ScientificQuestion,
)
from runtime_kernel.runtime.scientific.question_generator import generate_questions
from runtime_kernel.runtime.scientific.hypothesis_layer import form_hypotheses
from runtime_kernel.runtime.scientific.experiment_planner import design_experiment
from runtime_kernel.runtime.scientific.causal_analyzer import (
    analyze_experiment,
    generate_insights,
)
from runtime_kernel.runtime.scientific.theory_updater import (
    compute_theory_delta,
    update_world_model,
)


class ScientificLoop:
    """Orchestrates the scientific method loop.

    Integrates with RuntimeEngine: receives LLM callbacks, action executor
    references, and session state. Produces structured scientific output
    that feeds into the agent's knowledge and world model.

    The loop does NOT modify code or system prompts. It only produces
    structured data that the agent can use.
    """

    def __init__(
        self,
        llm_callback: Callable,
        action_executor: Optional[Any] = None,
        interval: int = 10,  # Run science cycle every N rounds
    ) -> None:
        self._llm = llm_callback
        self._action_executor = action_executor
        self._interval = interval
        self._cycle_count: int = 0
        self._last_cycle_round: int = 0
        self._history: list[CycleSummary] = []

    @property
    def interval(self) -> int:
        return self._interval

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    def get_history(self) -> list[dict]:
        return [c.to_dict() for c in self._history]

    def should_run(self, current_round: int) -> bool:
        """Check if a science cycle should run at this round."""
        if current_round < 5:  # Don't run too early
            return False
        if current_round - self._last_cycle_round >= self._interval:
            return True
        return False

    def run_cycle(
        self,
        current_round: int,
        world_model: dict,
        uncertainties: list,
        open_questions: list[str],
        recent_failures: list[str],
        available_operations: list[dict],
        capabilities_context: str = "",
    ) -> CycleSummary:
        """Run one complete scientific cycle.

        Args:
            world_model: Current world model dict.
            uncertainties: Current uncertainty list.
            open_questions: Current open questions.
            recent_failures: Recent action failures.
            available_operations: Available capability operations.
            capabilities_context: Formatted capabilities for LLM context.

        Returns CycleSummary with all results.
        """
        self._cycle_count += 1
        cycle_num = self._cycle_count
        print(f"  [Science] 🔬 Cycle {cycle_num} starting", file=sys.stderr)

        summary = CycleSummary(cycle=cycle_num)

        # ── Step 1: Generate questions ──
        questions = generate_questions(
            self._llm, world_model, uncertainties,
            open_questions, recent_failures,
        )
        if not questions:
            print(f"  [Science] No questions generated", file=sys.stderr)
            self._history.append(summary)
            return summary

        top_q = questions[0]
        summary.question = top_q
        print(f"  [Science] ❓ Q{cycle_num}: {top_q.question[:60]}...", file=sys.stderr)

        # ── Step 2: Form hypotheses ──
        hypotheses = form_hypotheses(self._llm, top_q, max_hypotheses=2)
        if not hypotheses:
            print(f"  [Science] No hypotheses formed", file=sys.stderr)
            self._history.append(summary)
            return summary

        summary.hypotheses = hypotheses
        print(f"  [Science] 💡 {len(hypotheses)} hypothesis(es)", file=sys.stderr)

        # ── Step 3: Design experiments ──
        for h in hypotheses:
            steps = design_experiment(
                self._llm, h, available_operations, capabilities_context,
            )
            if not steps:
                continue

            print(f"  [Science] 🧪 Experiment: {len(steps)} step(s)", file=sys.stderr)

            # ── Step 4: Execute experiments via ActionExecutor ──
            results: list[ExperimentResult] = []
            for i, step in enumerate(steps):
                if not self._action_executor:
                    results.append(ExperimentResult(
                        step=i, capability=step.capability,
                        operation=step.operation, success=False,
                        observation="No action executor available",
                    ))
                    continue

                action = Action(
                    capability=step.capability,
                    operation=step.operation,
                    parameters=step.parameters,
                )

                t0 = time.time()
                obs = self._action_executor.execute(action, session_id="")
                elapsed = int((time.time() - t0) * 1000)

                obs_text = ""
                if obs.success and obs.content:
                    if isinstance(obs.content, list):
                        obs_text = " ".join(str(x)[:100] for x in obs.content[:3])
                    else:
                        obs_text = str(obs.content)[:200]

                results.append(ExperimentResult(
                    step=i,
                    capability=step.capability,
                    operation=step.operation,
                    success=obs.success,
                    observation=obs_text,
                    elapsed_ms=elapsed,
                ))
                print(
                    f"  [Science]   Step {i+1}: {step.capability}.{step.operation} "
                    f"{'✅' if obs.success else '❌'} ({elapsed}ms)",
                    file=sys.stderr,
                )

            summary.results = results

            # ── Step 5: Analyze experiment ──
            analysis = analyze_experiment(h, results)
            print(f"  [Science] 📊 Analysis: {analysis.get('reason', '')}", file=sys.stderr)

            # ── Step 6: Generate insights ──
            cycle_data = {
                "question": top_q.question,
                "hypothesis": h.statement,
                "results": [r.to_dict() for r in results],
                "analysis": analysis,
            }
            insights = generate_insights(self._llm, cycle_data)
            summary.insights = insights

            # ── Step 7: Compute theory update ──
            theory_delta = compute_theory_delta(h, analysis, insights, world_model)
            summary.theory_delta = theory_delta

            for ins in insights:
                print(f"  [Science] 💡 Insight: {ins[:80]}", file=sys.stderr)

            # Only test first hypothesis per cycle
            break

        self._last_cycle_round = current_round
        self._history.append(summary)
        print(f"  [Science] ✅ Cycle {cycle_num} complete", file=sys.stderr)
        return summary
