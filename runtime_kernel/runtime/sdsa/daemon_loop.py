"""
daemon_loop — Autonomous background daemon loop for SDSA Runtime.

This is the core of the self-driven system. A background thread runs
continuously:
    1. Generate research goals (from Fold/uncertainty)
    2. Enqueue experiments for top goals
    3. Dequeue and execute experiments (via Core Layer only)
    4. Collect observations
    5. Update causal graph and world model
    6. Sleep / budget control

The loop does NOT depend on user requests. It is truly autonomous.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Any, Callable, Optional

from runtime_kernel.runtime.sdsa.models import SDSACycleResult
from runtime_kernel.runtime.sdsa.goal_generator import generate_goals
from runtime_kernel.runtime.sdsa.experiment_queue import ExperimentQueue


class AutonomousDaemonLoop:
    """Background daemon that drives the self-scientific loop.

    Starts a daemon thread that continuously:
        1. Generates research goals
        2. Enqueues experiments
        3. Executes experiments through Core
        4. Updates causal graph + world model

    Usage:
        loop = AutonomousDaemonLoop(llm_callback=..., core_validator=..., ...)
        loop.start()   # starts background thread
        loop.stop()    # graceful shutdown
    """

    def __init__(
        self,
        llm_callback: Callable,
        core_validator: Any,
        causal_graph: Any,
        prob_wm: Any,
        experiment_queue: ExperimentQueue,
        session_provider: Callable,
        interval: float = 60.0,
        max_actions_per_cycle: int = 3,
        event_bus: Any = None,
    ) -> None:
        self._llm = llm_callback
        self._core = core_validator
        self._causal_graph = causal_graph
        self._prob_wm = prob_wm
        self._queue = experiment_queue
        self._session_provider = session_provider  # Callable[[], session]
        self._event_bus = event_bus

        self._interval = interval
        self._max_actions = max_actions_per_cycle

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._cycle_count = 0
        self._histories: list[SDSACycleResult] = []

    # ── Lifecycle ──

    def start(self) -> None:
        """Start the daemon loop in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="sdsa-daemon",
        )
        self._thread.start()
        print(f"  [SDSA] Daemon loop started (interval={self._interval}s)", file=sys.stderr)

    def stop(self) -> None:
        """Gracefully stop the daemon loop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        print(f"  [SDSA] Daemon loop stopped", file=sys.stderr)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    # ── The loop ──

    def _loop(self) -> None:
        """Main daemon loop body. Runs in background thread."""
        while self._running:
            try:
                session = self._session_provider()
                if session:
                    self._run_one_cycle(session)
            except Exception as e:
                print(f"  [SDSA] Cycle error: {e}", file=sys.stderr)

            # Sleep with early exit check
            for _ in range(int(self._interval * 2)):  # check every 0.5s
                if not self._running:
                    return
                time.sleep(0.5)

    def _run_one_cycle(self, session: Any) -> None:
        """Execute one complete SDSA cycle for a session."""
        self._cycle_count += 1
        cycle_num = self._cycle_count
        result = SDSACycleResult(cycle=cycle_num)

        print(f"  [SDSA] 🔄 Cycle {cycle_num} starting", file=sys.stderr)

        # ── 1. Gather state ──
        world_model = session.state.get("world_model", {})
        uncertainties = session.state.get("uncertainties", [])
        recent_failures = []

        existing_goals = getattr(session, "_sdsa_goals", [])
        if not hasattr(session, "_sdsa_goals"):
            session._sdsa_goals = []

        causal_ctx = self._causal_graph.format_for_prompt() if self._causal_graph else ""
        prob_ctx = self._prob_wm.format_for_prompt() if self._prob_wm else ""

        # ── 2. Generate goals ──
        new_goals = generate_goals(
            llm_complete=self._llm,
            world_model=world_model,
            uncertainties=uncertainties,
            recent_failures=recent_failures,
            existing_goals=session._sdsa_goals,
            causal_graph_context=causal_ctx,
            prob_wm_context=prob_ctx,
            max_goals=2,
        )

        if not new_goals:
            print(f"  [SDSA] No new goals, skipping cycle", file=sys.stderr)
            return

        # Add new goals to session
        top_goal = new_goals[0]
        session._sdsa_goals.append(top_goal)
        result.goal = top_goal
        print(f"  [SDSA] 🎯 Goal: {top_goal.statement[:60]}...", file=sys.stderr)

        # ── 3. Enqueue experiment ──
        # Simple experiment: test the top hypothesis by executing a Search action
        hypothesis = f"通过搜索验证: {top_goal.statement[:60]}"
        variants = [
            {
                "name": "search",
                "action": {
                    "capability": "Search",
                    "operation": "web_search",
                    "parameters": {"query": top_goal.statement[:100]},
                },
            },
        ]
        entry = self._queue.enqueue(
            goal_id=top_goal.id,
            hypothesis=hypothesis,
            variants=variants,
            cost_estimate=2000,
            expected_information_gain=top_goal.information_gain_estimate,
        )

        # ── 4. Dequeue and execute ──
        exp = self._queue.dequeue()
        if not exp or not self._core:
            return

        actions_executed = 0
        for variant in exp.variants[:self._max_actions]:
            action_data = variant.get("action", {})
            if not action_data:
                continue

            from runtime_kernel.runtime.action import Action
            action = Action(
                capability=action_data.get("capability", "Search"),
                operation=action_data.get("operation", "web_search"),
                parameters=action_data.get("parameters", {}),
            )

            observation = self._core.validate_and_execute(action, session_id=session.id)
            actions_executed += 1

            # Record result
            exp.results.append({
                "variant": variant.get("name", "?"),
                "success": observation.success,
                "observation": str(observation.content)[:200] if observation.content else "",
            })

            # Update probabilistic WM
            if self._prob_wm:
                self._prob_wm.observe(
                    concept=f"{action.capability} → {action.operation}",
                    outcome=1.0 if observation.success else 0.0,
                )

            # Emit event
            self._emit("sdsa_action", {
                "cycle": cycle_num,
                "capability": action.capability,
                "operation": action.operation,
                "success": observation.success,
            })

        result.actions_executed = actions_executed

        # ── 5. Complete experiment ──
        self._queue.mark_completed(exp.id, exp.results, conclusion=exp.hypothesis)
        result.experiments_run = 1

        # ── 6. Causal update ──
        if self._causal_graph:
            self._causal_graph.observe_support("Search", "knowledge", 0.1)
            self._causal_graph.observe_support("Experiment", "knowledge", 0.15)
            result.causal_updates = 2

        # ── 7. World model update ──
        if self._prob_wm:
            result.world_model_updates = 1

        # Store in session for access
        if not hasattr(session, "_sdsa_history"):
            session._sdsa_history = []
        session._sdsa_history.append(result)

        self._histories.append(result)
        print(
            f"  [SDSA] ✅ Cycle {cycle_num} done: "
            f"{actions_executed} actions, "
            f"{result.causal_updates} causal updates",
            file=sys.stderr,
        )

        # Emit cycle complete
        self._emit("sdsa_cycle", {
            "cycle": cycle_num,
            "goal": top_goal.statement[:80],
            "actions": actions_executed,
        })

    def _emit(self, event_type: str, payload: dict) -> None:
        """Emit an agent event."""
        if not self._event_bus:
            return
        try:
            from runtime_kernel.runtime.agent_events import AgentEvent
            sid = payload.pop("session_id", "sdsa")
            self._event_bus.emit(AgentEvent(
                session_id=sid,
                type=event_type,
                payload=payload,
            ))
        except Exception:
            pass

    # ── Status ──

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "interval": self._interval,
            "queue": self._queue.get_stats(),
        }

    def get_history(self, limit: int = 10) -> list[dict]:
        return [r.to_dict() for r in self._histories[-limit:]]
