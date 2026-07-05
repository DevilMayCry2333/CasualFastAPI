"""
scheduler — Step scheduler for periodic agent execution.

The Runtime Engine does NOT contain infinite while True loops.
Instead, the Scheduler manages the execution cadence.

Key design: **wall-clock pacing**.
  - step() is called
  - We measure how long it took (including LLM inference time)
  - We sleep for the remainder of the interval

This ensures that no matter how fast or slow inference runs,
the GPU gets a guaranteed idle window between steps.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional


class Scheduler:
    """Manages periodic execution of engine steps.

    Uses wall-clock timing to ensure consistent spacing:
        total_cycle = max(step_duration, interval)
        sleep = interval - step_duration  (clamped to minimum 0.5s)

    This prevents GPU saturation even when step_duration < interval.
    """

    def __init__(self) -> None:
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ── Single-step execution helpers ──

    @staticmethod
    def sync_step(step_fn: Callable[[], None]) -> None:
        """Execute a single step synchronously.

        Args:
            step_fn: A callable that performs one engine step.
        """
        step_fn()

    # ── Interval loop ──

    def run_loop(
        self,
        step_fn: Callable[[], None],
        interval: float = 60.0,
        daemon: bool = True,
    ) -> None:
        """Run step_fn every `interval` seconds in a background thread.

        **Wall-clock pacing**: measures step_fn duration and sleeps only
        the remainder of `interval`, so the minimum time between step
        starts is always `interval` seconds.

        This means if step_fn takes 12s and interval is 60s, we sleep
        48s — giving the GPU a guaranteed 80% idle window.

        Args:
            step_fn: Callable that performs one engine step.
            interval: Minimum seconds between step starts (≥ 1.0).
            daemon: If True, thread exits when main thread exits.
        """
        if self._running:
            return

        self._running = True
        effective_interval = max(1.0, interval)

        def _loop():
            while self._running:
                t0 = time.time()
                try:
                    step_fn()
                except Exception:
                    import sys
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                elapsed = time.time() - t0

                # Sleep for the remainder of the interval
                # Guarantees at least 0.5s idle even if step took longer
                # than interval (back-to-back prevention floor)
                remainder = effective_interval - elapsed
                if remainder > 0:
                    time.sleep(remainder)
                else:
                    # Step took longer than interval — still give GPU a
                    # minimal cooldown so it doesn't thrash
                    time.sleep(0.5)

        self._thread = threading.Thread(target=_loop, daemon=daemon)
        self._thread.start()

    def stop(self) -> None:
        """Signal the loop to stop and wait for the thread to finish."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Context manager support ──

    def __enter__(self) -> "Scheduler":
        return self

    def __exit__(self, *args) -> None:
        self.stop()
