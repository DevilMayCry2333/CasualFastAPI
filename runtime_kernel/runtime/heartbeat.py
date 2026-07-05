"""
heartbeat — HeartbeatManager: background aliveness for persistent existence.

The heartbeat is the agent's "subconscious." Between user interactions,
it runs on a background thread and:

  1. Drives drift — curiosity decays, boredom creeps up, belonging fades
  2. Thought pool refresh — if drives cross thresholds, regenerate thoughts
  3. Memory consolidation — ensure recent states are indexed
  4. Auto-save — the session persists even when nobody is talking to it

When the user next speaks to the agent, it doesn't wake from a frozen state.
It wakes from a state that has been *living* — drives have shifted, thoughts
have been brewing, the identity has been quietly simmering.

The interval is configurable. Default 30s balances responsiveness and cost.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional


class HeartbeatManager:
    """Background pulse that maintains session aliveness.

    The engine registers a callback that is invoked on each tick.
    The heartbeat does not own sessions or the engine — it just
    calls the callback.

    Usage:
        def on_tick(tick_count):
            # update drives, consolidate, etc.
            pass

        hb = HeartbeatManager(on_tick, interval=30.0)
        hb.start()
        ...
        hb.stop()
    """

    def __init__(
        self,
        callback: Optional[Callable[[int], None]] = None,
        interval: float = 30.0,
    ) -> None:
        self._callback = callback
        self._interval = max(5.0, interval)  # minimum 5 seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._tick_count: int = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def set_callback(self, callback: Callable[[int], None]) -> None:
        """Set or replace the tick callback."""
        self._callback = callback

    def start(self) -> None:
        """Start the background heartbeat thread."""
        if self._running:
            return
        if self._callback is None:
            raise RuntimeError(
                "HeartbeatManager cannot start without a callback. "
                "Set one via constructor or set_callback()."
            )

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the heartbeat to stop and wait for the thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None

    def _loop(self) -> None:
        """Main heartbeat loop."""
        while self._running:
            start = time.time()

            self._tick_count += 1
            try:
                if self._callback:
                    self._callback(self._tick_count)
            except Exception as e:
                # Never let the heartbeat die — log and continue
                import sys
                print(
                    f"  [heartbeat] tick {self._tick_count} error: {e}",
                    file=sys.stderr,
                )

            # Sleep for the remainder of the interval
            elapsed = time.time() - start
            sleep_time = max(0.5, self._interval - elapsed)
            time.sleep(sleep_time)

    # ── Context manager support ──

    def __enter__(self) -> HeartbeatManager:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
