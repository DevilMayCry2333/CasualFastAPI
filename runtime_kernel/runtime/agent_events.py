"""
agent_events — Agent Observability Event System.

Every step of agent execution emits structured events:
    Planner → Action → MCP Request → MCP Response → Observation → Memory

These events are stored per-session and streamed to WebSocket subscribers
for real-time Agent Timeline rendering in the frontend.

Event types:
    planner         🧠 Planner is thinking or made a decision
    action_start    🔧 Action execution started
    action_result   📎 Action completed (Observation received)
    mcp_request     🌍 MCP tool call sent
    mcp_response    🌍 MCP tool result received
    observation     👁️ Observation fed back to planner
    memory_update   💾 Memory/Fold updated
    world_action    🌐 Agent took a world action
    final_answer    ✨ Agent produced final output
    error           ❌ Error occurred
    system          ⚙️ System event (connection, etc.)
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Callable, Optional


class AgentEvent:
    """A single observable event in the agent's execution trace.

    Every event has a type, timestamp, session association, and payload.
    Events are immutable once created.
    """

    __slots__ = ("id", "session_id", "timestamp", "type", "payload", "round")

    def __init__(
        self,
        session_id: str,
        type: str,
        payload: Optional[dict] = None,
        round: int = 0,
    ) -> None:
        self.id: str = uuid.uuid4().hex[:12]
        self.session_id: str = session_id
        self.timestamp: float = time.time()
        self.type: str = type
        self.payload: dict = payload or {}
        self.round: int = round

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "type": self.type,
            "payload": self.payload,
            "round": self.round,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __repr__(self) -> str:
        return (
            f"AgentEvent(id={self.id[:8]}, type={self.type}, "
            f"round={self.round}, session={self.session_id[:8]})"
        )


class AgentEventBus:
    """Publish-subscribe event bus for agent observability.

    Ingest:   any sync code calls emit(event)
    Storage:  recent events kept per session (ring buffer)
    Stream:   async subscribers (WebSocket) receive events in real-time

    Usage:
        bus = AgentEventBus()
        bus.emit(AgentEvent(session, "planner", {"decision": "search"}))

        # WebSocket subscriber
        queue = bus.subscribe(session_id)  # returns asyncio.Queue
        event = await queue.get()
    """

    def __init__(self, max_events_per_session: int = 500) -> None:
        self._max = max_events_per_session
        self._events: dict[str, list[AgentEvent]] = {}
        self._async_queues: dict[str, list["Any"]] = {}  # session_id -> [asyncio.Queue]
        self._lock = threading.Lock()

    # ── Emit ──

    def emit(self, event: AgentEvent) -> None:
        """Publish an event to storage and all subscribers."""
        sid = event.session_id
        with self._lock:
            # Storage
            if sid not in self._events:
                self._events[sid] = []
            self._events[sid].append(event)
            if len(self._events[sid]) > self._max:
                self._events[sid] = self._events[sid][-self._max:]

            # Async subscribers (put_nowait is thread-safe)
            for q in list(self._async_queues.get(sid, [])):
                try:
                    q.put_nowait(event)
                except Exception:
                    # Queue full or closed — remove it
                    try:
                        self._async_queues[sid].remove(q)
                    except ValueError:
                        pass

    # ── History ──

    def get_events(
        self,
        session_id: str,
        since_id: Optional[str] = None,
        limit: int = 200,
    ) -> list[AgentEvent]:
        """Get stored events for a session, optionally since a specific event ID."""
        events = list(self._events.get(session_id, []))
        if since_id:
            found = False
            for i, e in enumerate(events):
                if e.id == since_id:
                    events = events[i + 1:]
                    found = True
                    break
            if not found:
                events = events[-limit:]
        else:
            events = events[-limit:]
        return events

    def get_events_since(self, session_id: str, timestamp: float) -> list[AgentEvent]:
        """Get events since a given Unix timestamp."""
        return [
            e for e in self._events.get(session_id, [])
            if e.timestamp >= timestamp
        ]

    def clear_session(self, session_id: str) -> None:
        """Clear all events for a session."""
        with self._lock:
            self._events.pop(session_id, None)
            self._async_queues.pop(session_id, None)

    # ── Async subscription (for WebSocket) ──

    def subscribe(self, session_id: str) -> "Any":
        """Create an async queue subscriber for a session.

        Returns an asyncio.Queue that receives new events in real-time.
        The subscriber must call unsubscribe() when done.
        """
        import asyncio
        q = asyncio.Queue(maxsize=500)
        with self._lock:
            if session_id not in self._async_queues:
                self._async_queues[session_id] = []
            self._async_queues[session_id].append(q)
        return q

    def unsubscribe(self, session_id: str, queue: "Any") -> None:
        """Remove an async queue subscriber."""
        with self._lock:
            try:
                self._async_queues[session_id].remove(queue)
            except (ValueError, KeyError):
                pass

    # ── Statistics ──

    def event_count(self, session_id: str) -> int:
        """Number of stored events for a session."""
        return len(self._events.get(session_id, []))

    def all_session_ids(self) -> list[str]:
        """Return all session IDs that have events."""
        return list(self._events.keys())
