"""
human_adapter — Human Interaction capability adapter.

Human is a standard Capability, same level as Search, Browser, etc.
Planner never knows about frontend or WebSocket — it only produces
Actions with capability="Human".

Operations:
    ask:  Ask the human a question, then wait for answer
    tell: Send a message to the human (fire-and-forget)

Flow:
    Planner → Action(Human.ask) → HumanAdapter → PendingQuestion
    → Frontend displays → User answers → Observation → Planner continues
"""

from __future__ import annotations

import sys
from typing import Any

from runtime_kernel.runtime.action.adapters import (
    CapabilityAdapter,
    register_capability,
)
from runtime_kernel.runtime.action.models import Action, Capability, Observation


class HumanAdapter(CapabilityAdapter):
    """Human Interaction capability.

    Stores pending questions in-memory. The web layer checks for
    pending questions and delivers answers via continue_with_answer().

    Planner sees Human as just another Capability — it decides:
        "I need more information → use Human.ask"
        "I have something to say → use Human.tell"
    """

    CAPABILITY_NAME = "Human"
    CAPABILITY_DESC = "与人类用户交互：提问或发送消息"

    def __init__(self, event_bus: Any = None) -> None:
        self._event_bus = event_bus
        # session_id → {question, reason, timestamp}
        self._pending_questions: dict[str, dict] = {}

    def execute(self, action: Action, session_id: str = "") -> Observation:
        """Execute a Human action.

        For 'ask': stores the question, returns Observation with
        metadata indicating a human response is needed.

        For 'tell': returns a simple Observation (fire-and-forget).
        """
        op = action.operation
        params = action.parameters

        if op == "ask":
            return self._handle_ask(params, session_id)
        elif op == "tell":
            return self._handle_tell(params, session_id)
        else:
            return Observation(
                success=False,
                error=f"Human operation '{op}' not supported. Use 'ask' or 'tell'.",
            )

    def _handle_ask(self, params: dict, session_id: str) -> Observation:
        """Store the question and return a waiting Observation."""
        question = params.get("question", "")
        reason = params.get("reason", "")

        question_data = {
            "question": question,
            "reason": reason,
            "session_id": session_id,
        }
        self._pending_questions[session_id] = question_data

        print(
            f"  [Human] ❓ Agent asks: {question[:80]}",
            file=sys.stderr,
        )

        # Emit event
        self._emit_event("human_question_created", {
            "session_id": session_id,
            "question": question,
            "reason": reason,
        })

        return Observation(
            success=True,
            content=f"等待人类回答: {question[:100]}",
            metadata={
                "pending_human": True,
                "question": question,
                "reason": reason,
            },
        )

    def _handle_tell(self, params: dict, session_id: str) -> Observation:
        """Send a message to the human (no answer needed)."""
        message = params.get("message", "")

        print(
            f"  [Human] 💬 Agent tells: {message[:80]}",
            file=sys.stderr,
        )

        self._emit_event("human_message", {
            "session_id": session_id,
            "message": message,
        })

        return Observation(
            success=True,
            content=message,
            metadata={"pending_human": False},
        )

    def get_pending_question(self, session_id: str) -> dict | None:
        """Get the pending question for a session, if any."""
        return self._pending_questions.get(session_id)

    def deliver_answer(self, session_id: str, answer: str) -> Observation | None:
        """Deliver a human answer, returning an Observation for the Planner.

        Returns None if there was no pending question.
        """
        question_data = self._pending_questions.pop(session_id, None)
        if not question_data:
            return None

        print(
            f"  [Human] 👤 Human answers: {answer[:80]}",
            file=sys.stderr,
        )

        self._emit_event("human_answer_received", {
            "session_id": session_id,
            "question": question_data.get("question", ""),
            "answer": answer,
        })

        return Observation(
            success=True,
            content=answer,
            metadata={
                "answered_question": question_data.get("question", ""),
                "question_reason": question_data.get("reason", ""),
            },
        )

    def list_operations(self) -> list[dict]:
        return [
            {
                "name": "ask",
                "description": "向人类用户提问，等待回答",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "向人类提出的问题"},
                        "reason": {"type": "string", "description": "提问的原因"},
                    },
                    "required": ["question"],
                },
            },
            {
                "name": "tell",
                "description": "向人类用户发送消息（无需回答）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "发送给人类的消息"},
                    },
                    "required": ["message"],
                },
            },
        ]

    def get_capability_info(self) -> Capability:
        return Capability(
            name=self.CAPABILITY_NAME,
            description=self.CAPABILITY_DESC,
            enabled=True,
        )

    def _emit_event(self, event_type: str, payload: dict) -> None:
        if not self._event_bus:
            return
        try:
            from runtime_kernel.runtime.agent_events import AgentEvent
            sid = payload.pop("session_id", "")
            if sid:
                self._event_bus.emit(AgentEvent(
                    session_id=sid,
                    type=event_type,
                    payload=payload,
                ))
        except Exception:
            pass


# Register in the global capability registry
register_capability("Human", HumanAdapter)
