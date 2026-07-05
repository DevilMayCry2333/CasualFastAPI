"""
communication — Event-driven multi-agent communication layer.

Design philosophy:
    Agent A does NOT send text directly into Agent B's prompt.
    Instead:
        Agent A acts → World Event → Communication Layer
        → Agent B perceives → interprets → decides to respond

    Communication is a CAUSAL EVENT, not a chat channel.
    Every Message becomes a CausalEntry in the causal chain.

Architecture:
    EventBus       — world events are published here, agents subscribe
    Mailbox        — each agent has one, incoming messages arrive here
    Message        — structured causal event between agents
    CommunicationManager — routes messages, manages subscriptions
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

from runtime_kernel.runtime.models import (
    EXTENDED_MESSAGE_TYPES,
    MAILBOX_MAX_SIZE,
    MessageType,
)


# ── Message ──


class Message:
    """A causal event traveling between agents.

    Message is NOT just text. It's a causal event that can change
    another agent's state. Every message is recorded as a CausalEntry.

    Fields:
        id: Unique identifier.
        from_agent: Sender session ID.
        to_agent: Recipient session ID (or "broadcast").
        msg_type: One of MessageType enum values.
        content: Structured content dict.
        timestamp: When the message was created.
        world_tick: World tick when this message was sent.
        causal_parent: Previous causal entry ID this message relates to.
        world_room: Room where this message originated.
    """

    __slots__ = (
        "id", "from_agent", "to_agent", "msg_type", "content",
        "timestamp", "world_tick", "causal_parent", "world_room",
    )

    def __init__(
        self,
        from_agent: str,
        to_agent: str,
        msg_type: str = "observation",
        content: Optional[dict] = None,
        world_tick: int = 0,
        causal_parent: str = "",
        world_room: str = "",
    ) -> None:
        self.id: str = uuid.uuid4().hex[:12]
        self.from_agent: str = from_agent
        self.to_agent: str = to_agent
        self.msg_type: str = msg_type if msg_type in EXTENDED_MESSAGE_TYPES else "observation"
        self.content: dict = content or {}
        self.timestamp: float = time.time()
        self.world_tick: int = world_tick
        self.causal_parent: str = causal_parent
        self.world_room: str = world_room

    @property
    def summary(self) -> str:
        """Short one-line summary for logging/prompt injection."""
        body = str(self.content.get("text", self.content.get("statement", "")))
        if len(body) > 80:
            body = body[:77] + "..."
        return f"[{self.msg_type}] {self.from_agent[:8]}→{self.to_agent[:8]}: {body}"

    @property
    def label(self) -> str:
        """Human-readable label for prompt display."""
        labels = {
            "observation": "👁 观察",
            "question": "❓ 提问",
            "answer": "💡 回答",
            "hypothesis": "⊕ 假设",
            "plan": "📋 计划",
            "request": "📩 请求",
            "warning": "⚠ 警告",
            "event": "⊙ 事件",
            "report": "📄 报告",
            "share_memory": "🔗 共享记忆",
            "broadcast": "📢 广播",
        }
        return labels.get(self.msg_type, f"[{self.msg_type}]")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "msg_type": self.msg_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "world_tick": self.world_tick,
            "causal_parent": self.causal_parent,
            "world_room": self.world_room,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        msg = cls(
            from_agent=d.get("from_agent", ""),
            to_agent=d.get("to_agent", ""),
            msg_type=d.get("msg_type", "observation"),
            content=d.get("content", {}),
            world_tick=int(d.get("world_tick", 0)),
            causal_parent=d.get("causal_parent", ""),
            world_room=d.get("world_room", ""),
        )
        msg.id = d.get("id", msg.id)
        msg.timestamp = float(d.get("timestamp", msg.timestamp))
        return msg

    def __repr__(self) -> str:
        return (
            f"Message(id={self.id[:8]}, {self.from_agent[:8]}→{self.to_agent[:8]}, "
            f"type={self.msg_type})"
        )


# ── Mailbox ──


class Mailbox:
    """Per-agent message inbox.

    Messages arrive here from CommunicationManager.
    The agent reads them during step() and decides how to respond.

    NOT a prompt. This is Runtime input.
    """

    def __init__(self, agent_id: str) -> None:
        self._agent_id: str = agent_id
        self._messages: list[Message] = []
        self._lock = threading.Lock()

    @property
    def messages(self) -> list[Message]:
        """All current messages (oldest first)."""
        with self._lock:
            return list(self._messages)

    def count(self) -> int:
        with self._lock:
            return len(self._messages)

    def has_unread(self) -> bool:
        with self._lock:
            return len(self._messages) > 0

    def receive(self, msg: Message) -> None:
        """Deposit a message into the mailbox.

        If the mailbox is full, discards the oldest message.
        """
        with self._lock:
            self._messages.append(msg)
            if len(self._messages) > MAILBOX_MAX_SIZE:
                self._messages = self._messages[-MAILBOX_MAX_SIZE:]

    def pop_all(self) -> list[Message]:
        """Retrieve and clear all messages.

        Called at the start of an agent step.
        Returns messages in FIFO order.
        """
        with self._lock:
            msgs = list(self._messages)
            self._messages.clear()
            return msgs

    def pop_by_type(self, msg_type: str) -> list[Message]:
        """Pop messages of a specific type."""
        kept = []
        matched = []
        with self._lock:
            for msg in self._messages:
                if msg.msg_type == msg_type:
                    matched.append(msg)
                else:
                    kept.append(msg)
            self._messages = kept
        return matched

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()

    def to_dict(self) -> list[dict]:
        return [m.to_dict() for m in self.messages]

    @classmethod
    def from_dict(cls, agent_id: str, data: list[dict]) -> "Mailbox":
        mb = cls(agent_id)
        for item in data:
            mb._messages.append(Message.from_dict(item))
        return mb


# ── EventBus ──


class EventBus:
    """World event broadcast system.

    World changes are automatically published as events here.
    Agents can subscribe to specific event types or receive all.

    Events are NOT direct messages. They are world observations
    that agents may or may not perceive.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscriptions: dict[str, set[str]] = defaultdict(set)
        # type -> set of agent_ids subscribed
        self._event_history: list[dict] = []
        self._max_history = 50

    def subscribe(self, agent_id: str, event_type: str = "*") -> None:
        """Subscribe an agent to an event type.

        Args:
            agent_id: The session ID of the subscribing agent.
            event_type: Event type to subscribe to, or "*" for all.
        """
        with self._lock:
            self._subscriptions[event_type].add(agent_id)

    def unsubscribe(self, agent_id: str, event_type: str = "*") -> None:
        """Remove an agent's subscription."""
        with self._lock:
            self._subscriptions[event_type].discard(agent_id)

    def unsubscribe_all(self, agent_id: str) -> None:
        """Remove an agent from all subscriptions."""
        with self._lock:
            for subs in self._subscriptions.values():
                subs.discard(agent_id)

    def publish(self, event: dict) -> list[str]:
        """Publish a world event to all subscribed agents.

        Args:
            event: Dict with keys:
                - type: Event type string
                - content: Event content dict
                - room: Where the event occurred
                - source: Agent or system that caused it
                - tick: World tick

        Returns:
            List of agent IDs that should receive this event.
        """
        with self._lock:
            event_type = event.get("type", "event")
            content = event.get("content", {})
            room = event.get("room", "")
            source = event.get("source", "world")
            tick = event.get("tick", 0)

            record = {
                "type": event_type,
                "content": content,
                "room": room,
                "source": source,
                "tick": tick,
                "timestamp": time.time(),
                "id": uuid.uuid4().hex[:12],
            }

            self._event_history.append(record)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

            # Determine recipients: subscribers to this type + subscribers to "*"
            recipients: set[str] = set()
            for etype in (event_type, "*"):
                if etype in self._subscriptions:
                    recipients.update(self._subscriptions[etype])

            return list(recipients)

    def get_recent_events(self, n: int = 10) -> list[dict]:
        """Get recent events for prompt context."""
        with self._lock:
            return list(self._event_history[-n:])

    def get_subscriber_count(self) -> dict[str, int]:
        """Get subscription counts per type (for debugging)."""
        with self._lock:
            return {k: len(v) for k, v in self._subscriptions.items()}

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "subscriptions": {k: list(v) for k, v in self._subscriptions.items()},
                "event_history": list(self._event_history[-20:]),
            }


# ── CommunicationManager ──


class CommunicationManager:
    """Central communication layer for multi-agent interaction.

    Responsibilities:
        - Message routing (direct, broadcast)
        - Mailbox management
        - EventBus management
        - Message → causal chain integration

    RuntimeEngine does NOT manage agent communication directly.
    All communication goes through CommunicationManager.

    The manager does NOT force agents to respond.
    It only delivers information. Agents decide autonomously.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mailboxes: dict[str, Mailbox] = {}
        self._event_bus: EventBus = EventBus()
        self._sent_messages: list[Message] = []
        self._max_sent_history = 100

    # ── Properties ──

    @property
    def event_bus(self) -> EventBus:
        """The shared EventBus."""
        return self._event_bus

    @property
    def mailboxes(self) -> dict[str, Mailbox]:
        """All mailboxes (read-only view)."""
        return dict(self._mailboxes)

    @property
    def sent_messages(self) -> list[Message]:
        """All messages sent (recent history)."""
        return list(self._sent_messages[-self._max_sent_history:])

    def get_sent_messages(self, agent_id: str, n: int = 20) -> list[Message]:
        """Get messages sent BY a specific agent.

        Args:
            agent_id: The sender's session ID.
            n: Max messages to return.

        Returns:
            List of Messages sent by this agent, newest first.
        """
        with self._lock:
            agent_msgs = [
                m for m in self._sent_messages
                if m.from_agent == agent_id
            ]
            return list(reversed(agent_msgs[-n:]))

    # ── Agent lifecycle ──

    def register_agent(self, agent_id: str) -> Mailbox:
        """Register an agent in the communication system.

        Creates a mailbox and subscribes to broadcast events.

        Args:
            agent_id: The session ID.

        Returns:
            The agent's Mailbox.
        """
        with self._lock:
            mb = Mailbox(agent_id)
            self._mailboxes[agent_id] = mb
            self._event_bus.subscribe(agent_id, "*")
            return mb

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the communication system."""
        with self._lock:
            self._mailboxes.pop(agent_id, None)
            self._event_bus.unsubscribe_all(agent_id)

    def get_mailbox(self, agent_id: str) -> Optional[Mailbox]:
        """Get an agent's mailbox."""
        with self._lock:
            return self._mailboxes.get(agent_id)

    # ── Message sending ──

    def send(
        self,
        from_agent: str,
        to_agent: str,
        msg_type: str = "observation",
        content: Optional[dict] = None,
        world_tick: int = 0,
        causal_parent: str = "",
        world_room: str = "",
    ) -> Optional[Message]:
        """Send a direct message to another agent.

        Args:
            from_agent: Sender session ID.
            to_agent: Recipient session ID.
            msg_type: One of MessageType enum values.
            content: Dict with message payload.
            world_tick: Current world tick.
            causal_parent: Previous causal entry ID for linking.
            world_room: Room where message was sent.

        Returns:
            The Message if delivered, None if recipient doesn't exist.
        """
        msg = Message(
            from_agent=from_agent,
            to_agent=to_agent,
            msg_type=msg_type,
            content=content,
            world_tick=world_tick,
            causal_parent=causal_parent,
            world_room=world_room,
        )

        with self._lock:
            mailbox = self._mailboxes.get(to_agent)
            if mailbox is None:
                return None
            mailbox.receive(msg)
            self._sent_messages.append(msg)
            if len(self._sent_messages) > self._max_sent_history:
                self._sent_messages = self._sent_messages[-self._max_sent_history:]

        return msg

    def broadcast(
        self,
        from_agent: str,
        msg_type: str = "broadcast",
        content: Optional[dict] = None,
        world_tick: int = 0,
        causal_parent: str = "",
        world_room: str = "",
    ) -> list[Message]:
        """Broadcast a message to all registered agents except sender.

        Args:
            from_agent: Sender session ID.
            msg_type: Message type.
            content: Message payload.
            world_tick: Current world tick.
            causal_parent: Causal parent ID.
            world_room: Room where broadcast originated.

        Returns:
            List of Messages sent.
        """
        sent = []
        with self._lock:
            for agent_id in list(self._mailboxes.keys()):
                if agent_id == from_agent:
                    continue
                msg = Message(
                    from_agent=from_agent,
                    to_agent=agent_id,
                    msg_type=msg_type,
                    content=content,
                    world_tick=world_tick,
                    causal_parent=causal_parent,
                    world_room=world_room,
                )
                self._mailboxes[agent_id].receive(msg)
                sent.append(msg)
                self._sent_messages.append(msg)

        # Trim history
        if len(self._sent_messages) > self._max_sent_history:
            self._sent_messages = self._sent_messages[-self._max_sent_history:]

        return sent

    def publish_world_event(
        self,
        event_type: str,
        content: dict,
        room: str = "",
        source: str = "world",
        tick: int = 0,
    ) -> list[str]:
        """Publish a world event and deliver it to subscribed agents' mailboxes.

        This is the bridge between World changes and agent perception.
        World events are NOT forced on agents — they land in the mailbox,
        and agents decide whether to process them.

        Args:
            event_type: Event type.
            content: Event content.
            room: Where the event occurred.
            source: What caused the event.
            tick: World tick.

        Returns:
            List of agent IDs that received the event.
        """
        event = {
            "type": event_type,
            "content": content,
            "room": room,
            "source": source,
            "tick": tick,
        }

        recipients = self._event_bus.publish(event)

        # Deliver to mailboxes as broadcast-type messages
        with self._lock:
            for agent_id in recipients:
                mailbox = self._mailboxes.get(agent_id)
                if mailbox:
                    msg = Message(
                        from_agent="🌍",
                        to_agent=agent_id,
                        msg_type="broadcast",
                        content={
                            "event_type": event_type,
                            "text": content.get("text", ""),
                            "room": room,
                            "source": source,
                        },
                        world_tick=tick,
                        world_room=room,
                    )
                    mailbox.receive(msg)

        return recipients

    # ── Prompt context ──

    def format_mailbox_for_prompt(
        self,
        agent_id: str,
        max_messages: int = 5,
    ) -> str:
        """Format mailbox contents for prompt injection.

        Messages shown as structured events, not chat dialogue.
        """
        mailbox = self.get_mailbox(agent_id)
        if not mailbox or not mailbox.has_unread():
            return ""

        messages = mailbox.messages[-max_messages:]
        lines = ["【收到的消息】"]

        for msg in messages:
            text = str(msg.content.get("text", msg.content.get("statement", "")))
            if len(text) > 100:
                text = text[:97] + "..."

            room_info = f" @{msg.world_room}" if msg.world_room else ""
            lines.append(
                f"  {msg.label} {msg.from_agent[:8]}{room_info}: {text}"
            )

        lines.append(f"  — 共 {mailbox.count()} 条未处理消息")
        return "\n".join(lines)

    @staticmethod
    def format_events_for_prompt(events: list[dict], max_events: int = 5) -> str:
        """Format world events for prompt injection."""
        if not events:
            return ""
        lines = ["【最近世界事件】"]
        for ev in events[-max_events:]:
            content = ev.get("content", {})
            text = str(content.get("text", content.get("event_type", "")))
            room = ev.get("room", "")
            source = ev.get("source", "?")
            tick = ev.get("tick", 0)
            lines.append(
                f"  [{tick}] {room}: {text[:80]} (源: {source[:8]})"
            )
        return "\n".join(lines)

    # ── Serialization ──

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "mailboxes": {
                    aid: mb.to_dict()
                    for aid, mb in self._mailboxes.items()
                },
                "event_bus": self._event_bus.to_dict(),
            }

    @classmethod
    def from_dict(cls, d: dict) -> "CommunicationManager":
        cm = cls()
        mailboxes_data = d.get("mailboxes", {})
        for aid, mb_data in mailboxes_data.items():
            cm._mailboxes[aid] = Mailbox.from_dict(aid, mb_data)
        return cm
