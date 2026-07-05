"""WorldModel — the agent's model of the external world.

Maintains objects, places, agents seen, events, environment rules,
and predictions. This is ONLY about "the world", not about "me".
"""

from __future__ import annotations

from typing import Any


class WorldModel:
    """The agent's understanding of the external world.

    Fields:
        objects: Known objects and their properties.
        places: Known places/rooms and their connections.
        agents_seen: Other agents observed and where.
        events: Significant world events remembered.
        environment_rules: Understood rules of the environment.
        predictions: Predicted future world states.
    """

    def __init__(self) -> None:
        self._objects: dict[str, dict] = {}
        self._places: dict[str, dict] = {}
        self._agents_seen: dict[str, dict] = {}
        self._events: list[dict] = []
        self._environment_rules: list[str] = []
        self._predictions: list[dict] = []

    @property
    def objects(self) -> dict[str, dict]:
        return dict(self._objects)

    @property
    def places(self) -> dict[str, dict]:
        return dict(self._places)

    @property
    def agents_seen(self) -> dict[str, dict]:
        return dict(self._agents_seen)

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    @property
    def environment_rules(self) -> list[str]:
        return list(self._environment_rules)

    @property
    def predictions(self) -> list[dict]:
        return list(self._predictions)

    def observe_object(self, name: str, properties: dict) -> None:
        self._objects[name] = dict(properties)

    def observe_place(self, name: str, description: str, exits: list[str] = None) -> None:
        self._places[name] = {
            "description": description[:100],
            "exits": exits or [],
            "visited_count": self._places.get(name, {}).get("visited_count", 0) + 1,
        }

    def observe_agent(self, agent_id: str, room: str = "") -> None:
        if agent_id not in self._agents_seen:
            self._agents_seen[agent_id] = {
                "first_seen_room": room,
                "observations": [],
            }
        entry = self._agents_seen[agent_id]
        entry.setdefault("observations", [])
        if room and (not entry.get("observations") or entry["observations"][-1].get("room") != room):
            entry["observations"].append({"room": room})

    def add_event(self, event: dict) -> None:
        self._events.append(event)
        if len(self._events) > 30:
            self._events = self._events[-30:]

    def add_rule(self, rule: str) -> None:
        if rule not in self._environment_rules:
            self._environment_rules.append(rule)

    def add_prediction(self, prediction: dict) -> None:
        self._predictions.append(prediction)
        if len(self._predictions) > 10:
            self._predictions = self._predictions[-10:]

    def format_for_prompt(self) -> str:
        parts = ["【世界模型】"]
        if self._places:
            places_str = ", ".join(
                f"{n}({p.get('visited_count', 0)}次)"
                for n, p in self._places.items()
            )
            parts.append(f"  已知场所: {places_str}")
        if self._objects:
            parts.append(f"  已知物品: {', '.join(self._objects.keys())}")
        if self._agents_seen:
            agents_str = ", ".join(
                f"{aid[:8]}" for aid in self._agents_seen
            )
            parts.append(f"  见过的其他存在: {agents_str}")
        if self._environment_rules:
            parts.append("  环境规则:")
            for r in self._environment_rules[-3:]:
                parts.append(f"    · {r[:60]}")
        if self._events:
            parts.append("  最近事件:")
            for e in self._events[-3:]:
                text = e.get("text", e.get("content", {}).get("text", str(e)[:60]))
                parts.append(f"    · {str(text)[:60]}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "objects": dict(self._objects),
            "places": dict(self._places),
            "agents_seen": dict(self._agents_seen),
            "events": list(self._events[-20:]),
            "environment_rules": list(self._environment_rules),
            "predictions": list(self._predictions),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorldModel":
        m = cls()
        m._objects = dict(d.get("objects", {}))
        m._places = dict(d.get("places", {}))
        m._agents_seen = dict(d.get("agents_seen", {}))
        m._events = list(d.get("events", []))
        m._environment_rules = list(d.get("environment_rules", []))
        m._predictions = list(d.get("predictions", []))
        return m
