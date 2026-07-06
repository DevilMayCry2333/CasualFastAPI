"""
models — Action System core data structures.

Capability: What an agent CAN do (e.g. Search, Browser, Python)
Action:     What an agent DECIDES to do (capability + operation + parameters)
Observation: What HAPPENED as a result (success + content + metadata)

These are the ONLY data types that cross the Agent ↔ Runtime boundary
for action execution. No MCP types, no adapter internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Capability:
    """A capability the agent has — what it can do.

    This is the agent's answer to "what tools do I have?"
    Not MCP tools. Capabilities are abstract: Search, Browser, Python.

    Attributes:
        name: Short name, e.g. "Search", "Browser", "Python"
        description: What this capability lets the agent do
        enabled: Whether this capability is currently available
    """

    name: str
    description: str = ""
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
        }


@dataclass
class Action:
    """An action the agent decides to take.

    The Planner produces Actions. The ActionExecutor executes them.
    The Planner never knows HOW an action is executed — only that
    it results in an Observation.

    Attributes:
        capability: Which capability to use, e.g. "Search"
        operation: Which operation within that capability, e.g. "web_search"
        parameters: Arguments for the operation, e.g. {"query": "..."}
    """

    capability: str
    operation: str
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "capability": self.capability,
            "operation": self.operation,
            "parameters": self.parameters,
        }

    @staticmethod
    def from_dict(data: dict) -> Action:
        return Action(
            capability=data.get("capability", ""),
            operation=data.get("operation", ""),
            parameters=data.get("parameters", {}),
        )


@dataclass
class Observation:
    """The result of executing an Action.

    Every Action produces exactly one Observation.
    The Planner reads Observations to update its knowledge.

    Attributes:
        success: Whether the action completed without error
        content: The action's output (varies by capability/operation)
        metadata: Extra info (execution time, source, etc.)
        error: Human-readable error if success=False
    """

    success: bool = True
    content: Any = None
    metadata: dict = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "content": self.content,
            "metadata": self.metadata,
            "error": self.error,
        }
