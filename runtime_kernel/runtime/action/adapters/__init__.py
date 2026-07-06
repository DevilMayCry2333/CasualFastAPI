"""
adapters — Capability Adapter abstract base and registry.

Each capability (Search, Browser, Python, etc.) has an Adapter
that knows how to execute Actions for that capability.

The ActionExecutor discovers adapters via the CAPABILITY_REGISTRY.
Adding a new capability = writing a new Adapter subclass and
registering it. No changes to Planner or Runtime core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from runtime_kernel.runtime.action.models import Action, Capability, Observation


# ── Registry ──

CAPABILITY_REGISTRY: dict[str, type["CapabilityAdapter"]] = {}


def register_capability(name: str, adapter_cls: type["CapabilityAdapter"]) -> None:
    """Register a capability adapter class by capability name."""
    CAPABILITY_REGISTRY[name] = adapter_cls


# ── Abstract Base ──


class CapabilityAdapter(ABC):
    """Base class for all capability adapters.

    Each adapter handles ONE capability (Search, Browser, etc.).
    The adapter knows HOW to execute operations within that capability.
    The Planner never knows which adapter is used.

    Subclasses must:
        - Call register_capability() at module level
        - Implement execute()
        - Implement list_operations()
    """

    @abstractmethod
    def execute(self, action: Action) -> Observation:
        """Execute the given action and return an Observation.

        Args:
            action: The Action to execute (capability, operation, parameters).

        Returns:
            Observation with success status and content.
        """
        ...

    @abstractmethod
    def list_operations(self) -> list[dict]:
        """List all available operations for this capability.

        Returns a list of dicts with:
            name: Operation name (e.g. "web_search")
            description: Human-readable description
            parameters: JSON Schema-like dict of accepted params
        """
        ...

    @abstractmethod
    def get_capability_info(self) -> Capability:
        """Return capability metadata for this adapter."""
        ...
