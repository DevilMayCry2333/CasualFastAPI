"""
action — Agent Action System.

Architecture:
    Agent (Planner) → Action → ActionExecutor → CapabilityAdapter → Observation → Agent (Memory)

    ActionExecutor is the ONLY entry point for capability execution.
    CapabilityAdapter is the pluggable interface for each capability.
    Each adapter wraps its own backend (MCP, subprocess, API, etc.)

Capabilities vs Tools:
    Agent never knows about MCP, HTTP, or subprocess.
    Agent only knows:
        - It has the "Search" capability (for finding information)
        - It has the "Browser" capability (for browsing web pages)
        - etc.
    How each capability is implemented is invisible to the Agent.

Phase 1:
    Only Search capability is implemented.
    SearchAdapter wraps MCP Runtime internally.
    Adding Browser, Python, Filesystem = new adapter + registration.
"""

from runtime_kernel.runtime.action.models import Capability, Action, Observation
from runtime_kernel.runtime.action.executor import ActionExecutor
from runtime_kernel.runtime.action.adapters import CapabilityAdapter
from runtime_kernel.runtime.action.adapters.search_adapter import SearchAdapter
from runtime_kernel.runtime.action.adapters.human_adapter import HumanAdapter

__all__ = [
    "Capability",
    "Action",
    "Observation",
    "ActionExecutor",
    "CapabilityAdapter",
    "SearchAdapter",
    "HumanAdapter",
]
