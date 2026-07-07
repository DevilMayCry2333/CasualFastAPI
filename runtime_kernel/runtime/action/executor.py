"""
executor — ActionExecutor: unified entry point for all agent actions.

The ActionExecutor is the ONLY class the engine/agent interacts with
for executing capability actions. It routes Action → Adapter → Observation.

Architecture:
    Agent/Planner
        │
        ▼
    ActionExecutor.execute(action)
        │
        ├── SearchAdapter      (wraps MCP Runtime internally)
        ├── BrowserAdapter     (future)
        ├── PythonAdapter      (future)
        ├── FilesystemAdapter  (future)
        └── GitHubAdapter      (future)

Key rules:
    - Planner never knows which adapter executes an action.
    - Adding a new capability = registering a new adapter. Zero engine changes.
    - Every action returns an Observation — never a raw value.
"""

from __future__ import annotations

import sys
from typing import Any

from runtime_kernel.runtime.action.adapters import CapabilityAdapter, CAPABILITY_REGISTRY
from runtime_kernel.runtime.action.models import Action, Capability, Observation


class ActionExecutor:
    """Routes actions to the right capability adapter and returns observations.

    Usage:
        executor = ActionExecutor()
        executor.register("Search", SearchAdapter(mcp_configs))
        obs = executor.execute(Action(capability="Search", operation="web_search",
                                      parameters={"query": "..."}))
        # obs.content → search results
    """

    def __init__(self) -> None:
        self._adapters: dict[str, CapabilityAdapter] = {}
        self._initialized: bool = False

    # ── Registration ──

    def register(self, capability_name: str, adapter: CapabilityAdapter) -> None:
        """Register a capability adapter.

        Args:
            capability_name: e.g. "Search", "Browser"
            adapter: An instance of a CapabilityAdapter subclass.
        """
        self._adapters[capability_name] = adapter
        print(
            f"  [Action] Registered capability: {capability_name}",
            file=sys.stderr,
        )

    def is_initialized(self) -> bool:
        """Whether at least one capability is registered."""
        return len(self._adapters) > 0

    # ── Capability discovery ──

    def list_capabilities(self) -> list[Capability]:
        """Return metadata for all registered capabilities."""
        result: list[Capability] = []
        for name, adapter in self._adapters.items():
            try:
                info = adapter.get_capability_info()
                result.append(info)
            except Exception:
                result.append(Capability(name=name, description="", enabled=False))
        return result

    def get_operations(self, capability_name: str) -> list[dict]:
        """Return all operations available for a given capability.

        Args:
            capability_name: e.g. "Search"

        Returns list of operation dicts with name, description, parameters.
        """
        adapter = self._adapters.get(capability_name)
        if not adapter:
            return []
        try:
            return adapter.list_operations()
        except Exception:
            return []

    def has_capability(self, name: str) -> bool:
        """Check if a capability is registered."""
        return name in self._adapters

    # ── Execution ──

    def execute(self, action: Action, session_id: str = "") -> Observation:
        """Execute an action and return an observation.

        This is the ONLY execution method. All actions go through here.

        Args:
            action: The Action to execute (capability, operation, parameters).
            session_id: Optional session ID for event emission.

        Returns:
            Observation with success status and content. Never raises.

        Error handling:
            - Unknown capability → Observation(success=False, error=...)
            - Adapter fails     → Observation(success=False, error=...)
            - Unexpected error  → Observation(success=False, error=...)
        """
        adapter = self._adapters.get(action.capability)
        if not adapter:
            available = ", ".join(sorted(self._adapters.keys()))
            return Observation(
                success=False,
                error=(
                    f"Capability '{action.capability}' not available. "
                    f"Available capabilities: [{available}]"
                ),
                metadata={"action": action.to_dict()},
            )

        try:
            if hasattr(adapter, "execute") and session_id:
                # Pass session_id for event emission if adapter supports it
                return adapter.execute(action, session_id=session_id)
            return adapter.execute(action)
        except Exception as e:
            print(
                f"  [Action] Error executing {action.capability}.{action.operation}: {e}",
                file=sys.stderr,
            )
            return Observation(
                success=False,
                content=None,
                metadata={"action": action.to_dict()},
                error=f"Execution error: {e}",
            )

    # ── Prompt formatting ──

    def format_for_prompt(self) -> str:
        """Format capabilities and operations for LLM prompt injection.

        Returns a formatted string or empty string if no capabilities.
        """
        caps = self.list_capabilities()
        if not caps:
            return ""

        parts: list[str] = ["【可用能力】"]
        for cap in caps:
            if not cap.enabled:
                continue
            ops = self.get_operations(cap.name)
            if not ops:
                continue

            parts.append(f"\n{cap.name}: {cap.description}")

            for op in ops:
                name = op.get("name", "?")
                desc = op.get("description", "")
                params = op.get("parameters", {})
                props = params.get("properties", {})
                param_str = ", ".join(
                    f"{k}: {v.get('type', 'any')}"
                    for k, v in props.items()
                )
                if param_str:
                    parts.append(f"  • {name}({param_str}): {desc[:100]}")
                else:
                    parts.append(f"  • {name}: {desc[:100]}")

        # Build a dynamic example using the first discovered operation
        example_op = "tavily_search"
        for cap in caps:
            if cap.name == "Search":
                search_ops = self.get_operations(cap.name)
                if search_ops:
                    example_op = search_ops[0].get("name", example_op)
                break
        parts.append("")
        parts.append(
            "需要外部信息时，使用以下格式调用能力：\n"
            f'action: {{"capability": "Search", "operation": "{example_op}", '
            f'"parameters": {{"query": "..."}}}}'
        )
        return "\n".join(parts)

    # ── Cleanup ──

    def close_all(self) -> None:
        """Close all capability adapters and release resources."""
        for name, adapter in self._adapters.items():
            try:
                if hasattr(adapter, "close"):
                    adapter.close()
                print(f"  [Action] Closed: {name}", file=sys.stderr)
            except Exception as e:
                print(f"  [Action] Error closing {name}: {e}", file=sys.stderr)
        self._adapters.clear()
        print("  [Action] All adapters closed", file=sys.stderr)

    def __enter__(self) -> "ActionExecutor":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close_all()
