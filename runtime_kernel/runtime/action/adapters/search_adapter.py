"""
search_adapter — Search capability adapter.

Wraps MCP Runtime internally. The Planner never knows MCP exists.
Supports: web_search, fetch_url via any Search MCP server.

Future: Browser, Filesystem, GitHub — each gets its own adapter.
SearchAdapter only knows Search. No MCP types leak out.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from runtime_kernel.runtime.action.adapters import (
    CapabilityAdapter,
    register_capability,
)
from runtime_kernel.runtime.action.models import Action, Capability, Observation
from runtime_kernel.runtime.mcp import MCPConfig, MCPRuntime


class SearchAdapter(CapabilityAdapter):
    """Search capability — wraps MCP Runtime for search tools.

    Internally connects to one or more MCP servers that provide
    search-related tools (web_search, fetch_url, etc.).
    Externally exposes a clean Capability interface.
    """

    CAPABILITY_NAME = "Search"
    CAPABILITY_DESC = "搜索网络信息，获取网页内容"

    def __init__(self, mcp_configs: list[MCPConfig], event_bus: Any = None) -> None:
        self._runtimes: list[MCPRuntime] = []
        self._operation_map: dict[str, MCPRuntime] = {}
        self._event_bus = event_bus

        for config in mcp_configs:
            try:
                runtime = MCPRuntime(config)
                runtime.connect()
                tools = runtime.discover_tools()
                for tool in tools:
                    self._operation_map[tool.name] = runtime
                self._runtimes.append(runtime)
                print(
                    f"  [Action] SearchAdapter: {len(tools)} tools from {config.command or config.url[:50]}",
                    file=sys.stderr,
                )
            except Exception as e:
                print(
                    f"  [Action] SearchAdapter: failed to connect {config.command or config.url[:50]}: {e}",
                    file=sys.stderr,
                )

        total = len(self._operation_map)
        print(
            f"  [Action] SearchAdapter ready: {total} operation(s): "
            f"{', '.join(sorted(self._operation_map.keys()))}",
            file=sys.stderr,
        )

    def execute(self, action: Action, session_id: str = "") -> Observation:
        """Execute a search action via MCP.

        Args:
            action: Action with capability="Search" and an operation
                    like "web_search" or "fetch_url".
            session_id: Optional session ID for event emission.

        Returns:
            Observation with search results as content.
        """
        operation = action.operation
        params = action.parameters

        runtime = self._operation_map.get(operation)
        if not runtime:
            available = ", ".join(sorted(self._operation_map.keys()))
            return Observation(
                success=False,
                error=(
                    f"Search operation '{operation}' not available. "
                    f"Available: [{available}]"
                ),
            )

        print(f"  [Action] Search: {operation}({params})", file=sys.stderr)

        # Emit MCP request event
        self._emit_event("mcp_request", {
            "session_id": session_id,
            "capability": "Search",
            "tool": operation,
            "arguments": params,
        })

        t0 = time.time()
        result = runtime.execute(operation, params)
        elapsed = int((time.time() - t0) * 1000)

        if result.success:
            print(f"  [Action] Search result received ({len(result.content)} items)", file=sys.stderr)
            # Emit MCP response event
            self._emit_event("mcp_response", {
                "session_id": session_id,
                "tool": operation,
                "elapsed_ms": elapsed,
                "success": True,
                "result": result.content,
            })
            return Observation(
                success=True,
                content=result.content,
                metadata={"operation": operation, "parameters": params, "elapsed_ms": elapsed},
            )
        else:
            print(f"  [Action] Search failed: {result.error}", file=sys.stderr)
            self._emit_event("mcp_response", {
                "session_id": session_id,
                "tool": operation,
                "elapsed_ms": elapsed,
                "success": False,
                "error": result.error,
            })
            return Observation(
                success=False,
                content=None,
                metadata={"operation": operation, "parameters": params},
                error=result.error,
            )

    def list_operations(self) -> list[dict]:
        """List all search operations available via MCP.

        Returns operation descriptors with name, description, and parameters.
        """
        ops: list[dict] = []
        seen: set[str] = set()
        for runtime in self._runtimes:
            for tool in runtime.get_tools():
                if tool.name not in seen:
                    seen.add(tool.name)
                    ops.append({
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    })
        return ops

    def get_capability_info(self) -> Capability:
        return Capability(
            name=self.CAPABILITY_NAME,
            description=self.CAPABILITY_DESC,
            enabled=len(self._runtimes) > 0,
        )

    def _emit_event(self, event_type: str, payload: dict) -> None:
        """Emit an agent event via the event bus (if configured)."""
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

    def close(self) -> None:
        """Close all MCP connections."""
        for runtime in self._runtimes:
            try:
                runtime.close()
            except Exception:
                pass
        self._runtimes.clear()
        self._operation_map.clear()


# Register this adapter in the global capability registry
register_capability("Search", SearchAdapter)
