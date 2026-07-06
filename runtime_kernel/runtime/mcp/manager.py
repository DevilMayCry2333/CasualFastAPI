"""
manager — ToolManager: unified agent-facing interface for all MCP tools.

The ToolManager is the ONLY class that agents (or the engine) interact with.

Architecture:
    Agent  →  ToolManager.list_tools() / execute()
                  │
                  ├── MCPRuntime (Search MCP)
                  ├── MCPRuntime (Browser MCP, future)
                  └── MCPRuntime (Filesystem MCP, future)

Key design decisions:
    - Tools are auto-discovered via MCP's tools/list. No manual registration.
    - Adding a new MCP server = adding a config entry. Zero code changes.
    - The agent never knows which runtime provides a tool.
    - All tool schemas are cached and can be refreshed at any time.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

from runtime_kernel.runtime.mcp.models import MCPConfig, ToolInfo, ToolResult
from runtime_kernel.runtime.mcp.runtime import MCPRuntime


class ToolManager:
    """Unified tool interface for agents.

    Aggregates multiple MCPRuntimes (one per MCP server) and provides
    a single list/execute API. Tools are identified by name across all
    runtimes — name collisions are detected at initialize time.

    Usage:
        manager = ToolManager()
        manager.initialize([
            MCPConfig(command="uvx", args=["mcp-server-tavily"]),
        ])
        all_tools = manager.list_tools()
        result = manager.execute("web_search", {"query": "..."})
        manager.close_all()
    """

    def __init__(self) -> None:
        # All connected runtimes
        self._runtimes: list[MCPRuntime] = []

        # tool_name → runtime mapping for O(1) dispatch
        self._tool_registry: dict[str, MCPRuntime] = {}

        # Whether initialize() has been called
        self._initialized: bool = False

    # ── Initialization ──

    def initialize(self, mcp_configs: list[MCPConfig]) -> None:
        """Connect to all configured MCP servers and discover tools.

        Each MCP config becomes a separate MCPRuntime. Tools are
        discovered and registered in a unified namespace.

        Connection failures are logged but do NOT block the remaining
        servers. This ensures one broken MCP server doesn't disable
        the entire tool layer.

        Args:
            mcp_configs: List of MCP server configurations.
        """
        self._tool_registry.clear()
        self._runtimes.clear()

        if not mcp_configs:
            print("  [MCP] No MCP servers configured", file=sys.stderr)
            self._initialized = True
            return

        for config in mcp_configs:
            runtime = MCPRuntime(config)
            try:
                runtime.connect()
                tools = runtime.discover_tools()
                for tool in tools:
                    if tool.name in self._tool_registry:
                        print(
                            f"  [MCP] Warning: tool name collision — "
                            f"{tool.name!r} already registered, "
                            f"overriding with {config.command}",
                            file=sys.stderr,
                        )
                    self._tool_registry[tool.name] = runtime
                    print(f"  [MCP]   → {tool.name}: {tool.description[:60]}", file=sys.stderr)
                self._runtimes.append(runtime)
                print(
                    f"  [MCP] ✅ {len(tools)} tools from {config.command}",
                    file=sys.stderr,
                )
            except Exception as e:
                print(
                    f"  [MCP] ❌ Failed: {config.command} — {e}",
                    file=sys.stderr,
                )
                # Graceful degradation: skip this server, continue with others
                try:
                    runtime.close()
                except Exception:
                    pass

        total = len(self._tool_registry)
        servers = len(self._runtimes)
        print(f"  [MCP] Ready: {total} tools across {servers} server(s)", file=sys.stderr)
        self._initialized = True

    def is_initialized(self) -> bool:
        """Check if the manager has been initialized."""
        return self._initialized

    def server_count(self) -> int:
        """Return the number of connected MCP servers."""
        return len(self._runtimes)

    # ── Tool discovery ──

    def list_tools(self) -> list[ToolInfo]:
        """Return all discovered tools across all servers.

        Returns a list of ToolInfo with name, description, and parameters.
        This is the agent's view of available tools. Sorted by name.
        """
        seen: set[str] = set()
        result: list[ToolInfo] = []
        for runtime in self._runtimes:
            for tool in runtime.get_tools():
                if tool.name not in seen:
                    seen.add(tool.name)
                    result.append(tool)
        result.sort(key=lambda t: t.name)
        return result

    def get_tool_schemas(self) -> list[dict]:
        """Return tool schemas formatted for LLM prompt injection.

        Returns a list of dicts, each containing the ToolInfo fields
        plus a formatted JSON Schema for the LLM to understand.
        """
        return [t.to_dict() for t in self.list_tools()]

    def refresh(self) -> int:
        """Rediscover tools from all connected servers.

        Use this after adding/removing MCP servers without restarting.

        Returns the number of tools discovered.
        """
        self._tool_registry.clear()
        for runtime in self._runtimes:
            try:
                tools = runtime.discover_tools()
                for tool in tools:
                    self._tool_registry[tool.name] = runtime
            except Exception as e:
                print(
                    f"  [MCP] Refresh failed for {runtime}: {e}",
                    file=sys.stderr,
                )
        total = len(self._tool_registry)
        print(f"  [MCP] Refresh complete: {total} tools", file=sys.stderr)
        return total

    # ── Tool execution ──

    def execute(self, name: str, arguments: dict) -> ToolResult:
        """Execute a tool by name across all servers.

        Finds the runtime that provides this tool and delegates execution.
        This is the method agents call — they never touch MCPRuntime.

        Args:
            name: The tool name (e.g. "web_search").
            arguments: Tool arguments as a dict.

        Returns a ToolResult. On success, content contains the result.
        On failure, success=False with an error message.

        Handles:
            - Tool not found → ToolResult(success=False, error=...)
            - MCP server disconnected → ToolResult(success=False, error=...)
            - Tool execution error → ToolResult(success=False, error=...)
            - Timeout → ToolResult(success=False, error=...)
        """
        runtime = self._tool_registry.get(name)
        if runtime is None:
            available = ", ".join(sorted(self._tool_registry.keys()))
            return ToolResult(
                success=False,
                error=f"Tool '{name}' not found. Available tools: [{available}]",
            )
        return runtime.execute(name, arguments)

    # ── Cleanup ──

    def close_all(self) -> None:
        """Close all MCP connections and clear the registry.

        Safe to call multiple times. After this, initialize() must
        be called again before use.
        """
        for runtime in self._runtimes:
            try:
                runtime.close()
            except Exception as e:
                print(f"  [MCP] Error closing {runtime}: {e}", file=sys.stderr)
        self._runtimes.clear()
        self._tool_registry.clear()
        self._initialized = False
        print("  [MCP] All connections closed", file=sys.stderr)

    def has_tool(self, name: str) -> bool:
        """Check if a tool name is available."""
        return name in self._tool_registry

    def __enter__(self) -> "ToolManager":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close_all()
