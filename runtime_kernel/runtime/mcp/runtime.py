"""
runtime — MCPRuntime: manages one MCP server connection lifecycle.

Responsibilities:
    - Connect to an MCP server (establish session)
    - Discover available tools (tools/list)
    - Execute a tool by name (tools/call)
    - Cache tool schemas for inspectability
    - Reconnect on connection loss (future)
    - Graceful shutdown

MCPRuntime wraps MCPClient and adds tool schema caching and
result formatting. It knows nothing about specific tools.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

from runtime_kernel.runtime.mcp.client import MCPClient, MCPError
from runtime_kernel.runtime.mcp.models import MCPConfig, ToolInfo, ToolResult


class MCPRuntime:
    """Manages the lifecycle of one MCP server connection.

    Usage:
        runtime = MCPRuntime(MCPConfig(command="uvx", args=["mcp-server-tavily"]))
        runtime.connect()
        tools = runtime.discover_tools()
        result = runtime.execute("web_search", {"query": "..."})
        runtime.close()
    """

    def __init__(self, config: MCPConfig) -> None:
        self._config = config
        self._client: Optional[MCPClient] = None
        self._tools: list[ToolInfo] = []
        self._server_info: dict = {}

    # ── Connection lifecycle ──

    def connect(self) -> None:
        """Connect to the MCP server and perform handshake.

        Raises MCPError if connection fails.
        """
        print(f"  [MCP] Connecting: {self._config.command} {' '.join(self._config.args)}", file=sys.stderr)
        self._client = MCPClient(self._config)
        try:
            self._server_info = self._client.connect()
            print(
                f"  [MCP] Connected — server: {self._server_info.get('serverInfo', {}).get('name', 'unknown')}",
                file=sys.stderr,
            )
        except MCPError:
            self._client = None
            raise

    def discover_tools(self) -> list[ToolInfo]:
        """Discover available tools from the MCP server.

        Calls tools/list and caches the results as ToolInfo objects.

        Returns a list of ToolInfo.

        Raises MCPError if the server returns an error or is not connected.
        """
        if not self._client:
            raise MCPError("MCP not connected — call connect() first")

        raw_tools = self._client.list_tools()
        self._tools = []

        for t in raw_tools:
            info = ToolInfo(
                name=t.get("name", "unknown"),
                description=t.get("description", ""),
                parameters=t.get("inputSchema", {}),
            )
            self._tools.append(info)

        print(f"  [MCP] Discovered {len(self._tools)} tools", file=sys.stderr)
        return self._tools

    def execute(self, tool_name: str, arguments: dict) -> ToolResult:
        """Execute a tool on the MCP server.

        Args:
            tool_name: The name of the tool to execute.
            arguments: Tool arguments as a dict.

        Returns a ToolResult. On success, result.content is the raw
        MCP content list. On error, result.success is False with an
        error message.

        This never raises — all errors are captured in ToolResult.
        """
        if not self._client:
            return ToolResult(
                success=False,
                error="MCP not connected — call connect() first",
            )

        print(f"  [MCP] Execute {tool_name}", file=sys.stderr)

        try:
            raw = self._client.call_tool(tool_name, arguments)
        except MCPError as e:
            print(f"  [MCP] Error executing {tool_name}: {e}", file=sys.stderr)
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            print(f"  [MCP] Unexpected error executing {tool_name}: {e}", file=sys.stderr)
            return ToolResult(success=False, error=f"Unexpected error: {e}")

        # Normalize MCP content to a unified format
        content = raw.get("content", [])
        is_error = raw.get("isError", False)

        # Extract text from MCP content items for easier consumption
        extracted = []
        for item in content:
            item_type = item.get("type", "text")
            if item_type == "text":
                extracted.append(item.get("text", ""))
            elif item_type == "resource":
                extracted.append(str(item.get("resource", "")))
            else:
                extracted.append(str(item))

        result = ToolResult(
            success=not is_error,
            content=extracted,
            error=raw.get("error", "") if is_error else "",
        )

        print(f"  [MCP] Result received ({len(extracted)} items)", file=sys.stderr)
        return result

    def close(self) -> None:
        """Close the MCP connection and clean up resources."""
        if self._client:
            print(f"  [MCP] Closing connection", file=sys.stderr)
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._tools = []

    def get_tools(self) -> list[ToolInfo]:
        """Return cached tool list (from last discover_tools call)."""
        return self._tools

    def get_server_info(self) -> dict:
        """Return server info from the handshake."""
        return self._server_info

    def is_connected(self) -> bool:
        """Check if the runtime is currently connected."""
        return self._client is not None and self._client.is_connected()

    def __repr__(self) -> str:
        return (
            f"MCPRuntime({self._config.command} {' '.join(self._config.args)})"
        )
