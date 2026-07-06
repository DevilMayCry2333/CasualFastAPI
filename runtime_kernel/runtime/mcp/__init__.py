"""
mcp — MCP Tool Runtime for Agent Runtime Kernel.

Architecture:
    ToolManager (agent-facing, unified)
        │
        ├── MCPRuntime 1 (Search MCP)
        ├── MCPRuntime 2 (Browser MCP, future)
        └── MCPRuntime N (...)

Each MCPRuntime wraps one MCP server connection via MCPClient.
The ToolManager aggregates all runtimes and provides a unified
list/execute interface. Agents never talk to MCP directly.

Usage:
    manager = ToolManager()
    manager.initialize([MCPConfig(command="uvx", args=["mcp-server-tavily"])])
    tools = manager.list_tools()       # [ToolInfo, ...]
    result = manager.execute("web_search", {"query": "..."})
    manager.close_all()
"""

from runtime_kernel.runtime.mcp.models import MCPConfig, ToolInfo, ToolResult
from runtime_kernel.runtime.mcp.client import MCPClient, MCPError
from runtime_kernel.runtime.mcp.runtime import MCPRuntime
from runtime_kernel.runtime.mcp.manager import ToolManager

__all__ = [
    "MCPConfig",
    "MCPError",
    "ToolInfo",
    "ToolResult",
    "MCPClient",
    "MCPRuntime",
    "ToolManager",
]
