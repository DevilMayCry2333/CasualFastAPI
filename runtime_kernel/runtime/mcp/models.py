"""
models — Data models for the MCP Tool Runtime.

These are pure data containers (not pydantic, just dataclasses).
They isolate the rest of the system from the MCP protocol details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolInfo:
    """Unified representation of a Tool, independent of MCP protocol.

    All tools discovered via MCP are normalized to this format.
    No MCP SDK types leak into the rest of the system.

    Attributes:
        name: Tool identifier, e.g. "web_search"
        description: Human-readable description of what the tool does
        parameters: JSON Schema dict describing expected arguments
    """

    name: str
    description: str = ""
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ToolResult:
    """Unified result from any tool execution.

    Attributes:
        success: Whether the tool executed without errors
        content: The tool's output (varies by tool)
        error: Human-readable error message if success=False
    """

    success: bool = True
    content: Any = None
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "content": self.content,
            "error": self.error,
        }


@dataclass
class MCPConfig:
    """Configuration for connecting to an MCP server.

    Supports both stdio-based and URL-based MCP servers.

    For stdio transport (most common):
        command = "uvx"
        args = ["mcp-server-tavily"]
        env = {"TAVILY_API_KEY": "..."}

    For URL/SSE transport (future):
        url = "http://localhost:8080/sse"
    """

    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    timeout: float = 30.0
