"""
schemas — Pydantic models for REST API request/response validation.

These are pure data contracts. No runtime_kernel imports here.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MCPToolConfig(BaseModel):
    """Configuration for an MCP server connection.

    Can specify either a command (stdio transport) or a URL (SSE transport).
    """
    command: str = Field(default="", description="MCP server command to execute")
    args: list[str] = Field(default=[], description="Command-line arguments")
    env: dict[str, str] = Field(default={}, description="Environment variables")
    url: str = Field(default="", description="MCP server URL (SSE transport)")
    timeout: float = Field(default=30.0, description="Connection timeout in seconds")


class ConnectRequest(BaseModel):
    api_url: str = Field(default="http://localhost:28000/v1/chat/completions")
    api_key: str = Field(default="")
    model: str = Field(default="deepseek-v4-flash")
    temperature: float = Field(default=0.85, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=64, le=32768)
    mcp_tools: list[MCPToolConfig] = Field(
        default=[],
        description="MCP server configurations for tool usage",
    )


class ConnectResponse(BaseModel):
    session_id: str


class SessionActionRequest(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class SnapshotRequest(BaseModel):
    session_id: str
    filepath: str = ""


class RestoreRequest(BaseModel):
    filepath: str


class StateResponse(BaseModel):
    state: dict
    round: int
    fold_count: int
    history_count: int
    status: str
    identity_anchor: dict | None = None
    compressed_state: dict | None = None


class StepResponse(BaseModel):
    type: str
    session_id: str
    round: int
    state: dict
    identity_anchor: dict | None = None
    state_compressed: dict | None = None
    extra: dict | None = None
    # World Model v2 fields
    hypothesis_count: int = 0
    evidence_count: int = 0
    world_model_updates: dict | None = None
    # Communication v3 fields
    messages_sent: int = 0
    mailbox_unread: int = 0
    # Human interaction (waiting_human response — agent asks a question)
    question: str = ""
    reason: str = ""


class ChatResponse(BaseModel):
    type: str
    session_id: str
    round: int
    human_input: str
    nl_response: str
    state: dict
    identity_anchor: dict | None = None
    state_compressed: dict | None = None
    # World Model v2 fields
    hypothesis_count: int = 0
    evidence_count: int = 0
    world_model_updates: dict | None = None
    # Communication v3 fields
    messages_sent: int = 0
    mailbox_unread: int = 0


class FoldResponse(BaseModel):
    type: str
    session_id: str
    fold_count: int
    identity_anchor: dict
    round: int


class TranslateResponse(BaseModel):
    translation: str


class SnapshotResponse(BaseModel):
    filepath: str


class SendMessageRequest(BaseModel):
    session_id: str
    to_agent: str
    msg_type: str = "observation"
    content: dict = {}


class SendBroadcastRequest(BaseModel):
    session_id: str
    msg_type: str = "broadcast"
    content: dict = {}


class ErrorResponse(BaseModel):
    detail: str


class OperationInfo(BaseModel):
    """Operation within a capability."""
    name: str
    description: str = ""
    parameters: dict = {}


class CapabilityInfo(BaseModel):
    """Capability metadata."""
    name: str
    description: str = ""
    enabled: bool = True
    operations: list[OperationInfo] = []


class CapabilitiesResponse(BaseModel):
    capabilities: list[CapabilityInfo] = []


class ActionRequest(BaseModel):
    """Execute an action."""
    session_id: str
    capability: str
    operation: str
    parameters: dict = {}


class ObservationResponse(BaseModel):
    """Result of executing an action."""
    success: bool
    content: Any = None
    metadata: dict = {}
    error: str = ""


class ActionSystemStatusResponse(BaseModel):
    initialized: bool
    capability_count: int
    capabilities: list[str] = []


class HumanAnswerRequest(BaseModel):
    """Submit a human answer to a pending question."""
    session_id: str
    answer: str


class PendingQuestionResponse(BaseModel):
    """A pending question from an agent waiting for human input."""
    session_id: str
    has_pending: bool = False
    question: str = ""
    reason: str = ""
