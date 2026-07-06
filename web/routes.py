"""
routes — FastAPI router that delegates everything to RuntimeEngine.

Every endpoint:
  1. Reads the request
  2. Calls RuntimeEngine method(s)
  3. Returns the result

No agent logic here. No prompt construction. No state parsing.
"""

from __future__ import annotations

import json
import os

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from runtime_kernel import (
    Action,
    ActionExecutor,
    AgentEvent,
    AgentEventBus,
    CommunicationManager,
    HumanAdapter,
    MCPConfig,
    Observation,
    RuntimeEngine,
    SearchAdapter,
    SharedKnowledge,
    VirtualEnvironment,
    SessionNotFoundError,
)

from web.schemas import (
    ActionRequest,
    ActionSystemStatusResponse,
    CapabilitiesResponse,
    CapabilityInfo,
    ChatRequest,
    ChatResponse,
    ConnectRequest,
    ConnectResponse,
    ErrorResponse,
    FoldResponse,
    HumanAnswerRequest,
    ObservationResponse,
    OperationInfo,
    PendingQuestionResponse,
    RestoreRequest,
    SendBroadcastRequest,
    SendMessageRequest,
    SessionActionRequest,
    SnapshotRequest,
    SnapshotResponse,
    StateResponse,
    StepResponse,
    TranslateResponse,
)

# ── Shared virtual world ──
# All agents coexist in this environment
_world = VirtualEnvironment()

# ── Shared communication layer ──
# ALL agents share one CommunicationManager so they can talk to each other
_communication = CommunicationManager()

# ── Shared knowledge ──
# ALL agents share one SharedKnowledge for cross-agent consensus
_shared_knowledge = SharedKnowledge()

# Wire world events and agent messages to the shared communication layer
_world.set_event_callback(_communication.publish_world_event)
_world.set_message_callback(
    lambda from_agent, to_agent, text, room="", tick=0:
        _communication.send(from_agent, to_agent, "observation",
                          {"text": text}, tick, "", room)
)

# ── Shared AgentEventBus (Observability) ──
# All agent events across all sessions flow through this bus
_agent_events = AgentEventBus()

# ── Shared ActionExecutor (Agent Action System) ──
# Configured lazily on first /api/connect with mcp_tools config
_action_executor = ActionExecutor()

# ── In-memory engine store ──
# Key: session_id, Value: RuntimeEngine
_engines: dict[str, RuntimeEngine] = {}

router = APIRouter()


# ── Helpers ──


def _get_engine(session_id: str) -> RuntimeEngine:
    """Look up engine by session_id or raise 404."""
    engine = _engines.get(session_id)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return engine


def _cleanup_on_error(session_id: str) -> None:
    """Remove engine from store if something went wrong."""
    _engines.pop(session_id, None)


# ── Page ──


@router.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main Agent Console page."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()


# ── Connect ──


@router.post(
    "/api/connect",
    response_model=ConnectResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def connect(req: ConnectRequest):
    """Create a RuntimeEngine and a Session with the user's LLM config.

    This is the only endpoint that creates a new engine + session.
    Returns a session_id that must be used for all subsequent calls.

    If mcp_tools are provided, initializes the shared ToolManager.
    """
    llm_config = {
        "api_url": req.api_url,
        "api_key": req.api_key,
        "model": req.model,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }
    try:
        # Initialize Action System if MCP configs provided
        if req.mcp_tools:
            mcp_configs = [
                MCPConfig(
                    command=mc.command,
                    args=list(mc.args),
                    env=dict(mc.env),
                    url=mc.url,
                    timeout=mc.timeout,
                )
                for mc in req.mcp_tools
            ]
            if not _action_executor.is_initialized():
                search_adapter = SearchAdapter(mcp_configs, event_bus=_agent_events)
                _action_executor.register("Search", search_adapter)
                # Human capability is always available (no external config needed)
                human_adapter = HumanAdapter(event_bus=_agent_events)
                _action_executor.register("Human", human_adapter)
            else:
                if not _action_executor.has_capability("Human"):
                    human_adapter = HumanAdapter(event_bus=_agent_events)
                    _action_executor.register("Human", human_adapter)
        else:
            # No MCP tools, but Human capability should still be available
            if not _action_executor.is_initialized():
                human_adapter = HumanAdapter(event_bus=_agent_events)
                _action_executor.register("Human", human_adapter)

        engine = RuntimeEngine(
            llm_config=llm_config, world=_world, auto_save=False,
            communication=_communication, shared_knowledge=_shared_knowledge,
            action_executor=_action_executor if _action_executor.is_initialized() else None,
            event_bus=_agent_events,
        )
        session = engine.create_session()
        _engines[session.id] = engine
        return ConnectResponse(session_id=session.id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect: {e}")


# ── Step (autonomous thinking) ──


@router.post(
    "/api/step",
    response_model=StepResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def step(req: SessionActionRequest):
    """Execute one autonomous thinking step."""
    engine = _get_engine(req.session_id)
    try:
        result = engine.step(req.session_id)
        return StepResponse(
            type=result["type"],
            session_id=result["session_id"],
            round=result["round"],
            state=result["state"],
            identity_anchor=result.get("identity_anchor"),
            state_compressed=result.get("state_compressed"),
            extra=result.get("extra"),
        )
    except Exception as e:
        _cleanup_on_error(req.session_id)
        raise HTTPException(status_code=500, detail=f"Step failed: {e}")


# ── Chat (human interruption) ──


@router.post(
    "/api/chat",
    response_model=ChatResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def chat(req: ChatRequest):
    """Send a human message to the agent (interrupt + respond)."""
    engine = _get_engine(req.session_id)
    try:
        result = engine.interrupt(req.session_id, req.message)
        return ChatResponse(
            type=result["type"],
            session_id=result["session_id"],
            round=result["round"],
            human_input=result["human_input"],
            nl_response=result.get("nl_response", ""),
            state=result["state"],
            identity_anchor=result.get("identity_anchor"),
            state_compressed=result.get("state_compressed"),
        )
    except Exception as e:
        _cleanup_on_error(req.session_id)
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


# ── Get State ──


@router.get(
    "/api/state/{session_id}",
    response_model=StateResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_state(session_id: str):
    """Get the current agent state summary."""
    engine = _get_engine(session_id)
    try:
        session = engine.get_session(session_id)
        return StateResponse(
            state=session.state.to_dict_complete(),
            round=session.round,
            fold_count=session.fold_count,
            history_count=len(session.history),
            status=session.status.value,
            identity_anchor=session.identity_anchor,
            compressed_state=(
                session.state_compressed.to_dict_complete()
                if session.state_compressed
                else None
            ),
        )
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get state: {e}")


# ── Fold ──


@router.post(
    "/api/fold",
    response_model=FoldResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def fold(req: SessionActionRequest):
    """Trigger manual Fold compression."""
    engine = _get_engine(req.session_id)
    try:
        result = engine.fold(req.session_id)
        return FoldResponse(
            type=result["type"],
            session_id=result["session_id"],
            fold_count=result["fold_count"],
            identity_anchor=result["identity_anchor"],
            round=result["round"],
        )
    except Exception as e:
        _cleanup_on_error(req.session_id)
        raise HTTPException(status_code=500, detail=f"Fold failed: {e}")


# ── Translate ──


@router.post(
    "/api/translate",
    response_model=TranslateResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def translate(req: SessionActionRequest):
    """Translate current state to natural language."""
    engine = _get_engine(req.session_id)
    try:
        text = engine.translate(req.session_id)
        return TranslateResponse(translation=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translate failed: {e}")


# ── Snapshot ──


@router.post(
    "/api/snapshot",
    response_model=SnapshotResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def snapshot(req: SnapshotRequest):
    """Save a session snapshot to disk."""
    engine = _get_engine(req.session_id)
    try:
        filepath = req.filepath or None
        saved_path = engine.snapshot(req.session_id, filepath=filepath)
        return SnapshotResponse(filepath=saved_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Snapshot failed: {e}")


# ── Restore ──


@router.post(
    "/api/restore",
    response_model=ConnectResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def restore(req: RestoreRequest):
    """Restore a session from a snapshot file on disk."""
    try:
        engine = RuntimeEngine(world=_world, auto_save=False)
        session = engine.restore(req.filepath)
        _engines[session.id] = engine
        return ConnectResponse(session_id=session.id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restore failed: {e}")


# ── Delete Session ──


@router.delete(
    "/api/session/{session_id}",
    responses={404: {"model": ErrorResponse}},
)
async def delete_session(session_id: str):
    """Delete an agent session and clean up resources."""
    engine = _get_engine(session_id)
    try:
        engine.delete_session(session_id)
        _engines.pop(session_id, None)
        return {"ok": True, "session_id": session_id}
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")


# ── World ──


@router.get("/api/world")
async def get_world():
    """Get shared world summary."""
    return _world.get_world_summary()


@router.get("/api/world/context/{session_id}")
async def get_world_context(session_id: str):
    """Get environmental context for a specific agent."""
    try:
        context = _world.get_context(session_id)
        position = _world.agent_position(session_id)
        return {"position": position, "context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── List All Sessions ──


@router.get("/api/sessions")
async def list_all_sessions():
    """Get all active sessions with full state info (for multi-agent display)."""
    result = []
    dead_ids = []
    for sid, engine in _engines.items():
        try:
            session = engine.get_session(sid)
            result.append({
                "id": session.id,
                "round": session.round,
                "fold_count": session.fold_count,
                "history_count": len(session.history),
                "status": session.status.value,
                "state": session.state.to_dict(),
                "drives": session.drives,
                "identity_anchor": session.identity_anchor,
                "thought_pool": session.thought_pool,
                "introspections": session.introspections[-3:] if session.introspections else [],
                "position": _world.agent_position(sid) if _world else "",
                "rounds_since_human": session.rounds_since_human,
                "causal_chain_last": [e.to_dict() for e in session.causality.get_recent(session.id, n=10)],
                "causal_chain_count": len(session.causality.get_chain(session.id)),
                # World Model fields
                "hypothesis_count": len(session.hypothesis_manager.active_hypotheses),
                "evidence_count": len(session.evidence_manager.all_evidence),
                "world_model": session.state.get("world_model", {}),
                "confidence": session.state.get("confidence", 0.0),
                # Communication fields
                "mailbox_count": session.communication.get_mailbox(session.id).count()
                    if session.communication else 0,
                "shared_knowledge_count": len(session.shared_knowledge.public_knowledge)
                    if session.shared_knowledge else 0,
            })
        except Exception:
            dead_ids.append(sid)
    # Clean up dead sessions
    for sid in dead_ids:
        _engines.pop(sid, None)
    return {"sessions": result, "world": _world.get_world_summary() if _world else {}}


# ── Causal Chain ──


@router.get("/api/causal-chain/{session_id}")
async def get_causal_chain(session_id: str):
    """Get the full causal chain for a session."""
    engine = _get_engine(session_id)
    try:
        session = engine.get_session(session_id)
        chain = session.causality.get_chain(session_id)
        return {
            "session_id": session_id,
            "entries": [e.to_dict() for e in chain],
            "count": len(chain),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/causal-chain/{session_id}/trace")
async def trace_causal_path(session_id: str, from_round: int = 1, to_round: int = 0):
    """Trace the causal path between two rounds."""
    engine = _get_engine(session_id)
    try:
        session = engine.get_session(session_id)
        if to_round <= 0:
            to_round = session.round
        path = session.causality.trace_path(session_id, from_round, to_round)
        return {
            "session_id": session_id,
            "from_round": from_round,
            "to_round": to_round,
            "entries": [e.to_dict() for e in path],
            "count": len(path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Communication ──


@router.post("/api/send-message")
async def send_message(req: SendMessageRequest):
    """Send a direct message from one agent to another.

    The message lands in the recipient's mailbox.
    The recipient decides whether to process it.
    """
    engine = _get_engine(req.session_id)
    try:
        session = engine.get_session(req.session_id)
        cm = session.communication
        if not cm:
            raise HTTPException(status_code=400, detail="Communication layer not available")
        msg = cm.send(
            from_agent=req.session_id,
            to_agent=req.to_agent,
            msg_type=req.msg_type,
            content=req.content,
            world_tick=(
                _world.get_world_summary()["meta"]["tick"]
                if _world else 0
            ),
            world_room=_world.agent_position(req.session_id) if _world else "",
        )
        if msg is None:
            raise HTTPException(status_code=404, detail=f"Agent {req.to_agent} not found")
        return {"ok": True, "message": msg.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/broadcast")
async def broadcast(req: SendBroadcastRequest):
    """Broadcast a message to all agents."""
    engine = _get_engine(req.session_id)
    try:
        session = engine.get_session(req.session_id)
        cm = session.communication
        if not cm:
            raise HTTPException(status_code=400, detail="Communication layer not available")
        sent = cm.broadcast(
            from_agent=req.session_id,
            msg_type=req.msg_type,
            content=req.content,
            world_tick=(
                _world.get_world_summary()["meta"]["tick"]
                if _world else 0
            ),
            world_room=_world.agent_position(req.session_id) if _world else "",
        )
        return {"ok": True, "sent_count": len(sent)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/mailbox/{session_id}")
async def get_mailbox(session_id: str, max_items: int = 20):
    """Get the current mailbox contents for an agent."""
    engine = _get_engine(session_id)
    try:
        session = engine.get_session(session_id)
        cm = session.communication
        if not cm:
            return {"session_id": session_id, "messages": [], "count": 0}
        mb = cm.get_mailbox(session_id)
        if not mb:
            return {"session_id": session_id, "messages": [], "count": 0}
        all_msgs = mb.messages
        return {
            "session_id": session_id,
            "messages": [m.to_dict() for m in all_msgs[-max_items:]],
            "count": len(all_msgs),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/sent-messages/{session_id}")
async def get_sent_messages(session_id: str, max_items: int = 20):
    """Get messages sent BY a specific agent (visible in sender's communication panel)."""
    engine = _get_engine(session_id)
    try:
        session = engine.get_session(session_id)
        cm = session.communication
        if not cm:
            return {"session_id": session_id, "messages": [], "count": 0}
        msgs = cm.get_sent_messages(session_id, n=max_items)
        return {
            "session_id": session_id,
            "messages": [m.to_dict() for m in msgs],
            "count": len(msgs),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/shared-knowledge")
async def get_shared_knowledge(domain: str = ""):
    """Get public shared knowledge, optionally filtered by domain."""
    # Get the first available engine's shared knowledge
    if not _engines:
        return {"entries": [], "candidates": []}
    first_engine = next(iter(_engines.values()))
    try:
        session = first_engine.get_session(next(iter(_engines)))
        sk = session.shared_knowledge
        if not sk:
            return {"entries": [], "candidates": []}
        entries = sk.public_knowledge
        if domain:
            entries = [e for e in entries if e.domain == domain]
        return {
            "entries": [e.to_dict() for e in entries],
            "candidates": [e.to_dict() for e in sk.candidates],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/event-bus/{session_id}")
async def get_event_bus(session_id: str):
    """Get recent world events from the EventBus."""
    engine = _get_engine(session_id)
    try:
        session = engine.get_session(session_id)
        cm = session.communication
        if not cm:
            return {"events": []}
        return {
            "events": cm.event_bus.get_recent_events(n=20),
            "subscriptions": cm.event_bus.get_subscriber_count(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── World Model (hypotheses / evidence) ──


@router.get("/api/hypotheses/{session_id}")
async def get_hypotheses(session_id: str):
    """Get all hypotheses for a session."""
    engine = _get_engine(session_id)
    try:
        session = engine.get_session(session_id)
        return {
            "session_id": session_id,
            "active": [h.to_dict() for h in session.hypothesis_manager.active_hypotheses],
            "all": session.hypothesis_manager.to_dict(),
            "contradictions": [
                {"a": a.statement, "b": b.statement}
                for a, b in session.hypothesis_manager.get_contradictions()
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/evidence/{session_id}")
async def get_evidence(session_id: str):
    """Get all evidence for a session."""
    engine = _get_engine(session_id)
    try:
        session = engine.get_session(session_id)
        return {
            "session_id": session_id,
            "evidence": [e.to_dict() for e in session.evidence_manager.all_evidence],
            "count": len(session.evidence_manager.all_evidence),
            "domain_confidence": session.evidence_manager.get_domain_confidence(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Runtime Statistics & Evolution ──


@router.get("/api/stats/{session_id}")
async def get_runtime_stats(session_id: str):
    """Get runtime statistics for an agent."""
    engine = _get_engine(session_id)
    try:
        stats = engine._runtime_stats.get_agent_stats(session_id)
        return {
            "session_id": session_id,
            "stats": stats.to_dict(),
            "global": engine._runtime_stats.get_global_stats(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/evolution/{session_id}")
async def get_evolution_report(session_id: str):
    """Get evolution engine reports and current parameters."""
    engine = _get_engine(session_id)
    try:
        return {
            "session_id": session_id,
            "params": engine._evolution_params.to_dict(),
            "reports": engine._evolution_engine.reports,
            "last_run_round": engine._evolution_engine.last_run_round,
            "interval": engine._evolution_engine.interval,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Health ──


@router.get("/api/health")
async def health():
    """Health check (does NOT verify LLM connectivity)."""
    world_summary = _world.get_world_summary() if _world else None
    return {
        "status": "ok",
        "active_sessions": len(_engines),
        "world": {
            "rooms": world_summary["rooms"] if world_summary else 0,
            "agents": world_summary["agents"] if world_summary else 0,
            "ticks": world_summary["meta"]["tick"] if world_summary else 0,
        },
    }


# ── Action System ──


@router.get(
    "/api/capabilities",
    response_model=CapabilitiesResponse,
)
async def list_capabilities():
    """List all available capabilities and their operations."""
    if not _action_executor.is_initialized():
        return CapabilitiesResponse(capabilities=[])

    caps = _action_executor.list_capabilities()
    result = []
    for cap in caps:
        ops_raw = _action_executor.get_operations(cap.name)
        ops = [
            OperationInfo(
                name=o.get("name", "?"),
                description=o.get("description", ""),
                parameters=o.get("parameters", {}),
            )
            for o in ops_raw
        ]
        result.append(CapabilityInfo(
            name=cap.name,
            description=cap.description,
            enabled=cap.enabled,
            operations=ops,
        ))
    return CapabilitiesResponse(capabilities=result)


@router.get(
    "/api/capabilities/status",
    response_model=ActionSystemStatusResponse,
)
async def action_system_status():
    """Get Action System status (capabilities list)."""
    caps = _action_executor.list_capabilities() if _action_executor.is_initialized() else []
    return ActionSystemStatusResponse(
        initialized=_action_executor.is_initialized(),
        capability_count=len(caps),
        capabilities=[c.name for c in caps],
    )


@router.post(
    "/api/actions/execute",
    response_model=ObservationResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def execute_action(req: ActionRequest):
    """Execute an action on a capability.

    The action is routed to the correct capability adapter.
    Session ID is required to verify the engine is running.
    """
    engine = _get_engine(req.session_id)
    if not _action_executor.is_initialized():
        raise HTTPException(
            status_code=400,
            detail="Action system not initialized — provide mcp_tools config on connect",
        )
    try:
        action = Action(
            capability=req.capability,
            operation=req.operation,
            parameters=req.parameters or {},
        )
        observation = _action_executor.execute(action)
        return ObservationResponse(
            success=observation.success,
            content=observation.content,
            metadata=observation.metadata,
            error=observation.error,
        )
    except Exception as e:
        return ObservationResponse(success=False, error=str(e))


# ── Human Interaction ──


@router.get(
    "/api/human/pending/{session_id}",
    response_model=PendingQuestionResponse,
)
async def get_pending_human_question(session_id: str):
    """Check if an agent has a pending question waiting for human answer."""
    engine = _get_engine(session_id)
    try:
        session = engine.get_session(session_id)
        from runtime_kernel import SessionStatus
        if session.status != SessionStatus.WAITING_HUMAN:
            return PendingQuestionResponse(session_id=session_id, has_pending=False)
        q = session.pending_human_question or {}
        return PendingQuestionResponse(
            session_id=session_id,
            has_pending=True,
            question=q.get("question", ""),
            reason=q.get("reason", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/api/human/answer",
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def submit_human_answer(req: HumanAnswerRequest):
    """Submit a human answer to an agent's pending question.

    The agent's step will be continued with the answer as Observation.
    """
    engine = _get_engine(req.session_id)
    try:
        result = engine.continue_with_human_answer(req.session_id, req.answer)
        return result
    except Exception as e:
        _cleanup_on_error(req.session_id)
        raise HTTPException(status_code=500, detail=str(e))


# ── Policy Engine ──


@router.get("/api/policy/{session_id}")
async def get_policy_biases(session_id: str):
    """Get current policy biases for an agent."""
    engine = _get_engine(session_id)
    try:
        biases = engine._policy_engine.get_biases()
        prompt_context = engine._policy_engine.format_for_prompt()
        return {
            "session_id": session_id,
            "biases": biases,
            "prompt_context": prompt_context,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/policy/reset/{session_id}")
async def reset_policy(session_id: str):
    """Reset policy biases to defaults."""
    engine = _get_engine(session_id)
    try:
        engine._policy_engine.reset()
        return {"ok": True, "session_id": session_id, "biases": engine._policy_engine.get_biases()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Three-Layer Architecture (Core / Cognitive / Exploration) ──


@router.get("/api/core/status/{session_id}")
async def get_core_status(session_id: str):
    """Get Stable Core Layer status (validator stats + safety rules)."""
    engine = _get_engine(session_id)
    try:
        validator_stats = engine._core_validator.get_stats() if engine._core_validator else {}
        safety = engine._safety_rules.get_status()
        return {
            "session_id": session_id,
            "validator": validator_stats,
            "safety": safety,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/cognitive/graph/{session_id}")
async def get_causal_graph(session_id: str):
    """Get causal graph for a session."""
    engine = _get_engine(session_id)
    try:
        return {
            "session_id": session_id,
            "causal_graph": engine._causal_graph.to_dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/cognitive/beliefs/{session_id}")
async def get_probabilistic_beliefs(session_id: str):
    """Get probabilistic world model beliefs."""
    engine = _get_engine(session_id)
    try:
        return {
            "session_id": session_id,
            "beliefs": engine._probabilistic_wm.to_dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/exploration/experiments/{session_id}")
async def list_experiments(session_id: str):
    """List experiments scheduled for a session."""
    engine = _get_engine(session_id)
    try:
        return {
            "session_id": session_id,
            "experiments": engine._experiment_scheduler.list_experiments(),
            "stats": engine._experiment_scheduler.get_stats(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/exploration/worlds/{session_id}")
async def get_multi_worlds(session_id: str):
    """Get multi-world simulation results."""
    engine = _get_engine(session_id)
    try:
        return {
            "session_id": session_id,
            "worlds": [w.to_dict() for w in engine._multi_world.get_worlds()],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Self-Driven Scientific Agent (SDSA) ──


@router.get("/api/sdsa/status/{session_id}")
async def get_sdsa_status(session_id: str):
    """Get SDSA daemon loop status."""
    engine = _get_engine(session_id)
    try:
        daemon = engine._sdsa_daemon
        status = daemon.get_status() if daemon else {"running": False}
        queue = engine._sdsa_queue.get_stats()
        return {
            "session_id": session_id,
            "daemon": status,
            "queue": queue,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/sdsa/history/{session_id}")
async def get_sdsa_history(session_id: str):
    """Get SDSA cycle history."""
    engine = _get_engine(session_id)
    try:
        daemon = engine._sdsa_daemon
        history = daemon.get_history() if daemon else []
        return {
            "session_id": session_id,
            "cycles": history,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/sdsa/goals/{session_id}")
async def get_sdsa_goals(session_id: str):
    """Get SDSA research goals for a session."""
    engine = _get_engine(session_id)
    try:
        session = engine.get_session(session_id)
        goals = getattr(session, "_sdsa_goals", [])
        return {
            "session_id": session_id,
            "goals": [g.to_dict() for g in goals],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Scientific Loop ──


@router.get("/api/science/history/{session_id}")
async def get_science_history(session_id: str):
    """Get scientific cycle history for an agent."""
    engine = _get_engine(session_id)
    try:
        return {
            "session_id": session_id,
            "cycles": engine._scientific_loop.get_history(),
            "cycle_count": engine._scientific_loop.cycle_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/science/status/{session_id}")
async def get_science_status(session_id: str):
    """Get science loop status for an agent."""
    engine = _get_engine(session_id)
    try:
        session = engine.get_session(session_id)
        world_model = session.state.get("world_model", {})
        findings = world_model.get("scientific_findings", [])
        insights = world_model.get("insights", [])
        return {
            "session_id": session_id,
            "cycle_count": engine._scientific_loop.cycle_count,
            "interval": engine._scientific_loop.interval,
            "should_run": engine._scientific_loop.should_run(session.round),
            "round": session.round,
            "last_cycle_round": engine._scientific_loop._last_cycle_round,
            "findings_count": len(findings),
            "insights_count": len(insights),
            "latest_insights": insights[-5:],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Agent Observability Events ──


@router.get("/api/events/{session_id}")
async def get_agent_events(session_id: str, since: str = "", limit: int = 200):
    """Get stored agent events for a session.

    Optional `since` parameter filters to events after a specific event ID.
    """
    events = _agent_events.get_events(session_id, since_id=since or None, limit=limit)
    return {
        "session_id": session_id,
        "events": [e.to_dict() for e in events],
        "count": len(events),
    }


@router.websocket("/ws/events/{session_id}")
async def websocket_agent_events(websocket: WebSocket, session_id: str):
    """Stream agent events in real-time via WebSocket.

    1. Sends all stored events for the session (catch-up)
    2. Then streams new events as they happen
    3. Client reconnects by providing the last received event ID
    """
    await websocket.accept()

    # Send stored events first
    stored = _agent_events.get_events(session_id, limit=200)
    for event in stored:
        try:
            await websocket.send_json(event.to_dict())
        except Exception:
            return  # Client disconnected during catch-up

    # Subscribe to new events
    queue = _agent_events.subscribe(session_id)
    try:
        while True:
            event = await queue.get()
            try:
                await websocket.send_json(event.to_dict())
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        _agent_events.unsubscribe(session_id, queue)
