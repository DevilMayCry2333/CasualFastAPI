# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Agent Runtime Kernel** — a pure Python library for managing AI agent lifecycle with autonomous persistence, recursive self-improvement, and a scientific method loop.

The system's core goal is **building, maintaining, and correcting a World Model**, not generating conversation. Each agent continuously evolves a JSON state object through LLM self-dialogue, driven by evidence accumulation and hypothesis testing. Agents operate in a shared virtual world with capabilities (Search, Human, etc.) via an **Action System** that abstracts all external interactions.

Core equation: `f(x) = f(x-1) + t` — each round's state derives from the previous round plus new LLM-generated thought, with every transition recorded as a traceable causal edge.

## Two-Layer Architecture

```
runtime_kernel/     ← Framework-agnostic library (no HTTP imports, pure Python)
web/                ← FastAPI REST + HTML frontend wrapping the kernel
main.py             ← uvicorn entry point
pyproject.toml      ← Dependencies: fastapi, uvicorn, requests (Python 3.11+)
```

The `runtime_kernel` can be used standalone. The web layer is a thin REST/HTML shell.

## Project Structure

```
runtime_kernel/
  __init__.py                  # Public API exports

  runtime/
    engine.py                  # RuntimeEngine — SINGLE entry point, orchestrates all modules
    session.py                 # AgentSession — pure data container (NO LLM logic)
    state.py                   # State — JSON wrapper with world model fields
    llm.py                     # LLMClient — OpenAI-compatible API (sync + streaming)
    prompt.py                  # PromptBuilder — ALL prompts centralized (Chinese)
    parser.py                  # Stateless functions: extract → parse → repair → State
    models.py                  # Enums (StateCause, SessionStatus), constants
    exceptions.py              # Exception hierarchy (all inherit RuntimeError)

    # World Model (v2)
    hypothesis.py              # HypothesisManager — lifecycle management
    evidence.py                # EvidenceManager — observations become Evidence

    # Multi-Agent Communication (v3)
    communication.py           # CommunicationManager, Message, Mailbox, EventBus
    shared_memory.py           # SharedKnowledge — consensus-based public knowledge

    # Causal State Engine
    causality.py               # CausalityManager + CausalEntry — causal chain
    causal_physics.py          # CausalVector, force computation, state integration
    identity_manager.py        # IdentityManager — self-reflection
    introspection.py           # Introspector — world model evolution analysis
    self_modification.py       # SelfModificationManager

    # Drive System
    drive.py                   # DriveModel — curiosity/boredom/belonging
    goal_generator.py          # GoalGenerator — thought pool from drives
    heartbeat.py               # HeartbeatManager — background aliveness daemon

    # Memory
    embedding.py               # EmbeddingClient — separate embedding API
    memory_manager.py          # MemoryManager — RAG store/retrieve
    memory_storage.py          # MemoryStorage ABC + InMemoryMemoryStorage

    # Cognitive Architecture (v4)
    cognitive/
      __init__.py
      self_model.py, world_model.py, social_model.py, knowledge_model.py
      theory_of_mind.py, working_memory.py, perception.py, attention.py

    # Evolution Engine (v5)
    runtime_statistics.py      # RuntimeStatistics, AgentStats
    evolution.py               # EvolutionEngine, RuntimeParameters

    # Action System (v6) — NEW
    action/
      __init__.py              # Exports: Action, Observation, Capability, ActionExecutor
      models.py                # Action, Capability, Observation dataclasses
      executor.py              # ActionExecutor — unified execution entry point
      adapters/                # Pluggable capability adapters
        __init__.py            # CapabilityAdapter ABC + CAPABILITY_REGISTRY
        search_adapter.py      # SearchAdapter — wraps MCP Runtime internally
        human_adapter.py       # HumanAdapter — ask/tell operations

    # MCP Runtime — NEW
    mcp/
      __init__.py
      models.py                # MCPConfig, ToolInfo, ToolResult
      client.py                # MCPClient — JSON-RPC over stdio/SSE/HTTP
      runtime.py               # MCPRuntime — connection lifecycle manager

    # Agent Observability — NEW
    agent_events.py            # AgentEvent, AgentEventBus (pub/sub + WebSocket)

    # Policy Engine — NEW
    policy_engine.py           # PolicyEngine, OutcomeEvaluator — causal policy evolution

    # Autonomous Scientific Agent — NEW
    scientific/
      __init__.py
      models.py                # ScientificQuestion, Hypothesis, ExperimentStep, CausalEdge
      question_generator.py    # QuestionGenerator — generate questions from world model
      hypothesis_layer.py      # HypothesisLayer — questions → testable hypotheses
      experiment_planner.py    # ExperimentPlanner — hypotheses → experiment steps
      causal_analyzer.py       # CausalAnalyzer — experiment results analysis
      theory_updater.py        # TheoryUpdater — world model updates from findings
      loop.py                  # ScientificLoop — orchestrates the scientific cycle

    # World
    environment.py             # VirtualEnvironment — shared world
    persistence.py             # Persistence — snapshot/restore JSON
    scheduler.py               # Scheduler — wall-clock-timed step loop

web/
    __init__.py
    routes.py                  # FastAPI router + shared singletons
    schemas.py                 # Pydantic models
    templates/index.html       # Single-page frontend (~1900 lines, all JS inline)

main.py                        # uvicorn entry: `app = FastAPI()` + CORS + router
```

## Action System (v6)

The Action System is the **unified interface for all external interactions**. It replaces the earlier ad-hoc tool handling with a clean Capability architecture.

### Architecture

```
Planner (LLM) → action: {"capability": "Search", "operation": "web_search", "parameters": {...}}
    ↓
ActionExecutor.execute(action)
    ↓
SearchAdapter (wraps MCP)  |  HumanAdapter  |  BrowserAdapter (future)
    ↓
Observation(success, content, metadata, error)
    ↓
Planner continues reasoning
```

### CapabilityAdapter Protocol

Every capability adapter implements:
- `execute(action, session_id) → Observation`
- `list_operations() → list[dict]` (name, description, parameters)
- `get_capability_info() → Capability`

### Action Data Flow (Multi-turn)

```
① LLM outputs action field with capability/operation/parameters
② Engine detects capability_action in wm_updates
③ ActionExecutor routes to correct adapter
④ Adapter executes (Search → MCP, Human → store question)
⑤ Observation returned to engine
⑥ Engine builds continuation prompt with Observation
⑦ LLM continues reasoning (may use more actions or finish)
```

### LLM Output Format

```json
{
  "action": {
    "capability": "Search",
    "operation": "web_search",
    "parameters": {"query": "..."}
  }
}
```

The `action` field can also be a string (world action, backward compatible).

## MCP Runtime

The MCP Runtime is **internal to SearchAdapter**. No other component interacts with it directly.

Three transport modes auto-detected from config:

| Mode | Config | Use Case |
|---|---|---|
| `stdio` | `command` + `args` | Local subprocess (uvx mcp-server-tavily) |
| `sse` | `url` + SSE response | Standard MCP HTTP transport |
| `direct` | `url` + auto-fallback | Direct HTTP POST (Tavily MCP) |

## Human Interaction Capability

Human is a standard Capability, same level as Search.

### Operations

| Operation | Behavior |
|---|---|
| `ask` | Stores pending question, session status → `WAITING_HUMAN`, returns early |
| `tell` | Fire-and-forget message to human |

### Flow

```
Planner → action: {capability: "Human", operation: "ask", parameters: {question: "..."}}
    → HumanAdapter stores question
    → session.status = WAITING_HUMAN
    → step() returns {type: "waiting_human", question: "..."}
    → Frontend shows question + answer input
    → User answers → POST /api/human/answer
    → Engine.continue_with_human_answer()
        → Builds continuation prompt with Observation
        → LLM continues reasoning (may ask more, search, or finish)
```

### API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/human/pending/{id}` | Check for pending question |
| `POST` | `/api/human/answer` | Submit human answer |

## Agent Observability (Event Stream)

All agent execution steps emit structured events via `AgentEventBus`.

### Event Types

| Type | Icon | Meaning |
|---|---|---|
| `planner` | 🧠 | LLM decision point |
| `action_start` | 🔧 | Action execution begins |
| `action_result` | 📎 | Action completed |
| `mcp_request` | 🌍 | MCP tool call sent |
| `mcp_response` | 🌍 | MCP result received (+ elapsed ms) |
| `observation` | 👁️ | Observation fed back to planner |
| `memory_update` | 💾 | State stored in long-term memory |
| `world_action` | 🌐 | Agent acted in virtual world |
| `final_answer` | ✨ | Step completed |
| `human_question_created` | ❓ | Agent asked a question |
| `waiting_human` | ⏳ | Agent waiting for human answer |
| `human_answer_received` | 👤 | Human answered |
| `science_cycle` | 🔬 | Scientific cycle completed |

### WebSocket

```
WS /ws/events/{session_id}
  → Sends all stored events (catch-up on connect)
  → Then streams new events in real-time
```

## Policy Engine (Causal Policy Evolution)

Safe, traceable policy learning — NOT self-modifying code.

### How it works

```
Action results (per step)
    ↓
OutcomeEvaluator.evaluate(action) → success, efficiency, failures
    ↓
PolicyEngine.update_from_step(action_results)
    ↓
Updated biases: preferred_capabilities, avoided_operations, efficiency_score
    ↓
format_for_prompt() → injected as 【策略偏好】context
    ↓
Planner sees biases as data (may adjust decisions)
```

### What changes (data only, not code)

- `preferred_capabilities`: capabilities with >60% success rate
- `avoided_operations`: operations that failed ≥2 times
- `consecutive_failures`: triggers "建议更换策略" warning
- `efficiency_score`: blended success rate + timing

### API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/policy/{id}` | Get current biases |
| `POST` | `/api/policy/reset/{id}` | Reset to defaults |

## Autonomous Scientific Agent (ASA)

Runs every 10 rounds as part of the autonomous step.

### Complete Cycle

```
QuestionGenerator (from World Model + uncertainties)
    ↓
HypothesisLayer (question → testable hypotheses with predictions)
    ↓
ExperimentPlanner (hypothesis → action sequences)
    ↓
ActionExecutor.execute() (runs experiment steps)
    ↓
CausalAnalyzer (analyze results, generate insights)
    ↓
TheoryUpdater (compute theory delta, update World Model)
    ↓
Cycle repeats → new questions from updated World Model
```

### API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/science/history/{id}` | All cycles for session |
| `GET` | `/api/science/status/{id}` | Current status + insights |

## Web API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Serve frontend HTML |
| POST | `/api/connect` | Create engine + session |
| POST | `/api/step` | Autonomous thinking step |
| POST | `/api/chat` | Human interruption |
| GET | `/api/state/{id}` | Single session state |
| POST | `/api/fold` | Identity reflection |
| POST | `/api/translate` | State → natural language |
| POST | `/api/snapshot` | Save session to JSON |
| POST | `/api/restore` | Restore from JSON |
| DELETE | `/api/session/{id}` | Delete session |
| GET | `/api/world` | World summary |
| GET | `/api/world/context/{id}` | Agent's environment context |
| GET | `/api/sessions` | All sessions + world model data |
| GET | `/api/causal-chain/{id}` | Full causal chain |
| GET | `/api/causal-chain/{id}/trace` | Trace causal path |
| GET | `/api/hypotheses/{id}` | Active hypotheses |
| GET | `/api/evidence/{id}` | Evidence list |
| POST | `/api/send-message` | Direct message between agents |
| POST | `/api/broadcast` | Broadcast to all agents |
| GET | `/api/mailbox/{id}` | Agent's mailbox |
| GET | `/api/sent-messages/{id}` | Messages sent by agent |
| GET | `/api/shared-knowledge` | Public shared knowledge |
| GET | `/api/event-bus/{id}` | Recent world events |
| GET | `/api/stats/{id}` | Runtime statistics |
| GET | `/api/evolution/{id}` | Evolution report |
| GET | `/api/health` | Server health |
| GET | `/api/capabilities` | Available capabilities + operations |
| GET | `/api/capabilities/status` | Action System status |
| POST | `/api/actions/execute` | Execute a capability action |
| GET | `/api/human/pending/{id}` | Check pending human question |
| POST | `/api/human/answer` | Submit human answer |
| GET | `/api/policy/{id}` | Current policy biases |
| POST | `/api/policy/reset/{id}` | Reset policy biases |
| GET | `/api/science/history/{id}` | Scientific cycle history |
| GET | `/api/science/status/{id}` | Scientific loop status |
| GET | `/api/events/{id}` | Stored agent events |
| WS | `/ws/events/{id}` | Real-time event stream |

## Frontend Patterns (index.html ~1900 lines, all JS inline)

- **Left panel** (340px): config, agent list + drives/goals, world info
- **Right panel**: state bar, tabbed panels (State / Causal Chain / Communication / Timeline), output log, chat input
- **Agent list cards**: per-agent drives bars, goal/belief/topic, causal chain count, expandable state
- **Timeline tab** (📋): real-time WebSocket event stream with per-type icons, timestamps, expandable details
- **Human interaction**: when agent uses Human.ask, question + answer input appears inline
- **Auto-think loop**: cycles ALL agents via `/api/step` with 300ms inter-agent pause, 30s between rounds
- **State refresh**: GET `/api/sessions` every 3s
- **Event types visible in Timeline**: 🧠 planner, 🔧 action, 🌍 mcp, 👁️ observation, 💾 memory, ❓ human question, ⏳ waiting, 🔬 science

## Default Config

| Setting | Default |
|---|---|
| LLM API | `http://localhost:28000/v1/chat/completions` |
| Model | `deepseek-v4-flash` |
| Temperature | 0.85 (chat), 0.7 (auto step) |
| Embedding API | `http://0.0.0.0:28001/v1/embeddings` |
| Identity reflection | Every 5 rounds |
| Heartbeat | 30s daemon thread |
| Scientific cycle | Every 10 rounds |
| Session serialization | version 5 |

## Commands

```bash
# Start web server (dev with hot reload)
uv run uvicorn main:app --reload

# Quick inline test (no LLM needed)
uv run python -c "
from runtime_kernel import RuntimeEngine
engine = RuntimeEngine(demo=True)
s = engine.create_session(demo=True)
r = engine.step(s.id)
print(r['state'])
"

# Test Action System
uv run python -c "
from runtime_kernel import ActionExecutor, Action, Capability, CapabilityAdapter, Observation

class MockAdapter(CapabilityAdapter):
    def execute(self, action, session_id=''):
        return Observation(success=True, content=[f'Result: {action.operation}'])
    def list_operations(self):
        return [{'name': 'mock_op', 'description': '', 'parameters': {}}]
    def get_capability_info(self):
        return Capability(name='Mock', description='Test')

executor = ActionExecutor()
executor.register('Mock', MockAdapter())
obs = executor.execute(Action(capability='Mock', operation='mock_op'))
print(f'Observation: {obs.content}')
"

# Test HumanAdapter
uv run python -c "
from runtime_kernel import HumanAdapter, Action
ha = HumanAdapter()
obs = ha.execute(Action(capability='Human', operation='ask', parameters={'question': 'What model?'}), session_id='test')
print(f'Pending: {ha.get_pending_question(\"test\")}')
ans = ha.deliver_answer('test', 'GPT-5')
print(f'Answer: {ans.content}')
"

# Test PolicyEngine
uv run python -c "
from runtime_kernel import PolicyEngine
pe = PolicyEngine()
pe.update_from_step([
    {'capability': 'Search', 'operation': 'web_search', 'success': True, 'elapsed_ms': 300, 'error': ''},
    {'capability': 'Search', 'operation': 'fetch_url', 'success': False, 'elapsed_ms': 5000, 'error': 'timeout'},
])
print(pe.format_for_prompt())
"

# Verify causal chain on running server
curl -s http://localhost:8000/api/sessions | python3 -c "
import sys, json
d = json.load(sys.stdin)
for s in d['sessions']:
    print(f'{s[\"id\"][:8]} R{s[\"round\"]}')
"

# Create agent with MCP+Human capabilities
curl -X POST http://localhost:8000/api/connect \
  -H 'Content-Type: application/json' \
  -d '{
    "api_url": "http://localhost:28000/v1/chat/completions",
    "mcp_tools": [{"url": "https://mcp.tavily.com/mcp/?key=xxx"}]
  }'

# List capabilities
curl -s http://localhost:8000/api/capabilities | python3 -m json.tool

# Kill server
pkill -f "uvicorn main:app"
```

## Development

- **Python 3.11+**, managed with `uv` (`uv sync` creates `.venv/`)
- **No test framework** — tests run inline with `uv run python -c "..."` or temp scripts
- **No type checker / linter** — no mypy, no ruff
- **Swagger docs** at `/docs` when server is running
- **All modules** use `from __future__ import annotations` for deferred evaluation
- **Sessions are in-memory** — restart loses everything unless snapshots were saved
- **No database** — all state in Python dicts. Persistence via JSON file snapshots.
- **Port conflicts**: `pkill -f "uvicorn main:app"` then restart

## Key Architecture Decisions

- **Single entry point**: `RuntimeEngine` is the ONLY public API. No external code modifies sessions.
- **No LLM in Session**: `AgentSession` is a pure data container. LLM calls only in `RuntimeEngine.step()`.
- **Encapsulated prompts**: All prompt strings in `PromptBuilder` (Chinese). No scattered prompt strings.
- **MCP is internal**: Only SearchAdapter uses MCP. No other component knows about it.
- **Capabilities, not tools**: Planner sees Search/Human/Browser, never web_search/fetch_url directly.
- **Planner produces Actions**: The LLM outputs `action: {capability, operation, parameters}`. Engine routes execution.
- **Observation is universal**: Every action returns Observation. Planner never reads raw tool output.
- **Policy is data, not code**: Policy biases are structured data injected as prompt context. LLM never modifies its own prompt.
- **Human = Capability**: Human interaction is the same abstraction as Search. No special-case chat logic.
- **All steps emit events**: AgentEventBus records every decision, action, and observation. Frontend Timeline subscribes via WebSocket.
- **Causal chain as primary history**: Every transition recorded as `CausalEntry`. Immutable, traceable.
- **Scientific method loop**: Every 10 rounds the system generates questions → hypotheses → experiments → theory updates. No self-modifying code.
- **In-memory by default**: No database. All state in Python dicts. Persistence via JSON file snapshots.
- **Dependency injection**: All modules injected via constructor params.

## Debugging

### Action System

```python
# Check available capabilities
for cap in engine._action_executor.list_capabilities():
    print(f"  {cap.name}: enabled={cap.enabled}")
    for op in engine._action_executor.get_operations(cap.name):
        print(f"    - {op['name']}: {op.get('description', '')[:60]}")

# Execute an action directly
from runtime_kernel import Action
obs = engine._action_executor.execute(
    Action(capability="Search", operation="web_search", parameters={"query": "AI"}),
    session_id=s.id,
)
print(f"Success: {obs.success}, Content: {obs.content}")
```

### Agent Events

```python
# Check stored events
events = engine._event_bus.get_events(session.id)
for e in events[-5:]:
    print(f"  [{e.type}] R{e.round} {e.payload.get('decision', '')[:40]}")

# Check event count
print(f"Total events: {engine._event_bus.event_count(session.id)}")
```

### Policy Biases

```python
biases = engine._policy_engine.get_biases()
print(f"Success rate: {biases['successful_actions']}/{biases['total_actions']}")
print(f"Preferred: {biases['preferred_capabilities']}")
print(f"Avoided: {biases['avoided_operations']}")
print(f"Efficiency: {biases['efficiency_score']}")
```

### Human Interaction

```python
# Check if waiting for human
session = engine.get_session(session_id)
print(f"Status: {session.status.value}")
print(f"Pending: {session.pending_human_question}")

# Check HumanAdapter pending questions
human_adapter = engine._action_executor._adapters.get("Human")
if human_adapter:
    for sid, q in human_adapter._pending_questions.items():
        print(f"  {sid[:8]}: {q.get('question', '')[:60]}")
```

### World Model

```python
wm = session.state.get("world_model", {})
print(f"Scientific findings: {len(wm.get('scientific_findings', []))}")
print(f"Insights: {len(wm.get('insights', []))}")
```
