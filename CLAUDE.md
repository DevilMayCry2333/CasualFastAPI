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

## Three-Layer Architecture (inside RuntimeEngine)

```
Exploration Layer       ← High stochasticity, proposes experiments/candidates
    ↑↓
Cognitive Layer         ← 8 agent-internal cognitive models
    ↑↓
Stable Core Layer       ← ONLY layer that executes actions, pure constraint enforcement
```

| Layer | Module | Role | Can execute actions? |
|---|---|---|---|
| Stable Core | `core/` | `ActionValidator` + `SafetyRules` — validates and executes every action | Yes (only) |
| Cognitive | `cognitive/` | 8 models (SelfModel, WorldModel, SocialModel, KnowledgeModel, TheoryOfMind, WorkingMemory, Attention, Perception) | No |
| Exploration | `exploration/` | `StochasticHypothesisGenerator`, `ExperimentScheduler`, `MultiWorldSimulator` | No |

**Invariants:**
- Cognitive Layer cannot execute MCP directly
- Exploration Layer cannot execute MCP directly
- Only Stable Core can invoke ActionExecutor
- All actions pass through validator before execution
- No learning happens in Stable Core — pure constraint enforcement
- Exploration outputs are "candidates" for Cognitive Layer, never applied directly

## Key Subsystems

### Engine (2700 lines — the monolith)

`runtime_kernel/runtime/engine.py` is a single class `RuntimeEngine` that orchestrates everything. It is the ONLY public API. All agent logic lives here — prompt building, LLM calls, state parsing, force integration, identity reflection, memory storage, event emission, action execution, scientific cycles, evolution, heartbeats.

**The three phases of `step()`:**
1. **Autonomous step** (`_handle_autonomous_step`): No human input. Agent thinks, acts, updates world model. ~1200 lines.
2. **Human interrupt** (`_handle_human_interrupt`): Human message treated as **evidence** first, goes through evidence pipeline. ~300 lines.
3. **Continue with human answer** (`continue_with_human_answer`): Resumes after Human.ask pause, agent sees answer as Observation. ~200 lines.

### Evidence Pipeline (World Model v2 design)

```
human_input / observation
    ↓
Evidence (always the entry point)
    ↓
Hypothesis (accumulated support)
    ↓
Belief Update (only when hypothesis confidence > 0.7)
```

**Rules:**
- LLM cannot directly set `belief` — only through the evidence pipeline
- If LLM proposes a belief change without hypothesis backing, the engine checks for supporting evidence
- If force < 0.6 and no supporting evidence, belief stays unchanged
- Hypothesis needs ≥ 3 supporting evidence to be "belief-ready"
- If contradiction/support ratio > 0.5, belief update is blocked

### Action System (v6)

```
Planner (LLM) → action: {capability, operation, parameters}
    ↓
ActionExecutor.execute(action)  [routed via Core Validator]
    ↓
SearchAdapter (wraps MCP internally)  |  HumanAdapter (ask/tell)  |  BrowserAdapter (future)
    ↓
Observation(success, content, metadata, error)
    ↓
Planner continues reasoning (max 3 actions per step)
```

Every action returns an `Observation`. The planner never reads raw tool output.

### Cognitive Architecture (v4)

8 specialized models, each with `format_for_prompt()` that gets injected into the LLM prompt:

| Model | File | Purpose |
|---|---|---|
| SelfModel | `cognitive/self_model.py` | "who I am" — identity, beliefs, goals, drives |
| WorldModel | `cognitive/world_model.py` | "what the world is" — places, objects, events, rules |
| SocialModel | `cognitive/social_model.py` | "who others are" — trust, cooperation, interaction history |
| KnowledgeModel | `cognitive/knowledge_model.py` | "what I know" — facts, hypotheses, evidence, contradictions |
| TheoryOfMind | `cognitive/theory_of_mind.py` | "what others believe" — perceived beliefs, goals, confidence |
| WorkingMemory | `cognitive/working_memory.py` | "what I'm thinking about now" — current focus, active question |
| Attention | `cognitive/attention.py` | "what I notice" — salience-based event filtering |
| Perception | `cognitive/perception.py` | "how I perceive" — event-to-cognition pipeline |

**Prompt order (cognitive-first):**
`SelfModel → WorldModel → SocialModel → KnowledgeModel → WorkingMemory → Perception → TheoryOfMind → Environment → Drives → Memory → State → Capabilities`

All prompts are in Chinese, centralized in `prompt.py`. The prompt is a cognitive snapshot — it reflects the current state of the models, not a chat template.

### MCP Architecture

```
ToolManager (agent-facing, unified)
    │
    ├── MCPRuntime 1 (Search MCP)
    │       └── MCPClient (JSON-RPC over stdio/SSE/HTTP)
    └── MCPRuntime 2 (future)
```

- `mcp/manager.py` (`ToolManager`): the ONLY class agents/engine interact with. Aggregates multiple MCPRuntimes.
- `mcp/runtime.py` (`MCPRuntime`): lifecycle of one MCP server connection — connect, discover tools, execute, close.
- `mcp/client.py` (`MCPClient`): low-level JSON-RPC transport. Three modes auto-detected from config:

| Mode | Config | Use Case |
|---|---|---|
| `stdio` | `command` + `args` | Local subprocess (uvx mcp-server-tavily) |
| `sse` | `url` + SSE response | Standard MCP HTTP transport |
| `direct` | `url` + auto-fallback | Direct HTTP POST (Tavily MCP) |

Connection failures log but don't block remaining servers (graceful degradation). Tool name collisions are detected at initialization.

### SDSA — Self-Driven Scientific Agent

Background daemon loop that runs continuously (every 60s):
```
Generate Research Goals → Enqueue Experiments → Execute through Core → Causal Update → World Model Update
```
Stored in `runtime_kernel/runtime/sdsa/`. Independent of user requests. Only starts when an `ActionExecutor` is configured.

### Agent Observability (Event Stream)

All steps emit structured events via `AgentEventBus`. Event types and icons:

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

WebSocket at `/ws/events/{session_id}` — sends stored events on connect, then streams new ones. The frontend Timeline tab subscribes to this.

### Policy Engine (Causal Policy Evolution)

Data-only policy learning (not self-modifying code):
- `preferred_capabilities`: capabilities with >60% success rate
- `avoided_operations`: operations that failed ≥2 times
- `consecutive_failures`: triggers "建议更换策略" warning
- `efficiency_score`: blended success rate + latency

### Self-Modification System

The `SelfModificationManager` extracts `drive_params` and `thought_templates` from the identity anchor during reflection. These can override:
- `DriveModel.after_step()` and `DriveModel.tick()` behavior
- `GoalGenerator.generate()` template strings
- The agent's own `identity_anchor` values

The LLM never modifies its own prompt. Self-modifications are structured data extracted from the identity anchor.

### Experience & Identity Model

An `Experience` is the atomic unit of an agent's existence, distinct from a raw state transition:

- `round`, `session_id` — where/when
- `perception` — what the agent observed in the environment
- `action` — what the agent did ("look", "move garden", etc.)
- `observation` — world feedback / result
- `meaning` — initially empty, populated later by Reflection
- `state_before`, `state_after` — full snapshots

Key insight: **An agent IS its sequence of Experiences, not its state.** State is working memory; Experience is the life lived.

`compute_identity_maturity()` combines:
- Rounds lived (25% weight, denominator 150)
- Experiences accumulated (35% weight, denominator 80)
- Reflections performed (40% weight, denominator 20)

Each component saturates independently. Maturity ranges 0.0 (newborn) → 1.0 (stable personality).

### Duplicate WorkingMemory Warning

There's an OLD `runtime_kernel/runtime/memory.py` (`WorkingMemory` — history truncation wrapper, 91 lines) and the ACTIVE `runtime_kernel/runtime/cognitive/working_memory.py` (one of the 8 cognitive models). The cognitive one is the one used throughout `engine.py`. The old one is likely dead code.

## Session Data Model

`AgentSession` (in `session.py`, ~580 lines) holds everything:

```
_state                  State object (topic, belief, goal + world model fields)
_state_compressed       Fold/compressed state
_identity_anchor        Self-identity (emerges from experience)
_drives                 {curiosity, boredom, belonging} floats
_thought_pool           Candidate goals from GoalGenerator
_self_modifications     Override dicts for drives + templates
_causality              CausalityManager — all causal entries
_hypothesis_manager     HypothesisManager — active hypotheses lifecycle
_evidence_manager       EvidenceManager — all observations as evidence
_self_model             Cognitive SelfModel
_world_model_cog        Cognitive WorldModel
_social_model           SocialModel
_knowledge_model        KnowledgeModel
_theory_of_mind         TheoryOfMind
_working_memory         WorkingMemory
_history                List of all state transitions
_interactions           Human interaction records
_introspections         Introspection records
_experiences            Experience objects (round-scoped)
_communication          CommunicationManager reference
_shared_knowledge       SharedKnowledge reference
```

Held entirely in-memory. No database. Restart loses everything unless snapshots saved.

## LLM Output Format

```json
{
  "delta_topic": "新主题",
  "delta_belief": "新信念",
  "delta_goal": "新目标",
  "force": 0.65,
  "action": "要执行的动作",
  "source": "curiosity | boredom | memory | observation",
  "action": {"capability": "Search", "operation": "web_search", "parameters": {"query": "..."}},
  "new_hypotheses": [{"statement": "...", "domain": "..."}],
  "new_evidence": [{"statement": "...", "source": "observation", "confidence": 0.8}],
  "hypothesis_updates": [{"id": "...", "supports": true}],
  "world_model_update": {"key": "value"},
  "open_questions": ["..."],
  "uncertainties": [{"domain": "...", "description": "..."}],
  "confidence": 0.75,
  "self_update": {"belief": "...", "confidence": 0.7},
  "social_update": {"agent_id": "...", "cooperative": true},
  "send_message": [{"to_agent": "...", "type": "observation", "content": {"text": "..."}}],
  "share_knowledge": {"statement": "...", "domain": "..."}
}
```

Parsed by `parser.py` (~620 lines) — stateless functions: `extract_state()`, `extract_causal_vector()`, `extract_world_model_updates()`, `repair_state()`, `detect_loop()`.

Force integration (`causal_physics.py`, ~640 lines): combines LLM force + memory force + identity force + world force using `compute_*_force()` functions and `integrate_state()`.

## Default Config

| Setting | Default |
|---|---|
| LLM API | `http://localhost:28000/v1/chat/completions` |
| Model | `deepseek-v4-flash` |
| Temperature | 0.85 (chat), 0.7 (auto step) |
| Embedding API | `http://0.0.0.0:28001/v1/embeddings` |
| Memory retrieval top_k | 3 |
| Max actions per step | 3 |
| Identity reflection | Every 5 rounds |
| Introspection | Every 20 rounds |
| Scientific cycle | Every 10 rounds |
| Auto-save | Every 10 rounds (to `autosave.json`) |
| Heartbeat | 30s daemon thread |
| SDSA daemon | 60s cycle |
| Session serialization | version 5 |

## Complete API Endpoints

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
| GET | `/api/core/status/{id}` | Core layer validator + safety status |
| GET | `/api/cognitive/graph/{id}` | Causal graph state |
| GET | `/api/cognitive/beliefs/{id}` | Probabilistic world model beliefs |
| GET | `/api/exploration/experiments/{id}` | Scheduled experiments |
| GET | `/api/exploration/worlds/{id}` | Multi-world simulation results |
| GET | `/api/sdsa/status/{id}` | SDSA daemon loop status |
| GET | `/api/sdsa/history/{id}` | SDSA cycle history |
| GET | `/api/sdsa/goals/{id}` | SDSA research goals |
| WS | `/ws/events/{id}` | Real-time event stream |

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

# Check 3-layer architecture
curl -s http://localhost:8000/api/core/status/{id} | python3 -m json.tool
curl -s http://localhost:8000/api/cognitive/graph/{id} | python3 -m json.tool
curl -s http://localhost:8000/api/exploration/experiments/{id} | python3 -m json.tool
curl -s http://localhost:8000/api/sdsa/status/{id} | python3 -m json.tool

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
- **Port conflicts**: `pkill -f "uvicorn main:app"` then restart. LLM on :28000, Embeddings on :28001.
- **engine.py is 2700 lines** — single file, all orchestration logic. Refactor carefully.
- **All prompts in Chinese** — `PromptBuilder` in `prompt.py`. Don't scatter prompt strings.

## Key Architecture Decisions

- **Single entry point**: `RuntimeEngine` is the ONLY public API. No external code modifies sessions.
- **No LLM in Session**: `AgentSession` is a pure data container. LLM calls only in `RuntimeEngine.step()`.
- **Evidence pipeline**: human input and observations always enter as **Evidence**, go through Hypothesis, and only then affect Belief. No direct belief setting.
- **Causal chain as primary history**: Every transition recorded as `CausalEntry`. Immutable, traceable.
- **MCP is internal**: Only `SearchAdapter` uses MCP. No other component knows about it.
- **Capabilities, not tools**: Planner sees Search/Human/Browser, never web_search/fetch_url directly.
- **Planner produces Actions**: The LLM outputs `action: {capability, operation, parameters}`. Engine routes execution.
- **Observation is universal**: Every action returns Observation. Planner never reads raw tool output.
- **Policy is data, not code**: Policy biases are structured data injected as prompt context. LLM never modifies its own prompt.
- **Human = Capability**: Human interaction is the same abstraction as Search. No special-case chat logic.
- **All steps emit events**: `AgentEventBus` records every decision, action, and observation.
- **Scientific method loop**: Every 10 rounds: questions → hypotheses → experiments → theory updates.
- **Self-modification through identity**: `SelfModificationManager` extracts structured overrides from the identity anchor. The LLM generates new identity deltas; the engine extracts actionable modifications.
- **Loop detection**: `detect_loop()` in parser.py compares topic/belief fingerprints over a window of 8 rounds.
- **3-layer separation**: Core executes (no learning), Cognitive reasons (no execution), Exploration proposes (no execution). Each layer's invariant is enforced by code structure.
- **In-memory by default**: No database. All state in Python dicts. Persistence via JSON file snapshots.
- **Dependency injection**: All modules injected via constructor params.

## Debugging

### Check what's running
```bash
# All agents
curl -s http://localhost:8000/api/sessions | python3 -m json.tool
# Engine health + world state
curl -s http://localhost:8000/api/health
```

### Action System
```python
engine = _get_engine(session_id)  # from routes.py pattern
for cap in engine._action_executor.list_capabilities():
    print(f"  {cap.name}: enabled={cap.enabled}")
    for op in engine._action_executor.get_operations(cap.name):
        print(f"    - {op['name']}: {op.get('description', '')[:60]}")
```

### Session State
```python
session = engine.get_session(session_id)
print(f"Status: {session.status.value}, Round: {session.round}")
print(f"Drives: {session.drives}")
print(f"Belief: {session.state.belief}")
```

### World Model
```python
wm = session.state.get("world_model", {})
print(f"Findings: {len(wm.get('scientific_findings', []))}")
print(f"Insights: {len(wm.get('insights', []))}")
```

### Agent Events
```python
events = engine._event_bus.get_events(session.id)
for e in events[-5:]:
    print(f"  [{e.type}] R{e.round} {e.payload.get('decision', '')[:40]}")
print(f"Total events: {engine._event_bus.event_count(session.id)}")
```

### Policy Biases
```python
biases = engine._policy_engine.get_biases()
print(f"Success: {biases['successful_actions']}/{biases['total_actions']}")
print(f"Preferred: {biases['preferred_capabilities']}")
print(f"Efficiency: {biases['efficiency_score']}")
```

### Human Interaction
```python
session = engine.get_session(session_id)
print(f"Status: {session.status.value}")
print(f"Pending: {session.pending_human_question}")
# Check HumanAdapter
human_adapter = engine._action_executor._adapters.get("Human")
if human_adapter:
    for sid, q in human_adapter._pending_questions.items():
        print(f"  {sid[:8]}: {q.get('question', '')[:60]}")
```

## Exception Hierarchy

All inherit from `runtime_kernel.runtime.exceptions.RuntimeError` (which inherits from builtin `Exception`, NOT `RuntimeError`):

```
RuntimeError (base)
├── SessionNotFoundError
├── SessionAlreadyExistsError
├── LLMError
├── StateValidationError
├── StateParseError
├── ConfigurationError
├── PersistenceError
├── EmbeddingError
└── MemoryError
```
