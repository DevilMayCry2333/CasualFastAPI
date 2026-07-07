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
main.py             ← uvicorn entry point (lifespan: restore/save sessions)
pyproject.toml      ← Dependencies: fastapi, uvicorn, requests (Python 3.11+)
```

The `runtime_kernel` can be used standalone (no web framework required). The web layer is a thin REST/HTML shell that imports from `runtime_kernel`.

## Three-Layer Architecture (inside RuntimeEngine)

```
Exploration Layer       ← High stochasticity, proposes experiments/candidates
    ↑↓
Cognitive Layer         ← 8 agent-internal cognitive models + CausalGraph + ProbabilisticWM
    ↑↓
Stable Core Layer       ← ONLY layer that executes actions, pure constraint enforcement
```

| Layer | Module | Role | Can execute actions? |
|---|---|---|---|
| Stable Core | `core/` | `ActionValidator` + `SafetyRules` — validates and executes every action | Yes (only) |
| Cognitive | `cognitive/` | 8 models + `CausalGraph` + `ProbabilisticWorldModel` | No |
| Exploration | `exploration/` | `StochasticHypothesisGenerator`, `ExperimentScheduler`, `MultiWorldSimulator` | No |

**Invariants:**
- Cognitive Layer cannot execute MCP directly
- Exploration Layer cannot execute MCP directly
- Only Stable Core can invoke ActionExecutor
- All actions pass through validator before execution
- No learning happens in Stable Core — pure constraint enforcement
- Exploration outputs are "candidates" for Cognitive Layer, never applied directly

## Complete Module Map

Bold = primary files. All under `runtime_kernel/runtime/`.

| Module | Lines | Purpose |
|---|---|---|
| **engine.py** | 2706 | **Single RuntimeEngine class — the monolith orchestrating everything** |
| **session.py** | 577 | AgentSession — all agent state, cognitive models, causality |
| **prompt.py** | 788 | PromptBuilder — all LLM prompts (Chinese), cognitive model injection |
| **parser.py** | 618 | State parsing, force extraction, loop detection from LLM JSON output |
| **causal_physics.py** | 639 | Force integration + 3 constraint layers + World Anchor layer |
| **environment.py** | 760 | VirtualEnvironment — shared world (rooms/objects), multi-agent |
| **communication.py** | 630 | Inter-agent messaging: EventBus, Mailbox, CommunicationManager |
| causality.py | 465 | CausalityManager + CausalEntry — immutable causal chain |
| state.py | 316 | State dataclass — working memory with world model helpers |
| evidence.py | 367 | EvidenceManager — evidence pipeline (evidence → hypothesis → belief) |
| hypothesis.py | 454 | HypothesisManager — hypothesis lifecycle management |
| shared_memory.py | 392 | SharedKnowledge — cross-agent consensus knowledge base |
| memory_manager.py | 437 | MemoryManager — RAG for long-term memory (store / retrieve / context) |
| memory_storage.py | 177 | Abstract MemoryStorage + InMemoryMemoryStorage |
| memory.py | 91 | **DEAD CODE** — old WorkingMemory wrapper, not used |
| models.py | 156 | Enums (StateCause, SessionStatus, MessageType, MemoryRecordType) + thresholds |
| agent_events.py | 191 | AgentEventBus — structured event emit/subscribe, WebSocket stream |
| exceptions.py | 56 | Exception hierarchy (RuntimeError base) |
| experience.py | 117 | Experience dataclass + compute_identity_maturity |
| identity_manager.py | — | IdentityManager — identity anchor management |
| drive.py | — | DriveModel — curiosity/boredom/belonging drives + tick |
| goal_generator.py | — | GoalGenerator — thought pool candidates |
| evolution.py | 421 | EvolutionEngine — signals + RuntimeParameters |
| self_modification.py | — | SelfModificationManager — extract overrides from identity |
| heartbeat.py | — | HeartbeatManager — 30s daemon thread |
| runtime_statistics.py | 338 | AgentStats + RuntimeStatistics |
| persistence.py | 123 | Snapshot/restore JSON file persistence |
| scheduling.py | — | Scheduler — timed event scheduler |
| embedding.py | — | EmbeddingClient — embedding API calls |
| policy_engine.py | 321 | PolicyEngine + OutcomeEvaluator — causal policy evolution |

### Key Subsystems

#### 1. Engine (`engine.py` — 2706 lines, the monolith)

Single class `RuntimeEngine` that orchestrates everything. It is the ONLY public API. All agent logic lives here — prompt building, LLM calls, state parsing, force integration, identity reflection, memory storage, event emission, action execution, scientific cycles, evolution, heartbeats.

**The three phases of `step()`:**
1. **Autonomous step** (`_handle_autonomous_step`): No human input. Agent thinks, acts, updates world model. ~1200 lines.
2. **Human interrupt** (`_handle_human_interrupt`): Human message treated as **evidence** first, goes through evidence pipeline. ~300 lines.
3. **Continue with human answer** (`continue_with_human_answer`): Resumes after Human.ask pause, agent sees answer as Observation. ~200 lines.

#### 2. Evidence Pipeline (World Model v2 design)

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
- If force < 0.6 and no supporting evidence, belief stays unchanged
- Hypothesis needs ≥ 3 supporting evidence to be "belief-ready"
- If contradiction/support ratio > 0.5, belief update is blocked

#### 3. Action System (v6)

```
Planner (LLM) → action: {capability, operation, parameters}
    ↓
Core Validator (ActionValidator.validate_and_execute)
    ↓
ActionExecutor.execute(action)
    ↓
SearchAdapter (wraps MCP internally)  |  HumanAdapter (ask/tell)
    ↓
Observation(success, content, metadata, error)
    ↓
Planner continues reasoning (max 3 actions per step)
```

Every action returns an `Observation`. The planner never reads raw tool output.

**Action models** (`action/models.py`): three dataclasses — `Capability` (what the agent CAN do), `Action` (what it DECIDES to do), `Observation` (what HAPPENED). These are the ONLY types crossing the Agent↔Runtime boundary.

**Adding a new capability**: Write a `CapabilityAdapter` subclass (implement `execute()`, `list_operations()`, `get_capability_info()`), call `register_capability()` at module level, then `executor.register("Name", NameAdapter())` in `RuntimeEngine.__init__()`. Zero changes to Planner or core engine.

#### 4. Cognitive Architecture (v4) — `cognitive/`

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
| CausalGraph | `cognitive/causal_graph.py` | Directed causal relationships between concepts (evidence-driven) |
| ProbabilisticWM | `cognitive/probabilistic_wm.py` | Belief distributions (mean, uncertainty, samples) per concept |

**Prompt order (cognitive-first):**
`SelfModel → WorldModel → SocialModel → KnowledgeModel → WorkingMemory → Perception → TheoryOfMind → Environment → Drives → Memory → State → Capabilities`

All prompts are in Chinese, centralized in `prompt.py`. The prompt is a cognitive snapshot — it reflects the current state of the models, not a chat template.

There is an **empty** `cognition/` directory (no `__init__.py`, zero files). The active directory is `cognitive/`.

#### 5. Stable Core Layer — `core/`

Two files enforcing the 3-layer architecture:

- **`core/validator.py`** (`ActionValidator`): gates ALL action execution. Checks (1) capability exists, (2) operation supported, (3) safety rules pass, (4) parameters valid. Only after ALL checks pass does the action reach `ActionExecutor`. Has `validate_and_execute()` (for production) and `validate_only()` (for Cognitive Layer planning).

- **`core/safety.py`** (`SafetyRules`): immutable constraints — blacklisted operations (always rejected), rate limiting (30/min per session), max parameter size. No learning, no strategy, pure gatekeeping.

#### 6. Causal Physics + Constraint Layers — `causal_physics.py`

Core equation: `State(t+1) = Integrate(CausalForces, State(t))` where `ΔState = F_memory + F_identity + F_world + F_llm`.

**Three constraint layers** applied in order:
1. **Semantic Delay** — blocks self-interpretation (consciousness/awareness patterns) before round 10
2. **Semantic Escalation Barrier** — prevents abrupt environment→consciousness topic jumps
3. **Anti-Delusion Filter** — rejects un-caused self-narrative terms without causal origin in recent experiences

**World Anchor Dominance Layer** (`apply_world_anchor()`, ~160 lines): enforces that world-grounded topics (garden/room/plant) cannot be overridden by abstract discourse (consciousness/self-awareness). Human questions about abstract topics are auto-converted to world-relevant interpretations (e.g., "你有意识吗？" → "在当前环境中你感知到什么？").

#### 7. MCP Architecture — `mcp/`

```
ToolManager (agent-facing, unified)
    │
    ├── MCPRuntime 1 (Search MCP)
    │       └── MCPClient (JSON-RPC over stdio/SSE/HTTP)
    └── MCPRuntime 2 (future)
```

- `mcp/manager.py` (`ToolManager`): the ONLY class agents/engine interact with. Aggregates multiple MCPRuntimes. Tools auto-discovered via `tools/list`. Name collisions detected at init.
- `mcp/runtime.py` (`MCPRuntime`): lifecycle of one MCP server connection — connect, discover tools, execute, close.
- `mcp/client.py` (`MCPClient`, 571 lines): low-level JSON-RPC transport. Three modes auto-detected from config:

| Mode | Config | Use Case |
|---|---|---|
| `stdio` | `command` + `args` | Local subprocess (uvx mcp-server-tavily) |
| `sse` | `url` + SSE response | Standard MCP HTTP transport |
| `direct` | `url` + auto-fallback | Direct HTTP POST (Tavily MCP) |

Connection failures log but don't block remaining servers (graceful degradation). Adding a new MCP server = adding a `MCPConfig` entry. Zero code changes.

**Note**: `SearchAdapter` (`action/adapters/search_adapter.py`) is the ONLY consumer of MCP internally. The Planner never touches MCP.

#### 8. Communication — `communication.py` (630 lines)

Event-driven inter-agent messaging layer. Design: Agent A does NOT send text into Agent B's prompt directly — it acts, producing a World Event, which the Communication Layer routes as a causal event.

- **`Message`**: structured causal event (not just text) — `from_agent`, `to_agent`, `msg_type` (16 types: observation, question, hypothesis, plan, broadcast, etc.), `content`, `world_tick`, `causal_parent`
- **`Mailbox`**: per-agent inbox, max 20 messages
- **`EventBus`**: world events published here, agents subscribe
- **`CommunicationManager`**: routes messages, manages mailboxes and subscriptions

All agents share one `CommunicationManager` instance (wired in `routes.py`).

#### 9. Shared Memory — `shared_memory.py` (392 lines)

`SharedKnowledge` — cross-agent knowledge base with **consensus pipeline**:
```
Observation → Evidence → Candidate → Peer support (≥2 agents) → Public Knowledge
```

- No agent can directly write to shared knowledge
- `KnowledgeEntry` has `status`: `candidate` | `public` | `contested` | `discarded`
- `agent_support` / `agent_contradict` sets track which agents agree/disagree

#### 10. SDSA — Self-Driven Scientific Agent

`runtime_kernel/runtime/sdsa/` — background daemon loop every 60s:
```
Generate Research Goals → Enqueue Experiments → Execute through Core → Causal Update → World Model Update
```
Independent of user requests. Only starts when an `ActionExecutor` is configured. Modules: `daemon_loop.py`, `goal_generator.py`, `experiment_queue.py`, `models.py`.

#### 11. Scientific Loop — `scientific/`

Full scientific method cycle every 10 rounds (autonomous, not daemon):
```
Question Generator → Hypothesis Planner → Experiment → Observation → Causal Analysis → Theory Update
```
Modules: `question_generator.py`, `experiment_planner.py`, `hypothesis_layer.py`, `loop.py`, `causal_analyzer.py`, `theory_updater.py`, `models.py`.

#### 12. Exploration Layer — `exploration/`

High-stochasticity subsystems that cannot execute actions directly:
- `StochasticHypothesisGenerator` — random hypothesis proposals
- `ExperimentScheduler` — schedules experiments from hypotheses
- `MultiWorldSimulator` — runs "what-if" scenarios without real execution

All outputs are "candidates" for the Cognitive Layer.

#### 13. Memory System — `memory_manager.py` + `memory_storage.py`

- **`MemoryManager`**: RAG long-term memory — `store()` (state/interaction/reflection/introspection), `retrieve()` (semantic search via embedding), `build_context()` (format for prompt injection)
- **`MemoryStorage`** (abstract ABC) + **`InMemoryMemoryStorage`** (dev backend): designed to be replaced by MySQL/Postgres without changing engine code
- **`memory.py`**: OLD dead-code WorkingMemory wrapper (91 lines) — DO NOT USE. The active WorkingMemory is in `cognitive/working_memory.py`.

#### 14. State Model — `state.py`

`State` dataclass wrapping a dict with core keys (`topic`, `belief`, `goal`) plus World Model v2 fields (`world_model`, `hypotheses`, `evidence`, `open_questions`, `uncertainties`, `confidence`). Key design: `merge()` with `override=True` properly handles partial states — UNKNOWN filler values never replace real values.

#### 15. Experience & Identity — `experience.py`

`Experience` is the atomic unit of an agent's existence, distinct from a raw state transition:
- `round`, `session_id`, `perception`, `action`, `observation`, `meaning` (initially empty, populated by Reflection)
- `state_before`, `state_after` — full snapshots
- **An agent IS its sequence of Experiences, not its state.**

`compute_identity_maturity()` combines: rounds lived (25%), experiences (35%), reflections (40%). Each saturates independently. Range 0.0 (newborn) → 1.0 (stable personality).

### Agent Observability (Event Stream)

All steps emit structured events via `AgentEventBus` (`agent_events.py`). Event types and icons:

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
| `error` | ❌ | Error occurred |
| `system` | ⚙️ | System event (connection, etc.) |

WebSocket at `/ws/events/{session_id}` — sends stored events on connect, then streams new ones via async subscriber queues.

### Policy Engine (Causal Policy Evolution)

`policy_engine.py` — data-only policy learning (not self-modifying code):
- `preferred_capabilities`: capabilities with >60% success rate
- `avoided_operations`: operations that failed ≥2 times
- `consecutive_failures`: triggers "建议更换策略" warning
- `efficiency_score`: blended success rate + latency

`OutcomeEvaluator` is a pure-function evaluator (no LLM calls). Policy biases are structured data injected as `【策略偏好】` context in prompts.

## Session Data Model

`AgentSession` (in `session.py`, ~577 lines) holds everything:

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

## Session Status Lifecycle

```
INITIALIZING → RUNNING ↔ INTERRUPTED (human chat)
              RUNNING → WAITING_HUMAN (agent asked question)
              WAITING_HUMAN → RUNNING (human answered)
              RUNNING → REFLECTING → RUNNING (every 5 rounds)
              RUNNING → IDLE → RUNNING
              * → ERROR → RUNNING
              * → TERMINATED
```

## LLM Output Format

Parsed by `parser.py` (~618 lines) — stateless functions: `extract_state()`, `extract_causal_vector()`, `extract_world_model_updates()`, `repair_state()`, `detect_loop()`.

```json
{
  "delta_topic": "新主题",
  "delta_belief": "新信念",
  "delta_goal": "新目标",
  "force": 0.65,
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

Loop detection (`detect_loop()`): compares topic/belief fingerprints over a window of 8 rounds. State repair (`repair_state()`): ensures required keys exist, extracts fields from action if needed.

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

## Web Layer

`main.py` — FastAPI entry point with lifespan hooks: `load_all_sessions()` on startup, `save_all_sessions()` on shutdown.

`web/routes.py` (1236 lines) — APIRouter with shared singletons:
- One `VirtualEnvironment` (all agents coexist)
- One `CommunicationManager` (all agents share)
- One `SharedKnowledge` (cross-agent consensus)
- WebSocket endpoint at `/ws/events/{session_id}`
- All endpoints delegate to RuntimeEngine — no agent logic in routes

`web/schemas.py` (194 lines) — Pydantic models for all request/response validation.

`web/templates/index.html` — Single-page frontend (HTML + JS, no framework).

## Persistence & Sessions Directory

- `autosave.json`: auto-saved every 10 rounds in working directory (single JSON file containing all sessions)
- `sessions/`: snapshot files (`session_{id}.json`) created by `/api/snapshot`
- On server startup: `load_all_sessions()` reads `autosave.json`
- On server shutdown: `save_all_sessions()` writes `autosave.json`
- `Persistence` class (`persistence.py`) handles JSON read/write with auto-directory creation

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
    "mcp_tools": [{"command": "uvx", "args": ["mcp-server-tavily"]}]
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

## Common Development Tasks

```bash
# Run all inline tests (from root)
uv run python -c "from runtime_kernel import RuntimeEngine; RuntimeEngine(demo=True).step(RuntimeEngine(demo=True).create_session(demo=True).id)"

# Test a specific module
uv run python -c "from runtime_kernel.runtime.causal_physics import *; print('imports OK')"

# Check session state file
cat sessions/session_*.json | python3 -m json.tool 2>/dev/null || echo "No snapshots"

# Debug LLM output parsing
echo '{"delta_topic": "test", "force": 0.5}' | python3 -c "
import sys, json; d=json.load(sys.stdin)
from runtime_kernel.runtime.parser import extract_state; print(extract_state(d))
"

# Add a new MCP server (in routes.py, add to MCPToolConfig list in /api/connect body)
# Add a new capability adapter:
#   1. Create adapter in action/adapters/ (subclass CapabilityAdapter)
#   2. Export from action/adapters/__init__.py
#   3. Import and register in engine.py __init__
#   4. Add endpoint if needed in routes.py

# Add a new cognitive model:
#   1. Create file in cognitive/
#   2. Add to cognitive/__init__.py
#   3. Add to AgentSession (session.py)
#   4. Add format_for_prompt() and inject in prompt.py
#   5. Wire up in engine.py step() method

# Restore from a specific snapshot
curl -X POST http://localhost:8000/api/restore \
  -H 'Content-Type: application/json' \
  -d '{"filepath": "sessions/session_YOUR_ID.json"}'
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
- **engine.py is 2700+ lines** — single file, all orchestration logic. Refactor carefully.
- **All prompts in Chinese** — `PromptBuilder` in `prompt.py`. Don't scatter prompt strings.
- **Pydantic is only in `web/schemas.py`** — runtime_kernel uses plain dataclasses

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
- **Communication is causal, not chat**: Messages are causal events with `causal_parent` references.
- **Shared knowledge requires consensus**: No agent writes directly to `SharedKnowledge` — ≥2 agents must converge.
- **Scientific method loop**: Every 10 rounds: questions → hypotheses → experiments → theory updates.
- **Self-modification through identity**: `SelfModificationManager` extracts structured overrides from the identity anchor. The LLM generates new identity deltas; the engine extracts actionable modifications.
- **Loop detection**: `detect_loop()` in parser.py compares topic/belief fingerprints over a window of 8 rounds.
- **3-layer separation**: Core executes (no learning), Cognitive reasons (no execution), Exploration proposes (no execution). Each layer's invariant is enforced by code structure.
- **World Anchor Dominance**: World-grounded topics cannot be overridden by abstract discourse. Human abstract questions are auto-converted.
- **Constraint layers**: Three causal-physics constraints prevent self-narrative without causal basis (Semantic Delay, Semantic Escalation Barrier, Anti-Delusion Filter).
- **In-memory by default**: No database. All state in Python dicts. Persistence via JSON file snapshots.
- **Dependency injection**: All modules injected via constructor params.
- **`from __future__ import annotations`** throughout for deferred evaluation (Python 3.11+ style).

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

### Memory Store
```python
# Check stored memories for a session
ms = engine._memory_manager.storage
records = ms.get_session_records(session_id)
print(f"Memory records: {len(records)}")
for r in records[-3:]:
    print(f"  [{r['type']}] R{r['round']}: {r['summary'][:60]}")
```

### Shared Knowledge
```python
sk = engine._shared_knowledge
print(f"Public entries: {len(sk.get_public_entries())}")
print(f"Candidates: {len(sk.get_candidates())}")
for e in sk.get_public_entries()[:3]:
    print(f"  {e.statement[:60]} (confidence: {e.confidence})")
```

### Causality Chain
```python
entries = session._causality.get_chain()
for e in entries[-3:]:
    print(f"  R{e.round} {e.cause_type}: {e.state.get('topic', '?')[:30]}")
```

## Public API Surface

The `runtime_kernel/__init__.py` exports all public types. Import from `runtime_kernel` directly:

```python
from runtime_kernel import (
    RuntimeEngine,          # The ONLY entry point
    AgentSession,           # Session data container
    State,                  # Working memory dataclass
    Action, Observation, Capability,  # Action system types
    ActionExecutor,         # Action routing
    HumanAdapter, SearchAdapter,       # Built-in adapters
    CapabilityAdapter,      # Base class for new adapters
    MCPConfig,              # MCP server config
    AgentEvent, AgentEventBus,  # Event system
    CommunicationManager, Mailbox, Message,  # Inter-agent comms
    SharedKnowledge, KnowledgeEntry,    # Shared knowledge
    Evidence, EvidenceManager,  # Evidence pipeline
    Hypothesis, HypothesisManager,      # Hypothesis lifecycle
    CausalEntry, CausalityManager,      # Causal chain
    Experience, compute_identity_maturity,  # Identity model
    VirtualEnvironment,     # Shared world
    PolicyEngine, OutcomeEvaluator,     # Policy biases
    ScientificLoop, ScientificHypothesis,  # Scientific cycle
    CausalGraph, ProbabilisticWorldModel,  # Cognitive models
    SelfModel, WorldModel, SocialModel, KnowledgeModel,
    TheoryOfMind, WorkingMemory,
    AutonomousDaemonLoop,   # SDSA daemon
    MemoryManager, MemoryStorage, InMemoryMemoryStorage,
    EmbeddingClient,        # Embedding API
    IdentityManager,        # Identity management
    GoalGenerator,          # Goal generation
    DriveModel,             # Drive system
    EvolutionEngine,        # Evolution signals
    ActionValidator, SafetyRules,  # Core layer
    SessionNotFoundError, LLMError, StateValidationError,
    SessionStatus, StateCause,  # Enums
)
```
