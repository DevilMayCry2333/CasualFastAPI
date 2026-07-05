# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Agent Runtime Kernel** — a pure Python library for managing AI agent lifecycle with autonomous persistence and recursive self-improvement.

The system's core goal is **building, maintaining, and correcting a World Model**, not generating conversation or maintaining a persona. Each agent continuously evolves a JSON state object through LLM self-dialogue, driven by evidence accumulation and hypothesis testing. Agents are situated in a shared virtual world ("微光之境").

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
    session.py                 # AgentSession — pure data container, NO LLM logic
    state.py                   # State — JSON wrapper with world model fields
    llm.py                     # LLMClient — OpenAI-compatible API (sync + streaming)
    prompt.py                  # PromptBuilder — ALL prompts centralized (Chinese), world-first order
    parser.py                  # Stateless functions: extract → parse → repair → State
    models.py                  # Enums (StateCause, SessionStatus), constants
    exceptions.py              # Exception hierarchy (all inherit RuntimeError)

    # World Model (v2)
    hypothesis.py              # HypothesisManager — hypothesis lifecycle (proposed→testing→supported→contradicted→discarded)
    evidence.py                # EvidenceManager — observations become Evidence before influencing Belief

    # Multi-Agent Communication (v3)
    communication.py           # CommunicationManager, Message, Mailbox, EventBus — event-driven inter-agent layer
    shared_memory.py           # SharedKnowledge, KnowledgeEntry — consensus-based public knowledge

    # Causal State Engine
    causality.py               # CausalityManager + CausalEntry — causal chain recording/retrieval
    causal_physics.py          # CausalVector, force computation, state integration
    identity_manager.py        # IdentityManager — self-reflection, identity anchor update
    introspection.py           # Introspector — world model evolution analysis (every 20 rounds)
    self_modification.py       # SelfModificationManager — validate recursive self-improvements

    # Drive System
    drive.py                   # DriveModel — curiosity/boredom/belonging (pure functions)
    goal_generator.py          # GoalGenerator — thought pool from drives + identity gaps
    heartbeat.py               # HeartbeatManager — background aliveness (30s daemon thread)

    # Memory
    embedding.py               # EmbeddingClient — separate embedding API (not LLM)
    memory_manager.py          # MemoryManager — RAG store/retrieve, hypothesis-relevant retrieval
    memory_storage.py          # MemoryStorage ABC + InMemoryMemoryStorage

    # Cognitive Architecture (v4)
    cognitive/                 # Cognitive model package
      __init__.py              # Export all cognitive models
      self_model.py            # SelfModel — identity, beliefs, goals, drives
      world_model.py           # WorldModel — places, objects, events, environment rules
      social_model.py          # SocialModel — trust, cooperation, interaction history
      knowledge_model.py       # KnowledgeModel — facts, hypotheses, evidence, contradictions
      theory_of_mind.py        # TheoryOfMind — beliefs about other agents' beliefs
      working_memory.py        # WorkingMemory — current focus, active question
      perception.py            # Perception — event-to-cognition pipeline
      attention.py             # Attention — salience-based event filtering

    # Evolution Engine (v5)
    runtime_statistics.py      # RuntimeStatistics, AgentStats — system self-observation metrics
    evolution.py               # EvolutionEngine, RuntimeParameters, EvolutionSignals — ecosystem manager

    # World
    environment.py             # VirtualEnvironment — shared world, 5 rooms, action engine
    persistence.py             # Persistence — snapshot/restore session JSON files
    scheduler.py               # Scheduler — wall-clock-timed step loop

web/
    __init__.py
    routes.py                  # FastAPI router + shared _world + _engines dict
    schemas.py                 # Pydantic models
    templates/index.html       # Single-page frontend (~790 lines, all JS inline)

main.py                        # uvicorn entry: `app = FastAPI()` + CORS + router
```

## World Model Architecture (v2)

The system no longer centers on self-model / identity. The core design philosophy:

> **Self is a part of the World Model, not the center of it.**

The LLM's role is to maintain a World Model — reduce uncertainty about the environment — by:
1. Collecting evidence (observations, human statements, deductions)
2. Proposing and testing hypotheses
3. Detecting contradictions
4. Updating belief only when evidence accumulates

### State Fields (beyond topic/belief/goal)

| Field | Type | Purpose |
|---|---|---|
| `world_model` | dict | Structured understanding of the world (rooms, objects, causal relationships) |
| `hypotheses` | list[dict] | Active hypotheses with confidence, support/contradiction counts |
| `evidence` | list[dict] | Observations with source (observation/human/deduction/world_event) and confidence |
| `open_questions` | list[str] | Questions the agent has not yet answered |
| `uncertainties` | list[dict\|str] | Areas with low confidence |
| `confidence` | float | Overall world model confidence (0.0-1.0) |

### Evidence → Hypothesis → Belief Pipeline

```
Observation → Evidence (with source + confidence)
    ↓
Evidence linked to Hypothesis (supports or contradicts)
    ↓
Hypothesis accumulates support_count / contradiction_count
    ↓
Hypothesis reaches confidence > 0.6, support ≥ 3 → "belief-ready"
    ↓
Belief update (LLM cannot directly write belief without evidence)
```

**Key constraint**: The LLM CANNOT directly set `belief`. All belief changes must pass through the evidence→hypothesis pipeline. If the LLM proposes a `delta_belief` without evidence support, the engine rejects it unless `force > 0.6`.

### Hypothesis Lifecycle

```
proposed (conf=0.1, no evidence)
    ↓
testing (1+ supporting evidence)
    ↓
supported (3+ supporting evidence, conf>0.6)  → can influence belief
    ↓  OR
contradicted (30%+ contradictory evidence)  → triggers revision
    ↓
revised (reset counters, new statement)
    ↓
discarded (dead end, conf=0)
```

The `HypothesisManager` tracks all hypotheses, detects contradictions, and identifies belief-ready candidates. Each hypothesis has `id`, `statement`, `domain`, `source`, `confidence`, `support_count`, `contradiction_count`, and `evidence_ids`.

## Multi-Agent Communication (v3)

Event-driven inter-agent communication. Agents do NOT send text into each other's prompts. Instead:

```
Agent A acts → World Event → Communication Layer
    → Agent B perceives (via Mailbox) → interprets → decides
```

Communication IS a causal event. Every Message is recorded in the causal chain.

### Architecture

| Component | Purpose |
|---|---|
| `Message` | Causal event between agents. Fields: id, from_agent, to_agent, type, content, timestamp, world_tick, causal_parent, world_room |
| `Mailbox` | Per-agent inbox. Messages land here; agent pops them during step(). NOT a prompt — it's Runtime input. |
| `EventBus` | World event publish/subscribe. World changes (plant grew, soil dried, item crafted) are auto-published. |
| `CommunicationManager` | Routes messages, manages mailboxes and subscriptions. Engine does NOT manage communication directly. |

### Message Types (NOT chat)

- `OBSERVATION` — "the plant grew today"
- `QUESTION` — "why did the plant stop growing?"
- `ANSWER` — "the soil is dry"
- `HYPOTHESIS` — "i think the room has a hidden door"
- `PLAN` — "i will investigate the garden"
- `REQUEST` — "please help move the tools"
- `WARNING` — "the garden needs water"
- `EVENT` — "someone entered the lab"
- `REPORT` — "survey of the east wing complete"
- `SHARE_MEMORY` — sharing evidence/hypothesis
- `BROADCAST` — world event notification

### Two Ways Agents Send Messages

**1. JSON field (recommended):** LLM outputs `send_message` as a top-level JSON key:
```json
{
  "action": "move north",
  "send_message": {"to_agent": "...", "type": "observation", "content": {"text": "..."}}
}
```

**2. World action (fallback — LLMs naturally use this):** LLM writes `send_message` in the `action` field:
```
send_message to 7b43d59c 你好，我是探索者
```
The environment's `_handle_send_message` parses this, extracts `to_agent` with prefix matching (`7b43d59c` matches `7b43d59cf6ee`), and routes it through the CommunicationManager. Also handles bare `send_message` (broadcasts to all in same room).

**3. Function-call format (additional fallback):** Some LLMs use `send_message(to_agent='...', type='...', content='...')` inside the action string. Handled by `_parse_action_messages()` in engine.py.

### Key Rules

- **No forced responses**: CommunicationManager only delivers messages. Agents decide whether to respond.
- **No auto-reply**: Runtime never forces an agent to answer.
- **Mailbox is FIFO**: Agent pops all messages at start of step, decides what to do.
- **Messages are causal**: Every sent message is a causal event. Stored in long-term memory.
- **Session ID prefix matching**: LLMs often truncate agent IDs to 8 chars (e.g. `7b43d59c` instead of `7b43d59cf6ee`). Both the environment handler and `get_sent_messages()` use prefix matching.

### Shared CommunicationManager

All agents across all `/api/connect` calls share ONE `CommunicationManager` and ONE `SharedKnowledge` instance. In `routes.py`:

```python
_communication = CommunicationManager()  # global, shared across all engines
_shared_knowledge = SharedKnowledge()    # global, shared across all engines
_world.set_message_callback(lambda from_agent, to_agent, text, room, tick:
    _communication.send(from_agent, to_agent, "observation", {"text": text}, tick, "", room))
```

This is critical: without shared instances, agents created by different engines can't communicate.

### Shared Knowledge (Consensus)

`SharedKnowledge` is a cross-agent knowledge base with a consensus mechanism:

```
Observation → Evidence → SharedKnowledge Candidate → Peer support → Public Knowledge
```

- Agents propose observations as shared knowledge candidates
- Knowledge only becomes "public" after N distinct agents support it (`SHARED_KNOWLEDGE_CONSENSUS_MIN = 2`)
- No agent can directly write to public knowledge
- Shared knowledge supports RAG retrieval for all agents

### World Events via EventBus

World changes automatically publish events:
- `plant_watered` — garden was watered
- `seeds_planted` — seeds planted, sprouts appear
- `soil_dried` — garden soil dried out
- `plants_grew` — plants matured
- `note_written` — note left in room
- `telescope_focused` — telescope used
- `item_crafted` — object built in workshop

Events flow: `VirtualEnvironment.tick()` → callback → `CommunicationManager.publish_world_event()` → subscribed agents' mailboxes.

### Prompt Sections

```
【收到的消息】       ← Recent mailbox contents (labels + from + summary)
【公共知识】         ← Consensus-approved knowledge from SharedKnowledge
【最近世界事件】     ← Recent EventBus events (tick, room, event text)
```

These sections appear right after Open Questions in the world-first prompt order.

### Agent Autonomy

The agent can output these fields in its LLM response:
- `send_message` (JSON): `{to_agent, type, content}` — sends a direct message
- `send_message to agent_id text` (action): written in the `action` field as a world action
- `share_knowledge`: `{statement, domain}` — proposes to SharedKnowledge
- `support_knowledge`: `entry_id` — supports existing shared knowledge
- `contradict_knowledge`: `entry_id` — contradicts existing shared knowledge

Extended `MessageType` values available: `observation`, `question`, `answer`, `hypothesis`, `plan`, `request`, `warning`, `event`, `report`, `share_memory`, `broadcast`, `inquiry`, `greeting`, `suggestion`, `response`, `information`.

## Cognitive Architecture (v4)

The system now maintains **multiple specialized cognitive models** instead of flat state. Each model has a clear boundary:

| Model | Content | NOT in this model |
|---|---|---|
| **SelfModel** | identity, beliefs, goals, drives, personality | World info, other agents |
| **WorldModel** | places, objects, agents_seen, events, rules | Self-beliefs, others' intent |
| **SocialModel** | trust, cooperation, reliability per agent | Others' knowledge state |
| **KnowledgeModel** | facts, hypotheses, evidence, contradictions | Self-beliefs, world structure |
| **TheoryOfMind** | perceived beliefs/goals of other agents | Self-beliefs, trust levels |
| **WorkingMemory** | current focus, active question, unresolved | Long-term memory, history |

### Perception Pipeline

```
World Events + Mailbox Messages
    ↓
Perception.perceive()        ← merge event sources
    ↓
Attention.filter_events()    ← filter by salience (curiosity, novelty, importance, uncertainty)
    ↓
WorkingMemory                ← set cognitive focus for this step
    ↓
PromptBuilder                ← organize as cognitive snapshot
```

Attention is bounded — the agent cannot attend to all events. Only the most salient (max 3 per step) enter working memory. This gives the agent **limited attention**, a core cognitive constraint.

### LLM Output Extensions for Cognitive Models

```json
{
  "delta_topic": "...",
  "delta_belief": "...",
  ...
  "self_update": {"belief": "updated belief", "confidence": 0.6},
  "social_update": {"agent_id": "...", "cooperative": true},
  "theory_update": {"agent_id": "...", "belief": "what they believe", "confidence": 0.5}
}
```

- `self_update` — updates SelfModel beliefs and tracks recent changes
- `social_update` — adjusts trust/cooperation for a known agent
- `theory_update` — updates TheoryOfMind about another agent's epistemic state

### Runtime vs LLM Separation

**Runtime** maintains: world, events, memory, causality, timing, communication, attention, cognitive models  
**LLM** handles: interpretation, reasoning, hypothesis generation, belief revision, decision making

The system runtime does NOT produce intelligence — it maintains the cognitive infrastructure. Intelligence emerges from the LLM operating within this structured environment.

## Evolution Engine (v5)

The Evolution Engine is not a cognitive model. It does not modify agent beliefs, goals, or reasoning. It only adjusts **Runtime Parameters** — the environmental context in which agents operate.

### Three-Layer Separation

```
Runtime (ecosystem)       ← maintains world, events, memory, causality
    ↓
LLM (lifeform)            ← interprets, reasons, decides, acts
    ↓
Evolution Engine (climate) ← observes trends, adjusts parameters
```

### RuntimeStatistics

Collects per-agent metrics each step over a rolling 100-round window:

| Metric | Description |
|---|---|
| `exploration_ratio` | Proportion of explore actions |
| `social_ratio` | Proportion of message/social actions |
| `observation_ratio` | Proportion of look/examine actions |
| `entropy` | Shannon entropy of action distribution (normalized 0-1) |
| `hypothesis_success_rate` | Supported / (supported + contradicted) |
| `evidence_efficiency` | Evidence collected / total rounds |
| `communication_density` | Messages sent+received / total rounds |
| `belief_revision_rate` | Belief changes / total rounds |
| `action_diversity` | Unique action types / total actions |
| `world_growth` | Distinct places + objects discovered |

### EvolutionSignals

Derived trends from statistics, computed every cycle:

- **Novelty** — Are agents finding new things?
- **Entropy** — Is behavior diverse?
- **Stability** — Is the system stable?
- **Exploration** — Are agents exploring?
- **Cooperation** — Are agents cooperating?
- **HypothesisCycling** — Are hypotheses repeating without converging?

### RuntimeParameters

All evolvable parameters with bounds:

```
attention_curiosity_weight   (0.05-0.5)
attention_novelty_weight     (0.05-0.5)
curiosity_decay              (0.9-0.999)
boredom_increment            (0.01-0.15)
hypothesis_contradiction_threshold (0.1-0.5)
belief_force_threshold       (0.3-0.9)
message_probability          (0.05-0.8)
```

The Evolution Engine makes small, gradual adjustments — never sudden jumps. Protected parameters (identity_reflection_interval, etc.) cannot be evolved.

### Evolution Report (every 50 rounds)

```text
【演化报告】
  新颖性: 0.20 — 趋于重复
  熵: 0.30 — 行为收敛
  探索: 0.10 — 探索不足
  知识增长: 0.30 — 知识停滞
  假设循环: 0.60 — 假设在空转
```

Based on these signals, the Engine adjusts parameters. Example: if exploration is low, it increases `attention_novelty_weight` and `curiosity_baseline`. If hypotheses are cycling, it raises `hypothesis_contradiction_threshold`.

## Key Architecture Decisions

- **Single entry point**: `RuntimeEngine` is the ONLY public API. No external code modifies sessions.
- **No LLM in Session**: `AgentSession` is a pure data container. LLM calls only in `RuntimeEngine.step()`.
- **Encapsulated prompts**: All prompt strings in `PromptBuilder` (Chinese). No scattered prompt strings.
- **Stateless parser**: `parser.py` has no state, just pure functions.
- **Pluggable storage**: `MemoryStorage` ABC — swap `InMemoryMemoryStorage` for MySQL without changing Engine.
- **Dependency injection**: All modules injected via constructor params.
- **World as shared state**: Multiple engine sessions share one `VirtualEnvironment`. Actions mutate the world.
- **In-memory by default**: No database. All state in Python dicts. Persistence via JSON file snapshots.
- **Causal chain as primary history**: Every transition recorded as `CausalEntry` with full before/after context.
- **Belief via evidence pipeline**: LLM proposes, EvidenceManager collects, HypothesisManager validates, belief only updates when evidence is sufficient.
- **Runtime vs LLM separation**: Runtime maintains cognitive infrastructure (world, events, memory, causality, attention, models). LLM handles interpretation, reasoning, hypothesis generation, and decision making. Runtime does not produce intelligence — it maintains the environment for cognition.

## Prompt Order (Cognitive Architecture, injected every step)

```
【自我模型】         ← SelfModel (identity, beliefs, goals)
【世界模型】         ← CognitiveWorldModel (places, objects, events)
【社会模型】         ← SocialModel (trust, cooperation per agent)
【知识模型】         ← KnowledgeModel (facts, hypotheses, evidence, contradictions)
【工作记忆】         ← WorkingMemory (current focus, active question, unresolved)
【感知输入】         ← Perception (attended events, environment summary)
【心智理论】         ← TheoryOfMind (perceived beliefs of other agents)
【内驱力状态】       ← Drive states + thought pool
【念头池】           ← Candidate thoughts/goals
【环境感知】         ← World context (room, objects, agents, inventory)
【因果链】           ← Causal Chain
【检索记忆】         ← RAG top-K (hypothesis-relevant)
【收到消息】         ← Mailbox messages (if attention selected them)
【公共知识】         ← SharedKnowledge
【最近世界事件】     ← EventBus events (if attention selected them)
【身份锚点】         ← Identity Anchor (last)
当前状态：           ← Current state JSON
```

Prompt is a **cognitive snapshot**, not a chat template. The runtime maintains the actual models; the prompt only reflects their current state. `identity_maturity` only controls whether Identity section is shown.

## Three Constraint Layers (World Model version)

The old anti-delusion / anti-self-narrative layers are retained but less central:

1. **Semantic Delay** — blocks self-interpretation before round 10 (unchanged)
2. **Semantic Escalation Barrier** — prevents env→consciousness jumps (unchanged)
3. **Anti-Delusion Filter** — blocks un-caused self-narrative (unchanged)

**New: Belief Guard** — LLM cannot directly write `belief` without evidence pipeline support. See `_check_belief_update()` in engine.py.

## Data Flow (One Step, Cognitive Architecture v4)

```
① Capture BEFORE snapshot
② Build cognitive model contexts (SelfModel, WorldModel, SocialModel, KnowledgeModel, WorkingMemory, TheoryOfMind)
③ Perception pipeline:
   - Collect mailbox messages + world events + env context
   - perceive() → Attention.filter_events() → attended events
   - Update WorkingMemory with current focus
④ Build prompt: cognitive models → environment → drives → memory → state
⑤ LLMClient.complete() — synchronous POST
⑥ extract_causal_vector() + extract_world_model_updates()
   — delta, force, new_hypotheses, new_evidence, hypothesis_updates,
     world_model_update, open_questions, uncertainties, confidence,
     self_update, social_update, theory_update
⑦ _process_world_model_updates()
⑧ _process_communication() — messages from JSON or action field
⑨ _update_cognitive_models() — apply self_update, social_update, theory_update
⑩ _check_belief_update() — evidence-backed belief only
⑪ Force integration → new state
⑫ VirtualEnvironment.act() — action execution + world events
⑬ CausalityManager.create_entry()
⑭ Experience + MemoryManager.store_state/evidence/hypothesis/message
⑮ DriveModel.after_step() → GoalGenerator.generate()
⑯ IdentityManager.reflect() — every 5 rounds
⑰ Introspector.introspect() — every 20 rounds
⑱ Persistence.snapshot() — every 10 rounds
```

## LLM Output Format (World Model Builder)

The LLM outputs a **causal vector** with state delta, world model updates, and cognitive model updates:

```json
{
  "delta_topic": "garden_sprouts",
  "delta_belief": "seeds germinate within 3 days in moist soil",
  "delta_goal": "monitor_sprout_growth",
  "force": 0.65,
  "action": "look garden",
  "source": "curiosity",
  "new_hypotheses": [{"statement": "...", "domain": "garden"}],
  "new_evidence": [{"statement": "i see sprouts", "source": "observation", "domain": "garden"}],
  "hypothesis_updates": [{"id": "...", "supports": true}],
  "world_model_update": {"sprouts": "visible"},
  "open_questions": ["how fast do they grow?"],
  "uncertainties": [{"domain": "growth_rate", "avg_confidence": 0.3}],
  "confidence": 0.6,
  "self_update": {"belief": "I am an observer in a strange world", "confidence": 0.7},
  "social_update": {"agent_id": "abc123", "cooperative": true},
  "theory_update": {"agent_id": "abc123", "belief": "they think the garden needs water", "confidence": 0.5}
}
```

Cognitive model fields:
- `self_update` — updates SelfModel.beliefs with confidence; appended to recent_changes
- `social_update` — adjusts TrustModel trust/reliability for a peer agent
- `theory_update` — updates TheoryOfMind perceived_beliefs for another agent

## Web API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Serve frontend HTML |
| POST | `/api/connect` | Create engine + session (returns session_id) |
| POST | `/api/step` | Autonomous thinking step |
| POST | `/api/chat` | Human interruption |
| GET | `/api/state/{id}` | Single session state summary |
| POST | `/api/fold` | Identity reflection |
| POST | `/api/translate` | State → natural language |
| POST | `/api/snapshot` | Save session to JSON |
| POST | `/api/restore` | Restore from JSON |
| DELETE | `/api/session/{id}` | Delete session |
| GET | `/api/world` | World summary |
| GET | `/api/world/context/{id}` | Agent's environment context |
| GET | `/api/sessions` | **All sessions** + world model data (hypothesis_count, evidence_count, confidence, world_model) |
| GET | `/api/causal-chain/{id}` | Full causal chain for a session |
| GET | `/api/causal-chain/{id}/trace?from=N&to=M` | Trace causal path between rounds |
| GET | `/api/hypotheses/{id}` | Active hypotheses, contradictions, belief-ready status |
| GET | `/api/evidence/{id}` | Evidence list, domain confidence |
| POST | `/api/send-message` | Direct message between agents (lands in mailbox) |
| POST | `/api/broadcast` | Broadcast message to all agents |
| GET | `/api/mailbox/{id}` | Agent's mailbox (received messages) |
| GET | `/api/sent-messages/{id}` | Messages sent BY the agent (visible in sender's communication tab) |
| GET | `/api/shared-knowledge` | Public shared knowledge (optional `?domain=`) |
| GET | `/api/event-bus/{id}` | Recent world events from EventBus |
| GET | `/api/health` | Server health |

## Causal Chain Architecture

Every state transition is a `CausalEntry` dataclass with parent links for traceability. The causal chain replaces flat working memory and serves as the agent's primary historical record.

Key `CausalityManager` methods:
- `create_entry()` — record + auto-compute parent_rounds
- `build_context()` — format as 【因果链】block for prompt injection
- `build_causal_context_for_query()` — causally relevant retrieval (5 dimensions)
- `trace_path()` — follow parent_rounds links between two rounds
- `get_recent()` — last N entries

## Frontend Patterns (index.html ~790 lines, all JS inline)

- **Two-panel layout**: Left (340px, scrollable) = config + agent list + world info. Right = tabs + output + chat.
- **Agent list cards**: drives bars, goal/belief/topic, thought pool, causal chain count, expandable state.
- **Tabbed panel**: "📊 State" (JSON), "🔗 因果链" (causal entries), and "💬 通信" (inter-agent messages + world events + shared knowledge).
- **Auto-think loop**: cycles ALL agents via `/api/step` with 300ms inter-agent pause, 30s between rounds.
- **State refresh**: GET `/api/sessions` every 3s (all data from single response).
- Key JS variables: `sessions[]`, `selectedSessionId`, `busy`, `autoThinking`, `autoCount`.

## Default Config

| Setting | Default |
|---|---|
| LLM API | `http://localhost:28000/v1/chat/completions` |
| Model | `deepseek-v4-flash` |
| Temperature | 0.85 (chat), 0.7 (auto step) |
| Embedding API | `http://0.0.0.0:28001/v1/embeddings` |
| Identity reflection | Every 5 rounds |
| Heartbeat | 30s daemon thread |
| Auto-save | Every 10 rounds |
| Session serialization | version 5 (adds hypotheses + evidence managers + communication refs) |
| Max active hypotheses | 5 |

## Core Concepts

- **World Model**: Structured understanding of the environment, stored in `state.world_model`. The agent's primary task is to maintain and improve this model.
- **Evidence**: Atomic observation unit. Every observation (including human statements) becomes Evidence first. Has source (observation/human/deduction/world_event), confidence, domain, and links to hypotheses.
- **Hypothesis**: A testable proposition about the world. Goes through lifecycle: proposed → testing → supported/contradicted → revised → discarded. Only hypotheses with sufficient support can influence belief.
- **Belief**: The agent's current best understanding. Cannot be directly set by LLM — must be supported by evidence→hypothesis pipeline.
- **State**: JSON with core keys `topic`, `belief`, `goal` + extensions. v2 adds `world_model`, `hypotheses`, `evidence`, `open_questions`, `uncertainties`, `confidence`.
- **Causal Chain**: `CausalEntry` per transition with parent links. Primary historical record.
- **Identity Anchor**: `{core_goal, identity, worldview, stable_values, recent_reflection, confidence}` — identity is a byproduct of world model evolution, not the system's starting point.
- **Drives**: `curiosity` (rises with prediction error), `boredom` (rises with monotony), `belonging` (decays without human). Values 0-1. DriveModel = pure functions.
- **Thought Pool**: 1-3 candidate goals from drives + identity gaps.
- **Cognitive Models**: SelfModel (identity/beliefs), WorldModel (places/objects/rules), SocialModel (trust/cooperation per agent), KnowledgeModel (facts/hypotheses/evidence), TheoryOfMind (beliefs about others' beliefs), WorkingMemory (current focus/question).
- **Perception Pipeline**: World events + mailbox messages → Attention filter (curiosity/novelty/importance) → Working Memory → cognitive snapshot in prompt.
- **Virtual World**: 5 rooms with objects/exits/mutable state. Actions: look, move, take, use, write, inventory, drop, read, exits.
- **Heartbeat**: Background daemon (30s) — drive drift, world tick, thought pool refresh, auto-save.
- **Loop detection**: Autocorrelation on `(topic, belief)` across 8-round window.

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
print(f'Hypotheses: {s.hypothesis_manager.active_hypotheses}')
print(f'Evidence: {len(s.evidence_manager.all_evidence)}')
print(f'Causal entries: {len(s.causality.get_chain(s.id))}')
"

# Test hypothesis lifecycle
uv run python -c "
from runtime_kernel import HypothesisManager, HypothesisStatus
hm = HypothesisManager()
h = hm.propose('seeds sprout in 3 days', 'observation', round_num=1)
hm.add_evidence(h.id, 'ev1', supports=True, round_num=2)
hm.add_evidence(h.id, 'ev2', supports=True, round_num=3)
hm.add_evidence(h.id, 'ev3', supports=True, round_num=4)
ready = hm.get_ready_for_belief()
print(f'Hypothesis: {h.status} conf={h.confidence:.2f}')
print(f'Belief-ready: {len(ready)}')
"

# Verify causal chain on running server
curl -s http://localhost:8000/api/sessions | python3 -c "
import sys, json
d = json.load(sys.stdin)
for s in d['sessions']:
    print(f'{s[\"id\"][:8]} R{s[\"round\"]} hyp:{s.get(\"hypothesis_count\",0)} ev:{s.get(\"evidence_count\",0)} causal:{s[\"causal_chain_count\"]}')
"

# Get hypotheses for a session
curl -s http://localhost:8000/api/hypotheses/SESSION_ID | python3 -m json.tool

# Get evidence for a session
curl -s http://localhost:8000/api/evidence/SESSION_ID | python3 -m json.tool

# Send a message between agents
curl -X POST http://localhost:8000/api/send-message \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"FROM_SESSION","to_agent":"TO_SESSION","msg_type":"observation","content":{"text":"the plant grew"}}'

# Check an agent's mailbox
curl -s http://localhost:8000/api/mailbox/SESSION_ID | python3 -m json.tool

# Get shared knowledge
curl -s http://localhost:8000/api/shared-knowledge | python3 -m json.tool

# Get recent world events
curl -s http://localhost:8000/api/event-bus/SESSION_ID | python3 -m json.tool

# Create agent via API
curl -X POST http://localhost:8000/api/connect \
  -H 'Content-Type: application/json' \
  -d '{"api_url":"http://localhost:28000/v1/chat/completions","model":"deepseek-v4-flash"}'

# Get causal chain
curl -s http://localhost:8000/api/causal-chain/SESSION_ID | python3 -m json.tool

# Kill server
pkill -f "uvicorn main:app"
```

]()## Development

- **Python 3.11+**, managed with `uv` (`uv sync` creates `.venv/`)
- **No test framework** — tests run inline with `uv run python -c "..."` or temp scripts
- **No type checker / linter** — no mypy, no ruff
- **Swagger docs** at `/docs` when server is running
- **All modules** use `from __future__ import annotations` for deferred evaluation
- **Sessions are in-memory** — restart loses everything unless snapshots were saved
- **Session serialization version**: v5 (adds hypotheses + evidence). v2/v3/v4 fallback works via `from_dict()`
- **Port conflicts**: If `0.0.0.0:8000` is in use, `pkill -f "uvicorn main:app"` then restart

### Debugging Cognitive Models

```python
# SelfModel
print(f"Identity: {session.self_model.identity}")
print(f"Beliefs: {session.self_model.beliefs}")
print(f"Drives: {session.self_model.drives}")

# WorldModel
print(f"Places: {list(session.world_model_cog.places.keys())}")
print(f"Objects: {list(session.world_model_cog.objects.keys())}")
print(f"Events: {len(session.world_model_cog.events)}")

# SocialModel
for aid, profile in session.social_model.profiles.items():
    print(f"  {aid[:8]}: trust={profile.trust:.2f}, rel={profile.reliability:.2f}, "
          f"interactions={profile.interaction_count}")

# KnowledgeModel
for h in session.knowledge_model.hypotheses.active_hypotheses:
    print(f"  [{h.status}] {h.statement[:40]} conf={h.confidence:.2f}")
for f, c in session.knowledge_model.facts.items():
    print(f"  fact: {f[:40]} conf={c:.2f}")

# TheoryOfMind
for aid, ms in session.theory_of_mind._mental_states.items():
    print(f"  {aid[:8]}: beliefs={dict(ms.perceived_beliefs)}")

# WorkingMemory
print(f"Focus: {session.working_memory.current_focus}")
print(f"Question: {session.working_memory.active_question}")
print(f"Unresolved: {session.working_memory.unresolved}")
```

### Debugging Communication

```python
# Check sent messages from an agent
sent = session.communication.get_sent_messages(session.id)
for m in sent:
    print(f"  [{m.msg_type}] → {m.to_agent[:8]}: {m.content.get('text', '')[:60]}")

# Check mailbox
mb = session.communication.get_mailbox(session.id)
for m in mb.messages:
    print(f"  [{m.msg_type}] {m.from_agent[:8]}: {m.content.get('text', '')[:60]}")

# Check shared knowledge
for e in session.shared_knowledge.public_knowledge:
    print(f"  [{e.status}] {e.statement[:60]} (conf={e.confidence}, {e.support_count} agents)")

# Send message directly via CommunicationManager
session.communication.send(
    from_agent=s1.id, to_agent=s2.id,
    msg_type="observation",
    content={"text": "hello"},
    world_tick=1, world_room="entrance",
)
```

### Debugging World Model

```python
# Check hypotheses
for h in session.hypothesis_manager.active_hypotheses:
    print(f"  [{h.status}] {h.statement[:60]} conf={h.confidence:.2f} "
          f"sup={h.support_count} con={h.contradiction_count}")

# Check contradictions
for a, b in session.hypothesis_manager.get_contradictions():
    print(f"  CONFLICT: {a.statement} vs {b.statement}")

# Check evidence
for ev in session.evidence_manager.all_evidence[-5:]:
    print(f"  [{ev.source}] {ev.statement[:60]} conf={ev.confidence:.2f}")

# Check belief-ready candidates
for h in session.hypothesis_manager.get_ready_for_belief():
    print(f"  READY: {h.statement[:60]}")

# Check world model state
print(f"World model: {session.state.get('world_model', {})}")
print(f"Confidence: {session.state.get('confidence', 0.0)}")
print(f"Open questions: {session.state.get('open_questions', [])}")
```
