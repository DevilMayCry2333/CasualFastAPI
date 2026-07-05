"""
engine — RuntimeEngine: unified entry point for the Agent Runtime Kernel.

The RuntimeEngine is the ONLY public API.

It orchestrates:
    Session → MemoryManager.retrieve() → PromptBuilder → LLMClient → Parser
    → MemoryManager.store() → IdentityManager.reflect() → Introspection

External code (Flask, FastAPI, CLI) must ONLY call these methods.
No external code should directly modify Session internals.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from runtime_kernel.runtime.embedding import EmbeddingClient
from runtime_kernel.runtime.exceptions import (
    ConfigurationError,
    LLMError,
    SessionNotFoundError,
    SessionAlreadyExistsError,
    StateValidationError,
)
from runtime_kernel.runtime.causal_physics import (
    CausalVector,
    apply_all_constraints,
    apply_world_anchor,
    convert_human_question,
    compute_memory_force,
    compute_identity_force,
    compute_world_force,
    compute_llm_force,
    integrate_state,
    compute_reachable_states,
    format_reachable_states,
)
from runtime_kernel.runtime.causality import CausalEntry
from runtime_kernel.runtime.communication import CommunicationManager, Message
from runtime_kernel.runtime.drive import DriveModel
from runtime_kernel.runtime.environment import VirtualEnvironment
from runtime_kernel.runtime.evidence import Evidence, EvidenceManager
from runtime_kernel.runtime.experience import Experience
from runtime_kernel.runtime.goal_generator import GoalGenerator
from runtime_kernel.runtime.heartbeat import HeartbeatManager
from runtime_kernel.runtime.hypothesis import Hypothesis, HypothesisManager
from runtime_kernel.runtime.self_modification import SelfModificationManager
from runtime_kernel.runtime.identity_manager import IdentityManager
from runtime_kernel.runtime.introspection import Introspector
from runtime_kernel.runtime.llm import LLMClient
from runtime_kernel.runtime.memory_manager import MemoryManager
from runtime_kernel.runtime.memory_storage import (
    InMemoryMemoryStorage,
    MemoryStorage,
)
from runtime_kernel.runtime.models import (
    IDENTITY_REFLECTION_INTERVAL,
    HYPOTHESIS_MAX_ACTIVE,
    WORLD_EVENT_BROADCAST_INTERVAL,
    SessionStatus,
    StateCause,
)
from runtime_kernel.runtime.parser import (
    align_nl_with_state,
    detect_loop,
    extract_causal_vector,
    extract_state,
    extract_world_model_updates,
    parse_state,
    repair_state,
)
from runtime_kernel.runtime.shared_memory import SharedKnowledge
from runtime_kernel.runtime.persistence import Persistence
from runtime_kernel.runtime.prompt import PromptBuilder
from runtime_kernel.runtime.scheduler import Scheduler
from runtime_kernel.runtime.evolution import EvolutionEngine, RuntimeParameters
from runtime_kernel.runtime.runtime_statistics import RuntimeStatistics
from runtime_kernel.runtime.session import AgentSession
from runtime_kernel.runtime.state import State


class RuntimeEngine:
    """Unified Agent Runtime Kernel.

    Usage:
        engine = RuntimeEngine(llm_config={...})
        session = engine.create_session()
        engine.step(session.id)
        engine.interrupt(session.id, "What is the meaning of life?")

    All methods are thread-safe for concurrent session access.
    """

    def __init__(
        self,
        llm_config: Optional[dict] = None,
        embedding_config: Optional[dict] = None,
        identity_interval: int = IDENTITY_REFLECTION_INTERVAL,
        memory_storage: Optional[MemoryStorage] = None,
        world: Optional[VirtualEnvironment] = None,
        demo: bool = False,
        auto_save: bool = True,
        auto_save_path: str = "autosave.json",
        introspection_interval: int = 20,
        auto_save_interval: int = 10,
        step_interval: float = 300.0,
        heartbeat_interval: float = 30.0,
        enable_heartbeat: bool = True,
        communication: Optional[CommunicationManager] = None,
        shared_knowledge: Optional[SharedKnowledge] = None,
    ) -> None:
        # LLM client
        if llm_config is None:
            llm_config = {}
        self._llm = LLMClient(
            api_url=llm_config.get("api_url", "http://localhost:28000/v1/chat/completions"),
            model=llm_config.get("model", "deepseek-v4-flash"),
            api_key=llm_config.get("api_key", ""),
            timeout=llm_config.get("timeout", 300),
            temperature=llm_config.get("temperature", 0.85),
            max_tokens=llm_config.get("max_tokens", 4096),
        )

        # Embedding client (separate from LLM)
        if embedding_config:
            self._embedding_client = EmbeddingClient(
                api_url=embedding_config.get(
                    "api_url",
                    "http://0.0.0.0:28001/v1/embeddings",
                ),
                model=embedding_config.get("model", "text-embedding-ada-002"),
                api_key=embedding_config.get("api_key", ""),
                timeout=embedding_config.get("timeout", 60),
            )
        else:
            self._embedding_client = None

        # Memory storage (plug-and-play: swap InMemoryMemoryStorage for MySQL version later)
        self._memory_storage = memory_storage or InMemoryMemoryStorage()

        # Virtual environment (shared world for multi-agent interaction)
        self._world = world

        # Communication layer (event-driven multi-agent)
        # If shared across engines, pass in via constructor
        self._communication = communication or CommunicationManager()

        # Shared knowledge (cross-agent consensus)
        # If shared across engines, pass in via constructor
        self._shared_knowledge = shared_knowledge or SharedKnowledge()

        # Wire event bus from communication layer into world
        if self._world:
            self._world.set_event_callback(self._on_world_event)
            self._world.set_message_callback(self._on_agent_message)

        # Modules — all injected via constructor or created with dependencies
        self._prompt_builder = PromptBuilder()
        self._identity_manager = IdentityManager(self._llm, identity_interval)
        self._memory_manager = MemoryManager(
            self._memory_storage,
            self._embedding_client,
        )
        self._drive_model = DriveModel()
        self._goal_generator = GoalGenerator()
        self._persistence = Persistence()
        self._introspector = Introspector(self._llm)
        self._scheduler = Scheduler()

        # Heartbeat (background aliveness)
        # Runtime Statistics (system self-observation)
        self._runtime_stats = RuntimeStatistics(window=100)

        # Evolution Engine (ecosystem manager)
        self._evolution_params = RuntimeParameters()
        self._evolution_engine = EvolutionEngine(
            params=self._evolution_params,
            interval=50,
        )

        self._heartbeat = HeartbeatManager(
            callback=self._heartbeat_pulse,
            interval=heartbeat_interval,
        )

        # Configuration
        self._demo = demo
        self._auto_save = auto_save
        self._auto_save_path = auto_save_path
        self._identity_interval = identity_interval
        self._introspection_interval = introspection_interval
        self._auto_save_interval = auto_save_interval
        self._step_interval = step_interval
        self._memory_retrieval_top_k = 3
        self._enable_heartbeat = enable_heartbeat

        # Session store
        self._sessions: dict[str, AgentSession] = {}

        # Action-based message counter (resets each step)
        self._action_messages_sent: int = 0

        # Start heartbeat
        if enable_heartbeat:
            self._heartbeat.start()

    # ── World Event Callbacks ──

    def _on_agent_message(
        self,
        from_agent: str,
        to_agent: str,
        text: str,
        room: str = "",
        tick: int = 0,
    ) -> None:
        """Callback from VirtualEnvironment when an agent uses send_message as an action.

        Routes the message through CommunicationManager so it gets delivered
        to the recipient's mailbox and tracked in sent_messages.
        """
        if self._communication and text:
            # Record stats for action-based messages
            self._runtime_stats.get_agent_stats(from_agent).record_message_sent()
            msg = self._communication.send(
                from_agent=from_agent,
                to_agent=to_agent,
                msg_type="observation",
                content={"text": text},
                world_tick=tick,
                world_room=room,
            )
            if msg:
                self._memory_manager.store(
                    session_id=from_agent,
                    round_num=0,
                    state_dict={},
                    record_type="message",
                    content=msg.summary,
                    summary=f"Sent to {to_agent[:8]} (via action)",
                    importance=0.6,
                )
                # Increment the action-based message counter for this step
                self._action_messages_sent += 1

    def _on_world_event(
        self,
        event_type: str,
        content: dict,
        room: str = "",
        source: str = "world",
        tick: int = 0,
    ) -> None:
        """Callback from VirtualEnvironment when a world event occurs.

        Publishes the event via CommunicationManager's EventBus.
        """
        if self._communication:
            self._communication.publish_world_event(
                event_type=event_type,
                content=content,
                room=room,
                source=source,
                tick=tick,
            )

    # ── Session lifecycle ──

    def create_session(
        self,
        session_id: Optional[str] = None,
        seed_state: Optional[dict | State] = None,
        demo: bool = False,
    ) -> AgentSession:
        """Create a new Agent session, optionally with a seed state.

        If no seed_state is provided, the engine will call LLM to
        generate the initial state (the "seed" prompt).

        Args:
            session_id: Optional explicit ID (auto-generated if omitted).
            seed_state: Optional initial state dict or State object.
            demo: If True, use a deterministic seed state.

        Returns the created session.
        """
        if session_id and session_id in self._sessions:
            raise SessionAlreadyExistsError(
                f"Session {session_id!r} already exists"
            )

        session = AgentSession(session_id=session_id)
        session.set_status(SessionStatus.INITIALIZING)

        if seed_state:
            # Use provided seed
            if isinstance(seed_state, State):
                state_dict = seed_state.to_dict()
            else:
                state_dict = dict(seed_state)
            session.update_state(state_dict, cause=StateCause.INIT.value)

        elif demo:
            # Deterministic demo seed
            state_dict = {
                "topic": "consciousness_and_causality",
                "belief": "consciousness_may_be_self_observation_node",
                "goal": "seek_origin",
            }
            session.update_state(state_dict, cause=StateCause.INIT.value)

        else:
            # Call LLM for initial seed
            messages = PromptBuilder.build_seed()
            response = self._llm.complete(
                messages, temperature=0.9, max_tokens=300
            )

            if response:
                _, parsed_state = extract_state(response)
                if parsed_state:
                    state_dict = parsed_state.to_dict()
                else:
                    state_dict = {
                        "init": True, "topic": "start",
                        "belief": "unknown", "goal": "init",
                    }
            else:
                state_dict = {
                    "init": True, "topic": "start",
                    "belief": "unknown", "goal": "init",
                }

            session.update_state(state_dict, cause=StateCause.INIT.value)

        # Store initial state in memory
        self._memory_manager.store_state(
            session_id=session.id,
            round_num=session.round,
            state_dict=session.state.to_dict(),
        )

        # Register in shared world
        if self._world:
            self._world.register(session.id)

        # Register in communication layer
        if self._communication:
            self._communication.register_agent(session.id)
            session.set_communication(self._communication)

        # Assign shared knowledge reference
        session.set_shared_knowledge(self._shared_knowledge)

        session.set_status(SessionStatus.RUNNING)
        self._sessions[session.id] = session

        if self._auto_save:
            self._persistence.snapshot(session, self._auto_save_path)

        return session

    def get_session(self, session_id: str) -> AgentSession:
        """Get a session by ID.

        Raises SessionNotFoundError if not found.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session not found: {session_id!r}")
        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session from the in-memory store.

        Does NOT delete snapshot files.

        Returns True if deleted, False if not found.
        """
        if session_id in self._sessions:
            session = self._sessions[session_id]
            session.set_status(SessionStatus.TERMINATED)
            del self._sessions[session_id]
            if self._world:
                self._world.unregister(session_id)
            if self._communication:
                self._communication.unregister_agent(session_id)
            return True
        return False

    def list_sessions(self) -> list[dict]:
        """Return summary info for all active sessions."""
        return [
            {
                "id": s.id,
                "round": s.round,
                "status": s.status.value,
                "fold_count": s.fold_count,
                "drives": s.drives,
                "thought_count": len(s.thought_pool),
                "position": self._world.agent_position(s.id) if self._world else "",
                "created_at": s.created_at,
                "hypothesis_count": len(s.hypothesis_manager.active_hypotheses),
                "evidence_count": len(s.evidence_manager.all_evidence),
                "confidence": s.state.get("confidence", 0.0),
            }
            for s in self._sessions.values()
        ]

    # ── Core step ──

    def step(
        self,
        session_id: str,
        human_input: str = "",
    ) -> dict[str, Any]:
        """Execute one agent step.

        This is the primary method. It handles both:
          - Autonomous thinking (no human_input)
          - Human interruption (with human_input)

        New flow (architecture v2):
            retrieve() → build_prompt() → LLM → append_history()
            → memory.store() → identity.reflect() → introspect()

        Args:
            session_id: Target session ID.
            human_input: Optional human message to interrupt with.

        Returns a result dict with step metadata.

        Raises SessionNotFoundError if session_id is invalid.
        """
        session = self.get_session(session_id)
        session.set_status(SessionStatus.RUNNING)

        if human_input:
            return self._handle_human_interrupt(session, human_input)
        else:
            return self._handle_autonomous_step(session)

    def interrupt(self, session_id: str, human_message: str) -> dict[str, Any]:
        """Inject a human message into a running session.

        Convenience wrapper around step() with human_input.

        Args:
            session_id: Target session ID.
            human_message: The human's message.

        Returns step result dict.
        """
        return self.step(session_id, human_input=human_message)

    def _handle_autonomous_step(self, session: AgentSession) -> dict[str, Any]:
        """Execute an autonomous thinking step (World Model focus).

        Flow (v2 — World Model):
            1. Capture pre-transition snapshot
            2. Build hypothesis + evidence context for prompt
            3. Call LLM → extract delta + world model updates
            4. Process new hypotheses, evidence, hypothesis updates
            5. Check for belief-ready hypotheses → update belief
            6. Execute action in world
            7. Record CausalEntry
            8. Store in memory (including evidence, hypothesis events)
            9. Update drives, reflect identity, introspect
        """
        # Reset action-based message counter
        self._action_messages_sent = 0

        # 1. Capture pre-transition snapshot
        state_before = session.state.to_dict()
        world_room_before = self._world.agent_position(session.id) if self._world else ""
        identity_before = session.identity_anchor
        drives_before = session.drives
        thought_pool_before = session.thought_pool
        world_tick = self._world.get_world_summary()["meta"]["tick"] if self._world else 0

        session.increment_rounds_since_human()

        # Loop detection
        loop_detected = detect_loop(session.history)

        # Introspection injection check
        should_inject = self._introspector.should_inject(session)
        introspections = session.introspections if should_inject else []

        # 2. Build hypothesis + evidence context for prompt
        hypothesis_context = session.hypothesis_manager.format_for_prompt(max_hypotheses=3)
        evidence_context = session.evidence_manager.format_for_prompt(max_evidence=5)
        contradiction_context = session.hypothesis_manager.format_contradictions_for_prompt()
        if contradiction_context:
            if evidence_context:
                evidence_context += "\n" + contradiction_context
            else:
                evidence_context = contradiction_context

        # 2b. Build communication context: mailbox messages + shared knowledge + world events
        mailbox_context = ""
        shared_knowledge_context = ""
        world_events_context = ""
        if self._communication:
            mailbox_context = self._communication.format_mailbox_for_prompt(
                session.id, max_messages=5,
            )
            # World events from EventBus
            recent_events = self._communication.event_bus.get_recent_events(n=5)
            world_events_context = CommunicationManager.format_events_for_prompt(recent_events)

        if self._shared_knowledge:
            shared_knowledge_context = self._shared_knowledge.format_for_prompt(max_entries=5)

        # 3. Build cognitive model contexts (v4 Cognitive Architecture)
        self_model_context = session.self_model.format_for_prompt()
        world_model_cog_context = session.world_model_cog.format_for_prompt()
        social_model_context = session.social_model.format_for_prompt()
        knowledge_model_context = session.knowledge_model.format_for_prompt()
        working_memory_context = session.working_memory.format_for_prompt()
        theory_of_mind_context = session.theory_of_mind.format_for_prompt()

        # 3b. Run perception pipeline: events → attention → context
        perception_context = ""
        if self._communication:
            mailbox_msgs = []
            mb = self._communication.get_mailbox(session.id)
            if mb:
                mailbox_msgs = [m.to_dict() for m in mb.messages]

            recent_events = self._communication.event_bus.get_recent_events(n=10)
            env_context_raw = self._world.get_context(session.id) if self._world else ""

            from runtime_kernel.runtime.cognitive.perception import perceive, format_perception_for_prompt
            perception_result = perceive(
                mailbox_messages=mailbox_msgs,
                world_events=recent_events,
                env_context=env_context_raw,
                drives=session.drives,
                world_model=session.world_model_cog,
                uncertain_areas=[u.get("domain", str(u)) for u in session.state.get("uncertainties", [])],
                recent_attended_events=[],  # TODO: track across steps
                max_events=3,
            )
            perception_context = format_perception_for_prompt(perception_result)

            # Update working memory based on perception and current topic
            session.working_memory.build_from_models(
                topic=session.state.topic,
                belief=session.state.belief,
                goal=session.state.goal,
                open_questions=session.state.get("open_questions", []),
                contradictions=session.knowledge_model.get_contradictions(),
            )
            working_memory_context = session.working_memory.format_for_prompt()

        # Get world model from state (legacy)
        world_model = session.state.get("world_model", {})

        # Build active hypothesis statements for memory retrieval
        hyp_statements = [
            h.statement for h in session.hypothesis_manager.active_hypotheses
        ]

        # 3c. Retrieve memories for RAG + causal context
        memory_context = ""
        if self._memory_manager:
            query_text = (
                f"{session.state.topic} {session.state.belief} {session.state.goal}"
            )
            memory_context = self._memory_manager.build_context(
                query_text,
                top_k=self._memory_retrieval_top_k,
                causality=session.causality,
                session_id=session.id,
                hypothesis_statements=hyp_statements,
            )

        # 3d. Build causal chain
        causal_context = session.causality.build_context(session.id, n=5)

        # 4. Build prompt with Cognitive Architecture
        env_context = self._world.get_context(session.id) if self._world else ""
        messages = PromptBuilder.build_step(
            state=session.state,
            identity_anchor=session.identity_anchor,
            drives=session.drives,
            thought_pool=session.thought_pool,
            env_context=env_context,
            memory_context=memory_context,
            causal_context=causal_context,
            identity_maturity=session.identity_maturity,
            history=session.history,
            introspections=introspections,
            rounds_since_human=session.rounds_since_human,
            loop_detected=loop_detected,
            hypothesis_context=hypothesis_context,
            evidence_context=evidence_context,
            world_model=world_model,
            mailbox_context=mailbox_context,
            shared_knowledge_context=shared_knowledge_context,
            world_events_context=world_events_context,
            # Cognitive model contexts
            self_model_context=self_model_context,
            world_model_cog_context=world_model_cog_context,
            social_model_context=social_model_context,
            knowledge_model_context=knowledge_model_context,
            working_memory_context=working_memory_context,
            perception_context=perception_context,
            theory_of_mind_context=theory_of_mind_context,
        )

        # 5. Call LLM
        llm_delta: dict[str, str] = {}
        llm_force: float = 0.0
        llm_action: str = ""
        llm_source: str = "llm"
        llm_nl: str = ""
        wm_updates: dict = {}

        if self._demo:
            llm_delta = {"topic": f"{session.state.topic}_demo",
                         "belief": f"{session.state.belief}_step",
                         "goal": f"{session.state.goal}_continue"}
            llm_force = 0.5
        else:
            response = self._llm.complete(
                messages, temperature=0.7, max_tokens=1200
            )
            if response:
                llm_nl, llm_delta, llm_force, llm_action, llm_source = extract_causal_vector(response)
                # Extract world model updates from the same response
                wm_updates = extract_world_model_updates(response)

        # 5b. Apply constraint layers (keep for backward compat but reduce disruption)
        if llm_delta and not self._demo:
            llm_delta, llm_force = apply_all_constraints(
                delta=llm_delta,
                force=llm_force,
                round_num=session.round,
                current_topic=session.state.topic,
                identity_mass=session.identity_maturity,
                recent_experiences=session.experiences,
            )
            llm_delta = apply_world_anchor(
                delta=llm_delta,
                current_topic=session.state.topic,
                env_context=env_context,
                human_input="",
            )

        # 6. Process world model updates from LLM
        self._process_world_model_updates(session, wm_updates)

        # 6b. Process communication (messages from LLM output)
        self._process_communication(session, wm_updates, world_tick, world_room_before, llm_action)

        # 7. Check for belief-ready hypotheses → update belief if warranted
        self._check_belief_update(session, llm_delta, llm_force)

        # 7b. Update cognitive models from LLM output
        self._update_cognitive_models(session, wm_updates)

        # 8. Compute causal forces
        f_memory = compute_memory_force(
            current_state=session.state.to_dict(),
            recent_experiences=session.experiences,
        )
        f_identity = compute_identity_force(
            current_state=session.state.to_dict(),
            identity_mass=session.identity_maturity,
            llm_delta=llm_delta or {},
            llm_strength=llm_force,
        )
        f_world = compute_world_force(env_context)

        if llm_delta:
            llm_vector = compute_llm_force(
                delta=llm_delta,
                strength=llm_force,
                action=llm_action,
                source=llm_source,
            )
        else:
            llm_vector = CausalVector(source="llm", strength=0.0)

        # 9. Integrate forces
        state_dict = integrate_state(
            current_state=session.state.to_dict(),
            llm_vector=llm_vector,
            memory_vector=f_memory,
            identity_vector=f_identity,
            world_vector=f_world,
            identity_mass=session.identity_maturity,
        )

        # Merge world model fields back into state dict
        if "world_model" in wm_updates or session.state.get("world_model"):
            state_dict["world_model"] = session.state.get("world_model", {})
        # Keep hypotheses in state for serialization + display
        state_dict["hypotheses"] = [
            h.to_dict() for h in session.hypothesis_manager.active_hypotheses
        ]
        state_dict["evidence"] = [
            e.to_dict() for e in session.evidence_manager.all_evidence[-20:]
        ]
        state_dict["open_questions"] = session.state.get("open_questions", [])
        state_dict["uncertainties"] = session.state.get("uncertainties", [])
        state_dict["confidence"] = session.state.get("confidence", 0.0)

        state = repair_state(State(state_dict))
        session.update_state(state.to_dict(), cause=StateCause.SELF.value)

        # 10. Execute action in shared world (if any)
        action = ""
        if self._world:
            action = llm_action or state_dict.get("action", "")
            if action:
                self._world.act(session.id, action)

        # 10b. Record runtime statistics
        if action:
            self._runtime_stats.get_agent_stats(session.id).record_action(action)
        self._runtime_stats.get_agent_stats(session.id).set_round(session.round)
        self._runtime_stats.snapshot_if_needed(session.id, session.round)

        # 10c. Run evolution engine if interval reached
        if self._evolution_engine.should_run(session.round):
            report = self._evolution_engine.run(session.round, self._runtime_stats)
            # Apply parameter changes to engine configuration
            self._apply_evolution_params(session)

        # 11. Record causal entry
        causal_entry = session.causality.create_entry(
            session_id=session.id,
            round_num=session.round,
            cause=StateCause.SELF.value,
            state_before=state_before,
            state_after=state.to_dict(),
            action=action,
            world_room=world_room_before,
            world_tick=world_tick,
            identity_anchor=identity_before,
            drives=drives_before,
            thought_pool=thought_pool_before,
            reasoning=f"force:{llm_force:.2f} source:{llm_source} "
                      f"hypotheses:{len(session.hypothesis_manager.active_hypotheses)} "
                      f"evidence:{len(session.evidence_manager.all_evidence)}",
        )

        # 12. Record Experience
        env_perception = env_context[:200] if env_context else f"在 {world_room_before}"
        session.add_experience(Experience(
            round=session.round,
            session_id=session.id,
            perception=env_perception,
            action=action,
            observation=self._world.get_context(session.id)[:200] if action and self._world else "",
            meaning="",
            cause=StateCause.SELF.value,
            state_before=state_before,
            state_after=state.to_dict(),
            room=world_room_before,
        ))

        # 13. Store state in long-term memory
        self._memory_manager.store_state(
            session_id=session.id,
            round_num=session.round,
            state_dict=state.to_dict(),
        )

        # 14. Update drives + generate thought pool
        self._update_drives_and_goals(session, state.to_dict())

        # Token length warning
        too_long = IdentityManager.check_token_length(state)
        extra: dict[str, Any] = {"token_warning": too_long, "causal_round": causal_entry.round}

        # 15. Identity reflection (replaces Fold)
        if self._identity_manager.should_reflect(session.round):
            self._perform_identity_reflection(session)

        # 16. Auto-save
        if session.round % self._auto_save_interval == 0 and self._auto_save:
            self._persistence.snapshot(session, self._auto_save_path)

        # 17. Introspection
        if self._introspector.should_introspect(session) and not self._demo:
            summary = self._introspector.introspect(session)
            if summary:
                session.add_introspection(summary)
                self._memory_manager.store_introspection(
                    session_id=session.id,
                    round_num=session.round,
                    state_dict=state.to_dict(),
                    summary=summary,
                )

        session.set_status(SessionStatus.IDLE)

        return {
            "type": "autonomous",
            "session_id": session.id,
            "round": session.round,
            "state": state.to_dict(),
            "world_model_updates": wm_updates,
            "identity_anchor": session.identity_anchor,
            "drives": session.drives,
            "thought_pool": session.thought_pool,
            "state_compressed": session.state_compressed.to_dict() if session.state_compressed else None,
            "hypothesis_count": len(session.hypothesis_manager.active_hypotheses),
            "evidence_count": len(session.evidence_manager.all_evidence),
            "messages_sent": len(wm_updates.get("send_messages", [])) + self._action_messages_sent,
            "mailbox_unread": self._communication.get_mailbox(session.id).count() if self._communication else 0,
            "extra": extra,
        }

    # ── Communication Processing ──

    def _process_communication(
        self,
        session: AgentSession,
        wm_updates: dict,
        world_tick: int,
        world_room: str,
        action: str = "",
    ) -> None:
        """Process communication-related outputs from LLM.

        The LLM can output:
        - send_message: dict with to_agent, type, content — sends a message
        - share_knowledge: dict with statement, domain — proposes shared knowledge
        - support_knowledge: entry_id — supports existing shared knowledge

        Also handles LLMs that embed send_message inside the `action` field:
            action = "send_message(to_agent='xxx', type='obs', content='...')，然后做其他事"

        The agent decides whether to communicate.
        Runtime does NOT force responses.
        """
        if not wm_updates:
            wm_updates = {}
        if not self._communication:
            return

        # ── Fallback: parse send_message(...) from action text ──
        if action:
            self._parse_action_messages(session, action, wm_updates, world_tick, world_room)

        # 1. Send messages
        raw_messages = wm_updates.get("send_message", wm_updates.get("send_messages", []))
        if isinstance(raw_messages, dict):
            raw_messages = [raw_messages]
        if isinstance(raw_messages, list):
            for msg_data in raw_messages:
                to_agent = msg_data.get("to_agent", "")
                msg_type = msg_data.get("type", "observation")
                content = msg_data.get("content", msg_data)
                if isinstance(content, str):
                    content = {"text": content}

                if to_agent == "*" or to_agent == "broadcast":
                    # Broadcast to all
                    sent = self._communication.broadcast(
                        from_agent=session.id,
                        msg_type=msg_type,
                        content=content,
                        world_tick=world_tick,
                        world_room=world_room,
                    )
                    for msg in sent:
                        self._memory_manager.store(
                            session_id=session.id,
                            round_num=session.round,
                            state_dict={},
                            record_type="message",
                            content=msg.summary,
                            summary=f"Sent {msg_type} to {msg.to_agent[:8]}",
                            importance=0.6,
                        )
                else:
                    # Direct message
                    msg = self._communication.send(
                        from_agent=session.id,
                        to_agent=to_agent,
                        msg_type=msg_type,
                        content=content,
                        world_tick=world_tick,
                        causal_parent="",
                        world_room=world_room,
                    )
                    if msg:
                        self._memory_manager.store(
                            session_id=session.id,
                            round_num=session.round,
                            state_dict={},
                            record_type="message",
                            content=msg.summary,
                            summary=f"Sent {msg_type} to {to_agent[:8]}",
                            importance=0.6,
                        )

            # Record statistics for sent messages
            stats = self._runtime_stats.get_agent_stats(session.id)
            stats.record_message_sent()

        # 2. Propose shared knowledge
        share_data = wm_updates.get("share_knowledge", wm_updates.get("share_memory", None))
        if share_data and self._shared_knowledge:
            statement = share_data.get("statement", share_data.get("text", ""))
            if statement:
                domain = share_data.get("domain", session.state.topic)
                entry = self._shared_knowledge.propose(
                    statement=statement,
                    agent_id=session.id,
                    domain=domain,
                    source="observation",
                    tick=world_tick,
                )
                if entry:
                    self._memory_manager.store(
                        session_id=session.id,
                        round_num=session.round,
                        state_dict={},
                        record_type="shared_knowledge",
                        content=statement,
                        summary=f"Shared knowledge: {statement[:100]}",
                        importance=0.8,
                    )

        # 3. Support existing shared knowledge
        support_id = wm_updates.get("support_knowledge", "")
        if support_id and self._shared_knowledge:
            self._shared_knowledge.support(support_id, session.id)

        # 4. Contradict existing shared knowledge
        contradict_id = wm_updates.get("contradict_knowledge", "")
        if contradict_id and self._shared_knowledge:
            self._shared_knowledge.contradict(contradict_id, session.id)

        # 5. Store mailbox messages as memory (causal)
        mailbox = self._communication.get_mailbox(session.id)
        if mailbox and mailbox.has_unread():
            for msg in mailbox.messages:
                self._memory_manager.store(
                    session_id=session.id,
                    round_num=session.round,
                    state_dict={},
                    record_type="message",
                    content=f"From {msg.from_agent[:8]}: {msg.summary}",
                    summary=f"Received {msg.msg_type} from {msg.from_agent[:8]}",
                    importance=0.5,
                )

    def _parse_action_messages(
        self,
        session: AgentSession,
        action: str,
        wm_updates: dict,
        world_tick: int,
        world_room: str,
    ) -> None:
        """Parse send_message(...) patterns embedded in the `action` field.

        Some LLMs naturally write communication inside the action field:
            action = "send_message(to_agent='xxx', type='observation', content='...')，然后做其他事"

        This is a fallback parser for that pattern.
        """
        if not action or not self._communication:
            return

        # Pattern: send_message(to_agent='...', type='...', content='...')
        # Also handles "send_message(...)" at the start or middle of action string
        sm_match = re.search(
            r'send_message\s*\(\s*'
            r"to_agent\s*=\s*['\"]([^'\"]+)['\"]\s*,?\s*"
            r"type\s*=\s*['\"]([^'\"]+)['\"]\s*,?\s*"
            r"content\s*=\s*['\"]([^'\"]+)['\"]",
            action,
            re.DOTALL,
        )
        if not sm_match:
            # Try alternate format: send_message to_agent type content
            sm_match = re.search(
                r'send_message\s+'
                r"['\"]?([a-zA-Z0-9_]+)['\"]?\s+"
                r"['\"]?([a-zA-Z_]+)['\"]?\s+"
                r"['\"]([^'\"]+)['\"]",
                action,
            )

        if not sm_match:
            return

        to_agent = sm_match.group(1)
        msg_type = sm_match.group(2)
        content_text = sm_match.group(3)

        # Normalize message type
        valid_types = {"observation", "question", "answer", "hypothesis",
                       "plan", "request", "warning", "event", "report",
                       "share_memory", "broadcast", "inquiry", "greeting",
                       "suggestion", "response", "information"}
        if msg_type not in valid_types:
            msg_type = "observation"

        # Send the message
        msg = self._communication.send(
            from_agent=session.id,
            to_agent=to_agent,
            msg_type=msg_type,
            content={"text": content_text},
            world_tick=world_tick,
            world_room=world_room,
        )
        if msg:
            self._memory_manager.store(
                session_id=session.id,
                round_num=session.round,
                state_dict={},
                record_type="message",
                content=msg.summary,
                summary=f"Sent {msg_type} to {to_agent[:8]} (from action)",
                importance=0.6,
            )
            # Also add to wm_updates so the frontend sees it
            if "send_messages" not in wm_updates:
                wm_updates["send_messages"] = []
            wm_updates["send_messages"].append({
                "to_agent": to_agent,
                "type": msg_type,
                "content": {"text": content_text},
            })

    # ── World Model Update Processing ──

    def _process_world_model_updates(
        self,
        session: AgentSession,
        wm_updates: dict,
    ) -> None:
        """Process world model updates from LLM output.

        Handles:
        - new_hypotheses → propose to HypothesisManager
        - new_evidence → add to EvidenceManager
        - hypothesis_updates → link evidence to hypotheses (support/contradict)
        - world_model_update → merge into state world_model
        - open_questions → update state
        - uncertainties → update state
        - confidence → update state
        """
        if not wm_updates:
            return

        # Track evidence IDs created this step for linking
        step_evidence_ids: list[str] = []

        # 1. Process new evidence
        new_evidence = wm_updates.get("new_evidence", [])
        for ev_data in new_evidence:
            ev = session.evidence_manager.add_observation(
                statement=ev_data.get("statement", ""),
                source=ev_data.get("source", "observation"),
                confidence=float(ev_data.get("confidence", 0.5)),
                domain=ev_data.get("domain", ""),
                round_num=session.round,
                raw_context=ev_data.get("raw_context", ""),
            )
            step_evidence_ids.append(ev.id)

            # Store in long-term memory
            self._memory_manager.store_evidence(
                session_id=session.id,
                round_num=session.round,
                evidence=ev.to_dict(),
            )

        # 2. Process new hypotheses
        new_hypotheses = wm_updates.get("new_hypotheses", [])
        for hyp_data in new_hypotheses:
            # Enforce max active hypotheses
            if len(session.hypothesis_manager.active_hypotheses) >= HYPOTHESIS_MAX_ACTIVE:
                break

            hyp = session.hypothesis_manager.propose(
                statement=hyp_data.get("statement", ""),
                source=hyp_data.get("source", "observation"),
                round_num=session.round,
                domain=hyp_data.get("domain", ""),
            )

            # Store in long-term memory
            self._memory_manager.store_hypothesis(
                session_id=session.id,
                round_num=session.round,
                hypothesis=hyp.to_dict(),
            )

            # Link step evidence to new hypothesis if in same domain
            for ev_id in step_evidence_ids:
                ev = session.evidence_manager.get(ev_id)
                if ev and (ev.domain == hyp.domain or not hyp.domain):
                    session.evidence_manager.link_to_hypothesis(
                        ev_id, hyp.id, supports=True,
                    )
                    session.hypothesis_manager.add_evidence(
                        hyp.id, ev_id, supports=True, round_num=session.round,
                    )

        # 3. Process hypothesis updates (support/contradict)
        hyp_updates = wm_updates.get("hypothesis_updates", [])
        for update in hyp_updates:
            hyp_id = update.get("id", "")
            supports = update.get("supports", True)

            # Use step evidence if available, otherwise create a deduction
            if step_evidence_ids:
                ev_id = step_evidence_ids[0]
            else:
                # Create a deduction evidence for the update
                hyp = session.hypothesis_manager.get(hyp_id)
                if hyp:
                    ev = session.evidence_manager.add_deduction(
                        statement=f"Observation related to: {hyp.statement[:50]}",
                        domain=hyp.domain,
                        round_num=session.round,
                    )
                    ev_id = ev.id
                    step_evidence_ids.append(ev_id)
                else:
                    continue

            session.evidence_manager.link_to_hypothesis(ev_id, hyp_id, supports=supports)
            session.hypothesis_manager.add_evidence(
                hyp_id, ev_id, supports=supports, round_num=session.round,
            )

        # 4. Update world model
        wm_update = wm_updates.get("world_model_update")
        if wm_update:
            current_wm = dict(session.state.get("world_model", {}))
            current_wm.update(wm_update)
            session.state.set_world_model(current_wm)

        # 5. Update open questions
        questions = wm_updates.get("open_questions")
        if questions is not None:
            session.state.set_open_questions(questions)

        # 6. Update uncertainties
        uncertainties = wm_updates.get("uncertainties")
        if uncertainties is not None:
            session.state.set_uncertainties(uncertainties)

        # 7. Update confidence
        confidence = wm_updates.get("confidence")
        if confidence is not None:
            session.state.set_confidence(confidence)

        # Record statistics for evidence and hypotheses
        stats = self._runtime_stats.get_agent_stats(session.id)
        if new_evidence:
            for _ in new_evidence:
                stats.record_evidence()
        if new_hypotheses:
            for _ in new_hypotheses:
                stats.record_hypothesis_event("proposed")
        # Record belief changes from state diff
        if session.history and len(session.history) >= 2:
            prev = session.history[-2].get("state", {}).get("belief", "")
            curr = session.state.belief
            if prev and curr and prev != curr:
                stats.record_belief_change()

    # ── Evolution Engine ──

    def _apply_evolution_params(self, session: AgentSession) -> None:
        """Apply evolved runtime parameters to the current configuration."""
        params = self._evolution_params
        self._memory_retrieval_top_k = params.get("memory_retrieval_top_k", 3)
        # Drives, attention, and hypothesis thresholds are read
        # at usage time from self._evolution_params by other components.

    # ── Cognitive Model Updates ──

    def _update_cognitive_models(
        self,
        session: AgentSession,
        wm_updates: dict,
    ) -> None:
        """Update cognitive models from LLM output.

        The LLM can output:
        - self_update: {belief, confidence} — update self-model
        - social_update: {agent_id, cooperative, ...} — update social model
        - theory_update: {agent_id, belief} — update theory of mind
        """
        if not wm_updates:
            return

        # Self-model update
        self_update = wm_updates.get("self_update")
        if isinstance(self_update, dict):
            belief = self_update.get("belief", "")
            if belief:
                confidence = float(self_update.get("confidence", 0.5))
                session.self_model.set_belief(belief, confidence)
                session.self_model.add_change(f"信念更新: {belief[:40]}")

        # Social model update (judgment about another agent)
        social_update = wm_updates.get("social_update")
        if isinstance(social_update, dict):
            agent_id = social_update.get("agent_id", "")
            if agent_id:
                profile = session.social_model.get_or_create(agent_id)
                cooperative = social_update.get("cooperative", True)
                if isinstance(cooperative, bool):
                    profile.record_interaction(
                        round_num=session.round,
                        cooperative=cooperative,
                    )

        # Theory of mind update (belief about another agent's belief)
        theory_update = wm_updates.get("theory_update")
        if isinstance(theory_update, dict):
            agent_id = theory_update.get("agent_id", "")
            belief = theory_update.get("belief", "")
            if agent_id and belief:
                confidence = float(theory_update.get("confidence", 0.5))
                ms = session.theory_of_mind.get_or_create(agent_id)
                ms.observe_belief(belief, confidence)

    def _check_belief_update(
        self,
        session: AgentSession,
        llm_delta: dict[str, str],
        llm_force: float,
    ) -> None:
        """Check if any hypotheses are ready to update belief.

        Belief is updated through the evidence pipeline:
            Evidence → Hypothesis (repeated support) → Belief Update

        The LLM cannot directly set belief. Only hypotheses that have
        accumulated enough supporting evidence can influence belief.

        This method enforces the rule by:
        1. Checking for belief-ready hypotheses
        2. If found, merging the strongest hypothesis into belief
        3. If LLM proposed a delta_belief, checking if it has evidence support
        """
        belief_ready = session.hypothesis_manager.get_ready_for_belief()

        if belief_ready:
            # A hypothesis has enough support → it can influence belief
            strongest = max(belief_ready, key=lambda h: h.confidence)
            if llm_delta:
                # Only override if the hypothesis is solid enough
                if strongest.confidence > 0.7:
                    llm_delta["belief"] = strongest.statement[:100]

        elif llm_delta and llm_delta.get("belief"):
            # LLM proposed a belief change without hypothesis backing
            # Check if any evidence supports this proposed belief
            proposed = llm_delta["belief"]
            supporting_evidence = [
                ev for ev in session.evidence_manager.all_evidence
                if proposed.lower() in ev.statement.lower()
            ]
            if not supporting_evidence and llm_force < 0.6:
                # No evidence supports this belief change and force is weak
                # Don't change belief — keep current
                current_belief = session.state.belief
                if current_belief and current_belief != "unknown":
                    llm_delta["belief"] = current_belief

    def _handle_human_interrupt(
        self,
        session: AgentSession,
        human_input: str,
    ) -> dict[str, Any]:
        """Handle a human interruption (World Model aware).

        Human input is treated as EVIDENCE first, not a direct belief update.
        The input goes through the evidence pipeline:
            human_input → Evidence → Hypothesis → (maybe) Belief

        Flow:
            1. Convert abstract questions, capture pre-transition snapshot
            2. Store human input as evidence
            3. Build prompt with hypothesis + evidence context
            4. Call LLM
            5. Process world model updates from response
            6. Update session, record causal entry
        """
        # Reset action-based message counter
        self._action_messages_sent = 0

        # 1. Capture pre-transition snapshot
        state_before = session.state.to_dict()
        world_room_before = self._world.agent_position(session.id) if self._world else ""
        identity_before = session.identity_anchor
        drives_before = session.drives
        thought_pool_before = session.thought_pool
        world_tick = self._world.get_world_summary()["meta"]["tick"] if self._world else 0

        # Convert abstract human questions
        grounded_input = convert_human_question(human_input, session.state.topic)
        prompt_input = grounded_input

        session.set_status(SessionStatus.INTERRUPTED)

        # 2. Store human input as evidence (high confidence)
        ev = session.evidence_manager.add_human_statement(
            statement=human_input[:200],
            confidence=0.7,
            domain="human_input",
            round_num=session.round,
        )
        self._memory_manager.store_evidence(
            session_id=session.id,
            round_num=session.round,
            evidence=ev.to_dict(),
        )

        # Loop detection
        loop_detected = detect_loop(session.history)

        should_inject = self._introspector.should_inject(session)
        introspections = session.introspections if should_inject else []

        # Build hypothesis + evidence context
        hypothesis_context = session.hypothesis_manager.format_for_prompt(max_hypotheses=3)
        evidence_context = session.evidence_manager.format_for_prompt(max_evidence=5)
        world_model = session.state.get("world_model", {})

        # Build communication context
        mailbox_context = ""
        shared_knowledge_context = ""
        world_events_context = ""
        if self._communication:
            mailbox_context = self._communication.format_mailbox_for_prompt(
                session.id, max_messages=5,
            )
            recent_events = self._communication.event_bus.get_recent_events(n=5)
            world_events_context = CommunicationManager.format_events_for_prompt(recent_events)
        if self._shared_knowledge:
            shared_knowledge_context = self._shared_knowledge.format_for_prompt(max_entries=5)

        hyp_statements = [
            h.statement for h in session.hypothesis_manager.active_hypotheses
        ]

        # Retrieve memories for RAG + causal context
        memory_context = ""
        if self._memory_manager:
            query_text = f"{session.state.topic} {session.state.belief} {prompt_input[:200]}"
            memory_context = self._memory_manager.build_context(
                query_text,
                top_k=self._memory_retrieval_top_k,
                causality=session.causality,
                session_id=session.id,
                hypothesis_statements=hyp_statements,
            )

        # Build causal context
        causal_context = session.causality.build_context(session.id, n=5)

        # Build cognitive model contexts
        self_model_context = session.self_model.format_for_prompt()
        world_model_cog_context = session.world_model_cog.format_for_prompt()
        social_model_context = session.social_model.format_for_prompt()
        knowledge_model_context = session.knowledge_model.format_for_prompt()
        working_memory_context = session.working_memory.format_for_prompt()
        theory_of_mind_context = session.theory_of_mind.format_for_prompt()
        perception_context = ""  # Skip perception pipeline for interrupt (human input replaces it)

        # Build prompt with Cognitive Architecture
        env_context = self._world.get_context(session.id) if self._world else ""
        messages = PromptBuilder.build_interrupt(
            state=session.state,
            human_input=prompt_input,
            identity_anchor=session.identity_anchor,
            drives=session.drives,
            thought_pool=session.thought_pool,
            env_context=env_context,
            memory_context=memory_context,
            causal_context=causal_context,
            identity_maturity=session.identity_maturity,
            history=session.history,
            introspections=introspections,
            loop_detected=loop_detected,
            hypothesis_context=hypothesis_context,
            evidence_context=evidence_context,
            world_model=world_model,
            mailbox_context=mailbox_context,
            shared_knowledge_context=shared_knowledge_context,
            world_events_context=world_events_context,
            self_model_context=self_model_context,
            world_model_cog_context=world_model_cog_context,
            social_model_context=social_model_context,
            knowledge_model_context=knowledge_model_context,
            working_memory_context=working_memory_context,
            perception_context=perception_context,
            theory_of_mind_context=theory_of_mind_context,
        )

        # Call LLM
        nl_text = ""
        state_dict = session.state.to_dict()
        wm_updates: dict = {}

        if self._demo:
            nl_text = f"（回应人类：{human_input[:60]}...）"
        else:
            response = self._llm.complete(
                messages, temperature=0.85, max_tokens=2048
            )

            if response:
                nl_text, parsed_state = extract_state(response)
                state_dict = (
                    parsed_state.to_dict() if parsed_state
                    else session.state.to_dict()
                )
                # Extract world model updates
                wm_updates = extract_world_model_updates(response)
            else:
                nl_text = ""
                state_dict = session.state.to_dict()

            # Post-process: align NL with state
            nl_text = align_nl_with_state(session.state, nl_text)

        # Process world model updates from LLM response
        self._process_world_model_updates(session, wm_updates)

        # Process communication from LLM response (also check action field)
        world_tick = self._world.get_world_summary()["meta"]["tick"] if self._world else 0
        world_room = self._world.agent_position(session.id) if self._world else ""
        llm_action = state_dict.get("action", "")
        self._process_communication(session, wm_updates, world_tick, world_room, llm_action)

        # Update cognitive models from LLM output
        self._update_cognitive_models(session, wm_updates)

        # Store interaction
        session.add_interaction(human_input, nl_text)
        # Merge world model fields back into state
        if session.state.get("world_model") or state_dict:
            state_dict["world_model"] = session.state.get("world_model", {})
        state_dict["hypotheses"] = [
            h.to_dict() for h in session.hypothesis_manager.active_hypotheses
        ]
        state_dict["evidence"] = [
            e.to_dict() for e in session.evidence_manager.all_evidence[-20:]
        ]
        state_dict["open_questions"] = session.state.get("open_questions", [])
        state_dict["uncertainties"] = session.state.get("uncertainties", [])
        state_dict["confidence"] = session.state.get("confidence", 0.0)

        state = repair_state(State(state_dict), session.state.to_dict())
        session.update_state(state.to_dict(), cause=StateCause.HUMAN.value)
        session.reset_rounds_since_human()

        # Execute action in shared world (if any)
        action = ""
        if self._world:
            action = state_dict.get("action", "")
            if action:
                self._world.act(session.id, action)

        # Record causal entry
        causal_entry = session.causality.create_entry(
            session_id=session.id,
            round_num=session.round,
            cause=StateCause.HUMAN.value,
            state_before=state_before,
            state_after=state.to_dict(),
            action=action,
            world_room=world_room_before,
            world_tick=world_tick,
            identity_anchor=identity_before,
            drives=drives_before,
            thought_pool=thought_pool_before,
            reasoning=f"人类输入: {human_input[:100]}",
            human_input=human_input,
            nl_response=nl_text,
        )

        # Record Experience for human interaction
        session.add_experience(Experience(
            round=session.round,
            session_id=session.id,
            perception=f"人类说: {human_input[:100]}",
            action=action or "respond",
            observation=nl_text[:200] if nl_text else "",
            meaning="",
            cause=StateCause.HUMAN.value,
            state_before=state_before,
            state_after=state.to_dict(),
            room=world_room_before,
        ))

        # Update drives
        self._update_drives_and_goals(session, state.to_dict(), is_human_interaction=True)

        # Store in long-term memory
        self._memory_manager.store_interrupt(
            session_id=session.id,
            round_num=session.round,
            state_dict=state.to_dict(),
            human_input=human_input,
            ai_response=nl_text,
        )

        # Identity reflection check
        if self._identity_manager.should_reflect(session.round):
            self._perform_identity_reflection(session)

        # Auto-save
        if self._auto_save:
            self._persistence.snapshot(session, self._auto_save_path)

        session.set_status(SessionStatus.IDLE)

        return {
            "type": "interrupt",
            "session_id": session.id,
            "round": session.round,
            "human_input": human_input,
            "nl_response": nl_text,
            "state": state.to_dict(),
            "world_model_updates": wm_updates,
            "identity_anchor": session.identity_anchor,
            "drives": session.drives,
            "thought_pool": session.thought_pool,
            "state_compressed": session.state_compressed.to_dict() if session.state_compressed else None,
            "hypothesis_count": len(session.hypothesis_manager.active_hypotheses),
            "evidence_count": len(session.evidence_manager.all_evidence),
            "messages_sent": len(wm_updates.get("send_messages", [])) + self._action_messages_sent,
            "mailbox_unread": self._communication.get_mailbox(session.id).count() if self._communication else 0,
        }

    # ── Identity Reflection (replaces Fold) ──

    def reflect_identity(self, session_id: str) -> dict[str, Any]:
        """Manually trigger identity reflection.

        Args:
            session_id: Target session ID.

        Returns reflection metadata.
        """
        session = self.get_session(session_id)
        return self._perform_identity_reflection(session)

    # Backward compat alias
    def fold(self, session_id: str) -> dict[str, Any]:
        """Legacy alias for reflect_identity()."""
        return self.reflect_identity(session_id)

    def _perform_identity_reflection(
        self,
        session: AgentSession,
    ) -> dict[str, Any]:
        """Execute delta-based identity reflection.

        Flow:
            1. Ask LLM: "what recent experiences changed me?"
            2. Get IdentityDelta (small change, not full rewrite)
            3. Apply delta to current anchor
            4. Store delta in session
            5. Record causal entry + update maturity

        Args:
            session: The session to reflect on.

        Returns reflection metadata.
        """
        session.set_status(SessionStatus.REFLECTING)

        mods = session.self_modifications

        # Get recent experiences for reflection
        recent_experiences = session.experiences[-8:]

        # Produce IdentityDelta (returns None if no change detected)
        delta = self._identity_manager.reflect(
            current_anchor=session.identity_anchor,
            recent_experiences=recent_experiences,
            demo=self._demo,
            current_drive_params=mods.get("drive_params"),
            current_templates=mods.get("thought_templates"),
        )

        if delta and not self._demo:
            # Apply delta to anchor
            new_anchor = IdentityManager.apply_delta(session.identity_anchor, delta)

            # Store delta in session
            session.add_identity_delta(delta)

            # Extract self-modifications from delta (if any)
            extracted = SelfModificationManager.extract(new_anchor)
            if extracted:
                merged = dict(mods)
                if "drive_params" in extracted:
                    merged["drive_params"] = extracted["drive_params"]
                if "thought_templates" in extracted:
                    merged["thought_templates"] = extracted["thought_templates"]
                session.set_self_modifications(merged)
                import sys
                print(
                    f"  ⚡ [self-mod] Agent modified its own parameters: {extracted}",
                    file=sys.stderr,
                )
        else:
            # No change — keep current anchor but still record the attempt
            new_anchor = session.identity_anchor

        session.set_identity_anchor(new_anchor)
        session.increment_fold_count()

        # Recompute identity maturity (reflection increases it)
        from runtime_kernel.runtime.experience import compute_identity_maturity
        session.set_identity_maturity(
            compute_identity_maturity(
                round_count=session.round,
                experience_count=session.experience_count,
                reflection_count=session.fold_count,
            )
        )

        # Record causal entry for identity reflection
        reflection_text = delta.change if delta else "no_change_detected"
        session.causality.create_entry(
            session_id=session.id,
            round_num=session.round,
            cause=StateCause.REFLECT.value,
            state_before=session.state.to_dict(),
            state_after=session.state.to_dict(),
            action=f"identity_reflection_{session.fold_count}",
            world_room=self._world.agent_position(session.id) if self._world else "",
            identity_anchor=new_anchor,
            drives=session.drives,
            thought_pool=session.thought_pool,
            reasoning=reflection_text,
        )

        # Store reflection in long-term memory
        safe_anchor = dict(new_anchor)
        if safe_anchor.get("recent_reflection") is None:
            safe_anchor["recent_reflection"] = reflection_text
        self._memory_manager.store_reflection(
            session_id=session.id,
            round_num=session.round,
            state_dict=session.state.to_dict(),
            anchor=safe_anchor,
        )

        session.set_status(SessionStatus.IDLE)

        if self._auto_save:
            self._persistence.snapshot(session, self._auto_save_path)

        return {
            "type": "reflect",
            "session_id": session.id,
            "reflection_count": session.fold_count,
            "identity_anchor": new_anchor,
            "identity_delta": delta.to_dict() if delta else None,
            "identity_maturity": session.identity_maturity,
            "round": session.round,
        }

    # ── Drive + Goal Integration ──

    def _update_drives_and_goals(
        self,
        session: AgentSession,
        state_dict: dict,
        is_human_interaction: bool = False,
    ) -> None:
        """Update drive states and regenerate thought pool after a step.

        Args:
            session: The target session.
            state_dict: The state dict produced this step.
            is_human_interaction: Whether this step was a human interruption.
        """
        # Read self-modification overrides
        mods = session.self_modifications
        drive_params = mods.get("drive_params", {})
        templates_override = mods.get("thought_templates", {})

        if is_human_interaction:
            new_drives = DriveModel.on_human_interaction(
                session.drives, params=drive_params,
            )
        else:
            new_drives = DriveModel.after_step(
                drives=session.drives,
                history=session.history,
                state_dict=state_dict,
                rounds_since_human=session.rounds_since_human,
                params=drive_params,
            )

        session.set_drives(new_drives)

        # Regenerate thought pool (with template overrides)
        thoughts = GoalGenerator.generate(
            drives=new_drives,
            identity_anchor=session.identity_anchor,
            state_topic=session.state.topic,
            templates_override=templates_override,
        )
        session.set_thought_pool(thoughts)

    # ── Heartbeat ──

    def _heartbeat_pulse(self, tick_count: int) -> None:
        """Called by HeartbeatManager on each background tick.

        Drives drift, thoughts simmer, world ticks, memory consolidates —
        all without a step() call.

        Args:
            tick_count: The current heartbeat tick number.
        """
        # Advance the shared world
        if self._world:
            self._world.tick()

        if not self._sessions:
            return

        for session_id, session in list(self._sessions.items()):
            try:
                mods = session.self_modifications
                drive_params = mods.get("drive_params", {})
                templates_override = mods.get("thought_templates", {})

                # Drive drift
                drifted = DriveModel.tick(
                    session.drives,
                    rounds_since_human=session.rounds_since_human,
                    params=drive_params,
                )
                session.set_drives(drifted)

                # Refresh thought pool if drives have shifted
                thoughts = GoalGenerator.generate(
                    drives=drifted,
                    identity_anchor=session.identity_anchor,
                    state_topic=session.state.topic,
                    templates_override=templates_override,
                )
                session.set_thought_pool(thoughts)

                # Periodic auto-save during downtime
                if self._auto_save and tick_count % 6 == 0:
                    self._persistence.snapshot(session, self._auto_save_path)

            except Exception as e:
                import sys
                print(
                    f"  [heartbeat] session {session_id}: {e}",
                    file=sys.stderr,
                )

    def start_heartbeat(self) -> None:
        """Start the background heartbeat if not already running."""
        self._heartbeat.start()

    def stop_heartbeat(self) -> None:
        """Stop the background heartbeat."""
        self._heartbeat.stop()

    def get_heartbeat_info(self) -> dict:
        """Return heartbeat status information."""
        return {
            "running": self._heartbeat.is_running,
            "ticks": self._heartbeat.tick_count,
        }

    # ── Snapshot / Restore ──

    def snapshot(self, session_id: str, filepath: Optional[str] = None) -> str:
        """Save a full session snapshot.

        Args:
            session_id: Target session ID.
            filepath: Optional explicit path. Auto-generated if omitted.

        Returns the filepath written.
        """
        session = self.get_session(session_id)
        return self._persistence.snapshot(session, filepath)

    def restore(self, filepath: str) -> AgentSession:
        """Restore a session from a snapshot file.

        The restored session is added to the in-memory store.

        Args:
            filepath: Path to the snapshot JSON file.

        Returns the restored session.
        """
        session = self._persistence.restore(filepath)
        self._sessions[session.id] = session
        session.set_status(SessionStatus.RUNNING)
        return session

    def restore_from_backup(self, filepath: str) -> AgentSession:
        """Restore from a legacy CausalChain backup (auto-save format).

        Handles the older format produced by causal_chain.py.

        Args:
            filepath: Path to the backup JSON file.

        Returns the restored session.
        """
        data = self._persistence.load_json(filepath)
        if data is None:
            raise SessionNotFoundError(f"No backup found at {filepath}")
        # Convert legacy format to session
        session = AgentSession.from_dict(data)
        self._sessions[session.id] = session
        session.set_status(SessionStatus.RUNNING)
        return session

    # ── Translate ──

    def translate(self, session_id: str) -> str:
        """Translate the current session state to natural language.

        Args:
            session_id: Target session ID.

        Returns natural language translation.
        """
        session = self.get_session(session_id)
        if not session.state:
            return "(无状态可翻译)"

        if self._demo:
            return f"[demo] State: topic={session.state.topic}, belief={session.state.belief}, goal={session.state.goal}"

        messages = PromptBuilder.build_translate(session.state)
        response = self._llm.complete(
            messages, temperature=0.5, max_tokens=300
        )
        return response or "(翻译失败)"

    # ── Introspection (manual trigger) ──

    def introspect(self, session_id: str) -> Optional[str]:
        """Manually trigger introspection.

        Args:
            session_id: Target session ID.

        Returns the summary or None.
        """
        session = self.get_session(session_id)
        summary = self._introspector.introspect(session)
        if summary:
            session.add_introspection(summary)
            self._memory_manager.store_introspection(
                session_id=session.id,
                round_num=session.round,
                state_dict=session.state.to_dict(),
                summary=summary,
            )
        return summary

    # ── Scheduler ──

    def start_loop(
        self,
        session_id: str,
        interval: Optional[float] = None,
    ) -> None:
        """Start a background loop that calls step() on interval.

        Args:
            session_id: Session to step.
            interval: Seconds between steps (defaults to step_interval from
                       constructor, which defaults to 60s).
        """
        effective_interval = self._step_interval if interval is None else interval

        def _step():
            try:
                self.step(session_id)
            except Exception as e:
                # Log but don't crash the loop
                import sys
                print(f"  [scheduler] step error: {e}", file=sys.stderr)

        self._scheduler.run_loop(_step, effective_interval)

    def stop_loop(self) -> None:
        """Stop the background scheduler loop."""
        self._scheduler.stop()

    # ── Utilities ──

    def update_llm_config(self, **kwargs: Any) -> None:
        """Update LLM configuration at runtime."""
        self._llm.update_config(**kwargs)

    def health_check(self) -> bool:
        """Check LLM API connectivity."""
        return self._llm.health_check()

    def clear_sessions(self) -> None:
        """Remove all sessions from memory (does not delete snapshots)."""
        self._sessions.clear()

    # ── Memory inspection ──

    def get_memory_records(self, session_id: Optional[str] = None) -> list[dict]:
        """Inspect memory records (for debugging/admin).

        Args:
            session_id: Optional filter by session.

        Returns list of memory records.
        """
        if not self._memory_manager:
            return []
        return self._memory_manager.storage.list_all(session_id)

    def get_identity_anchor(self, session_id: str) -> dict:
        """Get the current identity anchor for a session.

        Args:
            session_id: Target session ID.

        Returns identity anchor dict.
        """
        session = self.get_session(session_id)
        return session.identity_anchor

    def __enter__(self) -> "RuntimeEngine":
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop_loop()
        self.stop_heartbeat()
        # Auto-save all sessions on exit
        if self._auto_save:
            for session in self._sessions.values():
                try:
                    self._persistence.snapshot(session, self._auto_save_path)
                except Exception:
                    pass
