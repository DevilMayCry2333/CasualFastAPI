"""
session — AgentSession: the runtime state of a single running Agent.

A Session holds:
  - Current State (working memory)
  - Compressed state (from Fold)
  - History of state transitions
  - Human interactions
  - Introspection records
  - Statistics (round count, fold count, etc.)

Session does NOT call LLM.
Session does NOT build prompts.
Session is purely a data container + serialization.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

from runtime_kernel.runtime.causality import CausalEntry, CausalityManager
from runtime_kernel.runtime.cognitive import (
    KnowledgeModel,
    SelfModel,
    SocialModel,
    TheoryOfMind,
    WorkingMemory,
    WorldModel as CognitiveWorldModel,
)
from runtime_kernel.runtime.communication import CommunicationManager, Mailbox, Message
from runtime_kernel.runtime.drive import DriveModel
from runtime_kernel.runtime.evidence import Evidence, EvidenceManager
from runtime_kernel.runtime.experience import Experience, IdentityDelta, compute_identity_maturity
from runtime_kernel.runtime.hypothesis import Hypothesis, HypothesisManager
from runtime_kernel.runtime.models import (
    DEFAULT_IDENTITY_ANCHOR,
    SessionStatus,
    StateCause,
    HistoryEntry,
    InteractionEntry,
    IntrospectionEntry,
)
from runtime_kernel.runtime.shared_memory import SharedKnowledge
from runtime_kernel.runtime.state import State
from runtime_kernel.runtime.exceptions import StateValidationError


class AgentSession:
    """Represents a running Agent.

    All mutation goes through RuntimeEngine.
    External code reads; only RuntimeEngine writes.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        state: Optional[State | dict] = None,
        causality_manager: Optional[CausalityManager] = None,
    ) -> None:
        self._id: str = session_id or uuid.uuid4().hex[:12]

        self._state: State = State(state) if state else State()
        self._state_compressed: State = State()

        # Identity anchor (self-reflection, not history compression)
        self._identity_anchor: dict = dict(DEFAULT_IDENTITY_ANCHOR)

        # Drive states (curiosity, boredom, belonging)
        self._drives: dict[str, float] = DriveModel.initial()

        # Thought pool (candidate goals from GoalGenerator)
        self._thought_pool: list[dict] = []

        # Self-modifications (recursive self-improvement overrides)
        self._self_modifications: dict = {}

        # Causal chain — every state transition is a traceable causal edge
        self._causality: CausalityManager = causality_manager or CausalityManager()

        # Hypothesis manager — world model hypotheses lifecycle
        self._hypothesis_manager: HypothesisManager = HypothesisManager()

        # Evidence manager — all observations become evidence first
        self._evidence_manager: EvidenceManager = EvidenceManager()

        # Cognitive models (v4)
        self._self_model: SelfModel = SelfModel()
        self._world_model_cog: CognitiveWorldModel = CognitiveWorldModel()
        self._social_model: SocialModel = SocialModel()
        self._knowledge_model: KnowledgeModel = KnowledgeModel(
            hypothesis_manager=self._hypothesis_manager,
            evidence_manager=self._evidence_manager,
        )
        self._theory_of_mind: TheoryOfMind = TheoryOfMind()
        self._working_memory: WorkingMemory = WorkingMemory()

        # Communication layer
        self._communication: Optional[CommunicationManager] = None

        # Shared knowledge reference
        self._shared_knowledge: Optional[SharedKnowledge] = None

        # Experiences — the atomic units of agent life (replaces flat history)
        self._experiences: list[Experience] = []

        # Identity maturity (0.0-1.0) — how formed the agent's sense of self is
        self._identity_maturity: float = 0.0

        # Identity deltas — small changes that accumulate into identity
        self._identity_deltas: list[IdentityDelta] = []

        self._history: list[HistoryEntry] = []
        self._interactions: list[InteractionEntry] = []
        self._introspections: list[IntrospectionEntry] = []

        self._status: SessionStatus = SessionStatus.INITIALIZING
        self._round: int = 0
        self._rounds_since_human: int = 0
        self._fold_count: int = 0

        self._created_at: float = time.time()
        self._updated_at: float = time.time()

        # Statistics (introspection data)
        self._statistics: dict[str, Any] = {}

    # ── Read-only properties ──

    @property
    def id(self) -> str:
        return self._id

    @property
    def state(self) -> State:
        return self._state

    @property
    def state_compressed(self) -> State:
        return self._state_compressed

    @property
    def identity_anchor(self) -> dict:
        """The agent's persistent identity anchor (self-reflection)."""
        return dict(self._identity_anchor)

    @property
    def identity_maturity(self) -> float:
        """How formed the agent's sense of self is (0.0-1.0)."""
        return self._identity_maturity

    @property
    def identity_deltas(self) -> list[IdentityDelta]:
        """Identity deltas accumulated through reflection."""
        return list(self._identity_deltas)

    @property
    def experiences(self) -> list[Experience]:
        """All experiences this agent has lived."""
        return list(self._experiences)

    @property
    def experience_count(self) -> int:
        """Total number of experiences."""
        return len(self._experiences)

    @property
    def drives(self) -> dict[str, float]:
        """Current drive states (curiosity, boredom, belonging)."""
        return dict(self._drives)

    @property
    def thought_pool(self) -> list[dict]:
        """Current candidate thoughts/goals."""
        return list(self._thought_pool)

    @property
    def self_model(self) -> SelfModel:
        """Cognitive self-model (identity, beliefs, goals, drives)."""
        return self._self_model

    @property
    def world_model_cog(self) -> CognitiveWorldModel:
        """Cognitive world model (places, objects, events, rules)."""
        return self._world_model_cog

    @property
    def social_model(self) -> SocialModel:
        """Cognitive social model (trust, cooperation with other agents)."""
        return self._social_model

    @property
    def knowledge_model(self) -> KnowledgeModel:
        """Cognitive knowledge model (facts, hypotheses, evidence)."""
        return self._knowledge_model

    @property
    def theory_of_mind(self) -> TheoryOfMind:
        """Theory of mind (beliefs about other agents' beliefs)."""
        return self._theory_of_mind

    @property
    def working_memory(self) -> WorkingMemory:
        """Working memory (current cognitive focus)."""
        return self._working_memory

    @property
    def causality(self) -> CausalityManager:
        """Causal chain manager for this session."""
        return self._causality

    @property
    def hypothesis_manager(self) -> HypothesisManager:
        """Hypothesis lifecycle manager for world model."""
        return self._hypothesis_manager

    @property
    def evidence_manager(self) -> EvidenceManager:
        """Evidence collection manager."""
        return self._evidence_manager

    @property
    def communication(self) -> Optional[CommunicationManager]:
        """Communication layer reference (set by engine)."""
        return self._communication

    @property
    def shared_knowledge(self) -> Optional[SharedKnowledge]:
        """Shared knowledge reference (set by engine)."""
        return self._shared_knowledge

    @property
    def self_modifications(self) -> dict:
        """Current self-modification overrides (drive/template params)."""
        return dict(self._self_modifications)

    @property
    def history(self) -> list[HistoryEntry]:
        return list(self._history)

    @property
    def interactions(self) -> list[InteractionEntry]:
        return list(self._interactions)

    @property
    def introspections(self) -> list[IntrospectionEntry]:
        return list(self._introspections)

    @property
    def status(self) -> SessionStatus:
        return self._status

    @property
    def round(self) -> int:
        return self._round

    @property
    def rounds_since_human(self) -> int:
        return self._rounds_since_human

    @property
    def fold_count(self) -> int:
        return self._fold_count

    @property
    def statistics(self) -> dict:
        return dict(self._statistics)

    @property
    def depth(self) -> int:
        return self._round

    @property
    def created_at(self) -> float:
        return self._created_at

    @property
    def updated_at(self) -> float:
        return self._updated_at

    # ── Mutations (called by RuntimeEngine only) ──

    def set_status(self, status: SessionStatus) -> None:
        self._status = status
        self._updated_at = time.time()

    def update_state(self, state_dict: dict, cause: str = "self") -> dict:
        """Record a new state, log the transition, increment round.

        Args:
            state_dict: The new state values.
            cause: Origin — init / self / human / fold / environment.

        Returns transition metadata.
        """
        self._state = State(state_dict)
        self._round += 1
        transition = {
            "state": self._state.to_dict(),
            "cause": cause,
            "round": self._round,
            "time": time.time(),
        }
        self._history.append(transition)
        self._updated_at = time.time()
        return {"id": self._round - 1, "content": self._state.to_dict()}

    def set_compressed_state(self, state: State | dict) -> None:
        if isinstance(state, dict):
            state = State(state)
        self._state_compressed = state
        self._updated_at = time.time()

    def set_identity_anchor(self, anchor: dict) -> None:
        """Update the identity anchor (called by IdentityManager)."""
        self._identity_anchor = dict(anchor)
        self._updated_at = time.time()

    def set_drives(self, drives: dict[str, float]) -> None:
        """Update drive states (called by DriveModel + Heartbeat)."""
        self._drives = dict(drives)
        self._updated_at = time.time()

    def set_thought_pool(self, thoughts: list[dict]) -> None:
        """Update thought pool (called by GoalGenerator)."""
        self._thought_pool = list(thoughts)
        self._updated_at = time.time()

    def add_experience(self, exp: Experience) -> None:
        """Record an experience and update identity maturity."""
        self._experiences.append(exp)
        self._identity_maturity = compute_identity_maturity(
            round_count=self._round,
            experience_count=len(self._experiences),
            reflection_count=self._fold_count,
        )
        self._updated_at = time.time()

    def add_identity_delta(self, delta: IdentityDelta) -> None:
        """Record an identity delta (produced by reflection)."""
        self._identity_deltas.append(delta)
        self._updated_at = time.time()

    def set_identity_maturity(self, maturity: float) -> None:
        """Manually set identity maturity."""
        self._identity_maturity = max(0.0, min(1.0, maturity))
        self._updated_at = time.time()

    def set_communication(self, cm: CommunicationManager) -> None:
        """Set the communication layer reference."""
        self._communication = cm

    def set_shared_knowledge(self, sk: SharedKnowledge) -> None:
        """Set the shared knowledge reference."""
        self._shared_knowledge = sk

    def set_self_modifications(self, mods: dict) -> None:
        """Update self-modification overrides (called by SelfModificationManager).

        Args:
            mods: Dict with optional keys "drive_params" and "thought_templates".
        """
        self._self_modifications = dict(mods)
        self._updated_at = time.time()

    def add_interaction(self, human_input: str, ai_response: str) -> dict:
        interaction = {
            "id": len(self._interactions),
            "human": human_input,
            "ai": ai_response,
            "after_round": self._round,
            "time": time.time(),
        }
        self._interactions.append(interaction)
        self._updated_at = time.time()
        return interaction

    def add_introspection(self, summary: str) -> dict:
        entry = {
            "round": self._round,
            "summary": summary,
            "time": time.time(),
        }
        self._introspections.append(entry)
        self._updated_at = time.time()
        return entry

    def increment_rounds_since_human(self) -> None:
        self._rounds_since_human += 1

    def reset_rounds_since_human(self) -> None:
        self._rounds_since_human = 0

    def increment_fold_count(self) -> None:
        self._fold_count += 1
        self._updated_at = time.time()

    def set_statistics(self, stats: dict) -> None:
        self._statistics = stats
        self._updated_at = time.time()

    # ── Serialization ──

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "state": self._state.to_dict(),
            "state_compressed": self._state_compressed.to_dict(),
            "identity_anchor": dict(self._identity_anchor),
            "drives": dict(self._drives),
            "thought_pool": list(self._thought_pool),
            "self_modifications": dict(self._self_modifications),
            "causal_chain": self._causality.to_dict(self._id) if self._causality else [],
            "hypotheses": self._hypothesis_manager.to_dict(),
            "evidence": self._evidence_manager.to_dict(),
            "cognitive": {
                "self_model": self._self_model.to_dict(),
                "world_model_cog": self._world_model_cog.to_dict(),
                "social_model": self._social_model.to_dict(),
                "knowledge_model": self._knowledge_model.to_dict(),
                "theory_of_mind": self._theory_of_mind.to_dict(),
                "working_memory": self._working_memory.to_dict(),
            },
            "experiences": [e.to_dict() for e in self._experiences[-50:]],
            "identity_maturity": self._identity_maturity,
            "identity_deltas": [d.to_dict() for d in self._identity_deltas[-30:]],
            "round": self._round,
            "rounds_since_human": self._rounds_since_human,
            "fold_count": self._fold_count,
            "status": self._status.value,
            "interactions": [
                {k: v for k, v in x.items() if k != "time"}
                for x in self._interactions[-50:]
            ],
            "history": [
                {k: v for k, v in h.items() if k != "time"}
                for h in self._history[-200:]
            ],
            "introspections": [
                {k: v for k, v in x.items() if k != "time"}
                for x in self._introspections[-20:]
            ],
            "statistics": self._statistics,
            "created_at": self._created_at,
            "updated_at": self._updated_at,
            "version": 5,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "AgentSession":
        session = cls(session_id=d.get("id"))
        # state
        state_raw = d.get("state", {})
        if isinstance(state_raw, str):
            try:
                session._state = State(json.loads(state_raw) if state_raw else {})
            except (json.JSONDecodeError, TypeError):
                session._state = State({"raw": state_raw})
        elif isinstance(state_raw, dict):
            session._state = State(state_raw)
        # compressed
        sc_raw = d.get("state_compressed", {})
        if isinstance(sc_raw, str):
            try:
                session._state_compressed = State(json.loads(sc_raw) if sc_raw else {})
            except (json.JSONDecodeError, TypeError):
                session._state_compressed = State()
        elif isinstance(sc_raw, dict):
            session._state_compressed = State(sc_raw)
        # identity anchor (v2+), fall back to defaults for older snapshots
        ia_raw = d.get("identity_anchor")
        if isinstance(ia_raw, dict) and ia_raw:
            session._identity_anchor = dict(ia_raw)
        else:
            session._identity_anchor = dict(DEFAULT_IDENTITY_ANCHOR)
        # drives (v3+)
        drives_raw = d.get("drives")
        if isinstance(drives_raw, dict) and drives_raw:
            session._drives = dict(drives_raw)
        else:
            session._drives = DriveModel.initial()
        # thought pool (v3+)
        tp_raw = d.get("thought_pool")
        if isinstance(tp_raw, list):
            session._thought_pool = list(tp_raw)
        else:
            session._thought_pool = []
        # self-modifications (v4+)
        sm_raw = d.get("self_modifications")
        if isinstance(sm_raw, dict):
            session._self_modifications = dict(sm_raw)
        else:
            session._self_modifications = {}
        # causal chain (v3+)
        cc_raw = d.get("causal_chain")
        if isinstance(cc_raw, list) and cc_raw:
            session._causality.from_dict(session._id, cc_raw)
        # hypotheses (v5+)
        hyp_raw = d.get("hypotheses")
        if isinstance(hyp_raw, list):
            session._hypothesis_manager = HypothesisManager.from_dict(hyp_raw)
        # evidence (v5+)
        ev_raw = d.get("evidence")
        if isinstance(ev_raw, list):
            session._evidence_manager = EvidenceManager.from_dict(ev_raw)
        # cognitive models (v6+)
        cog_raw = d.get("cognitive", {})
        if isinstance(cog_raw, dict):
            if "self_model" in cog_raw:
                session._self_model = SelfModel.from_dict(cog_raw["self_model"])
            if "world_model_cog" in cog_raw:
                session._world_model_cog = CognitiveWorldModel.from_dict(cog_raw["world_model_cog"])
            if "social_model" in cog_raw:
                session._social_model = SocialModel.from_dict(cog_raw["social_model"])
            if "knowledge_model" in cog_raw:
                session._knowledge_model = KnowledgeModel.from_dict(cog_raw["knowledge_model"])
            if "theory_of_mind" in cog_raw:
                session._theory_of_mind = TheoryOfMind.from_dict(cog_raw["theory_of_mind"])
            if "working_memory" in cog_raw:
                session._working_memory = WorkingMemory.from_dict(cog_raw["working_memory"])
        # Re-link knowledge model to hypothesis/evidence managers
        session._knowledge_model._hypotheses = session._hypothesis_manager
        session._knowledge_model._evidence = session._evidence_manager
        # experiences (v3+)
        exp_raw = d.get("experiences")
        if isinstance(exp_raw, list):
            session._experiences = [Experience.from_dict(e) for e in exp_raw]
        # identity maturity (v3+)
        session._identity_maturity = float(d.get("identity_maturity", 0.0))
        # identity deltas (v3+)
        delta_raw = d.get("identity_deltas")
        if isinstance(delta_raw, list):
            session._identity_deltas = [IdentityDelta.from_dict(d) for d in delta_raw]
        # counters
        session._round = int(d.get("round", 0))
        session._rounds_since_human = int(d.get("rounds_since_human", 0))
        session._fold_count = int(d.get("fold_count", 0))
        session._status = SessionStatus(d.get("status", SessionStatus.IDLE.value))
        # arrays
        session._interactions = [{**x, "time": 0.0} for x in d.get("interactions", [])]
        raw_history = d.get("history", [])
        session._history = [{**h, "time": 0.0} for h in raw_history] if raw_history else []
        raw_ints = d.get("introspections", [])
        session._introspections = [{**x, "time": 0.0} for x in raw_ints] if raw_ints else []
        session._statistics = d.get("statistics", {})
        # timestamps
        session._created_at = d.get("created_at", time.time())
        session._updated_at = d.get("updated_at", time.time())
        return session

    @classmethod
    def from_json(cls, text: str) -> "AgentSession":
        return cls.from_dict(json.loads(text))

    def __repr__(self) -> str:
        return (
            f"AgentSession(id={self._id!r}, round={self._round}, "
            f"status={self._status.value!r})"
        )
