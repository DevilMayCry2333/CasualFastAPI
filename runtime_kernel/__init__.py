"""
runtime_kernel — Agent Runtime Kernel

A pure Python library for managing Agent lifecycle.
Independent of HTTP, WebSocket, CLI, or any web framework.

Usage:
    engine = RuntimeEngine(llm_config={...})
    session = engine.create_session()
    engine.step(session.id)
"""

from runtime_kernel.runtime.causality import CausalEntry, CausalityManager
from runtime_kernel.runtime.cognitive import (
    KnowledgeModel,
    SelfModel,
    SocialModel,
    TheoryOfMind,
    WorkingMemory,
    WorldModel as CognitiveWorldModel,
    filter_events as cognitive_filter_events,
    perceive,
)
from runtime_kernel.runtime.communication import CommunicationManager, EventBus, Mailbox, Message, MessageType
from runtime_kernel.runtime.evidence import Evidence, EvidenceManager
from runtime_kernel.runtime.experience import Experience, IdentityDelta, compute_identity_maturity
from runtime_kernel.runtime.engine import RuntimeEngine
from runtime_kernel.runtime.hypothesis import Hypothesis, HypothesisManager, HypothesisStatus
from runtime_kernel.runtime.session import AgentSession
from runtime_kernel.runtime.shared_memory import SharedKnowledge, KnowledgeEntry
from runtime_kernel.runtime.state import State
from runtime_kernel.runtime.embedding import EmbeddingClient
from runtime_kernel.runtime.identity_manager import IdentityManager
from runtime_kernel.runtime.memory_manager import MemoryManager
from runtime_kernel.runtime.memory_storage import (
    MemoryStorage,
    InMemoryMemoryStorage,
)
from runtime_kernel.runtime.drive import DriveModel
from runtime_kernel.runtime.evolution import EvolutionEngine, EvolutionSignals, RuntimeParameters
from runtime_kernel.runtime.goal_generator import GoalGenerator
from runtime_kernel.runtime.runtime_statistics import AgentStats, RuntimeStatistics
from runtime_kernel.runtime.heartbeat import HeartbeatManager
from runtime_kernel.runtime.environment import VirtualEnvironment
from runtime_kernel.runtime.self_modification import SelfModificationManager
from runtime_kernel.runtime.action import Action, ActionExecutor, Capability, CapabilityAdapter, HumanAdapter, Observation, SearchAdapter
from runtime_kernel.runtime.agent_events import AgentEvent, AgentEventBus
from runtime_kernel.runtime.mcp import MCPConfig
from runtime_kernel.runtime.policy_engine import PolicyEngine, OutcomeEvaluator
from runtime_kernel.runtime.scientific import (
    CausalEdge,
    CycleSummary,
    ExperimentResult,
    ExperimentStep,
    Hypothesis as ScientificHypothesis,
    ScientificLoop,
    ScientificQuestion,
)
from runtime_kernel.runtime.models import (
    DEFAULT_IDENTITY_ANCHOR,
    StateCause,
    SessionStatus,
    LLMProvider,
    MemoryRecordType,
    MessageType,
)
from runtime_kernel.runtime.exceptions import (
    RuntimeError,
    SessionNotFoundError,
    LLMError,
    StateValidationError,
    EmbeddingError,
    MemoryError,
)

__all__ = [
    "CausalEntry",
    "CausalityManager",
    "CognitiveWorldModel",
    "cognitive_filter_events",
    "CommunicationManager",
    "EventBus",
    "EventBus",
    "Mailbox",
    "Message",
    "Evidence",
    "EvidenceManager",
    "Experience",
    "IdentityDelta",
    "compute_identity_maturity",
    "RuntimeEngine",
    "Hypothesis",
    "HypothesisManager",
    "HypothesisStatus",
    "AgentSession",
    "AgentStats",
    "EvolutionEngine",
    "EvolutionSignals",
    "KnowledgeEntry",
    "KnowledgeModel",
    "Mailbox",
    "Message",
    "MessageType",
    "perceive",
    "SelfModel",
    "SharedKnowledge",
    "SocialModel",
    "State",
    "TheoryOfMind",
    "WorkingMemory",
    "EmbeddingClient",
    "IdentityManager",
    "MemoryManager",
    "MemoryStorage",
    "InMemoryMemoryStorage",
    "DriveModel",
    "GoalGenerator",
    "HeartbeatManager",
    "SelfModificationManager",
    "VirtualEnvironment",
    "DEFAULT_IDENTITY_ANCHOR",
    "StateCause",
    "SessionStatus",
    "LLMProvider",
    "MemoryRecordType",
    "MessageType",
    "RuntimeError",
    "RuntimeParameters",
    "RuntimeStatistics",
    "SessionNotFoundError",
    "LLMError",
    "StateValidationError",
    "EmbeddingError",
    "MemoryError",
    "Action",
    "ActionExecutor",
    "AgentEvent",
    "AgentEventBus",
    "Capability",
    "CapabilityAdapter",
    "HumanAdapter",
    "MCPConfig",
    "OutcomeEvaluator",
    "CausalEdge",
    "CycleSummary",
    "ExperimentResult",
    "ExperimentStep",
    "OutcomeEvaluator",
    "PolicyEngine",
    "ScientificHypothesis",
    "ScientificLoop",
    "ScientificQuestion",
    "Observation",
    "SearchAdapter",
]
