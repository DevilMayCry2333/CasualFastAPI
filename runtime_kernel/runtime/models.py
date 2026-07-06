"""
models — Core data types and enumerations for the Runtime Engine.
"""

from __future__ import annotations

import enum
from typing import Any


class StateCause(str, enum.Enum):
    """Origin of a state transition."""
    INIT = "init"
    SELF = "self"
    HUMAN = "human"
    FOLD = "fold"
    REFLECT = "reflect"
    ENVIRONMENT = "environment"
    EVIDENCE = "evidence"        # New: belief updated via evidence pipeline
    HYPOTHESIS = "hypothesis"    # New: hypothesis proposed/revised/discarded
    MESSAGE = "message"          # New: state changed due to received message
    SHARED = "shared"            # New: shared knowledge update


class SessionStatus(str, enum.Enum):
    """Lifecycle status of an AgentSession."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    FOLDING = "folding"       # Deprecated: kept for backward compat
    REFLECTING = "reflecting"
    WAITING_HUMAN = "waiting_human"  # Agent asked a question, waiting for answer
    IDLE = "idle"
    ERROR = "error"
    TERMINATED = "terminated"


class LLMProvider(str, enum.Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    OPENAI_COMPATIBLE = "openai_compatible"


class MessageType(str, enum.Enum):
    """Types of inter-agent messages.

    NOT chat. These are causal events that travel through the world.
    """
    OBSERVATION = "observation"       # "the plant grew today"
    QUESTION = "question"             # "why did the plant stop growing?"
    ANSWER = "answer"                 # "the soil is dry"
    HYPOTHESIS = "hypothesis"         # "i think the room has a hidden door"
    PLAN = "plan"                     # "i will investigate the garden"
    REQUEST = "request"              # "please help move the tools"
    WARNING = "warning"              # "the garden needs water"
    EVENT = "event"                   # "someone entered the lab"
    REPORT = "report"                 # "survey of the east wing complete"
    SHARE_MEMORY = "share_memory"     # sharing an evidence/hypothesis
    BROADCAST = "broadcast"           # world event notification
    INQUIRY = "inquiry"               # inquiry/collaboration request
    GREETING = "greeting"             # greeting another agent
    SUGGESTION = "suggestion"         # suggesting a course of action
    RESPONSE = "response"             # general response
    INFORMATION = "information"       # sharing neutral information


class MemoryRecordType(str, enum.Enum):
    """Types of records stored in long-term memory."""
    STATE = "state"
    HUMAN_INTERRUPT = "human_interrupt"
    REFLECTION = "reflection"
    INTROSPECTION = "introspection"
    EVIDENCE = "evidence"           # New: observations as evidence
    HYPOTHESIS = "hypothesis"       # New: hypothesis lifecycle records
    CONTRADICTION = "contradiction"  # New: contradictions between evidence
    MESSAGE = "message"             # New: inter-agent messages as memory
    SHARED_KNOWLEDGE = "shared_knowledge"  # New: shared knowledge records


# Default identity anchor (used when no anchor has been established)
# Minimal by design — identity should emerge from experience, not be prescribed.
DEFAULT_IDENTITY_ANCHOR: dict[str, Any] = {
    "identity": "unknown",
    "core_goal": "observe_current_environment",
    "worldview": None,
    "stable_values": [],
    "recent_reflection": None,
    "confidence": 0.0,
}

# Identity maturity thresholds
IDENTITY_MATURITY_MIN = 0.0        # Newborn: no sense of self
IDENTITY_MATURITY_EARLY = 0.2      # Starting to form preferences
IDENTITY_MATURITY_MID = 0.5        # Identity influences decisions
IDENTITY_MATURITY_LATE = 0.8       # Identity is primary behavior driver
IDENTITY_MATURITY_MAX = 1.0        # Stable personality formed

# How much each factor contributes to identity maturity
IDENTITY_MATURITY_ROUND_WEIGHT = 0.25     # Rounds lived
IDENTITY_MATURITY_EXPERIENCE_WEIGHT = 0.35  # Experiences accumulated
IDENTITY_MATURITY_REFLECTION_WEIGHT = 0.40  # Reflections performed

# Round thresholds for identity maturity calculation
IDENTITY_MATURITY_ROUND_DENOM = 150      # ~150 rounds to saturate round component
IDENTITY_MATURITY_EXPERIENCE_DENOM = 80  # ~80 experiences to saturate experience component
IDENTITY_MATURITY_REFLECTION_DENOM = 20  # ~20 reflections to saturate reflection component

# Constants
IDENTITY_REFLECTION_INTERVAL = 5    # rounds between identity reflections
FOLD_INTERVAL = 5                    # kept for backward compat (maps to reflection interval)
LOOP_WINDOW = 8
PROTECTED_FIELDS = {"name", "core_belief", "identity"}
REQUIRED_STATE_KEYS = {"topic", "belief", "goal"}
REQUIRED_WORLD_MODEL_KEYS = {"world_model", "hypotheses", "evidence", "open_questions", "uncertainties", "confidence"}
CAUSE_LABELS: dict[str, str] = {
    "init": "⚡",
    "self": "⟳",
    "human": "◈",
    "fold": "⬡",
    "reflect": "◈",
    "environment": "⊙",
    "evidence": "◇",
    "hypothesis": "⊕",
    "message": "✉",
    "shared": "⊚",
}

# Message / Communication constants
MESSAGE_TYPES = tuple(mt.value for mt in MessageType)
# Extended type list for action field parsing
EXTENDED_MESSAGE_TYPES = MESSAGE_TYPES + ("inquiry", "greeting", "suggestion", "response", "information")
MAILBOX_MAX_SIZE = 20          # max messages kept per agent mailbox
SHARED_KNOWLEDGE_CONSENSUS_MIN = 2   # min agents supporting for knowledge promotion
SHARED_KNOWLEDGE_CANDIDATE_MAX = 50   # max candidates before cleanup
WORLD_EVENT_BROADCAST_INTERVAL = 3     # ticks between automated world event broadcasts

# World model defaults
DEFAULT_WORLD_MODEL: dict[str, str] = {}
DEFAULT_HYPOTHESIS_STATUS = ("proposed", "testing", "supported", "contradicted", "revised", "discarded")

# Hypothesis lifecycle thresholds
HYPOTHESIS_MIN_EVIDENCE_FOR_CONFIRM = 3
HYPOTHESIS_MIN_CONTRADICTION_FOR_DISCARD = 2
HYPOTHESIS_MAX_ACTIVE = 5

# Belief update thresholds
BELIEF_UPDATE_EVIDENCE_THRESHOLD = 3   # min supporting evidence count to update belief
BELIEF_CONTRADICTION_RATIO = 0.5       # if contradiction/support > this, block belief update


# History entry type
HistoryEntry = dict[str, Any]
InteractionEntry = dict[str, Any]
IntrospectionEntry = dict[str, Any]
