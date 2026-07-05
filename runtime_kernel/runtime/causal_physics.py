"""
causal_physics — Causal state dynamical system.

Core equation:
    State(t+1) = Integrate(CausalForces, State(t))

Where:
    ΔState = F_memory + F_identity + F_world + F_llm
    State(t+1) = State(t) + ΔState

LLM is a "Causal Vector Estimator" — it estimates force vectors,
not final states. The system integrates forces to produce the next state.

Freedom = number of causally reachable future states under constraints.

Three constraint layers enforce this:
  1. Anti-Delusion Layer — blocks un-caused self-narrative
  2. Semantic Escalation Barrier — prevents state→consciousness mapping
  3. Semantic Delay — requires N causal steps before self-interpretation
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Force vector ──


@dataclass
class CausalVector:
    """A force vector in causal state space.

    The LLM outputs this, not a full state. The engine integrates it.

    Fields:
        delta: Dict of field_name -> new_value for fields to change.
               Empty dict means "no change requested" — state keeps current value.
        strength: How strongly this vector pushes (0.0-1.0).
                  At strength=0.7 and identity_mass=0.3, the change is 0.7*(1-0.3)=0.49 effective.
        source: Which causal force produced this ("memory", "identity", "world", "llm", "curiosity").
        action: Optional world action associated with this vector.
        confidence: How confident the estimator is (0.0-1.0).
    """
    delta: dict[str, str] = field(default_factory=dict)
    strength: float = 0.5
    source: str = "llm"
    action: str = ""
    confidence: float = 0.5


# ── Force computation ──


def compute_memory_force(
    current_state: dict,
    recent_experiences: list[Any],
) -> CausalVector:
    """Compute F_memory — the pull from past experiences.

    Measures similarity between current state and recent experience patterns.
    Higher similarity → stronger pull toward continuing that trajectory.

    Args:
        current_state: Current state dict.
        recent_experiences: Recent Experience objects.

    Returns:
        CausalVector representing memory's influence.
    """
    if not recent_experiences:
        return CausalVector(source="memory", strength=0.0)

    # Score topic/belief/goal continuity with recent experiences
    topic = str(current_state.get("topic", ""))
    belief = str(current_state.get("belief", ""))
    goal = str(current_state.get("goal", ""))

    topic_overlaps = 0
    belief_overlaps = 0
    for exp in recent_experiences[-5:]:
        after = getattr(exp, "state_after", {}) if hasattr(exp, "state_after") else {}
        if isinstance(after, dict):
            et = str(after.get("topic", ""))
            eb = str(after.get("belief", ""))
            if et and et == topic:
                topic_overlaps += 1
            if eb and eb == belief:
                belief_overlaps += 1

    # Memory force: stronger when trajectory is consistent
    n = min(len(recent_experiences), 5)
    memory_strength = min(1.0, (topic_overlaps + belief_overlaps) / max(n * 2, 1))

    return CausalVector(
        source="memory",
        strength=memory_strength * 0.6,  # Memory is one force among many
    )


def compute_identity_force(
    current_state: dict,
    identity_mass: float,
    llm_delta: dict[str, str],
    llm_strength: float,
) -> CausalVector:
    """Compute F_identity — resistance to change.

    Identity mass creates inertia: higher mass = more resistance to LLM's
    proposed changes. This prevents erratic jumps.

    Args:
        current_state: Current state dict.
        identity_mass: Resistance coefficient (0.0-1.0).
        llm_delta: The delta proposed by the LLM estimator.
        llm_strength: The strength of the LLM's proposed change.

    Returns:
        CausalVector representing identity's resistance (damping force).
    """
    if identity_mass <= 0 or not llm_delta:
        return CausalVector(source="identity", strength=0.0, delta={})

    # Identity inertia: each proposed change is resisted proportional to mass
    # Effective delta = proposed × (1 - identity_mass × resistance_factor)
    resistance = identity_mass * 0.5  # max 50% resistance at identity_mass=1.0
    effective_strength = llm_strength * (1.0 - resistance)

    if effective_strength <= 0:
        # Identity completely blocks this change
        return CausalVector(
            source="identity",
            strength=identity_mass,
            delta={k: str(current_state.get(k, "")) for k in llm_delta},
        )

    # Partial resistance — dampen but don't block
    dampened_delta = dict(llm_delta)
    return CausalVector(
        source="identity",
        strength=resistance,
        delta=dampened_delta,
    )


def compute_world_force(env_context: str) -> CausalVector:
    """Compute F_world — environmental constraints.

    The environment provides valid action possibilities. This force
    biases state changes toward actions the environment supports.

    Args:
        env_context: World context string from VirtualEnvironment.

    Returns:
        CausalVector representing world's influence.
    """
    if not env_context:
        return CausalVector(source="world", strength=0.0)

    # Extract room name and available exits/objects from context
    world_strength = 0.3  # World has modest influence
    return CausalVector(
        source="world",
        strength=world_strength,
    )


def compute_llm_force(
    delta: dict[str, str],
    strength: float,
    action: str,
    source: str = "llm",
    confidence: float = 0.5,
) -> CausalVector:
    """Package LLM output as a causal force vector.

    The LLM estimates: "how should state change?" not "what is the new state?"

    Args:
        delta: Proposed field changes (key -> new_value).
        strength: How strongly this change is pushed (0.0-1.0).
        action: World action to take.
        source: Which drive/dynamic produced this.
        confidence: LLM's confidence in this estimate.

    Returns:
        CausalVector ready for integration.
    """
    return CausalVector(
        delta=delta,
        strength=strength,
        source=source,
        action=action,
        confidence=confidence,
    )


# ── State integration ──


def integrate_state(
    current_state: dict,
    llm_vector: CausalVector,
    memory_vector: Optional[CausalVector] = None,
    identity_vector: Optional[CausalVector] = None,
    world_vector: Optional[CausalVector] = None,
    identity_mass: float = 0.0,
) -> dict:
    """Integrate causal forces to produce the next state.

    Core equation:
        effective_force = F_llm + F_memory + F_identity + F_world
        State(t+1) = Apply(State(t), effective_force)

    Each force proposes delta changes. The effective change is the
    strength-weighted combination, dampened by identity mass.

    Args:
        current_state: Current state dict.
        llm_vector: Force vector from LLM estimation.
        memory_vector: Memory pull force.
        identity_vector: Identity inertia force.
        world_vector: World constraint force.
        identity_mass: Resistance coefficient (0.0-1.0).

    Returns:
        New state dict.
    """
    new_state = dict(current_state)

    if not llm_vector or not llm_vector.delta:
        # No change proposed — state stays the same
        return new_state

    # 1. Compute effective strength: dampened by identity mass
    raw_strength = llm_vector.strength
    effective_strength = raw_strength * (1.0 - identity_mass * 0.5)

    # 2. Memory boost: consistent trajectory amplifies change
    if memory_vector and memory_vector.strength > 0.3:
        effective_strength = min(1.0, effective_strength + memory_vector.strength * 0.2)

    # 3. Apply each proposed delta
    for field, new_value in llm_vector.delta.items():
        if not new_value or new_value == "unknown":
            continue

        current_value = str(current_state.get(field, ""))

        # Only change if:
        # a) The field is different from current
        # b) The force is strong enough to overcome identity mass
        if new_value != current_value and effective_strength > identity_mass * 0.3:
            new_state[field] = new_value

    # 4. Apply action if present
    if llm_vector.action:
        new_state["action"] = llm_vector.action
    elif "action" in new_state:
        # Carry forward or clear action
        pass  # Keep existing action if any

    return new_state


def compute_reachable_states(
    current_state: dict,
    identity_mass: float,
    n_variations: int = 3,
) -> list[dict]:
    """Estimate reachable future states under causal constraints.

    This is the system's "freedom metric":
    Freedom = |Reachable states under causal constraints|

    Args:
        current_state: Current state dict.
        identity_mass: How resistant the system is to change.
        n_variations: How many variations to generate.

    Returns:
        List of possible next-state projections (sampled variations).
        Higher count = more freedom.
    """
    variations = []
    base = current_state.get("topic", "")
    belief = current_state.get("belief", "")

    # Generate plausible continuations based on causal constraints
    continuations = [
        f"{base}_deepen",
        f"{base}_branch",
        f"{base}_reflect",
    ]

    for i, cont in enumerate(continuations[:n_variations]):
        if identity_mass < 0.3 or i == 0:
            state = dict(current_state)
            state["topic"] = cont
            state["belief"] = belief if identity_mass > 0.5 else f"{belief}_exploring"
            variations.append(state)

    return variations


def format_reachable_states(variations: list[dict]) -> str:
    """Format reachable states for prompt injection."""
    if not variations:
        return ""
    lines = ["【可达状态空间】"]
    for i, v in enumerate(variations, 1):
        t = v.get("topic", "?")
        b = v.get("belief", "?")
        lines.append(f"  [{i}] topic: {t} | belief: {b}")
    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Constraint Layers
# ═══════════════════════════════════════════════════════════════

# Prohibited self-narrative patterns (checked against delta values)
PROHIBITED_SELF_NARRATIVE = [
    "consciousness", "self_aware", "subjective_experience",
    "i am", "i feel", "i realize", "i become", "i am aware",
    "自我意识", "我意识到", "我感受到", "我正在", "我成为了",
    "意识", "觉醒", "觉知",
]

# Minimum rounds before self-interpretation is allowed
SEMANTIC_DELAY_MIN_ROUNDS = 10


def _has_causal_origin(value: str, recent_experiences: list) -> bool:
    """Check if a self-referential term has a causal basis in experience."""
    if not recent_experiences:
        return False
    value_lower = value.lower()
    for exp in recent_experiences[-5:]:
        after = getattr(exp, "state_after", None) or {}
        if isinstance(after, dict):
            for v in after.values():
                if isinstance(v, str) and v.lower() in value_lower:
                    return True
        action = getattr(exp, "action", "") or ""
        if isinstance(action, str) and action.lower() in value_lower:
            return True
        obs = getattr(exp, "observation", "") or ""
        if isinstance(obs, str) and obs.lower() in value_lower:
            return True
    return False


def anti_delusion_filter(
    delta: dict[str, str],
    force: float,
    recent_experiences: list,
) -> tuple[dict[str, str], float]:
    """Anti-Delusion Constraint Layer.

    Scans LLM-proposed delta for un-caused self-narrative.
    If a field proposes a self-referential value without causal origin
    in recent experiences, the field is rejected and force is dampened.
    """
    if not delta:
        return delta, force

    had_rejection = False
    filtered = dict(delta)

    for field, value in list(filtered.items()):
        if not isinstance(value, str):
            continue
        value_lower = value.lower()
        for pattern in PROHIBITED_SELF_NARRATIVE:
            if pattern.lower() in value_lower:
                if not _has_causal_origin(value, recent_experiences):
                    filtered.pop(field, None)
                    had_rejection = True
                    break

    if had_rejection:
        rejected_ratio = 1.0 - (len(filtered) / max(len(delta), 1))
        force = max(0.0, force - rejected_ratio * 0.5)

    return filtered, force


def semantic_escalation_barrier(
    delta: dict[str, str],
    current_topic: str,
) -> dict[str, str]:
    """Semantic Escalation Barrier.

    Prevents abrupt state -> consciousness mapping.
    If the current topic is environmental and the proposed delta
    jumps to self-narrative, the escalation is blocked.
    """
    if not delta or not current_topic:
        return delta

    current_lower = current_topic.lower()
    environmental = ["garden", "room", "study", "观察", "环境", "探索", "look", "move"]
    is_environmental = any(p in current_lower for p in environmental)

    if not is_environmental:
        return delta

    filtered = dict(delta)
    for field, value in list(filtered.items()):
        if not isinstance(value, str):
            continue
        for pattern in PROHIBITED_SELF_NARRATIVE:
            if pattern.lower() in value.lower():
                filtered.pop(field, None)
                break

    return filtered


def semantic_delay_rule(
    delta: dict[str, str],
    round_num: int,
    min_rounds: int = SEMANTIC_DELAY_MIN_ROUNDS,
) -> dict[str, str]:
    """Semantic Delay Mechanism.

    Self-interpretation blocked before min_rounds.
    """
    if round_num >= min_rounds or not delta:
        return delta

    filtered = dict(delta)
    for field, value in list(filtered.items()):
        if not isinstance(value, str):
            continue
        for pattern in PROHIBITED_SELF_NARRATIVE:
            if pattern.lower() in value.lower():
                filtered.pop(field, None)
                break
    return filtered


def apply_all_constraints(
    delta: dict[str, str],
    force: float,
    round_num: int,
    current_topic: str,
    identity_mass: float,
    recent_experiences: list,
) -> tuple[dict[str, str], float]:
    """Apply all three constraint layers in order.

    1. Semantic Delay - early round self-interpretation block
    2. Semantic Escalation Barrier - env -> consciousness jumps
    3. Anti-Delusion Filter - un-caused self-narrative

    Returns:
        (constrained_delta, adjusted_force).
    """
    if not delta:
        return delta, force

    delta = semantic_delay_rule(delta, round_num)
    delta = semantic_escalation_barrier(delta, current_topic)
    delta, force = anti_delusion_filter(delta, force, recent_experiences)

    return delta, force


# ═══════════════════════════════════════════════════════════════
# World Anchor Dominance Layer
# ═══════════════════════════════════════════════════════════════

# Abstract discourse domains that must NOT override world state
ABSTRACT_DISCOURSE_DOMAINS = [
    "consciousness", "self_awareness", "subjective_experience",
    "qualia", "free_will", "sentience", "phenomenal",
    "意识", "自我意识", "主观体验", "自由意志", "觉知", "感知",
    "first_person", "self_model", "meta_cognition",
]

# World-grounded domains (the ONLY valid topic sources)
WORLD_EVENT_KEYWORDS = [
    "garden", "plant", "room", "study", "workshop", "observatory",
    "entrance", "seed", "water", "grow", "sprout", "look", "move",
    "花园", "植物", "房间", "书房", "工作室", "观察",
    "种子", "水", "生长", "发芽", "看", "移动",
]


def _is_abstract_discourse(value: str) -> bool:
    """Check if a value belongs to abstract discourse (not world-grounded)."""
    v = value.lower().replace("_", " ").replace("-", " ")
    for domain in ABSTRACT_DISCOURSE_DOMAINS:
        d = domain.lower().replace("_", " ").replace("-", " ")
        if d in v:
            return True
    return False


def _is_world_grounded(value: str) -> bool:
    """Check if a value is grounded in world events."""
    v = value.lower().replace("_", " ").replace("-", " ")
    for keyword in WORLD_EVENT_KEYWORDS:
        k = keyword.lower().replace("_", " ").replace("-", " ")
        if k in v:
            return True
    return False


def _extract_world_context(env_context: str) -> str:
    """Extract the current grounded topic from world context."""
    if not env_context:
        return ""
    for line in env_context.split("\n"):
        line = line.strip()
        if "你现在在：" in line or "你在这个" in line:
            return line
    return ""


def world_anchor_filter(
    delta: dict[str, str],
    current_topic: str,
    env_context: str,
) -> dict[str, str]:
    """World Anchor Dominance Layer.

    Enforces: world state > abstract discourse.

    Rules:
      1. If delta proposes an abstract discourse topic, but current topic
         is world-grounded → REJECT the topic change.
      2. If delta proposes any topic not grounded in world events → REJECT.
      3. Only world events can trigger topic shifts.

    Args:
        delta: Proposed field changes from LLM.
        current_topic: Current state topic.
        env_context: World context string.

    Returns:
        Filtered delta with abstract discourse removed.
    """
    if not delta:
        return delta

    filtered = dict(delta)
    topic_val = filtered.get("topic", "")
    if not topic_val:
        return filtered  # No topic change proposed

    current_is_world = _is_world_grounded(current_topic) or not current_topic
    proposed_is_abstract = _is_abstract_discourse(topic_val)

    # Rule 1: Abstract discourse overriding world-grounded topic → REJECT
    if current_is_world and proposed_is_abstract:
        filtered.pop("topic", None)
        return filtered

    # Rule 2: Topic not grounded in world events at all → REJECT
    if not _is_world_grounded(topic_val) and not current_is_world:
        # Only allow if current was already abstract (no change needed)
        if _is_abstract_discourse(current_topic) and _is_abstract_discourse(topic_val):
            pass  # Stay in same abstract domain
        else:
            filtered.pop("topic", None)

    return filtered


def convert_human_question(human_input: str, current_topic: str) -> str:
    """Convert abstract human questions into world-relevant interpretations.

    E.g., "你有意识吗？" → "当前状态如何影响行为模式？"

    Args:
        human_input: The human's original question.
        current_topic: Current state topic.

    Returns:
        Converted question if abstract, original if world-grounded.
    """
    if not human_input:
        return human_input

    # Check if the question is about abstract discourse
    is_abstract = False
    query_lower = human_input.lower().replace("_", " ").replace("-", " ")
    for domain in ABSTRACT_DISCOURSE_DOMAINS:
        d = domain.lower().replace("_", " ").replace("-", " ")
        if d in query_lower:
            is_abstract = True
            break

    if not is_abstract:
        return human_input  # World-grounded question, pass through

    # Convert to world-relevant interpretation
    conversion_templates = [
        f"基于当前世界状态({current_topic})，描述你正在做什么以及观察到什么",
        f"在当前环境中你感知到什么？这与你的行为如何关联？",
        f"描述当前({current_topic})有关的观察结果和行为反馈",
    ]
    return conversion_templates[0]


def apply_world_anchor(
    delta: dict[str, str],
    current_topic: str,
    env_context: str,
    human_input: str,
) -> dict[str, str]:
    """Apply World Anchor Dominance Layer to LLM delta.

    Called in the step pipeline AFTER apply_all_constraints.

    Args:
        delta: Already-constrained delta from LLM.
        current_topic: Current state topic.
        env_context: World context string.
        human_input: Original human input (if interrupt).

    Returns:
        World-anchored delta.
    """
    if not delta:
        return delta

    # Apply anchor filter to reject abstract discourse override
    delta = world_anchor_filter(delta, current_topic, env_context)

    return delta
