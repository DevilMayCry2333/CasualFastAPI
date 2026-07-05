"""
causality — CausalityManager: record, organize, and retrieve causal chains.

Every state transition is recorded as a CausalEntry — a complete snapshot of
what changed, why, what the agent did, and what the world/identity/drives
looked like before and after.

The causal chain replaces flat "working memory" with a traceable directed graph.
Memory retrieval queries causality (topic continuity, action sequence, room
proximity) rather than semantic embedding summaries alone.

Key insight: an agent's experience IS its causal chain. Each entry carries its
own full context, so retrieval doesn't need external summaries — it looks at
the causal shape of the transition itself.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class CausalEntry:
    """A single node in the agent's causal chain.

    Every state transition is one causal edge. This record captures:
      - What changed (state diff)
      - Why it changed (cause + full input context)
      - What the agent did about it (action)
      - What external AND internal state looked like before/after
      - Which prior transitions influenced this one (parent_rounds)

    This is NOT a "memory summary" — it is the raw causal fact of a transition.
    """

    round: int
    cause: str                     # "init" | "self" | "human" | "environment" | "reflect"
    session_id: str

    # ── State transition ──
    state_before: dict = field(default_factory=dict)
    state_after: dict = field(default_factory=dict)

    # ── Agent action ──
    action: str = ""               # extracted from state["action"] if any

    # ── World context ──
    world_room: str = ""
    world_tick: int = 0

    # ── Internal state at transition time (snapshot) ──
    identity_anchor: Optional[dict] = None
    drives: dict = field(default_factory=dict)
    thought_pool: list = field(default_factory=list)

    # ── Causal narrative ──
    reasoning: str = ""            # LLM-extracted or auto-computed "why this happened"
    parent_rounds: list[int] = field(default_factory=list)

    # ── Human interaction ──
    human_input: str = ""
    nl_response: str = ""

    # ── Metadata ──
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()

    @property
    def state_diff(self) -> dict[str, tuple[Any, Any]]:
        """Auto-compute the diff: keys present in either state_before or state_after."""
        diff: dict[str, tuple[Any, Any]] = {}
        all_keys = set(self.state_before.keys()) | set(self.state_after.keys())
        for k in all_keys:
            before = self.state_before.get(k)
            after = self.state_after.get(k)
            if before != after:
                diff[k] = (before, after)
        return diff

    @property
    def core_diff_text(self) -> str:
        """Compact one-line description of what changed in core fields."""
        parts: list[str] = []
        for k in ("topic", "belief", "goal"):
            b = self.state_before.get(k, "?")
            a = self.state_after.get(k, "?")
            if b != a:
                parts.append(f"{k}: {b} → {a}")
            else:
                parts.append(f"{k}: {b}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state_diff"] = {k: list(v) for k, v in self.state_diff.items()}
        d["core_diff"] = self.core_diff_text
        return d

    def to_short_text(self) -> str:
        """Single-line text representation for prompt injection."""
        cause_symbols = {
            "init": "⚡", "self": "⟳", "human": "◈",
            "environment": "⊙", "reflect": "◇",
        }
        symbol = cause_symbols.get(self.cause, "?")
        action_text = f" | 行动: {self.action}" if self.action else ""
        drive_text = ""
        if self.drives:
            d = self.drives
            drive_text = f" | cur:{d.get('curiosity',0):.1f} bor:{d.get('boredom',0):.1f} bel:{d.get('belonging',0):.1f}"
        return (
            f"[R{self.round}] {symbol} {self.cause}{action_text}{drive_text}\n"
            f"    {self.core_diff_text}"
        )


class CausalityManager:
    """Records and retrieves causal chains for every session.

    Each session has an ordered list of CausalEntry nodes forming a directed
    chain (edges have parent_rounds links for causal tracing).

    Core operations:
      - record()         — store a new causal edge
      - get_recent()     — last N entries for prompt context
      - build_context()  — format causal chain for prompt injection
      - trace_path()     — trace causal path between two rounds
      - find_relevant()  — find causally relevant entries (replaces RAG)
    """

    def __init__(self) -> None:
        self._chains: dict[str, list[CausalEntry]] = {}

    # ── Recording ──

    def record(self, entry: CausalEntry) -> CausalEntry:
        """Record a new causal entry for a session.

        Args:
            entry: Fully constructed CausalEntry.

        Returns the entry (for chaining).
        """
        sid = entry.session_id
        if sid not in self._chains:
            self._chains[sid] = []
        entry.parent_rounds = self._find_causal_parents(entry)
        self._chains[sid].append(entry)
        return entry

    def create_entry(
        self,
        session_id: str,
        round_num: int,
        cause: str,
        state_before: dict,
        state_after: dict,
        action: str = "",
        world_room: str = "",
        world_tick: int = 0,
        identity_anchor: Optional[dict] = None,
        drives: Optional[dict] = None,
        thought_pool: Optional[list] = None,
        reasoning: str = "",
        human_input: str = "",
        nl_response: str = "",
    ) -> CausalEntry:
        """Create and record a causal entry in one call."""
        entry = CausalEntry(
            round=round_num,
            cause=cause,
            session_id=session_id,
            state_before=dict(state_before),
            state_after=dict(state_after),
            action=action,
            world_room=world_room,
            world_tick=world_tick,
            identity_anchor=dict(identity_anchor) if identity_anchor else None,
            drives=dict(drives) if drives else {},
            thought_pool=list(thought_pool) if thought_pool else [],
            reasoning=reasoning,
            human_input=human_input,
            nl_response=nl_response,
        )
        return self.record(entry)

    # ── Retrieval ──

    def get_chain(self, session_id: str) -> list[CausalEntry]:
        """Get the full causal chain for a session (ordered by round)."""
        return list(self._chains.get(session_id, []))

    def get_recent(self, session_id: str, n: int = 5) -> list[CausalEntry]:
        """Get the last N entries in the causal chain."""
        chain = self._chains.get(session_id, [])
        return chain[-n:]

    def get_by_round(self, session_id: str, round_num: int) -> Optional[CausalEntry]:
        """Get a specific entry by round number."""
        for e in self._chains.get(session_id, []):
            if e.round == round_num:
                return e
        return None

    # ── Context building (replaces flat working memory in prompts) ──

    def build_context(self, session_id: str, n: int = 5) -> str:
        """Build a causal chain context block for prompt injection.

        Replaces the old "【工作记忆】" section. Shows each transition as
        a causal edge with reason, action, and state diff.

        Args:
            session_id: Target session.
            n: Number of recent entries to include.

        Returns:
            Formatted string or empty string if no entries.
        """
        recent = self.get_recent(session_id, n)
        if not recent:
            return ""

        lines = ["【因果链】"]
        for entry in recent:
            lines.append(entry.to_short_text())
            if entry.reasoning:
                lines.append(f"    ⤷ {entry.reasoning[:100]}")
            # Show parent links if any (indicates causal influence)
            if len(entry.parent_rounds) > 1:
                parent_str = ",".join(str(r) for r in entry.parent_rounds[:-1])
                lines.append(f"    ← 受 R{parent_str} 影响")
        lines.append("")
        return "\n".join(lines)

    def build_causal_context_for_query(
        self,
        session_id: str,
        query_tokens: set[str],
        top_k: int = 3,
    ) -> str:
        """Find causally relevant entries matching a query.

        Instead of embedding similarity, scores entries by:
          - Topic/belief/goal token overlap with query
          - Action similarity
          - Cause type relevance
          - Recency bonus

        Args:
            session_id: Target session.
            query_tokens: Set of tokens/keywords from the query.
            top_k: Max entries to return.

        Returns:
            Formatted context string.
        """
        chain = self._chains.get(session_id, [])
        if not chain or not query_tokens:
            return ""

        scored: list[tuple[float, CausalEntry]] = []
        for entry in chain:
            score = self._causal_relevance_score(entry, query_tokens)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        if not top:
            return ""

        lines = ["【因果回溯】（因果相关的历史变迁）"]
        for score, entry in top:
            lines.append(f"  [{entry.round}|{entry.cause}] (匹配度:{score:.2f})")
            lines.append(f"    {entry.core_diff_text}")
            if entry.action:
                lines.append(f"    行动: {entry.action}")
        lines.append("")
        return "\n".join(lines)

    # ── Path tracing ──

    def trace_path(self, session_id: str, from_round: int, to_round: int) -> list[CausalEntry]:
        """Trace the causal path between two rounds.

        Follows parent_rounds links backward from to_round to find the
        directed path connecting them. If no direct path, returns all
        entries in the round range as a simple chronological sequence.

        Args:
            session_id: Target session.
            from_round: Start round (inclusive).
            to_round: End round (inclusive).

        Returns:
            List of CausalEntry in chronological order along the path.
        """
        chain = self._chains.get(session_id, [])
        if not chain:
            return []

        # Build a round -> entry map
        round_map = {e.round: e for e in chain}

        # Try to follow parent links backward
        visited: set[int] = set()
        path_set: set[int] = set()

        def follow_parents(r: int) -> None:
            if r in visited or r not in round_map:
                return
            visited.add(r)
            entry = round_map[r]
            for parent_r in entry.parent_rounds:
                if from_round <= parent_r <= to_round:
                    path_set.add(parent_r)
                    follow_parents(parent_r)

        follow_parents(to_round)
        path_set.add(to_round)

        if path_set:
            # We found a causal path via parent links
            result = [round_map[r] for r in sorted(path_set) if r >= from_round]
        else:
            # Fallback: chronological range
            result = [e for e in chain if from_round <= e.round <= to_round]

        return result

    # ── Serialization ──

    def to_dict(self, session_id: str) -> list[dict]:
        """Serialize a session's chain for persistence."""
        return [e.to_dict() for e in self._chains.get(session_id, [])]

    def from_dict(self, session_id: str, data: list[dict]) -> None:
        """Deserialize and load a chain from saved data."""
        entries = []
        for d in data:
            entry = CausalEntry(
                round=d.get("round", 0),
                cause=d.get("cause", "unknown"),
                session_id=session_id,
                state_before=d.get("state_before", {}),
                state_after=d.get("state_after", {}),
                action=d.get("action", ""),
                world_room=d.get("world_room", ""),
                world_tick=d.get("world_tick", 0),
                identity_anchor=d.get("identity_anchor"),
                drives=d.get("drives", {}),
                thought_pool=d.get("thought_pool", []),
                reasoning=d.get("reasoning", ""),
                parent_rounds=d.get("parent_rounds", []),
                human_input=d.get("human_input", ""),
                nl_response=d.get("nl_response", ""),
                timestamp=d.get("timestamp", 0.0),
            )
            entries.append(entry)
        self._chains[session_id] = entries

    # ── Internal helpers ──

    def _find_causal_parents(self, entry: CausalEntry) -> list[int]:
        """Determine which prior entries causally influenced this one.

        Uses a combination of:
        1. Topic/belief continuity (same topic chain)
        2. Action relevance (same type of action)
        3. Room persistence (acting in the same room)
        4. Recency (recent entries weighted higher)

        Returns a sorted list of round numbers (most relevant first).
        """
        chain = self._chains.get(entry.session_id, [])
        if not chain:
            return []

        # Build weighted scores for prior entries
        scored: list[tuple[float, int]] = []
        for prior in chain:
            if prior.round >= entry.round:
                continue
            score = 0.0

            # 1. Topic/belief continuity
            prior_topic = prior.state_after.get("topic", "")
            new_topic = entry.state_after.get("topic", "")
            if prior_topic and new_topic and prior_topic == new_topic:
                score += 3.0

            prior_belief = prior.state_after.get("belief", "")
            new_belief = entry.state_after.get("belief", "")
            if prior_belief and new_belief and prior_belief == new_belief:
                score += 2.0

            # 2. Action relevance (same or continuous action)
            if prior.action and prior.action == entry.action:
                score += 2.0

            # 3. Room persistence
            if prior.world_room and prior.world_room == entry.world_room:
                score += 1.5

            # 4. Recency bonus (0 to 1, linear decay, safe from division by zero)
            if chain and chain[-1].round > prior.round:
                max_r = max(chain[-1].round, 1)
                recency = 1.0 - (chain[-1].round - prior.round) / max_r
                score += max(0.0, recency) * 2.0

            # 5. Identity relevance (same identity context)
            if prior.identity_anchor and entry.identity_anchor:
                prior_core = prior.identity_anchor.get("core_goal", "")
                new_core = entry.identity_anchor.get("core_goal", "")
                if prior_core and new_core and prior_core == new_core:
                    score += 1.0

            if score > 0:
                scored.append((score, prior.round))

        # Sort by score descending, take top 3
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:3]]

    def _causal_relevance_score(self, entry: CausalEntry, query_tokens: set[str]) -> float:
        """Score how causally relevant an entry is to a query.

        Higher score = more likely to be causally relevant.
        """
        if not query_tokens:
            return 0.0

        score = 0.0

        # Token overlap with after-state values
        for val in entry.state_after.values():
            if isinstance(val, str):
                val_tokens = set(val.lower().split("_"))
                overlap = len(val_tokens & query_tokens)
                score += overlap * 0.5

        # Token overlap with action
        if entry.action:
            action_tokens = set(entry.action.lower().split())
            score += len(action_tokens & query_tokens) * 0.3

        # Cause type relevance
        cause_boost = {"self": 0.2, "human": 0.5, "environment": 0.3}
        score += cause_boost.get(entry.cause, 0.0)

        # Recency bonus (safe from division by zero)
        chain_len = len(self._chains.get(entry.session_id, []))
        denom = 1.0 + chain_len - entry.round
        if denom > 0:
            score += 1.0 / denom

        return score
