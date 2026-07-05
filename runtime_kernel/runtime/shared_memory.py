"""
shared_memory — SharedKnowledge: cross-agent knowledge base with consensus.

Design philosophy:
    No agent can directly write to shared knowledge.
    All contributions follow a consensus pipeline:
        Observation → Evidence → Candidate → Peer support → Public Knowledge

    Shared knowledge belongs to the WORLD, not to any single agent.
    It is the collective understanding that emerges from multiple
    independent observers converging on the same facts.

Key differences from private memory:
    - Private: per-agent, includes subjective interpretations
    - Shared: inter-agent, requires consensus, objective facts only
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Optional

from runtime_kernel.runtime.models import (
    SHARED_KNOWLEDGE_CANDIDATE_MAX,
    SHARED_KNOWLEDGE_CONSENSUS_MIN,
)


class KnowledgeEntry:
    """A single piece of shared knowledge.

    Fields:
        id: Unique identifier.
        statement: The knowledge statement.
        domain: Which world domain this belongs to.
        confidence: Consensus confidence (0.0-1.0).
        agent_support: Set of agent IDs that support this.
        agent_contradict: Set of agent IDs that contradict this.
        created_tick: World tick when proposed.
        promoted_tick: World tick when promoted from candidate to public.
        source: How it originated (observation, deduction, human, consensus).
        status: "candidate" | "public" | "contested" | "discarded"
        supporting_evidence_ids: Evidence IDs from any agent backing this.
    """

    __slots__ = (
        "id", "statement", "domain", "confidence",
        "agent_support", "agent_contradict",
        "created_tick", "promoted_tick", "source", "status",
        "supporting_evidence_ids",
    )

    def __init__(
        self,
        statement: str,
        domain: str = "",
        source: str = "observation",
        created_tick: int = 0,
    ) -> None:
        self.id: str = uuid.uuid4().hex[:12]
        self.statement: str = statement
        self.domain: str = domain
        self.confidence: float = 0.1  # starts low
        self.agent_support: set[str] = set()
        self.agent_contradict: set[str] = set()
        self.created_tick: int = created_tick
        self.promoted_tick: int = 0
        self.source: str = source
        self.status: str = "candidate"
        self.supporting_evidence_ids: list[str] = []

    @property
    def support_count(self) -> int:
        return len(self.agent_support)

    @property
    def is_public(self) -> bool:
        return self.status == "public"

    def add_support(self, agent_id: str) -> None:
        """An agent supports this knowledge.

        When enough distinct agents support it, it becomes public.
        """
        self.agent_support.add(agent_id)
        self.agent_contradict.discard(agent_id)
        self._recompute_confidence()

        # Promote to public when enough agents agree
        if self.support_count >= SHARED_KNOWLEDGE_CONSENSUS_MIN:
            if self.status == "candidate":
                self.status = "public"
                self.promoted_tick = int(time.time())

    def add_contradiction(self, agent_id: str) -> None:
        """An agent contradicts this knowledge."""
        self.agent_contradict.add(agent_id)
        self.agent_support.discard(agent_id)
        self._recompute_confidence()

        if self.status == "public" and self.support_count < SHARED_KNOWLEDGE_CONSENSUS_MIN:
            self.status = "contested"
            self.confidence = max(0.0, self.confidence - 0.3)

    def discard(self) -> None:
        self.status = "discarded"
        self.confidence = 0.0

    def _recompute_confidence(self) -> None:
        """Recompute confidence from support/contradiction balance.

        Formula: sigmoid(support * 2 - contradiction * 3) / num_agents factor
        """
        total = self.support_count + len(self.agent_contradict)
        if total == 0:
            self.confidence = 0.1
            return
        net = self.support_count * 2 - len(self.agent_contradict) * 3
        self.confidence = max(0.0, min(1.0, 1.0 / (1.0 + pow(2.718, -net * 0.5))))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement[:200],
            "domain": self.domain,
            "confidence": round(self.confidence, 2),
            "agent_support": list(self.agent_support),
            "agent_contradict": list(self.agent_contradict),
            "created_tick": self.created_tick,
            "promoted_tick": self.promoted_tick,
            "source": self.source,
            "status": self.status,
            "support_count": self.support_count,
            "supporting_evidence_ids": self.supporting_evidence_ids,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeEntry":
        ke = cls(
            statement=d.get("statement", ""),
            domain=d.get("domain", ""),
            source=d.get("source", "observation"),
            created_tick=int(d.get("created_tick", 0)),
        )
        ke.id = d.get("id", ke.id)
        ke.confidence = float(d.get("confidence", 0.1))
        ke.agent_support = set(d.get("agent_support", []))
        ke.agent_contradict = set(d.get("agent_contradict", []))
        ke.promoted_tick = int(d.get("promoted_tick", 0))
        ke.status = d.get("status", "candidate")
        ke.supporting_evidence_ids = list(d.get("supporting_evidence_ids", []))
        return ke

    def __repr__(self) -> str:
        return (
            f"KnowledgeEntry({self.status}, conf={self.confidence:.2f}, "
            f"support={self.support_count}, statement={self.statement[:40]})"
        )


class SharedKnowledge:
    """Cross-agent shared knowledge base.

    Agents contribute observations, evidence, and hypotheses.
    Knowledge only becomes "public" after consensus from multiple agents.

    No agent can directly write. All contributions are candidates first.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, KnowledgeEntry] = {}
        self._domain_index: dict[str, set[str]] = {}  # domain -> entry IDs

    # ── Read ──

    @property
    def candidates(self) -> list[KnowledgeEntry]:
        """Entries awaiting consensus."""
        with self._lock:
            return [e for e in self._entries.values() if e.status == "candidate"]

    @property
    def public_knowledge(self) -> list[KnowledgeEntry]:
        """Entries that have reached consensus."""
        with self._lock:
            return [e for e in self._entries.values() if e.is_public]

    @property
    def all_entries(self) -> list[KnowledgeEntry]:
        with self._lock:
            return list(self._entries.values())

    def get_by_domain(self, domain: str) -> list[KnowledgeEntry]:
        """Get knowledge entries for a specific domain."""
        with self._lock:
            ids = self._domain_index.get(domain, set())
            return [self._entries[eid] for eid in ids if eid in self._entries]

    def search(self, query: str, top_k: int = 5) -> list[KnowledgeEntry]:
        """Simple keyword search over shared knowledge.

        Args:
            query: Search keywords.
            top_k: Max results.

        Returns:
            Sorted by confidence descending.
        """
        query_lower = query.lower()
        scored = []
        with self._lock:
            for entry in self._entries.values():
                if entry.status == "discarded":
                    continue
                # Simple keyword overlap scoring
                statement_lower = entry.statement.lower()
                score = 0
                for token in query_lower.split():
                    if token in statement_lower:
                        score += 1
                if score > 0:
                    scored.append((score * entry.confidence, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    # ── Contribution (consensus pipeline) ──

    def propose(
        self,
        statement: str,
        agent_id: str,
        domain: str = "",
        source: str = "observation",
        tick: int = 0,
    ) -> Optional[KnowledgeEntry]:
        """Propose a new piece of knowledge.

        The agent observes something and proposes it as shared knowledge.
        It immediately counts as a supporter.

        Args:
            statement: The knowledge statement.
            agent_id: Proposing agent.
            domain: Domain this belongs to.
            source: How it originated.
            tick: Current world tick.

        Returns:
            The new KnowledgeEntry, or None if duplicate.
        """
        # Check for near-duplicate statements
        with self._lock:
            for existing in self._entries.values():
                if existing.status == "discarded":
                    continue
                # Simple duplicate check: 70% word overlap
                existing_words = set(existing.statement.lower().split())
                new_words = set(statement.lower().split())
                if len(existing_words) > 0 and len(new_words) > 0:
                    overlap = len(existing_words & new_words)
                    ratio = overlap / max(len(existing_words), len(new_words))
                    if ratio > 0.7:
                        # Near-duplicate — just add support
                        existing.add_support(agent_id)
                        return existing

            entry = KnowledgeEntry(
                statement=statement,
                domain=domain,
                source=source,
                created_tick=tick,
            )
            entry.add_support(agent_id)
            self._entries[entry.id] = entry

            # Index by domain
            if domain:
                if domain not in self._domain_index:
                    self._domain_index[domain] = set()
                self._domain_index[domain].add(entry.id)

            # Cleanup if too many candidates
            candidates = [e for e in self._entries.values() if e.status == "candidate"]
            if len(candidates) > SHARED_KNOWLEDGE_CANDIDATE_MAX:
                oldest = min(candidates, key=lambda e: e.created_tick)
                oldest.discard()

            return entry

    def support(self, entry_id: str, agent_id: str) -> bool:
        """An agent supports an existing knowledge entry.

        Args:
            entry_id: Knowledge entry ID.
            agent_id: Supporting agent.

        Returns:
            True if found.
        """
        with self._lock:
            entry = self._entries.get(entry_id)
            if not entry:
                return False
            entry.add_support(agent_id)
            return True

    def contradict(self, entry_id: str, agent_id: str) -> bool:
        """An agent contradicts an existing knowledge entry.

        Args:
            entry_id: Knowledge entry ID.
            agent_id: Contradicting agent.

        Returns:
            True if found.
        """
        with self._lock:
            entry = self._entries.get(entry_id)
            if not entry:
                return False
            entry.add_contradiction(agent_id)
            return True

    def discard_entry(self, entry_id: str) -> bool:
        """Discard a knowledge entry."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if not entry:
                return False
            entry.discard()
            return True

    # ── Prompt context ──

    def format_for_prompt(self, domain: str = "", max_entries: int = 5) -> str:
        """Format public knowledge for prompt injection.

        Args:
            domain: Optional domain filter.
            max_entries: Max entries to show.

        Returns:
            Formatted string, or empty if none.
        """
        entries = self.public_knowledge
        if domain:
            entries = [e for e in entries if e.domain == domain]

        if not entries:
            return ""

        entries.sort(key=lambda e: e.confidence, reverse=True)
        lines = ["【公共知识】"]
        for e in entries[:max_entries]:
            status_icon = "✓" if e.is_public else "○"
            lines.append(
                f"  {status_icon} [{e.domain or 'general'}] "
                f"{e.statement[:80]} "
                f"(conf:{e.confidence:.2f}, {e.support_count} agents)"
            )

        # Add candidate count
        candidates = self.candidates
        if candidates:
            lines.append(f"  — 还有 {len(candidates)} 条待共识知识")

        return "\n".join(lines)

    # ── Serialization ──

    def to_dict(self) -> list[dict]:
        with self._lock:
            return [e.to_dict() for e in self._entries.values()]

    @classmethod
    def from_dict(cls, data: list[dict]) -> "SharedKnowledge":
        sk = cls()
        for item in data:
            entry = KnowledgeEntry.from_dict(item)
            sk._entries[entry.id] = entry
            if entry.domain:
                if entry.domain not in sk._domain_index:
                    sk._domain_index[entry.domain] = set()
                sk._domain_index[entry.domain].add(entry.id)
        return sk

    def __len__(self) -> int:
        return len(self._entries)
