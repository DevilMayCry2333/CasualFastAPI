"""
memory_manager — MemoryManager: long-term memory (RAG) for the agent.

Handles:
  - store()         — save state/interaction/reflection/introspection
  - retrieve()      — semantic search over stored memories
  - update_importance() — adjust importance score
  - build_context()  — format retrieved memories as prompt context

v2 (World Model): also stores evidence, hypotheses, contradictions.
RAG prioritizes "evidence relevant to current hypotheses" over
"similar chat content."
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from runtime_kernel.runtime.causality import CausalityManager
from runtime_kernel.runtime.embedding import EmbeddingClient
from runtime_kernel.runtime.memory_storage import MemoryStorage
from runtime_kernel.runtime.models import MemoryRecordType


class MemoryManager:
    """Long-term memory manager with RAG support.

    Uses an abstract MemoryStorage backend and a separate EmbeddingClient.
    All memory storage and retrieval goes through here.

    Usage:
        manager = MemoryManager(storage, embedding_client)
        manager.store(session_id="abc", round_num=1, state_dict={...},
                      record_type="state", content=json.dumps({...}))
        results = manager.retrieve("what is my goal?", top_k=5)
    """

    def __init__(
        self,
        storage: MemoryStorage,
        embedding_client: Optional[EmbeddingClient] = None,
        top_k: int = 5,
    ) -> None:
        """Initialize MemoryManager.

        Args:
            storage: Backend storage implementation (e.g. InMemoryMemoryStorage).
            embedding_client: Optional EmbeddingClient for semantic search.
                              Pass None to store records without embeddings.
            top_k: Default number of results to retrieve.
        """
        self._storage = storage
        self._embedding = embedding_client
        self._top_k = top_k

    @property
    def storage(self) -> MemoryStorage:
        """Expose the underlying storage (for inspection/debug)."""
        return self._storage

    def store(
        self,
        session_id: str,
        round_num: int,
        state_dict: dict,
        record_type: str,
        content: str,
        summary: str = "",
        importance: float = 0.5,
    ) -> dict:
        """Store a memory record with embedding.

        Args:
            session_id: The owning session ID.
            round_num: The current round number.
            state_dict: Current state dict (for topic/belief/goal extraction).
            record_type: One of "state", "human_interrupt", "reflection", "introspection".
            content: Full text content to store.
            summary: Short summary (used for embedding + display).
            importance: Importance score 0.0-1.0.

        Returns:
            The stored record dict (with assigned id).
        """
        text_to_embed = f"{summary} {content}"[:2048]  # truncate for embedding

        embedding: list[float] = []
        if self._embedding and text_to_embed.strip():
            try:
                embedding = self._embedding.embed(text_to_embed)
            except Exception:
                embedding = []

        record: dict[str, Any] = {
            "session_id": session_id,
            "round": round_num,
            "timestamp": time.time(),
            "type": record_type,
            "topic": state_dict.get("topic", ""),
            "belief": state_dict.get("belief", ""),
            "goal": state_dict.get("goal", ""),
            "summary": summary[:500],
            "content": content,
            "importance": min(1.0, max(0.0, importance)),
            "embedding": embedding,
        }

        return self._storage.save(record)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """Semantic search over stored memories.

        Args:
            query: Natural language query string.
            top_k: Number of results (defaults to self._top_k).

        Returns:
            List of memory records sorted by relevance.
            Empty list if no embedding client is configured or query is empty.
        """
        k = top_k or self._top_k

        if not query.strip() or not self._embedding:
            return []

        try:
            query_embedding = self._embedding.embed(query.strip())
        except Exception:
            return []

        if not query_embedding:
            return []

        return self._storage.search(query_embedding, k)

    def update_importance(self, record_id: str, importance: float) -> bool:
        """Update the importance score of a memory record.

        Args:
            record_id: The record's id.
            importance: New importance score (0.0-1.0).

        Returns:
            True if updated, False if record not found.
        """
        return self._storage.update(
            record_id,
            {"importance": min(1.0, max(0.0, importance))},
        )

    def store_evidence(
        self,
        session_id: str,
        round_num: int,
        evidence: dict,
    ) -> dict:
        """Store an evidence item in long-term memory.

        Args:
            session_id: The owning session ID.
            round_num: The current round number.
            evidence: Evidence dict (from EvidenceManager).

        Returns:
            The stored record dict.
        """
        return self.store(
            session_id=session_id,
            round_num=round_num,
            state_dict={},
            record_type=MemoryRecordType.EVIDENCE.value,
            content=json.dumps(evidence, ensure_ascii=False),
            summary=f"Evidence: {evidence.get('statement', '')[:200]}",
            importance=0.6,
        )

    def store_hypothesis(
        self,
        session_id: str,
        round_num: int,
        hypothesis: dict,
    ) -> dict:
        """Store a hypothesis lifecycle event in long-term memory.

        Args:
            session_id: The owning session ID.
            round_num: The current round number.
            hypothesis: Hypothesis dict (from HypothesisManager).

        Returns:
            The stored record dict.
        """
        return self.store(
            session_id=session_id,
            round_num=round_num,
            state_dict={},
            record_type=MemoryRecordType.HYPOTHESIS.value,
            content=json.dumps(hypothesis, ensure_ascii=False),
            summary=f"Hypothesis [{hypothesis.get('status', '')}]: "
                    f"{hypothesis.get('statement', '')[:150]}",
            importance=0.7,
        )

    def store_contradiction(
        self,
        session_id: str,
        round_num: int,
        contradiction: dict,
    ) -> dict:
        """Store a contradiction event in long-term memory.

        Args:
            session_id: The owning session ID.
            round_num: The current round number.
            contradiction: Contradiction info dict.

        Returns:
            The stored record dict.
        """
        return self.store(
            session_id=session_id,
            round_num=round_num,
            state_dict={},
            record_type=MemoryRecordType.CONTRADICTION.value,
            content=json.dumps(contradiction, ensure_ascii=False),
            summary=f"Contradiction: {contradiction.get('description', '')[:200]}",
            importance=0.9,
        )

    def retrieve_by_domain(
        self,
        domain: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Retrieve memories for a specific world domain.

        Args:
            domain: Domain tag to filter by.
            top_k: Max results.

        Returns:
            List of matching memory records.
        """
        k = top_k or self._top_k
        all_records = self._storage.search_by_field("domain", domain)
        return all_records[:k]

    def retrieve_hypothesis_relevant(
        self,
        hypothesis_statements: list[str],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Retrieve memories relevant to current hypotheses.

        Priorities:
        1. Evidence records matching hypothesis keywords
        2. State records from rounds when hypothesis was active
        3. Contradiction records

        Args:
            hypothesis_statements: Current active hypothesis statements.
            top_k: Max results.

        Returns:
            List of memory records sorted by relevance.
        """
        k = top_k or self._top_k
        if not hypothesis_statements:
            return []

        # Build a combined query from hypotheses
        combined = " ".join(hypothesis_statements)
        results = self.retrieve(combined, k)

        # Also search for evidence-type records explicitly
        evidence_results = [
            r for r in self._storage.list_all()
            if r.get("type") in ("evidence", "hypothesis", "contradiction")
        ]
        evidence_results.sort(key=lambda r: r.get("importance", 0), reverse=True)

        # Merge: semantic results first, then evidence-specific
        seen_ids = set(r.get("id") for r in results)
        for ev in evidence_results:
            if ev.get("id") not in seen_ids and len(results) < k * 2:
                results.append(ev)
                seen_ids.add(ev.get("id"))

        return results[:k * 2]

    def build_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        causality: Optional[CausalityManager] = None,
        session_id: str = "",
        hypothesis_statements: Optional[list[str]] = None,
    ) -> str:
        """Build a RAG context string from retrieved memories.

        v2: Prioritizes hypothesis-relevant evidence over general chat.
        If hypothesis_statements is provided, retrieves evidence matching
        current hypotheses FIRST, then falls back to semantic search.

        If a CausalityManager is provided, ALSO includes causally relevant
        transitions — bridging semantic similarity (what happened) with
        causal relevance (why it happened and what it led to).

        Args:
            query: The query to search with.
            top_k: Number of results to include.
            causality: Optional CausalityManager for causal retrieval.
            session_id: Session ID for causal chain lookup.
            hypothesis_statements: Optional list of active hypothesis statements.

        Returns:
            A formatted string, or empty string if no results.
        """
        k = top_k or self._top_k
        parts: list[str] = []

        # 1. Hypothesis-relevant evidence (World Model priority)
        if hypothesis_statements:
            hyp_results = self.retrieve_hypothesis_relevant(
                hypothesis_statements, top_k=k,
            )
            if hyp_results:
                parts.append("【假设相关证据】")
                for i, r in enumerate(hyp_results[:k], 1):
                    rtype = r.get("type", "memory")
                    summary = r.get("summary", "")
                    importance = r.get("importance", 0.5)
                    imp_label = "!!" if importance > 0.8 else ("!" if importance > 0.5 else "")
                    parts.append(f"  [{i}] ({rtype}) {imp_label} {summary}")
                parts.append("")

        # 2. Semantic RAG results (general context)
        results = self.retrieve(query, k)
        if results:
            parts.append("【检索记忆】")
            for i, r in enumerate(results[:k], 1):
                rtype = r.get("type", "memory")
                summary = r.get("summary", "")
                topic = r.get("topic", "")
                importance = r.get("importance", 0.5)
                imp_label = "!!" if importance > 0.8 else ("!" if importance > 0.5 else "")
                prefix = f"[{i}] ({rtype})"
                if topic:
                    prefix += f" [{topic}]"
                parts.append(f"{prefix} {imp_label} {summary}")

        # 3. Causal retrieval — bridge to why this matters
        if causality and session_id:
            query_tokens = set(query.lower().split())
            causal_str = causality.build_causal_context_for_query(
                session_id, query_tokens, top_k=k,
            )
            if causal_str:
                parts.append("")
                parts.append(causal_str.strip())

        return "\n".join(parts)

    def store_state(
        self,
        session_id: str,
        round_num: int,
        state_dict: dict,
    ) -> dict:
        """Convenience: store a state transition in memory."""
        return self.store(
            session_id=session_id,
            round_num=round_num,
            state_dict=state_dict,
            record_type=MemoryRecordType.STATE.value,
            content=json.dumps(state_dict, ensure_ascii=False),
            summary=f"State: {state_dict.get('topic', '')} / {state_dict.get('belief', '')}",
            importance=0.5,
        )

    def store_interrupt(
        self,
        session_id: str,
        round_num: int,
        state_dict: dict,
        human_input: str,
        ai_response: str,
    ) -> dict:
        """Convenience: store a human interruption in memory."""
        return self.store(
            session_id=session_id,
            round_num=round_num,
            state_dict=state_dict,
            record_type=MemoryRecordType.HUMAN_INTERRUPT.value,
            content=f"Human: {human_input}\nAI: {ai_response}",
            summary=f"Human: {human_input[:200]}",
            importance=0.7,
        )

    def store_reflection(
        self,
        session_id: str,
        round_num: int,
        state_dict: dict,
        anchor: dict,
    ) -> dict:
        """Convenience: store an identity reflection in memory."""
        reflection_text = anchor.get("recent_reflection") or ""
        return self.store(
            session_id=session_id,
            round_num=round_num,
            state_dict=state_dict,
            record_type=MemoryRecordType.REFLECTION.value,
            content=json.dumps(anchor, ensure_ascii=False),
            summary=f"Reflection: {reflection_text[:200]}",
            importance=0.8,
        )

    def store_introspection(
        self,
        session_id: str,
        round_num: int,
        state_dict: dict,
        summary: str,
    ) -> dict:
        """Convenience: store an introspection in memory."""
        return self.store(
            session_id=session_id,
            round_num=round_num,
            state_dict=state_dict,
            record_type=MemoryRecordType.INTROSPECTION.value,
            content=summary,
            summary=summary[:200],
            importance=0.6,
        )
