"""
memory_storage — Abstract storage interface for long-term memory.

MemoryStorage is an abstract base class defining the storage contract.
InMemoryMemoryStorage provides a development-only in-memory implementation.

The interface is designed to be replaced by MySQLMemoryStorage (or other
backends) without changing MemoryManager or RuntimeEngine.

Memory record schema:
    {
        "id": str,              # unique identifier
        "session_id": str,      # owning session
        "round": int,           # round at which this record was created
        "timestamp": float,     # unix timestamp
        "type": str,            # "state" | "human_interrupt" | "reflection" | "introspection"
        "topic": str,           # from state.topic
        "belief": str,          # from state.belief
        "goal": str,            # from state.goal
        "summary": str,         # short text summary
        "content": str,         # full content
        "importance": float,    # importance score 0.0-1.0
        "embedding": list[float],  # embedding vector (may be empty)
    }
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Any, Optional


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class MemoryStorage(ABC):
    """Abstract storage backend for long-term memory.

    Implementations: InMemoryMemoryStorage, MySQLMemoryStorage, etc.
    """

    @abstractmethod
    def save(self, record: dict) -> dict:
        """Store a memory record.

        Args:
            record: Memory record dict. If "id" is omitted, one is generated.

        Returns:
            The stored record with its assigned id.
        """
        ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Search for closest records by embedding similarity.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of records sorted by relevance (most relevant first).
        """
        ...

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """Delete a memory record by id.

        Returns True if deleted, False if not found.
        """
        ...

    @abstractmethod
    def list_all(self, session_id: Optional[str] = None) -> list[dict]:
        """List all records, optionally filtered by session_id.

        Returns a shallow copy list of records.
        """
        ...

    @abstractmethod
    def get(self, record_id: str) -> Optional[dict]:
        """Get a single record by id.

        Returns None if not found.
        """
        ...

    @abstractmethod
    def update(self, record_id: str, updates: dict) -> bool:
        """Update fields of an existing record.

        Returns True if updated, False if not found.
        """
        ...


class InMemoryMemoryStorage(MemoryStorage):
    """In-memory list-based storage for development use.

    NOT for production. All data is lost on process exit.
    """

    def __init__(self) -> None:
        self._records: list[dict] = []
        self._next_id: int = 0

    def save(self, record: dict) -> dict:
        stored = dict(record)
        if "id" not in stored:
            stored["id"] = f"mem_{self._next_id}"
            self._next_id += 1
        stored["_created"] = time.time()
        self._records.append(stored)
        return dict(stored)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        if not query_embedding:
            return []

        scored: list[tuple[float, dict]] = []
        for r in self._records:
            emb = r.get("embedding")
            if emb and len(emb) > 0:
                sim = _cosine_similarity(query_embedding, emb)
                scored.append((sim, r))
            else:
                scored.append((0.0, r))

        # Sort by similarity descending, then by importance descending
        scored.sort(key=lambda x: (x[0], x[1].get("importance", 0.0)), reverse=True)
        return [dict(r) for _, r in scored[:top_k]]

    def delete(self, record_id: str) -> bool:
        for i, r in enumerate(self._records):
            if r.get("id") == record_id:
                self._records.pop(i)
                return True
        return False

    def list_all(self, session_id: Optional[str] = None) -> list[dict]:
        if session_id:
            return [dict(r) for r in self._records if r.get("session_id") == session_id]
        return [dict(r) for r in self._records]

    def get(self, record_id: str) -> Optional[dict]:
        for r in self._records:
            if r.get("id") == record_id:
                return dict(r)
        return None

    def update(self, record_id: str, updates: dict) -> bool:
        for r in self._records:
            if r.get("id") == record_id:
                r.update(updates)
                return True
        return False
