"""
causal_graph — Causal Graph for the Cognitive Layer.

Tracks directed causal relationships between concepts with confidence.
Supports edge strengthening, weakening, and removal based on evidence.

Every update is based on Fold evidence. No arbitrary modifications.
"""

from __future__ import annotations

from typing import Any


class CausalEdge:
    """A directed causal relationship with confidence tracking."""

    def __init__(self, source: str, target: str, confidence: float = 0.5):
        self.source = source
        self.target = target
        self.confidence = max(0.0, min(1.0, confidence))
        self.evidence_count: int = 0
        self.contradiction_count: int = 0

    def strengthen(self, amount: float = 0.1) -> None:
        """Increase confidence based on supporting evidence."""
        self.confidence = min(1.0, self.confidence + amount)
        self.evidence_count += 1

    def weaken(self, amount: float = 0.15) -> None:
        """Decrease confidence based on contradicting evidence."""
        self.confidence = max(0.0, self.confidence - amount)
        self.contradiction_count += 1
        if self.confidence <= 0.1 and self.contradiction_count >= 3:
            self.confidence = 0.0

    @property
    def is_active(self) -> bool:
        """Edge is active if confidence > 0.1."""
        return self.confidence > 0.1

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "confidence": round(self.confidence, 2),
            "evidence_count": self.evidence_count,
            "contradiction_count": self.contradiction_count,
            "active": self.is_active,
        }


class CausalGraph:
    """Manages causal relationships between concepts.

    Supports:
        - Edge CRUD
        - Confidence updates from evidence
        - Hidden variable inference (placeholder)
        - Graph export for prompt context
    """

    def __init__(self) -> None:
        # (source, target) → CausalEdge
        self._edges: dict[tuple[str, str], CausalEdge] = {}

    def add_edge(self, source: str, target: str, confidence: float = 0.5) -> CausalEdge:
        """Add or update a causal edge."""
        key = (source, target)
        if key in self._edges:
            edge = self._edges[key]
            edge.confidence = max(edge.confidence, confidence)
            return edge
        edge = CausalEdge(source, target, confidence)
        self._edges[key] = edge
        return edge

    def observe_support(self, source: str, target: str, amount: float = 0.1) -> None:
        """Strengthen edge based on supporting evidence."""
        key = (source, target)
        if key not in self._edges:
            self.add_edge(source, target, 0.5)
        self._edges[key].strengthen(amount)

    def observe_contradiction(self, source: str, target: str, amount: float = 0.15) -> None:
        """Weaken edge based on contradicting evidence."""
        key = (source, target)
        if key in self._edges:
            self._edges[key].weaken(amount)

    def get_active_edges(self) -> list[CausalEdge]:
        """Return all active edges (confidence > 0.1)."""
        return [e for e in self._edges.values() if e.is_active]

    def get_edge(self, source: str, target: str) -> CausalEdge | None:
        """Get a specific edge."""
        return self._edges.get((source, target))

    def to_dict(self) -> dict:
        """Export graph as serializable dict."""
        return {
            "edges": [e.to_dict() for e in self._edges.values()],
            "active_count": len(self.get_active_edges()),
            "total_count": len(self._edges),
        }

    def format_for_prompt(self, max_edges: int = 8) -> str:
        """Format active edges for LLM prompt injection."""
        edges = self.get_active_edges()
        if not edges:
            return ""
        edges.sort(key=lambda e: e.confidence, reverse=True)
        lines = ["【因果图】"]
        for e in edges[:max_edges]:
            arrow = "→" if e.confidence > 0.5 else "-/→"
            lines.append(
                f"  {e.source} {arrow} {e.target} "
                f"(conf={e.confidence:.2f}, ev={e.evidence_count})"
            )
        return "\n".join(lines)
