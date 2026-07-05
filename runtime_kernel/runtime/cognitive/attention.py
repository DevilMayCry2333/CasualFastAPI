"""Attention — what the agent actually pays attention to.

The agent cannot attend to all events, messages, and observations.
Attention filters based on:
  - curiosity (high curiosity → novel events get attention)
  - importance (world-changing events are hard to ignore)
  - novelty (repeated events get less attention)
  - relationship (events from known agents get more attention)
  - uncertainty (uncertain areas attract attention)

This gives the agent bounded attention — a core cognitive constraint.
"""

from __future__ import annotations

from typing import Any, Optional


# Default attention weights
DEFAULT_WEIGHTS = {
    "curiosity": 0.25,
    "importance": 0.25,
    "novelty": 0.20,
    "relationship": 0.15,
    "uncertainty": 0.15,
}


def _compute_novelty(event_content: str, recent_events: list[dict]) -> float:
    """Compute novelty score (0.0-1.0) based on similarity to recent events."""
    if not recent_events:
        return 1.0
    words = set(event_content.lower().split())
    if not words:
        return 0.5
    max_overlap = 0.0
    for ev in recent_events[-5:]:
        ev_text = str(ev.get("text", ev.get("content", {}).get("text", "")))
        ev_words = set(ev_text.lower().split())
        if ev_words and words:
            overlap = len(words & ev_words) / max(len(words), len(ev_words))
            max_overlap = max(max_overlap, overlap)
    return max(0.0, 1.0 - max_overlap)


def _compute_importance(event: dict) -> float:
    """Compute importance score based on event type and source."""
    event_type = event.get("type", event.get("event_type", ""))
    source = event.get("source", "")

    important_types = {"warning", "seeds_planted", "soil_dried", "item_crafted",
                       "plants_grew", "plant_watered", "telescope_focused"}
    if event_type in important_types:
        return 0.8
    if source == "human":
        return 0.9
    return 0.3


def _compute_relevance_to_uncertainty(
    event_content: str,
    uncertain_areas: list[str],
) -> float:
    """Compute relevance to current uncertainties."""
    if not uncertain_areas or not event_content:
        return 0.0
    content_lower = event_content.lower()
    matches = sum(1 for u in uncertain_areas if u.lower() in content_lower)
    return min(1.0, matches * 0.3)


def filter_events(
    events: list[dict],
    drives: dict[str, float],
    recent_events: list[dict],
    uncertain_areas: list[str],
    weights: Optional[dict[str, float]] = None,
    max_events: int = 3,
) -> list[dict]:
    """Filter events through attention, returning the most salient ones.

    Args:
        events: Incoming events to filter.
        drives: Current drive states.
        recent_events: Recently attended events (for novelty computation).
        uncertain_areas: Current areas of uncertainty.
        weights: Attention weight overrides.
        max_events: Max events to return.

    Returns:
        List of (score, event) tuples, sorted by salience descending.
    """
    if not events:
        return []

    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    curiosity = drives.get("curiosity", 0.5)
    scored: list[tuple[float, dict]] = []

    for event in events:
        event_text = str(event.get("text", event.get("content", {}).get("text", "")))

        # Novelty — repeated events score lower
        novelty = _compute_novelty(event_text, recent_events) * w["novelty"]

        # Importance — world-changing events score higher
        importance = _compute_importance(event) * w["importance"]

        # Curiosity-driven — high curiosity amplifies all scores
        curiosity_factor = curiosity * w["curiosity"]

        # Uncertainty relevance
        uncertainty = _compute_relevance_to_uncertainty(event_text, uncertain_areas) * w["uncertainty"]

        # Salience = weighted sum
        salience = novelty + importance + curiosity_factor + uncertainty

        scored.append((salience, event))

    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:max_events]]
