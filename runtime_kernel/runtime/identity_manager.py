"""
identity_manager — IdentityManager: emergent identity via delta accumulation.

Identity is no longer rewritten from scratch every N rounds.

Instead:
  1. Reflection reads recent Experiences and asks:
     "What recent experiences changed me?"
  2. It produces an IdentityDelta — a small, specific change.
  3. Deltas accumulate into the identity anchor over time.

Identity is the sediment of experience, not a prompt output.
"""

from __future__ import annotations

from typing import Any, Optional

from runtime_kernel.runtime.experience import Experience, IdentityDelta
from runtime_kernel.runtime.llm import LLMClient
from runtime_kernel.runtime.models import (
    DEFAULT_IDENTITY_ANCHOR,
    IDENTITY_REFLECTION_INTERVAL,
)
from runtime_kernel.runtime.parser import extract_state
from runtime_kernel.runtime.prompt import PromptBuilder
from runtime_kernel.runtime.state import State


class IdentityManager:
    """Manages emergent identity through delta-based reflection.

    Every N rounds, the LLM reviews recent experiences and produces an
    IdentityDelta — a small change like "started enjoying exploration" —
    rather than rewriting the entire identity anchor.

    Identity = sum(IdentityDeltas) + initial minimal anchor.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        identity_interval: int = IDENTITY_REFLECTION_INTERVAL,
    ) -> None:
        self._llm = llm_client
        self._identity_interval = max(3, identity_interval)  # min 3 rounds between reflections

    @property
    def identity_interval(self) -> int:
        return self._identity_interval

    # ── Reflection schedule ──

    def should_reflect(self, current_round: int) -> bool:
        """Check whether identity reflection is due based on round count."""
        return (
            current_round > 0
            and current_round % self._identity_interval == 0
        )

    # ── Delta-based reflection ──

    def reflect(
        self,
        current_anchor: dict,
        recent_experiences: list[Experience],
        demo: bool = False,
        current_drive_params: Optional[dict] = None,
        current_templates: Optional[dict] = None,
    ) -> Optional[IdentityDelta]:
        """Reflect on recent experiences and produce an IdentityDelta.

        Unlike the old approach which rewrote the entire anchor, this
        produces a small delta that captures what changed. Identity
        accumulates through these deltas over time.

        Args:
            current_anchor: Current identity anchor dict.
            recent_experiences: Recent Experiences to reflect on.
            demo: If True, skip LLM and return None.

        Returns:
            IdentityDelta if a change was detected, None otherwise.
        """
        if not current_anchor:
            current_anchor = dict(DEFAULT_IDENTITY_ANCHOR)

        if demo or not self._llm:
            return None

        if len(recent_experiences) < 1:
            return None

        # Try LLM-based reflection
        try:
            messages = PromptBuilder.build_identity_delta_reflection(
                current_anchor=current_anchor,
                recent_experiences=recent_experiences,
                current_drive_params=current_drive_params,
                current_templates=current_templates,
            )
            response = self._llm.complete(
                messages,
                temperature=0.5,
                max_tokens=300,
            )

            if response:
                _, parsed = extract_state(response)
                if parsed and parsed.to_dict():
                    delta_data = parsed.to_dict()
                    # Build IdentityDelta from LLM output
                    # Expected format: { change, because, affected_field, strength }
                    return IdentityDelta(
                        round=recent_experiences[-1].round if recent_experiences else 0,
                        session_id=recent_experiences[-1].session_id if recent_experiences else "",
                        change=delta_data.get("change", delta_data.get("identity", "")),
                        because=delta_data.get("because", ""),
                        affected_field=delta_data.get("affected_field", "identity"),
                        strength=min(1.0, max(0.0, float(delta_data.get("strength", 0.3)))),
                    )
        except Exception:
            pass

        return None

    # ── Apply delta to anchor ──

    @staticmethod
    def apply_delta(anchor: dict, delta: IdentityDelta) -> dict:
        """Apply an IdentityDelta to an identity anchor.

        The delta modifies the affected field of the anchor. Changes
        are cumulative — each delta adds nuance rather than replacing.

        Args:
            anchor: Current identity anchor.
            delta: IdentityDelta to apply.

        Returns:
            Updated identity anchor dict.
        """
        merged = dict(anchor)
        field = delta.affected_field

        if field == "identity":
            # Accumulate identity description
            existing = merged.get("identity", "")
            if existing and existing != "unknown":
                merged["identity"] = f"{existing}; {delta.change}"
            else:
                merged["identity"] = delta.change

        elif field == "core_goal":
            # Update or enrich core goal
            existing = merged.get("core_goal", "")
            if delta.strength > 0.6:
                merged["core_goal"] = delta.change
            elif existing and existing != "observe_current_environment":
                merged["core_goal"] = f"{existing}; {delta.change}"
            else:
                merged["core_goal"] = delta.change

        elif field == "worldview":
            existing = merged.get("worldview")
            if existing is None:
                merged["worldview"] = delta.change
            else:
                merged["worldview"] = f"{existing}; {delta.change}"

        elif field == "stable_values":
            existing = merged.get("stable_values", [])
            if not isinstance(existing, list):
                existing = []
            if delta.change not in existing:
                existing.append(delta.change)
            merged["stable_values"] = existing

        # Update recent_reflection with the delta's reasoning
        merged["recent_reflection"] = f"{delta.change} (因为: {delta.because})"

        # Gradually increase confidence as deltas accumulate
        current_conf = float(merged.get("confidence", 0.0))
        merged["confidence"] = min(1.0, current_conf + delta.strength * 0.15)

        return merged

    # ── Utility ──

    @staticmethod
    def check_token_length(
        state: State,
        warn_threshold_chars: int = 600,
    ) -> bool:
        """Check if state serialization is approaching token limits."""
        serialized = state.serialize()
        return len(serialized) > warn_threshold_chars

    @staticmethod
    def get_default_anchor() -> dict:
        """Return a fresh copy of the minimal default identity anchor."""
        return dict(DEFAULT_IDENTITY_ANCHOR)
