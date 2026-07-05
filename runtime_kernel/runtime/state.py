"""
state — State object for Agent working memory.

Encapsulates the agent's internal state dict with validation,
serialization, deserialization, merge, and update operations.

Core keys: topic, belief, goal
World Model keys (v2): world_model, hypotheses, evidence, open_questions,
                       uncertainties, confidence

The state now stores the agent's CURRENT UNDERSTANDING of the world,
not a self-narrative. Belief must have concrete content, not abstract
status labels like "updated".
"""

from __future__ import annotations

import json
from typing import Any, Optional

from runtime_kernel.runtime.exceptions import StateValidationError
from runtime_kernel.runtime.models import (
    REQUIRED_STATE_KEYS,
    REQUIRED_WORLD_MODEL_KEYS,
    DEFAULT_WORLD_MODEL,
)


UNKNOWN = "unknown"


class State:
    """Working memory state for an Agent.

    Wraps a dict. Core keys (topic, belief, goal) are special:

    - Properties return UNKNOWN ("unknown") when a core key is absent.
    - Construction does NOT auto-fill defaults — the data is stored as-is.
    - Call ensure_core_keys() explicitly when core keys must be materialized.
    - merge() with override=True properly handles partial states:
      only keys present in the source are overridden, UNKNOWN filler values
      from the source never replace real values in the target.

    This design means:
        State({"topic": "x"}).merge(session.state)
    correctly overrides only the topic, leaving belief and goal untouched.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Optional[dict | str] = None) -> None:
        self._data: dict[str, Any] = {}
        if data is not None:
            if isinstance(data, str):
                self._data = self._deserialize_dict(data)
            elif isinstance(data, dict):
                self._data = dict(data)
            else:
                raise StateValidationError(
                    f"State data must be dict or str, got {type(data).__name__}"
                )

    REQUIRED_KEYS = REQUIRED_STATE_KEYS

    # ── World Model field accessors ──

    @property
    def world_model(self) -> dict:
        """Current world model — structured understanding of the world."""
        return dict(self._data.get("world_model", DEFAULT_WORLD_MODEL))

    @property
    def hypotheses(self) -> list[dict]:
        """Active hypotheses the agent is tracking."""
        return list(self._data.get("hypotheses", []))

    @property
    def evidence(self) -> list[dict]:
        """Evidence collected (observations with source + confidence)."""
        return list(self._data.get("evidence", []))

    @property
    def open_questions(self) -> list[str]:
        """Questions the agent has not yet answered."""
        return list(self._data.get("open_questions", []))

    @property
    def uncertainties(self) -> list[dict]:
        """Areas where the agent has low confidence."""
        return list(self._data.get("uncertainties", []))

    @property
    def confidence(self) -> float:
        """Overall confidence in the current world model (0.0-1.0)."""
        return float(self._data.get("confidence", 0.0))

    # ── World Model mutation helpers ──

    def merge_hypothesis(self, hypothesis: dict) -> "State":
        """Add or update a hypothesis in the state's hypothesis list.

        If a hypothesis with the same id exists, updates it in-place.
        Otherwise appends.
        """
        hyps = list(self._data.get("hypotheses", []))
        hid = hypothesis.get("id", "")
        replaced = False
        for i, h in enumerate(hyps):
            if h.get("id") == hid:
                hyps[i] = hypothesis
                replaced = True
                break
        if not replaced:
            hyps.append(hypothesis)
        self._data["hypotheses"] = hyps
        return self

    def add_evidence(self, evidence_item: dict) -> "State":
        """Append a piece of evidence."""
        evd = list(self._data.get("evidence", []))
        evd.append(evidence_item)
        self._data["evidence"] = evd
        return self

    def set_world_model(self, wm: dict) -> "State":
        """Replace the world model dict."""
        self._data["world_model"] = dict(wm)
        return self

    def set_open_questions(self, questions: list[str]) -> "State":
        self._data["open_questions"] = list(questions)
        return self

    def set_uncertainties(self, uncertainties: list[dict]) -> "State":
        self._data["uncertainties"] = list(uncertainties)
        return self

    def set_confidence(self, confidence: float) -> "State":
        self._data["confidence"] = max(0.0, min(1.0, confidence))
        return self

    # ── Core key materialization ──

    def ensure_core_keys(self, defaults: Optional[dict] = None) -> "State":
        """Fill missing topic / belief / goal with defaults or UNKNOWN.

        This is called explicitly after parsing LLM output, NOT on construction.
        """
        for key in self.REQUIRED_KEYS:
            if key not in self._data:
                if defaults and key in defaults:
                    self._data[key] = defaults[key]
                else:
                    self._data[key] = UNKNOWN
        return self

    def has_core_keys(self) -> bool:
        """Check whether all three core keys exist in the underlying dict."""
        return all(k in self._data for k in self.REQUIRED_KEYS)

    # ── accessors ──

    @property
    def topic(self) -> str:
        return str(self._data.get("topic", UNKNOWN))

    @property
    def belief(self) -> str:
        return str(self._data.get("belief", UNKNOWN))

    @property
    def goal(self) -> str:
        return str(self._data.get("goal", UNKNOWN))

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)

    # ── core operations ──

    def validate(self) -> bool:
        """Check all required keys exist with non-empty, non-UNKNOWN values."""
        for key in self.REQUIRED_KEYS:
            val = self._data.get(key)
            if not val or val == UNKNOWN:
                return False
        return True

    def serialize(self) -> str:
        """Serialize to JSON string, ensuring core keys are present."""
        out = dict(self._data)
        for key in self.REQUIRED_KEYS:
            if key not in out:
                out[key] = UNKNOWN
        return json.dumps(out, ensure_ascii=False)

    def serialize_pretty(self) -> str:
        """Serialize to formatted JSON string, ensuring core keys are present."""
        out = dict(self._data)
        for key in self.REQUIRED_KEYS:
            if key not in out:
                out[key] = UNKNOWN
        return json.dumps(out, ensure_ascii=False, indent=2)

    @classmethod
    def deserialize(cls, text: str) -> "State":
        """Deserialize from JSON string."""
        return cls(data=text)

    def to_dict(self) -> dict:
        """Return a plain dict copy (raw data, no filler)."""
        return dict(self._data)

    def to_dict_complete(self) -> dict:
        """Return a dict with all core keys guaranteed present."""
        out = dict(self._data)
        for key in self.REQUIRED_KEYS:
            if key not in out:
                out[key] = UNKNOWN
        return out

    def merge(self, other: "State | dict", override: bool = True) -> "State":
        """Merge another state into this one.

        Args:
            other: State or dict to merge from.
            override: If True, keys from other override matching keys.
                      If False, only keys NOT already present are added.
                      In both cases, UNKNOWN values from other never replace
                      real (non-UNKNOWN) values in self.

        Returns self for chaining.
        """
        source = other._data if isinstance(other, State) else other
        if override:
            for k, v in source.items():
                current = self._data.get(k)
                # Don't let UNKNOWN filler replace a real value
                if v == UNKNOWN and current is not None and current != UNKNOWN:
                    continue
                # Don't overwrite with UNKNOWN if existing value is also UNKNOWN
                # (just preserve whatever is there in case it's more specific)
                self._data[k] = v
        else:
            for k, v in source.items():
                if k not in self._data:
                    self._data[k] = v
                else:
                    # If self has UNKNOWN but source has real value, take it
                    current = self._data.get(k)
                    if current == UNKNOWN and v != UNKNOWN:
                        self._data[k] = v
        return self

    def update(self, **kwargs: Any) -> "State":
        """Update state with keyword arguments.

        Returns self for chaining.
        """
        self._data.update(kwargs)
        return self

    def copy(self) -> "State":
        """Return a shallow copy (sufficient for JSON data)."""
        return State(data=dict(self._data))

    def fingerprint(self) -> tuple:
        """Return (topic, belief) tuple for loop detection."""
        return (self.topic, self.belief)

    def world_model_fingerprint(self) -> tuple:
        """Fingerprint including world model for richer loop detection."""
        wm_keys = tuple(sorted(self._data.get("world_model", {}).keys()))
        hyp_count = len(self._data.get("hypotheses", []))
        ev_count = len(self._data.get("evidence", []))
        return (self.topic, self.belief, wm_keys, hyp_count, ev_count)

    # ── internals ──

    @staticmethod
    def _deserialize_dict(text: str) -> dict:
        """Parse a JSON string into a dict."""
        if not text or not text.strip():
            return {}
        return json.loads(text)

    def __repr__(self) -> str:
        return f"State({self._data})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return self._data == other._data

    def __hash__(self) -> int:
        return hash(json.dumps(self._data, sort_keys=True, ensure_ascii=False))
