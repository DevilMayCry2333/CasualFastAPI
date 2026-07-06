"""
parser — Parse, validate, and repair LLM output into structured State objects.

Pipeline:
    LLM output text
        → parse_state()    (extract dict from JSON or other formats)
        → validate()       (ensure core keys present)
        → repair()         (fix common issues)
        → State object

All parsing logic from the original causal_chain.py is migrated here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Optional

from runtime_kernel.runtime.models import LOOP_WINDOW, REQUIRED_STATE_KEYS
from runtime_kernel.runtime.state import State


# ── Primary parser ──


def parse_state(text: str) -> State | None:
    """Parse LLM output text into a State object.

    Tries multiple strategies in order:
      1. Direct JSON parsing (with various fix attempts)
      2. Extract { ... } block from text
      3. Handle nested "note" strings
      4. Loose regex key-value extraction
      5. STATE{key:value} format

    Returns None if all strategies fail.
    """
    if not text:
        return None

    text = text.strip()
    state_dict = _parse_state_from_text(text)

    if state_dict is not None:
        return State(state_dict)

    return None


def extract_state(text: str) -> tuple[str, State | None]:
    """Separate natural language from state dict in AI output.

    Canonical path: ===STATE=== delimiter
    Fallback: parse JSON / STATE format
    Last resort: {"raw": text}

    Returns (nl_text, state_or_none).
    """
    if not text:
        return "", None

    nl = ""
    state_part = text

    # 1. ===STATE=== delimiter
    if "===STATE===" in text:
        parts = text.split("===STATE===", 1)
        nl = parts[0].strip()
        state_part = parts[1].strip()

    # 2. Parse dict from state part
    state_dict = _parse_state_from_text(state_part)
    if state_dict is not None:
        return nl, State(state_dict)

    # 3. Fallback: loose regex on full text
    loose_kv = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*"([^"]*)"', text)
    if loose_kv:
        return nl, State({k: v for k, v in loose_kv})

    return nl, None


# ── State text parsing (migrated from causal_chain.py) ──


def _parse_state_from_text(text: str) -> dict | None:
    """Extract a state dict from LLM output text.

    Supports truncated JSON recovery, nested string extraction,
    unquoted key names, and multiple format variations.
    """
    if not text:
        return None

    text = text.strip()

    # 1. Direct JSON parsing with various fix strategies
    for fix in _json_fix_functions():
        try:
            return json.loads(fix(text))
        except json.JSONDecodeError:
            continue

    # 2. Extract {...} block, handle truncation
    result = _extract_brace_block(text)
    if result is not None:
        return result

    # 3. Handle "note": "{\"topic\":..." nested strings
    result = _extract_nested_json(text)
    if result is not None:
        return result

    # 4. Loose regex: extract "key":"value" pairs
    kv_pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', text)
    if kv_pairs:
        return _ensure_core_keys_dict(dict(kv_pairs))

    # 5. STATE{key:value} format
    result = _extract_state_format(text)
    if result is not None:
        return result

    return None


def _json_fix_functions():
    """Return ordered list of JSON fix functions to try."""
    return [
        lambda s: s,
        lambda s: s.replace("'", '"'),
        lambda s: re.sub(r',\s*}', '}', s),
        lambda s: re.sub(
            r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:',
            r'\1"\2":',
            s,
        ),
    ]


def _extract_brace_block(text: str) -> dict | None:
    """Extract the first top-level {...} block and try to parse it.

    Handles truncated JSON (no closing brace) by appending one.
    """
    brace_start = text.find("{")
    if brace_start < 0:
        return None

    depth = 0
    for i in range(brace_start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[brace_start : i + 1]
                for fix in _json_fix_functions():
                    try:
                        return json.loads(fix(candidate))
                    except json.JSONDecodeError:
                        continue
                break  # matched closing brace but parse failed

    # No matching closing brace (truncated)
    candidate = text[brace_start:] + "}"
    for fix in [
        lambda s: re.sub(r',\s*$', "}", s.rstrip(",")),
        lambda s: s,
    ]:
        try:
            return json.loads(fix(candidate))
        except json.JSONDecodeError:
            continue

    return None


def _extract_nested_json(text: str) -> dict | None:
    """Handle "note": "{\"topic\":..." pattern where JSON is nested in a string."""
    note_match = re.search(r'"note"\s*:\s*"\\?\{', text)
    if not note_match:
        return None

    start = note_match.end() - 1
    inner = text[start:]
    if inner.count("{") > inner.count("}"):
        inner += "}"
    inner = inner.replace('\\"', '"').replace("\\n", "")
    try:
        return json.loads(inner)
    except json.JSONDecodeError:
        inner = re.sub(r',\s*}', "}", inner)
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass
    return None


def _extract_state_format(text: str) -> dict | None:
    """Extract STATE{key: value} format."""
    m = re.search(r"STATE\s*\{([^}]*)\}", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None

    body = m.group(1).strip()
    result = {}
    for line in body.split("\n"):
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            result[k.strip()] = v.strip()
    if result:
        return _ensure_core_keys_dict(result)
    return None


# ── Validation and repair ──


def _ensure_core_keys_dict(
    state_dict: dict,
    defaults: dict | None = None,
) -> dict:
    """Ensure topic, belief, goal are present; fill with defaults or 'unknown'."""
    for key in REQUIRED_STATE_KEYS:
        if key not in state_dict:
            if defaults and key in defaults:
                state_dict[key] = defaults[key]
            else:
                state_dict[key] = "unknown"
    return state_dict


def validate_state(state: State) -> bool:
    """Check that the state has all required keys with non-empty values."""
    for key in REQUIRED_STATE_KEYS:
        val = state.get(key)
        if not val:
            return False
    return True


def repair_state(state: State, defaults: dict | None = None) -> State:
    """Repair a state by ensuring core keys exist."""
    d = state.to_dict()
    d = _ensure_core_keys_dict(d, defaults)
    return State(d)


# ── Loop detection ──


def detect_loop(history: Sequence[dict]) -> bool:
    """Detect if recent state history shows a period-2 or period-3 cycle.

    Uses autocorrelation on (topic, belief) fingerprints.

    Args:
        history: List of history entries, each with {"state": {...}}.

    Returns True if a cycle is detected.
    """
    if len(history) < LOOP_WINDOW:
        return False

    # Extract fingerprints from last N entries
    fingerprints: list[tuple[str, str]] = []
    for h in history[-LOOP_WINDOW:]:
        s = h.get("state", {})
        fp = (str(s.get("topic", "")), str(s.get("belief", "")))
        fingerprints.append(fp)

    # Exclude stagnation (all same)
    unique_fps = set(fingerprints)
    if len(unique_fps) < 2:
        return False

    # Autocorrelation for period 2 and 3
    for period in (2, 3):
        if len(fingerprints) >= period * 2:
            periodic = all(
                fingerprints[i] == fingerprints[i + period]
                for i in range(len(fingerprints) - period)
            )
            if periodic and len(unique_fps) <= period:
                return True

    return False


# ── NL alignment ──


def align_nl_with_state(state: State, nl_text: str) -> str:
    """Force natural language response to be semantically consistent with state.

    1. Detect generic safety disclaimers → replace with template
    2. Short chat openers → pass through
    3. Long substantive responses → trust the LLM
    """
    if not nl_text or not state:
        return nl_text or ""

    nl_lower = nl_text.lower()

    # Detection of generic safety disclaimers
    refusal_patterns = [
        "我没有意识", "我不能", "我不认为自己", "只是一个程序",
        "no consciousness", "i cannot", "i do not have", "merely a",
    ]
    is_refusal = any(p in nl_lower for p in refusal_patterns)

    if is_refusal:
        topic = str(state.get("topic", ""))
        belief = str(state.get("belief", ""))
        parts = []
        if topic:
            parts.append(f"当前主题：{topic}")
        if belief:
            parts.append(f"认知状态：{belief}")
        return "；".join(parts) if parts else nl_text

    # Short text (< 40 chars) → chat opener, pass through
    if len(nl_text) <= 40:
        return nl_text

    # Long text → has substance, trust the LLM alignment
    return nl_text


# ── Causal force vector parsing (new: LLM outputs forces, not states) ──


def extract_causal_vector(text: str) -> tuple[str, Optional[dict], float, str, str]:
    """Extract a causal force vector from LLM output.

    The LLM is a World Model Builder. It outputs how to CHANGE the current
    state AND world model updates (hypotheses, evidence, etc.).

    Expected output format:
        {"delta": {"topic": "new_value", ...},
         "force": 0.65, "action": "look", "source": "curiosity",
         "new_hypotheses": [...], "new_evidence": [...],
         "hypothesis_updates": [...], "world_model_update": {...},
         "open_questions": [...], "uncertainties": [...],
         "confidence": 0.7}

    Returns:
        Tuple of (nl_text, delta_dict, force_strength, action, source).
        nl_text: Natural language reasoning (before ===STATE===).
        delta_dict: Dict of field -> new_value, or None.
        force_strength: 0.0-1.0 float.
        action: World action string.
        source: Force source string.
    """
    if not text:
        return "", None, 0.0, "", "llm"

    nl = ""
    state_part = text

    # 1. ===STATE=== delimiter (keep backward compat)
    if "===STATE===" in text:
        parts = text.split("===STATE===", 1)
        nl = parts[0].strip()
        state_part = parts[1].strip()
    else:
        nl = text.strip()

    # 2. Parse the state part as a causal vector
    state_dict = _parse_state_from_text(state_part)

    if not state_dict:
        return nl, None, 0.0, "", "llm"

    # 3. Extract delta (the "how to change" info)
    delta = state_dict.get("delta", None)
    if delta and isinstance(delta, dict):
        pass  # Use nested delta dict
    else:
        delta = {}

    # Check for delta_ prefix format
    for key in list(state_dict.keys()):
        if key.startswith("delta_") and len(key) > 6:
            field = key[6:]
            if field in ("topic", "belief", "goal", "name", "core_belief", "identity"):
                val = str(state_dict[key])
                if val and val != "unknown":
                    delta[field] = val

    # Direct field format (fallback)
    if not delta:
        for key in ("topic", "belief", "goal", "name", "core_belief", "identity"):
            if key in state_dict and key not in ("force", "action", "source", "confidence",
                                                  "new_hypotheses", "new_evidence",
                                                  "hypothesis_updates", "world_model_update",
                                                  "open_questions", "uncertainties"):
                delta[key] = str(state_dict[key])

    # 4. Extract force strength and metadata
    force = float(state_dict.get("force", state_dict.get("strength", 0.5)))
    action = str(state_dict.get("action", ""))
    source = str(state_dict.get("source", state_dict.get("drive", "llm")))

    return nl, delta, min(1.0, max(0.0, force)), action, source


def extract_world_model_updates(text: str) -> dict:
    """Extract world model updates from LLM output.

    Parses new_hypotheses, new_evidence, hypothesis_updates,
    world_model_update, open_questions, uncertainties, confidence
    from the raw LLM response text.

    Returns:
        Dict with optional keys:
            new_hypotheses: list[dict]
            new_evidence: list[dict]
            hypothesis_updates: list[dict]
            world_model_update: dict
            open_questions: list[str]
            uncertainties: list[str | dict]
            confidence: float
    """
    if not text:
        return {}

    state_part = text

    # Extract state part after ===STATE=== if present
    if "===STATE===" in text:
        parts = text.split("===STATE===", 1)
        state_part = parts[1].strip()
    elif "{" in text:
        # Try to find JSON by brace matching
        brace_start = text.find("{")
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(text)):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        state_part = text[brace_start:i + 1]
                        break

    state_dict = _parse_state_from_text(state_part)
    if not state_dict:
        return {}

    result = {}

    # New hypotheses proposed by LLM
    raw_hypotheses = state_dict.get("new_hypotheses", [])
    if isinstance(raw_hypotheses, list):
        result["new_hypotheses"] = [
            h for h in raw_hypotheses
            if isinstance(h, dict) and h.get("statement")
        ]

    # New evidence collected
    raw_evidence = state_dict.get("new_evidence", [])
    if isinstance(raw_evidence, list):
        result["new_evidence"] = [
            e for e in raw_evidence
            if isinstance(e, dict) and e.get("statement")
        ]

    # Hypothesis updates (support/contradict)
    raw_updates = state_dict.get("hypothesis_updates", [])
    if isinstance(raw_updates, list):
        result["hypothesis_updates"] = [
            u for u in raw_updates
            if isinstance(u, dict) and u.get("id")
        ]

    # World model key-value updates
    wm_update = state_dict.get("world_model_update")
    if isinstance(wm_update, dict):
        result["world_model_update"] = wm_update

    # Open questions
    raw_questions = state_dict.get("open_questions", [])
    if isinstance(raw_questions, list):
        result["open_questions"] = [str(q) for q in raw_questions if q]

    # Uncertainties
    raw_uncertainties = state_dict.get("uncertainties", [])
    if isinstance(raw_uncertainties, list):
        result["uncertainties"] = [str(u) if isinstance(u, str) else u
                                    for u in raw_uncertainties if u]

    # Confidence
    raw_confidence = state_dict.get("confidence")
    if raw_confidence is not None:
        try:
            result["confidence"] = max(0.0, min(1.0, float(raw_confidence)))
        except (ValueError, TypeError):
            pass

    # Messages to send to other agents
    raw_messages = state_dict.get("send_message", state_dict.get("send_messages", []))
    if isinstance(raw_messages, dict):
        raw_messages = [raw_messages]
    if isinstance(raw_messages, list):
        cleaned = []
        for m in raw_messages:
            if isinstance(m, dict) and (m.get("to_agent") or m.get("content")):
                # Normalize: ensure 'content' is a dict
                if isinstance(m.get("content"), str):
                    m["content"] = {"text": m["content"]}
                elif not isinstance(m.get("content"), dict):
                    m["content"] = {"text": str(m.get("content", ""))}
                cleaned.append(m)
        if cleaned:
            result["send_messages"] = cleaned

    # Tool usage
    tool_use = state_dict.get("tool_use")
    if isinstance(tool_use, dict) and tool_use.get("name"):
        result["tool_use"] = tool_use

    # Knowledge operations (share/support/contradict)
    share = state_dict.get("share_knowledge", state_dict.get("share_memory"))
    if isinstance(share, dict):
        result["share_knowledge"] = share
    support = state_dict.get("support_knowledge")
    if support:
        result["support_knowledge"] = str(support)
    contradict = state_dict.get("contradict_knowledge")
    if contradict:
        result["contradict_knowledge"] = str(contradict)

    # Tool usage detection (extracted alongside other fields)
    tool_use = state_dict.get("tool_use")
    if isinstance(tool_use, dict) and tool_use.get("name"):
        result["tool_use"] = tool_use

    # Capability action detection (Action System)
    # The LLM outputs:
    #   "action": {"capability": "Search", "operation": "web_search",
    #              "parameters": {"query": "..."}}
    # This replaces the older tool_use pattern.
    raw_action = state_dict.get("action")
    if isinstance(raw_action, dict) and raw_action.get("capability") and raw_action.get("operation"):
        result["capability_action"] = raw_action
        # Also set a readable string action for fallback
        cap = raw_action["capability"]
        op = raw_action["operation"]
        result["action"] = f"{cap}:{op}"

    return result


def extract_tool_use(text: str) -> Optional[dict]:
    """Extract the tool_use field from the latest LLM response.

    The LLM can output a tool_use field to request tool execution:
        "tool_use": {"name": "web_search", "arguments": {"query": "..."}}

    Returns:
        Dict with "name" and "arguments" keys, or None if no tool use requested.
    """
    if not text:
        return None

    state_part = text

    # Extract state part after ===STATE=== if present
    if "===STATE===" in text:
        parts = text.split("===STATE===", 1)
        state_part = parts[1].strip()
    elif "{" in text:
        # Try to find JSON by brace matching
        brace_start = text.find("{")
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(text)):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        state_part = text[brace_start:i + 1]
                        break

    state_dict = _parse_state_from_text(state_part)
    if not state_dict:
        return None

    tool_use = state_dict.get("tool_use")
    if isinstance(tool_use, dict) and tool_use.get("name"):
        return tool_use

    # Also check for a standalone tool_use in the non-state part
    if "===STATE===" not in text:
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict) and data.get("tool_use"):
                tool_use = data["tool_use"]
                if isinstance(tool_use, dict) and tool_use.get("name"):
                    return tool_use
        except (json.JSONDecodeError, ValueError):
            pass

    return None
