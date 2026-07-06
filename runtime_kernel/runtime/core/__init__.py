"""
core — Stable Core Layer: the ONLY layer that executes actions.

Rules:
    - Cognitive Layer cannot execute MCP directly
    - Exploration Layer cannot execute MCP directly
    - Only Stable Core can invoke ActionExecutor
    - All actions pass through validator before execution
    - No learning happens here — pure constraint enforcement
"""

from runtime_kernel.runtime.core.validator import ActionValidator
from runtime_kernel.runtime.core.safety import SafetyRules

__all__ = ["ActionValidator", "SafetyRules"]
