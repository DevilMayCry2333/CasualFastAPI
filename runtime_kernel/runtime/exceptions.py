"""
exceptions — Runtime Engine exception hierarchy.
"""

from __future__ import annotations


class RuntimeError(Exception):
    """Base exception for all Runtime Engine errors."""
    pass


class SessionNotFoundError(RuntimeError):
    """Raised when a session ID does not exist."""
    pass


class SessionAlreadyExistsError(RuntimeError):
    """Raised when trying to create a session with an existing ID."""
    pass


class LLMError(RuntimeError):
    """Raised when an LLM call fails."""
    pass


class StateValidationError(RuntimeError):
    """Raised when state validation fails."""
    pass


class StateParseError(RuntimeError):
    """Raised when parsing LLM output to State fails."""
    pass


class ConfigurationError(RuntimeError):
    """Raised on invalid engine configuration."""
    pass


class PersistenceError(RuntimeError):
    """Raised on snapshot/restore failures."""
    pass


class EmbeddingError(RuntimeError):
    """Raised when an embedding API call fails."""
    pass


class MemoryError(RuntimeError):
    """Raised on memory storage retrieval failures."""
    pass
