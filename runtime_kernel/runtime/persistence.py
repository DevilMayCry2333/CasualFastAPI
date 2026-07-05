"""
persistence — Snapshot, restore, and file-based persistence.

Provides:
  - snapshot() / restore() — save and load full session state
  - save_json() / load_json() — generic JSON file operations

The interface is designed to be replaceable:
today it writes JSON files; tomorrow it could use SQLite, Redis, or Postgres
without modifying the Runtime Engine.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from runtime_kernel.runtime.exceptions import PersistenceError
from runtime_kernel.runtime.session import AgentSession


class Persistence:
    """File-based persistence for AgentSessions.

    Currently uses JSON files; swap the backend by subclassing
    or replacing this class.
    """

    def __init__(self, base_dir: str = ".") -> None:
        self._base_dir = base_dir

    # ── Session persistence ──

    def snapshot(self, session: AgentSession, filepath: Optional[str] = None) -> str:
        """Save a full session snapshot to a JSON file.

        Args:
            session: The session to snapshot.
            filepath: Optional explicit path. Auto-generated if omitted.

        Returns the filepath written.
        """
        filepath = filepath or self._session_path(session.id)
        return self.save_json(session.to_dict(), filepath)

    def restore(self, filepath: str) -> AgentSession:
        """Load a session from a JSON snapshot file.

        Args:
            filepath: Path to the snapshot file.

        Returns a restored AgentSession.
        """
        data = self.load_json(filepath)
        if data is None:
            raise PersistenceError(f"Cannot restore from {filepath}: file not found or empty")
        # The file may contain a top-level dict (legacy CausalChain format)
        # or a session dict (new format with "id" key).
        if "id" in data or "state" in data:
            return AgentSession.from_dict(data)
        raise PersistenceError(f"Cannot restore from {filepath}: unrecognized format")

    def restore_from_dict(self, data: dict) -> AgentSession:
        """Restore a session from a dict (e.g., loaded from another source)."""
        return AgentSession.from_dict(data)

    # ── Generic JSON operations ──

    def save_json(self, data: Any, filepath: str) -> str:
        """Serialize data to a JSON file.

        Creates parent directories if needed.

        Returns the filepath.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return filepath
        except OSError as e:
            raise PersistenceError(f"Failed to write {filepath}: {e}")

    def load_json(self, filepath: str) -> Any:
        """Load data from a JSON file.

        Returns None if the file doesn't exist or is empty.
        """
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise PersistenceError(f"Failed to read {filepath}: {e}")

    # ── Path helpers ──

    def _session_path(self, session_id: str) -> str:
        return os.path.join(self._base_dir, f"session_{session_id}.json")

    def auto_save_path(self, name: str = "autosave") -> str:
        return os.path.join(self._base_dir, f"{name}.json")

    # ─── Directory operations ──

    def list_snapshots(self, pattern: str = "session_*.json") -> list[str]:
        """List session snapshot files in base directory."""
        import glob
        return sorted(glob.glob(os.path.join(self._base_dir, pattern)))

    def delete_snapshot(self, filepath: str) -> bool:
        """Delete a snapshot file.

        Returns True if deleted, False if not found.
        """
        try:
            os.remove(filepath)
            return True
        except OSError:
            return False
