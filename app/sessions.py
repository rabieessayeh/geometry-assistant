"""In-memory store of the models uploaded by each browser session.

Uploads are private to the session that sent them and are never written to
disk. Memory is bounded: each session keeps its most recent uploads, and the
least recently used session is dropped when there are too many. State lives
in the process and is lost on restart, which is enough for a demo.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import trimesh

from .geometry import Catalog


class SessionStore:
    """Uploaded models, keyed by session id."""

    def __init__(self, max_sessions: int, max_uploads: int) -> None:
        """Create a store keeping `max_uploads` models for each of `max_sessions` sessions."""
        self.max_sessions = max_sessions
        self.max_uploads = max_uploads
        self._sessions: OrderedDict[str, Catalog] = OrderedDict()
        self._lock = threading.Lock()

    def catalog(self, session_id: str | None) -> Catalog:
        """Models uploaded by a session; empty for an unknown session."""
        with self._lock:
            if session_id not in self._sessions:
                return Catalog()
            self._sessions.move_to_end(session_id)
            # A copy, so a request keeps a consistent view while another one uploads.
            return Catalog(dict(self._sessions[session_id].models))

    def add(
        self, session_id: str, name: str, mesh: trimesh.Trimesh, reserved: frozenset[str]
    ) -> str:
        """Store an upload and return its id, which avoids the `reserved` ids."""
        with self._lock:
            uploads = self._sessions.setdefault(session_id, Catalog())
            self._sessions.move_to_end(session_id)
            model_id = uploads.add(name, mesh, reserved=reserved)
            while len(uploads.models) > self.max_uploads:
                del uploads.models[next(iter(uploads.models))]  # the oldest upload
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)  # the least recently used session
            return model_id
