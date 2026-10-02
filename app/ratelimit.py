"""In-memory sliding-window rate limiter, used to protect the LLM quota on public demos.

State lives in the process: it is reset on restart and is not shared between
workers, which is enough for a single-container demo.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

MAX_TRACKED_KEYS = 10_000  # expired keys are swept once this many are tracked


class RateLimiter:
    """Allow at most `limit` hits per key in any window of `window_s` seconds."""

    def __init__(
        self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        """Create a limiter; `clock` is injectable for tests."""
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def retry_after(self, key: str) -> float:
        """Seconds until `key` may hit again; 0 if it may hit now. Does not record a hit."""
        with self._lock:
            hits = self._fresh_hits(key, self._clock())
            if len(hits) < self.limit:
                return 0.0
            return hits[0] + self.window_s - self._clock()

    def hit(self, key: str) -> None:
        """Record one hit for `key`."""
        with self._lock:
            now = self._clock()
            if len(self._hits) >= MAX_TRACKED_KEYS:
                self._sweep(now)
            self._fresh_hits(key, now).append(now)

    def _fresh_hits(self, key: str, now: float) -> deque[float]:
        """Return the hits of `key` still inside the window, dropping older ones."""
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= now - self.window_s:
            hits.popleft()
        return hits

    def _sweep(self, now: float) -> None:
        """Forget keys whose hits have all expired."""
        for key in list(self._hits):
            if not self._fresh_hits(key, now):
                del self._hits[key]
