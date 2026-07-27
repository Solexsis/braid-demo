"""Concurrency admission and per-IP rate limiting.

Deliberately small and in-process. One model can serve one generation at a
time, so the interesting question is not "how do we scale" but "what do we tell
the ninth person". Answer: 429 immediately, with a `Retry-After`, instead of a
queue that grows until the reverse proxy times out.

For real traffic, put the rate limiting in the reverse proxy and set
`BRAID_RATE_LIMIT_REQUESTS=0`; the interface below stays the same either way.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque


class Overloaded(RuntimeError):
    """No capacity: the caller should retry later."""

    def __init__(self, message: str, retry_after: int = 5) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ConcurrencyGate:
    """A bounded semaphore plus a bounded waiting room.

    `slots` requests generate at once; `queue_limit` more may wait. Anything
    beyond that is rejected immediately rather than accepted and starved.
    """

    def __init__(self, slots: int = 1, queue_limit: int = 8) -> None:
        self._semaphore = asyncio.Semaphore(slots)
        self._slots = slots
        self._queue_limit = queue_limit
        self._waiting = 0
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def waiting(self) -> int:
        return self._waiting

    async def __aenter__(self) -> ConcurrencyGate:
        if self._waiting >= self._queue_limit and self._semaphore.locked():
            raise Overloaded("the model is busy and the queue is full", retry_after=5)
        self._waiting += 1
        try:
            await self._semaphore.acquire()
        finally:
            self._waiting -= 1
        self._in_flight += 1
        return self

    async def __aexit__(self, *exc) -> bool:
        self._in_flight -= 1
        self._semaphore.release()
        return False


class RateLimiter:
    """Fixed-window-per-key counter. `requests <= 0` disables it."""

    def __init__(self, requests: int = 20, window_s: float = 60.0) -> None:
        self.requests = requests
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    @property
    def enabled(self) -> bool:
        return self.requests > 0

    def check(self, key: str, now: float | None = None) -> None:
        if not self.enabled:
            return
        now = now if now is not None else time.monotonic()
        hits = self._hits[key]
        cutoff = now - self.window_s
        while hits and hits[0] < cutoff:
            hits.popleft()
        if len(hits) >= self.requests:
            retry = max(1, int(hits[0] + self.window_s - now) + 1)
            raise Overloaded(
                f"rate limit exceeded ({self.requests} requests per {int(self.window_s)}s)",
                retry_after=retry,
            )
        hits.append(now)
        # Opportunistic cleanup so an idle server does not accumulate keys.
        if len(self._hits) > 4096:
            for stale in [k for k, v in self._hits.items() if not v]:
                del self._hits[stale]
