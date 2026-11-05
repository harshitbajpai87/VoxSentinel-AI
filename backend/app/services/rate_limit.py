"""In-memory bounded rate limiting and daily quota for provider-backed endpoints.

Deliberately simple: a sliding one-minute window per client host plus a
per-day quota counter. No locks beyond a single asyncio-friendly coarse lock;
the counters are advisory protection for the provider quota, not a security
boundary (clients behind a shared proxy NAT share one bucket, which fails
closed, the safer direction).
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class RateLimitDecision:
    allowed: bool
    reason: str | None = None


class SlidingWindowLimiter:
    """Per-key sliding-window limiter with an optional daily quota."""

    def __init__(self, *, per_minute: int, daily_quota: int, window_s: float = 60.0,
                 day_s: float = 86400.0) -> None:
        self._per_minute = max(1, per_minute)
        self._daily_quota = max(1, daily_quota)
        self._window_s = window_s
        self._day_s = day_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._day_counts: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> RateLimitDecision:
        """Non-consuming check: is `key` currently allowed to act?"""
        t = time.monotonic() if now is None else now
        with self._lock:
            self._prune(t)
            if len(self._hits.get(key, ())) >= self._per_minute:
                return RateLimitDecision(False, "per_minute")
            if len(self._day_counts.get(key, ())) >= self._daily_quota:
                return RateLimitDecision(False, "daily")
            return RateLimitDecision(True)

    def record(self, key: str, now: float | None = None) -> None:
        """Consume one slot after a successful/authorized action."""
        t = time.monotonic() if now is None else now
        with self._lock:
            self._prune(t)
            self._hits[key].append(t)
            self._day_counts.setdefault(key, []).append(t)

    def _prune(self, t: float) -> None:
        for hits in self._hits.values():
            while hits and hits[0] <= t - self._window_s:
                hits.popleft()
        stale_keys = [k for k, stamps in self._day_counts.items()
                      if stamps and stamps[0] <= t - self._day_s]
        for k in stale_keys:
            del self._day_counts[k]
            if k in self._hits and not self._hits[k]:
                pass
