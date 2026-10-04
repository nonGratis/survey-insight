"""Keep bulk catalog loading inside Google's per-user Forms API quotas.

Google allows 390 reads (forms.get) and 180 "expensive reads" (forms.responses.list)
per minute per user, and 975 / 450 per project
(https://developers.google.com/workspace/forms/api/limits). Beyond them Google answers
429, and it plans to charge for requests over quota later in 2026. The catalog needs one
call of each kind per form, so an account with ~200 forms hits the responses limit as
soon as loading is faster than about a minute.

The guard admits a call while the user made fewer than ``limit`` calls of that kind in the
last 60 seconds. The default limits stay below Google's to leave room for the other
pages. It counts per process: Cloud Run may run several API instances, so it is a brake,
not an exact meter, and a real 429 from Google is still handled as before.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

WINDOW_SECONDS = 60.0


class RollingWindowGuard:
    """Admit at most ``limit`` calls per key within any ``window_seconds``."""

    def __init__(
        self,
        limit: int,
        *,
        window_seconds: float = WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._clock = clock
        self._calls: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def try_acquire(self, key: str) -> bool:
        """Count a call for ``key`` and return True, or return False if the window is full."""
        now = self._clock()
        with self._lock:
            calls = self._calls.setdefault(key, deque())
            while calls and now - calls[0] >= self.window_seconds:
                calls.popleft()
            if len(calls) >= self.limit:
                return False
            calls.append(now)
            return True


@dataclass(frozen=True)
class FormsQuotaGuards:
    """One guard per Forms API quota the catalog spends."""

    reads: RollingWindowGuard
    response_lists: RollingWindowGuard

    @classmethod
    def from_env(cls) -> FormsQuotaGuards:
        return cls(
            reads=RollingWindowGuard(_env_limit("SI_FORMS_READS_PER_MINUTE", 300)),
            response_lists=RollingWindowGuard(
                _env_limit("SI_FORMS_RESPONSE_LISTS_PER_MINUTE", 140)
            ),
        )


def _env_limit(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default
