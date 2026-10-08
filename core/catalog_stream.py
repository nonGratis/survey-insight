"""Load a whole catalog of forms in one pass and report each result as soon as it is ready.

Runs where the Google calls are made: inside one streamed API request in production (one
set of credentials, one quota guard, no round trip per batch from the page) and in process
in the local demo. The page only draws what has arrived.

Order of work: every form's details first, then the response counts. The details give each
row its status, what the page shows first, and on a repeat visit they come from the cache
at once; a count queued ahead of them would hold them back behind a Google call each (in
production 4.8 s for 212 cached statuses). Counts the quota guard holds back
(RetryLaterError) run when their slots free.
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from core.forms_api import FormsApiError

LoadKind = Literal["summary", "stats"]
SUMMARY: LoadKind = "summary"
STATS: LoadKind = "stats"
# Google's own 429 (not our guard): wait and try again a few times before giving up.
GOOGLE_RATE_LIMIT_RETRY_SECONDS = 15.0
GOOGLE_RATE_LIMIT_ATTEMPTS = 3
# Longest silence on the stream: keeps proxies from closing it and the countdown fresh.
HEARTBEAT_SECONDS = 5.0

EventKind = Literal["summary", "stats", "waiting", "error", "done"]


class RetryLaterError(Exception):
    """Raised by a load that may succeed after ``seconds`` (a slot the quota guard keeps)."""

    seconds: float = 0.0


@dataclass(frozen=True)
class Loaded:
    """What a load returns: the data and where it came from."""

    data: dict[str, Any]
    fetched_at: str | None = None
    cache_hit: bool = False


@dataclass(frozen=True)
class CatalogEvent:
    """One line of the stream.

    ``summary`` and ``stats`` carry one form's details or response count (``status`` ok
    with ``data``, or an error status). ``waiting`` says how many loads wait for the Google
    quota and how long until the last of them runs. ``error`` stops the load (the Google
    grant is gone); ``done`` ends it.
    """

    event: EventKind
    form_id: str | None = None
    status: str | None = None
    error_code: str | None = None
    data: dict[str, Any] | None = None
    fetched_at: str | None = None
    cache_hit: bool | None = None
    seconds: float | None = None
    forms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def catalog_status(exc: FormsApiError) -> str:
    """Row status for a Google error."""
    if exc.status in {401, 403}:
        return "no_access"
    if exc.status == 404:
        return "deleted"
    if exc.status == 429:
        return "rate_limited"
    if exc.status == 400:
        return "unsupported"
    return "api_error"


# A load to run: which, for which form, after how many Google 429s.
_Task = tuple[LoadKind, str, int]


@dataclass(order=True)
class _Held:
    due: float
    seq: int
    kind: LoadKind = field(compare=False)
    form_id: str = field(compare=False)
    attempts: int = field(compare=False, default=0)


def load_catalog(
    form_ids: Sequence[str],
    *,
    load_summary: Callable[[str], Loaded],
    load_stats: Callable[[str], Loaded],
    workers: int = 10,
    deadline_seconds: float = 240.0,
    fatal: tuple[type[BaseException], ...] = (),
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[CatalogEvent]:
    """Events for every form: its details, then its response count; ``done`` at the end.

    A form whose details fail gets no count. Loads still pending at the deadline end as
    ``timeout``. An exception of a ``fatal`` type stops everything with an ``error`` event.
    """
    loads = {SUMMARY: load_summary, STATS: load_stats}
    started = clock()
    order = itertools.count()
    details: deque[str] = deque(form_ids)
    counts: deque[_Task] = deque()  # ready to run once no details are left to start
    held: list[_Held] = []
    running: dict[Future[Loaded], _Task] = {}
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        while details or counts or held or running:
            now = clock()
            left = deadline_seconds - (now - started)
            if left <= 0:
                yield from _timeouts(details, counts, held, running)
                break
            while held and held[0].due <= now:
                item = heapq.heappop(held)
                counts.append((item.kind, item.form_id, item.attempts))
            while len(running) < max(1, workers) and (counts or details):
                task: _Task = (SUMMARY, details.popleft(), 0) if details else counts.popleft()
                running[pool.submit(loads[task[0]], task[1])] = task
            if not running:
                # Everything left waits for the quota: say how long, then sleep until the
                # first of it is due (at most a heartbeat).
                yield _waiting(held, now)
                sleep(max(0.0, min(held[0].due - now, HEARTBEAT_SECONDS, left)))
                continue
            next_due = held[0].due - now if held else HEARTBEAT_SECONDS
            done, _ = wait(
                running, timeout=min(next_due, HEARTBEAT_SECONDS, left), return_when=FIRST_COMPLETED
            )
            if not done:
                yield _waiting(held, clock())
            for future in done:
                kind, form_id, attempts = running.pop(future)
                try:
                    loaded = future.result()
                except RetryLaterError as exc:
                    heapq.heappush(
                        held, _Held(clock() + exc.seconds, next(order), kind, form_id, attempts)
                    )
                    continue
                except fatal as exc:
                    yield CatalogEvent("error", error_code=type(exc).__name__)
                    return
                except FormsApiError as exc:
                    status = catalog_status(exc)
                    if status == "rate_limited" and attempts + 1 < GOOGLE_RATE_LIMIT_ATTEMPTS:
                        due = clock() + GOOGLE_RATE_LIMIT_RETRY_SECONDS
                        heapq.heappush(held, _Held(due, next(order), kind, form_id, attempts + 1))
                        continue
                    yield CatalogEvent(kind, form_id, status, exc.reason or type(exc).__name__)
                    continue
                except Exception as exc:  # noqa: BLE001 - one form must not stop the catalog
                    yield CatalogEvent(kind, form_id, "api_error", type(exc).__name__)
                    continue
                yield CatalogEvent(
                    kind,
                    form_id,
                    "ok",
                    data=loaded.data,
                    fetched_at=loaded.fetched_at,
                    cache_hit=loaded.cache_hit,
                )
                if kind == SUMMARY:
                    counts.append((STATS, form_id, 0))
        yield CatalogEvent("done")
    finally:
        # Also on a closed stream: do not wait for calls in flight, start no new ones.
        pool.shutdown(wait=False, cancel_futures=True)


def _waiting(held: list[_Held], now: float) -> CatalogEvent:
    last = max((item.due for item in held), default=now)
    return CatalogEvent("waiting", seconds=round(max(0.0, last - now), 1), forms=len(held))


def _timeouts(
    details: deque[str],
    counts: deque[_Task],
    held: list[_Held],
    running: dict[Future[Loaded], _Task],
) -> Iterator[CatalogEvent]:
    pending: list[tuple[LoadKind, str]] = [(SUMMARY, form_id) for form_id in details]
    pending += [(kind, form_id) for kind, form_id, _ in counts]
    pending += [(item.kind, item.form_id) for item in held]
    pending += [(kind, form_id) for kind, form_id, _ in running.values()]
    for kind, form_id in pending:
        yield CatalogEvent(kind, form_id, "timeout", "catalog_load_deadline")
