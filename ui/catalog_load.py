"""One catalog load in a background thread, and what has arrived of it so far.

The page starts a load and draws snapshot() on every run; it never waits for Google itself.
Plain Python on purpose: the thread only fills this object, it never calls Streamlit.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields
from typing import Any

from core.forms_catalog import FormEnrichment, ResponseStats
from ui.saas_api import GoogleTokenRevokedError, GoogleUnavailableError

CatalogEvents = Callable[[list[str]], Iterator[dict[str, Any]]]
# A load nobody has looked at for this long stops: its browser tab was closed or
# reloaded, or the user went to another page. Its stream closes, and with it the API's
# Google calls, which would otherwise eat the quota of the load that replaces it. The
# page looks every few seconds while it loads; the API sends a line at least every 5 s.
IDLE_STOP_SECONDS = 15.0


@dataclass(frozen=True)
class CatalogSnapshot:
    """The load as the page draws it.

    ``summaries`` maps a form to its details, or to None when they failed; ``finished``
    holds the forms whose loading is over (count arrived, or failed, or details failed);
    ``statuses`` is the last status of each form (``ok`` unless a load failed);
    ``wait_seconds`` is the time until the last count held by the Google quota runs.
    """

    summaries: dict[str, FormEnrichment | None]
    stats: dict[str, ResponseStats]
    statuses: dict[str, str]
    fetched_at: dict[str, str]
    finished: frozenset[str]
    wait_seconds: float | None
    done: bool
    error: Exception | None


class CatalogLoad:
    """Reads the events of one catalog load (core.catalog_stream) in a daemon thread."""

    def __init__(
        self,
        form_ids: list[str],
        events: CatalogEvents,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.form_ids = list(form_ids)
        self._events = events
        self._clock = clock
        self._lock = threading.Lock()
        self._summaries: dict[str, FormEnrichment | None] = {}
        self._stats: dict[str, ResponseStats] = {}
        self._statuses: dict[str, str] = {}
        self._fetched_at: dict[str, str] = {}
        self._finished: set[str] = set()
        self._wait_until: float | None = None
        self._done = False
        self._error: Exception | None = None
        self._seen = clock()
        self._stop = False
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="catalog-load", daemon=True)

    def start(self) -> CatalogLoad:
        self._thread.start()
        return self

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def stop(self) -> None:
        """Stop at the next line of the stream and close it."""
        with self._lock:
            self._stop = True

    @property
    def stopped(self) -> bool:
        """Stopped before its end (stop(), or nobody looked): a new load takes its place."""
        with self._lock:
            return self._stopped

    def snapshot(self) -> CatalogSnapshot:
        with self._lock:
            self._seen = self._clock()
            wait = None if self._wait_until is None else max(0.0, self._wait_until - self._clock())
            return CatalogSnapshot(
                summaries=dict(self._summaries),
                stats=dict(self._stats),
                statuses=dict(self._statuses),
                fetched_at=dict(self._fetched_at),
                finished=frozenset(self._finished),
                wait_seconds=wait,
                done=self._done,
                error=self._error,
            )

    def _run(self) -> None:
        events = self._events(self.form_ids)
        try:
            for event in events:
                with self._lock:
                    if self._stop or self._clock() - self._seen > IDLE_STOP_SECONDS:
                        self._stopped = True
                        break
                    self._apply(event)
        except Exception as exc:  # noqa: BLE001 - shown on the page, not lost in the thread
            with self._lock:
                self._error = exc
        finally:
            # Closing the generator closes the HTTP stream; the API stops calling Google.
            close = getattr(events, "close", None)
            if close is not None:
                close()
            with self._lock:
                self._done = True

    def _apply(self, event: dict[str, Any]) -> None:
        kind = event.get("event")
        form_id = str(event.get("form_id") or "")
        status = str(event.get("status") or "ok")
        if kind == "summary":
            self._statuses[form_id] = status
            if status == "ok":
                self._summaries[form_id] = _build(FormEnrichment, event.get("data") or {})
                if event.get("fetched_at"):
                    self._fetched_at[form_id] = str(event["fetched_at"])
            else:
                self._summaries[form_id] = None
                self._finished.add(form_id)
        elif kind == "stats":
            if status == "ok":
                self._stats[form_id] = _build(ResponseStats, event.get("data") or {})
            else:
                self._statuses[form_id] = status
            self._finished.add(form_id)
        elif kind == "waiting":
            seconds = event.get("seconds")
            waiting = bool(event.get("forms")) and isinstance(seconds, int | float)
            self._wait_until = self._clock() + float(seconds) if waiting else None
        elif kind == "error":
            self._error = _load_error(str(event.get("error_code") or ""))
        elif kind == "done":
            # The API is done even if the connection lingers a moment before it closes.
            self._done = True


def _build[T](cls: type[T], data: dict[str, Any]) -> T:
    """The dataclass from the fields it knows: a field the API adds must not break the page."""
    known = {field.name for field in fields(cls)}  # type: ignore[arg-type]
    return cls(**{key: value for key, value in data.items() if key in known})


def _load_error(code: str) -> Exception:
    """The error the page's API boundary knows for an ``error`` line of the stream."""
    if code == "GoogleTokenRevoked":
        return GoogleTokenRevokedError("Google revoked the grant during the catalog load.")
    if code == "GoogleTokenRefreshFailed":
        return GoogleUnavailableError("Google could not refresh the grant.")
    return RuntimeError(f"Catalog load stopped: {code or 'unknown error'}")
