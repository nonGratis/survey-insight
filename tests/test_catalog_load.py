from __future__ import annotations

import queue
from collections.abc import Iterator
from typing import Any

from ui.catalog_load import IDLE_STOP_SECONDS, CatalogLoad
from ui.saas_api import GoogleTokenRevokedError, GoogleUnavailableError


def _form(title: str, **extra: Any) -> dict[str, Any]:
    return {
        "title": title,
        "description": "",
        "sections_count": 1,
        "questions_count": 3,
        "linked_sheet_id": None,
        "is_published": True,
        "accepting_responses": True,
        **extra,
    }


def _count(total: int) -> dict[str, Any]:
    return {"total": total, "first_response": None, "second_response": None, "last_response": None}


def _finished(*events: dict[str, Any], clock: Any = None) -> CatalogLoad:
    def stream(form_ids: list[str]) -> Iterator[dict[str, Any]]:
        yield from events

    load = CatalogLoad(["a", "b"], stream, **({"clock": clock} if clock else {})).start()
    load.join(5)
    return load


def test_details_and_counts_land_in_the_snapshot() -> None:
    summary = _form("A", added_by_api=1)
    load = _finished(
        {
            "event": "summary",
            "form_id": "a",
            "status": "ok",
            "data": summary,
            "fetched_at": "2026-10-07T10:00:00+00:00",
        },
        {"event": "stats", "form_id": "a", "status": "ok", "data": _count(4)},
        {"event": "done"},
    )

    snapshot = load.snapshot()
    form = snapshot.summaries["a"]
    # A field the API added later is dropped, not a crash.
    assert form is not None and form.title == "A" and form.questions_count == 3
    assert snapshot.stats["a"].total == 4
    assert snapshot.statuses == {"a": "ok"}
    assert snapshot.fetched_at == {"a": "2026-10-07T10:00:00+00:00"}
    assert snapshot.finished == {"a"}
    assert snapshot.done and snapshot.error is None


def test_a_failed_form_is_finished_with_its_status() -> None:
    load = _finished(
        {"event": "summary", "form_id": "a", "status": "no_access", "error_code": "HttpError"},
        {"event": "summary", "form_id": "b", "status": "ok", "data": _form("B")},
        {"event": "stats", "form_id": "b", "status": "rate_limited"},
    )

    snapshot = load.snapshot()
    assert snapshot.summaries["a"] is None
    assert snapshot.statuses == {"a": "no_access", "b": "rate_limited"}
    assert "b" not in snapshot.stats
    assert snapshot.finished == {"a", "b"}


def test_the_wait_counts_down_and_clears() -> None:
    now = [100.0]

    def clock() -> float:
        return now[0]

    load = _finished({"event": "waiting", "seconds": 40.0, "forms": 2}, clock=clock)
    now[0] = 110.0
    assert load.snapshot().wait_seconds == 30.0
    now[0] = 200.0
    assert load.snapshot().wait_seconds == 0.0

    cleared = _finished(
        {"event": "waiting", "seconds": 40.0, "forms": 2},
        {"event": "waiting", "seconds": 0.0, "forms": 0},
    )
    assert cleared.snapshot().wait_seconds is None


def test_error_lines_become_the_errors_the_page_handles() -> None:
    revoked = _finished({"event": "error", "error_code": "GoogleTokenRevoked"})
    refresh = _finished({"event": "error", "error_code": "GoogleTokenRefreshFailed"})
    other = _finished({"event": "error", "error_code": "Boom"})

    assert isinstance(revoked.snapshot().error, GoogleTokenRevokedError)
    assert isinstance(refresh.snapshot().error, GoogleUnavailableError)
    assert isinstance(other.snapshot().error, RuntimeError)
    assert revoked.snapshot().done


def test_a_failing_stream_ends_the_load_with_its_error() -> None:
    def stream(form_ids: list[str]) -> Iterator[dict[str, Any]]:
        yield {"event": "summary", "form_id": "a", "status": "ok", "data": _form("A")}
        raise ConnectionError("API went away")

    load = CatalogLoad(["a"], stream).start()
    load.join(5)

    snapshot = load.snapshot()
    assert snapshot.done
    assert isinstance(snapshot.error, ConnectionError)
    assert snapshot.summaries["a"] is not None


def test_a_load_nobody_looks_at_stops_and_closes_its_stream() -> None:
    now = [0.0]
    closed: list[bool] = []

    def stream(form_ids: list[str]) -> Iterator[dict[str, Any]]:
        try:
            yield {"event": "summary", "form_id": "a", "status": "ok", "data": _form("A")}
            now[0] += IDLE_STOP_SECONDS + 1  # the browser tab is gone: no snapshot since
            yield {"event": "summary", "form_id": "b", "status": "ok", "data": _form("B")}
        finally:
            closed.append(True)

    load = CatalogLoad(["a", "b"], stream, clock=lambda: now[0]).start()
    load.join(5)

    assert load.stopped
    assert closed == [True]
    assert set(load.snapshot().summaries) == {"a"}


def test_stop_closes_the_stream_at_the_next_line() -> None:
    lines: queue.Queue[dict[str, Any]] = queue.Queue()
    closed: list[bool] = []

    def stream(form_ids: list[str]) -> Iterator[dict[str, Any]]:
        try:
            while True:
                yield lines.get(timeout=5)
        finally:
            closed.append(True)

    load = CatalogLoad(["a"], stream).start()
    load.stop()
    lines.put({"event": "waiting", "seconds": 1.0, "forms": 1})
    load.join(5)

    assert load.stopped and load.snapshot().done
    assert closed == [True]
    assert load.snapshot().wait_seconds is None
