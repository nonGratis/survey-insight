"""core.catalog_stream: one pass over the catalog, each result reported when it is ready."""

from __future__ import annotations

import threading

import pytest

from core import catalog_stream
from core.catalog_stream import CatalogEvent, Loaded, RetryLaterError, load_catalog
from core.forms_api import FormsApiError


def _hold(seconds: float) -> RetryLaterError:
    error = RetryLaterError("held by the quota guard")
    error.seconds = seconds
    return error


def _summary(form_id: str) -> Loaded:
    return Loaded({"title": form_id}, fetched_at="2026-10-07T10:00:00+00:00")


def _stats(form_id: str) -> Loaded:
    return Loaded({"total": 3}, cache_hit=True)


def _brief(events: list[CatalogEvent]) -> list[tuple]:
    return [(e.event, e.form_id, e.status) for e in events if e.event != "waiting"]


def test_every_forms_details_come_before_the_counts_and_the_load_ends_with_done() -> None:
    events = list(load_catalog(["a", "b"], load_summary=_summary, load_stats=_stats, workers=1))

    # Statuses first: no count, a Google call each, holds back the next form's details.
    assert _brief(events) == [
        ("summary", "a", "ok"),
        ("summary", "b", "ok"),
        ("stats", "a", "ok"),
        ("stats", "b", "ok"),
        ("done", None, None),
    ]
    assert events[0].data == {"title": "a"}
    assert events[0].fetched_at == "2026-10-07T10:00:00+00:00"
    assert events[2].cache_hit is True


def test_a_form_whose_details_fail_gets_no_count() -> None:
    def summary(form_id: str) -> Loaded:
        if form_id == "gone":
            raise FormsApiError("not found", status=404)
        return _summary(form_id)

    events = list(load_catalog(["gone", "ok"], load_summary=summary, load_stats=_stats))

    assert ("summary", "gone", "deleted") in _brief(events)
    assert ("stats", "gone", "ok") not in _brief(events)
    assert ("stats", "ok", "ok") in _brief(events)


def test_a_count_held_by_the_quota_runs_when_its_slot_frees() -> None:
    calls: list[str] = []

    def stats(form_id: str) -> Loaded:
        calls.append(form_id)
        if len(calls) == 1:
            raise _hold(0.2)
        return _stats(form_id)

    events = list(load_catalog(["a"], load_summary=_summary, load_stats=stats))

    assert _brief(events) == [("summary", "a", "ok"), ("stats", "a", "ok"), ("done", None, None)]
    assert calls == ["a", "a"]
    # While nothing else runs, the stream says how many loads wait and for how long.
    [waiting, *_] = [e for e in events if e.event == "waiting"]
    assert waiting.forms == 1
    assert 0 < waiting.seconds <= 0.2


def test_google_rate_limits_are_retried_a_few_times(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog_stream, "GOOGLE_RATE_LIMIT_RETRY_SECONDS", 0.01)
    calls: list[str] = []

    def stats(form_id: str) -> Loaded:
        calls.append(form_id)
        raise FormsApiError("rate limited", status=429)

    events = list(load_catalog(["a"], load_summary=_summary, load_stats=stats))

    assert len(calls) == catalog_stream.GOOGLE_RATE_LIMIT_ATTEMPTS
    assert ("stats", "a", "rate_limited") in _brief(events)


def test_a_fatal_error_stops_the_whole_load() -> None:
    class GrantRevokedError(Exception):
        pass

    def summary(form_id: str) -> Loaded:
        raise GrantRevokedError

    events = list(
        load_catalog(
            ["a", "b"],
            load_summary=summary,
            load_stats=_stats,
            workers=1,
            fatal=(GrantRevokedError,),
        )
    )

    assert _brief(events) == [("error", None, None)]
    assert events[-1].error_code == "GrantRevokedError"


def test_loads_pending_at_the_deadline_end_as_timeouts() -> None:
    def stats(form_id: str) -> Loaded:
        raise _hold(30)

    events = list(
        load_catalog(["a"], load_summary=_summary, load_stats=stats, deadline_seconds=0.2)
    )

    assert _brief(events) == [
        ("summary", "a", "ok"),
        ("stats", "a", "timeout"),
        ("done", None, None),
    ]


def test_a_closed_stream_starts_no_new_loads() -> None:
    started: list[str] = []
    release = threading.Event()

    def summary(form_id: str) -> Loaded:
        started.append(form_id)
        release.wait(2)
        return _summary(form_id)

    stream = load_catalog(["a", "b", "c"], load_summary=summary, load_stats=_stats, workers=1)
    release.set()
    next(stream)  # the first form's details
    stream.close()  # the page went away

    assert started == ["a"]


def test_events_serialise_without_empty_fields() -> None:
    event = CatalogEvent("stats", "a", "ok", data={"total": 1}, cache_hit=False)

    assert event.to_dict() == {
        "event": "stats",
        "form_id": "a",
        "status": "ok",
        "data": {"total": 1},
        "cache_hit": False,
    }
