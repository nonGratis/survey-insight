"""POST /v1/forms/catalog/stream: the whole catalog in one request, one JSON line per result."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient
from google.oauth2.credentials import Credentials

from api.google_data_cache import clear_api_cache
from api.google_quota import FormsQuotaGuards, RollingWindowGuard
from api.main import SESSION_COOKIE_NAME, create_api_app
from api.routes import google_forms as google_forms_routes
from tests.test_saas_api import (
    _AnyFormCountingClient,
    _client_with_valid_grant,
    _MidCallRefreshGoogleFormsClient,
    _refresh_error,
    _seed_google_grant,
    _seed_user_session,
    _test_container,
)

STREAM = "/v1/forms/catalog/stream"


def _client(guards: FormsQuotaGuards | None = None) -> tuple[TestClient, _AnyFormCountingClient]:
    clear_api_cache()
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    google = _AnyFormCountingClient()
    app = create_api_app(container, google_forms_client=google)
    if guards is not None:
        app.state.forms_quota = guards
    client = TestClient(app)
    client.cookies.set(SESSION_COOKIE_NAME, session_id)
    return client, google


def _events(response) -> list[dict]:  # type: ignore[no-untyped-def]
    return [json.loads(line) for line in response.text.splitlines() if line]


def test_the_stream_sends_each_forms_details_and_count_then_done(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, google = _client()
    caplog.set_level(logging.INFO, logger=google_forms_routes.log.name)

    response = client.post(STREAM, json={"form_ids": ["form_a", "form_b", "form_a"]})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = _events(response)
    finished = {(e["event"], e["form_id"]) for e in events if e.get("status") == "ok"}
    assert finished == {
        ("summary", "form_a"),
        ("stats", "form_a"),
        ("summary", "form_b"),
        ("stats", "form_b"),
    }
    summary = next(e for e in events if e["event"] == "summary")
    assert summary["data"]["questions_count"] == 5
    assert "fetched_at" in summary
    assert events[-1] == {"event": "done"}
    # Duplicates asked once; one telemetry line per stream.
    assert (google.summary_calls, google.stats_calls) == (2, 2)
    [line] = [r for r in caplog.records if r.getMessage() == "forms_catalog_stream_completed"]
    assert (line.forms, line.summary_ok_count, line.count_ok_count) == (2, 2, 2)
    assert line.counts_ms is not None and line.credentials_ms >= 0


def test_counts_held_by_the_quota_wait_inside_the_stream() -> None:
    # One response list per 0.3 s: the other counts wait for their slots, here, not on the page.
    client, google = _client(
        FormsQuotaGuards(
            reads=RollingWindowGuard(100), response_lists=RollingWindowGuard(1, window_seconds=0.3)
        )
    )

    events = _events(client.post(STREAM, json={"form_ids": ["form_a", "form_b", "form_c"]}))

    counts = [e for e in events if e["event"] == "stats"]
    assert [e["status"] for e in counts] == ["ok", "ok", "ok"]
    assert google.stats_calls == 3
    assert any(e["event"] == "waiting" and e["forms"] >= 1 for e in events)


def test_a_catalog_too_large_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(google_forms_routes, "CATALOG_STREAM_MAX_FORMS", 2)
    client, _ = _client()

    response = client.post(STREAM, json={"form_ids": ["a", "b", "c"]})

    assert response.status_code == 400


def test_a_grant_revoked_during_the_stream_ends_it_with_an_error_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, container = _client_with_valid_grant(_MidCallRefreshGoogleFormsClient())

    def refresh(self: Credentials, request: object) -> None:
        raise _refresh_error("invalid_grant")

    monkeypatch.setattr(Credentials, "refresh", refresh)
    clear_api_cache()

    response = client.post(STREAM, json={"form_ids": ["form_1"]})

    # The credentials passed the check before the first line; the stream has begun, so the
    # revoked grant arrives as its last line, and the grant record is dropped as elsewhere.
    assert response.status_code == 200
    assert _events(response)[-1] == {"event": "error", "error_code": "GoogleTokenRevoked"}
    assert container.tokens.get_by_user("user_1") is None
