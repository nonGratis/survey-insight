"""Analytics-ready logs: who and what each web record belongs to, page runs, Google time.

Cloud Logging reads ``severity`` (tests/test_service_logging.py). These tests cover the
fields the analysis groups by: ``session_ref`` and ``user_id`` on web records, one
``ui_page_run`` per page run, and the time split in ``forms_catalog_stream_completed``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from streamlit.runtime.scriptrunner_utils.exceptions import RerunException, StopException
from streamlit.runtime.scriptrunner_utils.script_requests import RerunData
from streamlit.testing.v1 import AppTest

from api.google_data_cache import clear_api_cache
from api.main import create_api_app
from api.routes import google_forms as google_forms_routes
from core.saas.security import log_user_ref
from tests.test_saas_api import (
    SESSION_COOKIE_NAME,
    _AnyFormCountingClient,
    _seed_google_grant,
    _seed_user_session,
    _test_container,
)
from ui.saas_api import GoogleAccess, SaaSApiClient
from ui.telemetry import page_run

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

_FILTER_PROBE = """
import logging
import threading

import streamlit as st

from core.logger import StreamlitContextFilter
from ui.telemetry import bind_user

st.session_state["user"] = {"id": "user-42"}
bind_user()


def probe() -> logging.LogRecord:
    record = logging.LogRecord("probe", logging.INFO, "", 0, "probe", None, None)
    StreamlitContextFilter().filter(record)
    return record


in_script = probe()
in_thread: list[logging.LogRecord] = []
worker = threading.Thread(target=lambda: in_thread.append(probe()))
worker.start()
worker.join()
st.text(f"script|{getattr(in_script, 'session_ref', '')}|{getattr(in_script, 'user_id', '')}")
st.text(f"thread|{getattr(in_thread[0], 'session_ref', '')}|{getattr(in_thread[0], 'user_id', '')}|")
"""


def test_web_records_carry_the_session_and_user_of_the_page_run() -> None:
    at = AppTest.from_string(_FILTER_PROBE)
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    script, thread = (text.value.split("|") for text in at.text)
    # Streamlit runs the page script outside the main thread: the old main-thread check
    # skipped every record, so no web log could be grouped by session or user.
    assert script[0] == "script"
    assert len(script[1]) == 12
    assert script[2] == log_user_ref("user-42")  # the same digest the API logs
    # A thread the page starts has no script context: no fields, and no session state read.
    assert thread == ["thread", "", "", ""]


def _page_runs(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.getMessage() == "ui_page_run"]


@pytest.mark.parametrize(
    ("raised", "outcome"),
    [
        (None, "completed"),
        (StopException(), "stopped"),
        (RerunException(RerunData()), "rerun"),
        (ValueError("boom"), "error"),
    ],
)
def test_a_page_run_is_logged_however_it_ends(
    caplog: pytest.LogCaptureFixture, raised: BaseException | None, outcome: str
) -> None:
    caplog.set_level(logging.INFO)

    with pytest.raises(type(raised)) if raised else mock.MagicMock(), page_run("catalog"):
        if raised:
            raise raised

    [record] = _page_runs(caplog)
    assert (record.page, record.run_kind, record.outcome) == ("catalog", "full", outcome)
    assert isinstance(record.duration_ms, float)


def test_a_fragment_inside_a_full_run_is_not_counted_twice(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    with page_run("catalog", fragment=True):
        pass

    assert _page_runs(caplog) == []


def test_the_app_logs_one_page_run_per_run(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("API_BASE_URL", "https://api.test")
    monkeypatch.setenv("APP_BASE_URL", "https://app.test")
    caplog.set_level(logging.INFO)
    from datetime import UTC, datetime

    with (
        mock.patch.object(
            SaaSApiClient,
            "check_google_access",
            return_value=GoogleAccess(ok=True, purpose="forms"),
        ),
        mock.patch.object(SaaSApiClient, "list_forms", return_value=[]),
    ):
        at = AppTest.from_file(str(APP_PATH), default_timeout=60)
        at.session_state["saas_session_id"] = "sid-analytics"
        at.session_state["saas_session_checked_at"] = datetime.now(UTC)
        at.session_state["user"] = {"id": "user_1", "email": "a@x.com", "name": "A"}
        at.run()

    assert not at.exception, [e.value for e in at.exception]
    # The line is written while st.stop() unwinds. A log filter that read st.session_state
    # then got the pending stop raised at it, and the record never reached any handler.
    [record] = _page_runs(caplog)
    # An account without forms: the catalog shows its empty state and stops there.
    assert (record.page, record.run_kind, record.outcome) == ("catalog", "full", "stopped")


def test_the_stream_line_splits_the_time_between_google_and_the_rest(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clear_api_cache()
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    client = TestClient(create_api_app(container, google_forms_client=_AnyFormCountingClient()))
    client.cookies.set(SESSION_COOKIE_NAME, session_id)
    caplog.set_level(logging.INFO, logger=google_forms_routes.log.name)

    response = client.post(
        "/v1/forms/catalog/stream", json={"form_ids": ["form_a", "form_b", "form_c"]}
    )

    assert response.status_code == 200
    [record] = [r for r in caplog.records if r.getMessage() == "forms_catalog_stream_completed"]
    assert (record.google_get_count, record.google_list_count) == (3, 3)
    for field in (
        "credentials_ms",
        "google_ms_total",
        "google_get_ms_p50",
        "google_get_ms_max",
        "google_list_ms_p50",
        "google_list_ms_max",
        "duration_ms",
    ):
        assert isinstance(getattr(record, field), float), field
    assert record.duration_ms >= record.google_get_ms_max
