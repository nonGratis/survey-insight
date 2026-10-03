"""Session flows exercised through Streamlit's real script runner (AppTest).

Unit tests with a patched ``st`` cannot show that ``st.rerun()``/``st.stop()``
behave inside cached functions, or that the whole app lands on the login UI.
These run the actual ``app.py`` with the network layer replaced.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import pytest
from streamlit.testing.v1 import AppTest

from ui.saas_api import (
    GoogleAccess,
    GoogleTokenRevokedError,
    SaaSApiClient,
    SaaSSession,
)

APP_TIMEOUT_SECONDS = 60
# Absolute on purpose: newer Streamlit resolves a relative script path against the calling
# file's directory (tests/), older versions against the working directory.
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def saas_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("API_BASE_URL", "https://api.test")
    monkeypatch.setenv("APP_BASE_URL", "https://app.test")


def _app_with_validated_session() -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT_SECONDS)
    at.session_state["saas_session_id"] = "sid-1"
    at.session_state["saas_session_checked_at"] = datetime.now(UTC)
    at.session_state["user"] = {"id": "user_a", "email": "a@x.com", "name": "A", "plan": "pilot"}
    return at


def test_revoked_grant_signs_out_through_the_api_and_shows_no_traceback() -> None:
    with (
        mock.patch.object(
            SaaSApiClient,
            "check_google_access",
            return_value=GoogleAccess(ok=True, purpose="forms"),
        ),
        mock.patch.object(
            SaaSApiClient, "list_forms", side_effect=GoogleTokenRevokedError("revoked")
        ),
        mock.patch.object(SaaSApiClient, "logout") as logout,
        mock.patch.object(
            SaaSApiClient, "read_session", return_value=SaaSSession(authenticated=False)
        ),
    ):
        at = _app_with_validated_session()
        at.run()
        first_pass_subheaders = [s.value for s in at.subheader]
        at.run()  # the cookie component answers and Streamlit reruns

    assert not at.exception
    assert logout.call_count == 1
    assert logout.call_args.args == ("sid-1",)
    assert "saas_session_id" not in at.session_state
    assert "Вхід" in [s.value for s in at.subheader], first_pass_subheaders


def test_api_cold_start_keeps_the_session_and_falls_back_to_login_after_retries() -> None:
    with (
        mock.patch.object(SaaSApiClient, "read_session", return_value=None) as read_session,
        # Replace only the module's own ``time`` name: patching ``time.sleep`` itself
        # would also hit Streamlit's internal polling loops.
        mock.patch("ui.components.auth_widget.time") as fake_time,
    ):
        at = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT_SECONDS)
        at.session_state["saas_session_id"] = "sid-1"
        at.run()

    assert not at.exception
    # first check + 3 retries, each retry preceded by a pause
    assert read_session.call_count == 4
    assert fake_time.sleep.call_count == 3
    assert "Вхід" in [s.value for s in at.subheader]
    # a slow API says nothing about the session: it must not be discarded
    assert at.session_state["saas_session_id"] == "sid-1"


def test_login_screen_shows_the_deployed_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_VERSION", "cd3b096")
    monkeypatch.setenv("APP_BUILD_DATE", "2026-10-03")
    with mock.patch.object(
        SaaSApiClient, "read_session", return_value=SaaSSession(authenticated=False)
    ):
        at = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT_SECONDS)
        at.session_state["saas_session_id"] = "sid-1"
        at.run()

    assert not at.exception
    assert "Вхід" in [s.value for s in at.subheader]
    assert "Версія cd3b096 · 03.10.2026" in [c.value for c in at.caption]


def test_signed_in_sidebar_shows_the_deployed_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_VERSION", "cd3b096")
    monkeypatch.delenv("APP_BUILD_DATE", raising=False)
    with (
        mock.patch.object(
            SaaSApiClient,
            "check_google_access",
            return_value=GoogleAccess(ok=True, purpose="forms"),
        ),
        mock.patch.object(SaaSApiClient, "list_forms", return_value=[]),
    ):
        at = _app_with_validated_session()
        at.run()

    assert not at.exception
    assert "Версія cd3b096" in [c.value for c in at.sidebar.caption]


_CACHED_BOUNDARY_APP = """
import streamlit as st
from ui.api_boundary import handle_api_errors
from ui.saas_api import ApiServerError, GoogleUnavailableError

@handle_api_errors
def fetch(kind):
    if kind == "502":
        raise GoogleUnavailableError("upstream")
    raise ApiServerError(503, "boom")

@st.cache_data
def cached(kind, token):
    return fetch(kind)

st.write("before")
cached(st.session_state["kind"], "tok")
st.write("after")
"""


@pytest.mark.parametrize(
    ("kind", "expected_warning", "expected_error"),
    [
        ("502", "Тимчасова проблема з Google API. Спробуй оновити сторінку.", None),
        ("503", None, "Сервіс тимчасово недоступний. Код для підтримки:"),
    ],
)
def test_boundary_stops_the_page_from_inside_a_cache_data_function(
    kind: str, expected_warning: str | None, expected_error: str | None
) -> None:
    at = AppTest.from_string(_CACHED_BOUNDARY_APP, default_timeout=APP_TIMEOUT_SECONDS)
    at.session_state["kind"] = kind
    at.run()

    assert not at.exception
    assert [m.value for m in at.markdown] == ["before"]  # st.stop() ended the script
    if expected_warning:
        assert [w.value for w in at.warning] == [expected_warning]
    if expected_error:
        assert len(at.error) == 1
        assert at.error[0].value.startswith(expected_error)
