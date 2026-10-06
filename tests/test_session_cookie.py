"""The sign-in cookie: written at login until the browser confirms it, deleted on sign-out.

Runs the real ``app.py`` (AppTest) with the network layer replaced. A cookie component
answers through its widget value; AppTest has no browser, so the tests set that value.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from unittest import mock

import pytest
from streamlit.testing.v1 import AppTest

from ui.saas_api import GoogleAccess, GoogleTokenRevokedError, SaaSApiClient, SaaSSession

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
APP_TIMEOUT_SECONDS = 60
COOKIE_COMPONENT_KEY = "saas_auth_cookies"
SESSION_COOKIE = "survey_insight_session_id"
SESSION = SaaSSession(
    authenticated=True,
    user_id="user_1",
    email="owner@example.com",
    name="Owner",
    plan="pilot",
    session_id="sid-cookie-login",
)
# A session id of its own: the catalog's st.cache_data is keyed by it and outlives a test.
REVOKED_SESSION = dataclasses.replace(SESSION, session_id="sid-cookie-revoked")


@pytest.fixture(autouse=True)
def saas_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("API_BASE_URL", "https://api.test")
    monkeypatch.setenv("APP_BASE_URL", "https://app.test")


def _cookie_calls(at: AppTest) -> list[str]:
    calls: list[str] = []

    def walk(node) -> None:  # type: ignore[no-untyped-def]
        for child in getattr(node, "children", {}).values():
            if getattr(child, "type", None) == "component_instance":
                calls.append(json.loads(child.proto.json_args)["method"])
            walk(child)

    walk(at._tree)
    return calls


def _api_with_an_empty_catalog():  # type: ignore[no-untyped-def]
    return (
        mock.patch.object(SaaSApiClient, "exchange_login_ticket", return_value=SESSION),
        mock.patch.object(
            SaaSApiClient,
            "check_google_access",
            return_value=GoogleAccess(ok=True, purpose="forms"),
        ),
        mock.patch.object(SaaSApiClient, "list_forms", return_value=[]),
    )


def test_the_login_cookie_is_written_until_the_browser_confirms_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    exchange, access, forms = _api_with_an_empty_catalog()
    with exchange, access, forms, caplog.at_level(logging.INFO):
        at = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT_SECONDS)
        at.query_params["login_ticket"] = "ticket-1"
        at.run()
        on_login = _cookie_calls(at)
        write = at.session_state["saas_cookie_write"]
        at.query_params.clear()
        at.run()  # the browser has not confirmed the write yet
        unconfirmed = _cookie_calls(at)
        at.session_state[write["key"]] = True  # the component answers once it wrote the cookie
        at.run()

    assert not at.exception, [e.value for e in at.exception]
    # One render can be removed by the next run before the browser carries it out (prod
    # 2026-10-06: the cookie was gone at the next visit), so the write stays until confirmed.
    assert "set" in on_login
    assert "set" in unconfirmed
    assert "set" not in _cookie_calls(at)
    assert "saas_cookie_write" not in at.session_state
    assert "ui_session_cookie_saved" in caplog.text


def test_a_revoked_google_grant_at_page_load_signs_out_without_an_error() -> None:
    with (
        mock.patch.object(SaaSApiClient, "read_session", return_value=REVOKED_SESSION),
        mock.patch.object(
            SaaSApiClient,
            "check_google_access",
            return_value=GoogleAccess(ok=True, purpose="forms"),
        ),
        mock.patch.object(
            SaaSApiClient, "list_forms", side_effect=GoogleTokenRevokedError("revoked")
        ),
        mock.patch.object(SaaSApiClient, "logout") as logout,
    ):
        at = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT_SECONDS)
        at.run()  # the cookie component has not answered yet
        at.session_state[COOKIE_COMPONENT_KEY] = {SESSION_COOKIE: "sid-cookie-revoked"}
        # Restores from the cookie (its component is on the page), then the catalog learns
        # that Google revoked the grant and signs out from inside a cached function.
        at.run()
        signed_out = [s.value for s in at.subheader]
        at.session_state[COOKIE_COMPONENT_KEY] = {
            SESSION_COOKIE: "sid-cookie-revoked",
            "other": "x",
        }
        at.run()  # the cookie component answers on the sign-in page
        deleting = _cookie_calls(at)
        delete_key = at.session_state["saas_cookie_delete"]
        at.session_state[delete_key] = True  # the browser confirms the delete
        # The cookie component on the page read the cookies before the delete and keeps
        # reporting that answer until the page reloads.
        at.session_state[COOKIE_COMPONENT_KEY] = {
            SESSION_COOKIE: "sid-cookie-revoked",
            "other": "x",
        }
        at.run()

    # Prod 2026-10-06: the sign-out rendered a second cookie component inside the cached
    # catalog load and failed with StreamlitDuplicateElementKey.
    assert not at.exception, [e.value for e in at.exception]
    assert logout.call_count == 1
    assert "Вхід" in signed_out
    assert "delete" in deleting
    assert "saas_cookie_delete" not in at.session_state
    # The stale answer does not sign the tab in again (nor start another delete).
    assert "Вхід" in [s.value for s in at.subheader]
    assert "delete" not in _cookie_calls(at)
