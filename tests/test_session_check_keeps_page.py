"""The session check must not move the page. Real web and API, fake Google.

The check renders hidden cookie components on some runs only. Anything that appears or goes
away above the page shifts it, and the browser builds the catalog table anew (it loses its
scroll and blinks), twice per click: once when the components appear, once when they go.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest import mock

from streamlit.testing.v1 import AppTest

from tests.test_catalog_page import (  # noqa: F401 - the fixture switches the web to production
    _app_with,
    _ManyOpenForms,
    _position_of_table,
    _production_web,
    _seed_google_grant,
    _seed_user_session,
    _signed_in_web,
    _test_container,
    _web_talking_to,
)
from tests.test_e2e_web_api import APP_PATH, APP_TIMEOUT_SECONDS
from ui.saas_api import GoogleAccess, SaaSApiClient, SaaSSession

COOKIE_COMPONENT_KEY = "saas_auth_cookies"
SESSION_COOKIE = "survey_insight_session_id"


class _LongCatalog(_ManyOpenForms):
    """More forms than two loading steps: the page keeps loading and does not rerun itself."""

    def list_forms(self, creds):  # type: ignore[no-untyped-def]
        return [
            {**form, "id": f"{form['id']}_{copy}", "name": f"{form['name']} {copy}"}
            for copy in range(3)
            for form in super().list_forms(creds)
        ]


def _cookie_calls(at: AppTest) -> list[str]:
    """What the cookie components rendered in the last run ask the browser to do."""
    calls: list[str] = []

    def walk(node) -> None:  # type: ignore[no-untyped-def]
        for child in getattr(node, "children", {}).values():
            if getattr(child, "type", None) == "component_instance":
                calls.append(json.loads(child.proto.json_args)["method"])
            walk(child)

    walk(at._tree)
    return calls


def _table(at: AppTest) -> tuple:
    return _position_of_table(at._tree), at.dataframe[0].proto.id


def _catalog_api():  # type: ignore[no-untyped-def]
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    _, api_app = _app_with(container, _LongCatalog())
    return session_id, api_app


def test_the_periodic_session_check_keeps_the_table_in_place() -> None:
    session_id, api_app = _catalog_api()

    with _web_talking_to(api_app):
        at = _signed_in_web(session_id)
        at.run()
        without_check = _table(at)
        # The check runs again once the last one is older than 30 s.
        stale = datetime.now(UTC) - timedelta(minutes=1)
        at.session_state["saas_session_checked_at"] = stale
        at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["saas_session_checked_at"] > stale  # the API was asked again
    assert _table(at) == without_check
    assert _cookie_calls(at) == []


def test_a_page_load_reads_the_cookie_without_rewriting_it() -> None:
    session_id, api_app = _catalog_api()

    with _web_talking_to(api_app):
        at = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT_SECONDS)
        at.run()  # the cookie component has not answered yet
        at.session_state[COOKIE_COMPONENT_KEY] = {SESSION_COOKIE: session_id}  # its answer
        at.run()
        on_load = (_cookie_calls(at), _table(at))
        at.run()
        next_run = (_cookie_calls(at), _table(at))

    assert not at.exception, [e.value for e in at.exception]
    # The component goes away on the next run; the table below must not move.
    assert next_run[1] == on_load[1]
    # The cookie lives as long as the API session (30 days from login): nothing to rewrite.
    assert on_load[0] == ["getAll"]
    assert next_run[0] == []


def test_login_writes_the_session_cookie() -> None:
    session = SaaSSession(
        authenticated=True,
        user_id="user_1",
        email="owner@example.com",
        name="Owner",
        plan="pilot",
        session_id="sid-1",
    )
    with (
        mock.patch.object(SaaSApiClient, "exchange_login_ticket", return_value=session),
        mock.patch.object(
            SaaSApiClient,
            "check_google_access",
            return_value=GoogleAccess(ok=True, purpose="forms"),
        ),
        mock.patch.object(SaaSApiClient, "list_forms", return_value=[]),
    ):
        at = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT_SECONDS)
        at.query_params["login_ticket"] = "ticket-1"
        at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["saas_session_id"] == "sid-1"
    assert "set" in _cookie_calls(at)
