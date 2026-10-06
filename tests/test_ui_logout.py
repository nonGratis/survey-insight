from __future__ import annotations

import logging

import httpx
import pytest

import ui.components.auth_widget as auth_widget


class _Client:
    def __init__(self, calls: list[str], error: Exception | None = None) -> None:
        self.calls = calls
        self.error = error

    def logout(self, session_id: str | None) -> None:
        self.calls.append(f"api_logout:{session_id}")
        if self.error:
            raise self.error


class _State(dict):
    def clear(self) -> None:
        self.calls.append("state_clear")
        super().clear()

    calls: list[str]


def _setup(
    monkeypatch: pytest.MonkeyPatch, error: Exception | None = None
) -> tuple[list[str], _State]:
    calls: list[str] = []
    state = _State({"saas_session_id": "sid", "user": {"id": "user_a"}})
    state.calls = calls
    monkeypatch.setattr(auth_widget.st, "session_state", state)
    monkeypatch.setattr(auth_widget, "_saas_client", lambda url: _Client(calls, error))
    monkeypatch.setattr(
        auth_widget, "clear_google_data_cache", lambda session_id=None: calls.append("cache_clear")
    )
    monkeypatch.setattr(auth_widget, "_clear_saas_session", lambda *a, **k: calls.append("cookie"))
    return calls, state


def test_logout_calls_api_before_clearing_local_state(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, state = _setup(monkeypatch)

    auth_widget._logout_saas_session()

    assert calls[0] == "api_logout:sid"
    assert calls.index("api_logout:sid") < calls.index("state_clear")
    assert "cache_clear" in calls
    assert state == {}


@pytest.mark.parametrize(
    "error", [httpx.ReadTimeout("slow"), httpx.ConnectError("down"), httpx.HTTPError("boom")]
)
def test_logout_still_clears_state_when_api_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, error: Exception
) -> None:
    calls, state = _setup(monkeypatch, error)

    with caplog.at_level(logging.WARNING):
        auth_widget._logout_saas_session()

    assert "saas_logout_failed" in caplog.text
    assert "cache_clear" in calls
    assert "state_clear" in calls
    assert state == {}


class _RerunError(Exception):
    pass


def _restore_setup(monkeypatch: pytest.MonkeyPatch, session_result: object) -> dict:
    state: dict = {"saas_session_id": "sid"}
    monkeypatch.setattr(auth_widget.st, "session_state", state)
    monkeypatch.setattr(auth_widget, "_query_param", lambda name: None)
    monkeypatch.setattr(auth_widget, "_handle_saas_login_ticket", lambda: False)
    monkeypatch.setattr(
        auth_widget, "_cookie_manager", lambda: type("M", (), {"get": lambda *a: None})()
    )
    monkeypatch.setattr(auth_widget, "_api_base_url", lambda: "https://api.example.com")
    monkeypatch.setattr(
        auth_widget,
        "_saas_client",
        lambda url: type("C", (), {"read_session": lambda self, sid: session_result})(),
    )
    monkeypatch.setattr(auth_widget.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(auth_widget.time, "sleep", lambda s: None)

    def rerun() -> None:
        raise _RerunError

    monkeypatch.setattr(auth_widget.st, "rerun", rerun)
    monkeypatch.setattr(
        auth_widget, "_clear_saas_session", lambda *a, **k: pytest.fail("session must be kept")
    )
    return state


def test_unavailable_api_retries_then_falls_back_to_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _restore_setup(monkeypatch, None)

    for _ in range(auth_widget.SESSION_RESTORE_MAX_RETRIES):
        with pytest.raises(_RerunError):
            auth_widget._restore_saas_session(manage_cookie=True)

    assert auth_widget._restore_saas_session(manage_cookie=True) is False
    assert state.get("saas_session_retries", 0) == 0
    assert state["saas_session_id"] == "sid"


def test_successful_session_check_resets_retry_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    from ui.saas_api import SaaSSession

    session = SaaSSession(authenticated=True, user_id="user_a", session_id="sid")
    state = _restore_setup(monkeypatch, session)
    state["saas_session_retries"] = 2
    monkeypatch.setattr(auth_widget, "_remember_saas_session", lambda *a, **k: None)

    assert auth_widget._restore_saas_session(manage_cookie=True) is True
    assert "saas_session_retries" not in state
