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
