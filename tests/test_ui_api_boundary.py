from __future__ import annotations

import logging

import pytest

import ui.api_boundary as boundary
from ui.saas_api import (
    ApiServerError,
    GoogleTokenRevokedError,
    GoogleUnavailableError,
    MissingGoogleScopesError,
    SessionExpiredError,
)


class _StopError(BaseException):
    pass


class _RerunError(BaseException):
    pass


@pytest.fixture
def ui_calls(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls: dict = {"logout": 0, "warning": [], "error": []}
    monkeypatch.setattr(boundary.st, "session_state", {"saas_session_id": "raw-sid"})
    monkeypatch.setattr(
        boundary, "_sign_out", lambda: calls.__setitem__("logout", calls["logout"] + 1)
    )
    monkeypatch.setattr(boundary.st, "warning", lambda msg, *a, **k: calls["warning"].append(msg))
    monkeypatch.setattr(boundary.st, "error", lambda msg, *a, **k: calls["error"].append(msg))

    def stop() -> None:
        raise _StopError

    def rerun() -> None:
        raise _RerunError

    monkeypatch.setattr(boundary, "_in_script_thread", lambda: True)
    monkeypatch.setattr(boundary.st, "stop", stop)
    monkeypatch.setattr(boundary.st, "rerun", rerun)
    return calls


def _raising(error: BaseException):
    @boundary.handle_api_errors
    def call() -> str:
        raise error

    return call


@pytest.mark.parametrize("error", [SessionExpiredError("x"), GoogleTokenRevokedError("x")])
def test_expired_or_revoked_signs_out_and_reruns(ui_calls: dict, error: Exception) -> None:
    with pytest.raises(_RerunError):
        _raising(error)()

    assert ui_calls["logout"] == 1


def test_google_unavailable_warns_and_stops(ui_calls: dict) -> None:
    with pytest.raises(_StopError):
        _raising(GoogleUnavailableError("x"))()

    assert ui_calls["warning"] == ["Тимчасова проблема з Google API. Спробуй оновити сторінку."]
    assert ui_calls["logout"] == 0


def test_other_5xx_logs_error_and_shows_reference_without_raw_session_id(
    ui_calls: dict, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR), pytest.raises(_StopError):
        _raising(ApiServerError(503, "boom"))()

    assert "ui_saas_api_server_error" in caplog.text
    assert len(ui_calls["error"]) == 1
    assert "raw-sid" not in ui_calls["error"][0]
    assert "raw-sid" not in caplog.text


def test_scope_errors_and_other_exceptions_pass_through_untouched(ui_calls: dict) -> None:
    scope = MissingGoogleScopesError(purpose="sheets", missing_scopes=[], connect_url="u")
    with pytest.raises(MissingGoogleScopesError):
        _raising(scope)()
    with pytest.raises(ValueError):
        _raising(ValueError("x"))()


def test_success_returns_the_value(ui_calls: dict) -> None:
    @boundary.handle_api_errors
    def call(x: int) -> int:
        return x + 1

    assert call(1) == 2


def test_errors_outside_the_script_thread_propagate_unchanged(
    ui_calls: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(boundary, "_in_script_thread", lambda: False)

    with pytest.raises(SessionExpiredError):
        _raising(SessionExpiredError("x"))()
    with pytest.raises(GoogleUnavailableError):
        _raising(GoogleUnavailableError("x"))()

    assert ui_calls["logout"] == 0
    assert ui_calls["warning"] == []


def test_every_google_data_facade_function_goes_through_the_boundary() -> None:
    import ast
    from pathlib import Path

    tree = ast.parse((Path(boundary.__file__).parent / "google_data.py").read_text("utf-8"))
    unguarded = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and "return google_data_client()." in ast.unparse(node)
        and not any("handle_api_errors" in ast.unparse(dec) for dec in node.decorator_list)
    ]

    assert unguarded == []
