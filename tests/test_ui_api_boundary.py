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
    calls: dict = {"logout": 0, "warning": [], "error": [], "link": []}
    monkeypatch.setattr(boundary.st, "session_state", {"saas_session_id": "raw-sid"})
    monkeypatch.setattr(
        boundary, "_sign_out", lambda: calls.__setitem__("logout", calls["logout"] + 1)
    )
    monkeypatch.setattr(boundary.st, "warning", lambda msg, *a, **k: calls["warning"].append(msg))
    monkeypatch.setattr(boundary.st, "error", lambda msg, *a, **k: calls["error"].append(msg))
    monkeypatch.setattr(
        boundary.st, "link_button", lambda label, url, **k: calls["link"].append((label, url))
    )

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


def _forms_scope_error(
    connect_url: str = "https://api.example.com/v1/auth/google/start",
) -> Exception:
    return MissingGoogleScopesError(purpose="forms", missing_scopes=[], connect_url=connect_url)


def test_forms_scope_error_offers_reconnect_instead_of_a_traceback(ui_calls: dict) -> None:
    """A token without the Forms scope must lead the user to grant it, not to a raw 403."""
    with pytest.raises(_StopError):
        _raising(_forms_scope_error())()

    assert len(ui_calls["warning"]) == 1
    assert ui_calls["link"] == [
        ("Підключити Google Forms", "https://api.example.com/v1/auth/google/start")
    ]
    assert ui_calls["logout"] == 0


def test_forms_scope_error_without_a_url_still_stops_with_a_message(ui_calls: dict) -> None:
    with pytest.raises(_StopError):
        _raising(_forms_scope_error(connect_url=""))()

    assert len(ui_calls["warning"]) == 1
    assert ui_calls["link"] == []


def test_access_check_leaves_scope_errors_to_its_caller(ui_calls: dict) -> None:
    """For the decision call a scope shortfall is the normal answer, not a failure."""

    @boundary.handle_access_check_errors
    def check() -> str:
        raise _forms_scope_error()

    with pytest.raises(MissingGoogleScopesError):
        check()

    assert ui_calls["link"] == []


def test_access_check_still_signs_out_on_an_invalid_session(ui_calls: dict) -> None:
    @boundary.handle_access_check_errors
    def check() -> str:
        raise SessionExpiredError("x")

    with pytest.raises(_RerunError):
        check()

    assert ui_calls["logout"] == 1
