"""The real Streamlit app talking to the real FastAPI app, in one process.

Only Google is faked. Per-service unit tests cannot see contract drift between
web and API (error codes, response fields, scope bookkeeping), which is exactly
where the "every account gets 403 on the catalog" bug lived.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from streamlit.testing.v1 import AppTest

from api.google_data_cache import clear_api_cache
from api.main import create_api_app
from core.forms_api import FormsApiError
from core.saas.container import SaaSContainer
from core.saas.errors import InvalidSession
from core.saas.google_scopes import IDENTITY_SCOPES
from tests.test_saas_api import (
    _FakeGoogleFormsClient,
    _seed_google_grant,
    _seed_user_session,
    _test_container,
)
from ui.saas_api import SaaSApiClient

APP_TIMEOUT_SECONDS = 60
# Absolute on purpose: newer Streamlit resolves a relative script path against the calling
# file's directory (tests/), older versions against the working directory.
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
RECONNECT_TEXT = "Цій сторінці потрібен доступ до Google Forms"


class _GoogleRejectsScopes(_FakeGoogleFormsClient):
    """Google's answer for a token that lacks the Drive/Forms scopes."""

    def list_forms(self, creds: Credentials) -> list[dict]:
        raise FormsApiError(
            "Не вдалося отримати список форм з Drive: Request had insufficient "
            "authentication scopes.",
            status=403,
            reason="ACCESS_TOKEN_SCOPE_INSUFFICIENT",
        )


class _GoogleRefreshesMidCall(_FakeGoogleFormsClient):
    """google-auth refreshes after a 401 in the middle of a Google call."""

    def list_forms(self, creds: Credentials) -> list[dict]:
        creds.refresh(None)
        return super().list_forms(creds)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("APP_BASE_URL", "https://app.example.com")
    clear_api_cache()


def _web_talking_to(api_app) -> mock._patch:
    """Make every SaaSApiClient call go to ``api_app`` instead of the network."""
    return mock.patch.object(
        SaaSApiClient,
        "_client",
        lambda self, timeout: TestClient(api_app, base_url=self.base_url),
    )


def _signed_in_web(session_id: str) -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT_SECONDS)
    at.session_state["saas_session_id"] = session_id
    at.session_state["saas_session_checked_at"] = datetime.now(UTC)
    at.session_state["user"] = {
        "id": "user_1",
        "email": "owner@example.com",
        "name": "Owner",
        "plan": "pilot",
    }
    return at


def _link_button_urls(at: AppTest) -> list[str]:
    return [element.proto.url for element in at.get("link_button")]


def _app_with(
    container: SaaSContainer, google_forms_client: _FakeGoogleFormsClient
) -> tuple[TestClient, object]:
    api_app = create_api_app(container, google_forms_client=google_forms_client)
    return TestClient(api_app), api_app


def test_a_record_that_overstates_its_scopes_leads_to_reconnect_not_a_raw_403() -> None:
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)  # claims Forms access, Google says the token lacks it
    _, api_app = _app_with(container, _GoogleRejectsScopes())

    with _web_talking_to(api_app):
        at = _signed_in_web(session_id)
        at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert not any("403" in error.value for error in at.error)
    assert any(RECONNECT_TEXT in warning.value for warning in at.warning)
    urls = _link_button_urls(at)
    assert len(urls) == 1
    assert urls[0].startswith("https://api.example.com/v1/auth/google/start?")
    assert "purpose=forms" in urls[0]
    assert "next_url=https%3A%2F%2Fapp.example.com%2F" in urls[0]


def test_a_plain_sign_in_is_offered_the_forms_connect_button() -> None:
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container, scopes=IDENTITY_SCOPES)
    _, api_app = _app_with(container, _FakeGoogleFormsClient())

    with _web_talking_to(api_app):
        at = _signed_in_web(session_id)
        at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert any(RECONNECT_TEXT in warning.value for warning in at.warning)
    assert len(_link_button_urls(at)) == 1


def test_revoked_consent_signs_the_user_out_of_both_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    _, api_app = _app_with(container, _GoogleRefreshesMidCall())

    def refresh(self: Credentials, request: object) -> None:
        raise RefreshError(
            "invalid_grant: Token has been expired or revoked.",
            {"error": "invalid_grant", "error_description": "Token has been expired or revoked."},
        )

    monkeypatch.setattr(Credentials, "refresh", refresh)

    with _web_talking_to(api_app):
        at = _signed_in_web(session_id)
        at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert "saas_session_id" not in at.session_state
    assert container.tokens.get_by_user("user_1") is None
    with pytest.raises(InvalidSession):
        container.session_service.validate(session_id)
