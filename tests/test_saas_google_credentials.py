from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import google_auth_httplib2
import httplib2
import pytest
from google.auth.exceptions import RefreshError, TransportError
from google.oauth2.credentials import Credentials

from core.saas.errors import (
    GoogleApiTemporaryError,
    GoogleTokenRefreshFailed,
    GoogleTokenRevoked,
    MissingRequiredScopes,
)
from core.saas.google_credentials import GoogleCredentialService
from core.saas.google_scopes import FORM_SCOPES, IDENTITY_SCOPES
from core.saas.inmemory import InMemoryTokenCrypto, InMemoryTokenRepository
from core.saas.models import OAuthAccount

CLIENT_CONFIG = json.dumps(
    {
        "web": {
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "client",
            "client_secret": "secret",
        }
    }
)


def _service() -> tuple[GoogleCredentialService, InMemoryTokenRepository, InMemoryTokenCrypto]:
    tokens = InMemoryTokenRepository()
    crypto = InMemoryTokenCrypto()
    service = GoogleCredentialService(
        tokens=tokens, token_crypto=crypto, client_config_json=CLIENT_CONFIG
    )
    return service, tokens, crypto


def _seed(
    tokens: InMemoryTokenRepository,
    crypto: InMemoryTokenCrypto,
    *,
    expiry: datetime | None,
    scopes: tuple[str, ...] = FORM_SCOPES,
) -> None:
    tokens.save(
        OAuthAccount(
            user_id="user_1",
            provider="google",
            google_sub="sub",
            email="owner@example.com",
            scopes=scopes,
            encrypted_access_token=crypto.encrypt("old-access"),
            encrypted_refresh_token=crypto.encrypt("refresh"),
            token_expiry=expiry,
            updated_at=datetime.now(UTC),
        )
    )


def _refresh_error(error: str, *, retryable: bool = False) -> RefreshError:
    """Shape of the error google-auth raises for a failed token-endpoint call."""
    return RefreshError(
        f"{error}: description",
        {"error": error, "error_description": "description"},
        retryable=retryable,
    )


def _fake_refresh(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    def refresh(self: Credentials, request: object) -> None:
        calls.append(self.token or "")
        self.token = "new-access"
        self.expiry = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1)

    monkeypatch.setattr(Credentials, "refresh", refresh)


def test_token_refresh_on_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    service, tokens, crypto = _service()
    _seed(tokens, crypto, expiry=datetime.now(UTC) - timedelta(minutes=5))
    calls: list[str] = []
    _fake_refresh(monkeypatch, calls)

    creds = service.credentials_for_user("user_1", required_scopes=FORM_SCOPES)

    assert calls == ["old-access"]
    assert creds.token == "new-access"
    saved = tokens.get_by_user("user_1")
    assert saved is not None
    assert crypto.decrypt(saved.encrypted_access_token or "") == "new-access"
    assert saved.token_expiry is not None
    assert saved.token_expiry > datetime.now(UTC).replace(tzinfo=None)


def test_token_refresh_happens_within_60_seconds_of_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, tokens, crypto = _service()
    _seed(tokens, crypto, expiry=datetime.now(UTC) + timedelta(seconds=30))
    calls: list[str] = []
    _fake_refresh(monkeypatch, calls)

    service.credentials_for_user("user_1", required_scopes=FORM_SCOPES)

    assert calls == ["old-access"]


def test_valid_token_is_not_refreshed(monkeypatch: pytest.MonkeyPatch) -> None:
    service, tokens, crypto = _service()
    _seed(tokens, crypto, expiry=datetime.now(UTC) + timedelta(minutes=30))
    calls: list[str] = []
    _fake_refresh(monkeypatch, calls)

    creds = service.credentials_for_user("user_1", required_scopes=FORM_SCOPES)

    assert calls == []
    assert creds.token == "old-access"


def test_refresh_error_deletes_oauth_record_and_raises_revoked(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    service, tokens, crypto = _service()
    _seed(tokens, crypto, expiry=datetime.now(UTC) - timedelta(minutes=5))

    def refresh(self: Credentials, request: object) -> None:
        raise _refresh_error("invalid_grant")

    monkeypatch.setattr(Credentials, "refresh", refresh)

    with caplog.at_level(logging.WARNING), pytest.raises(GoogleTokenRevoked):
        service.credentials_for_user("user_1", required_scopes=FORM_SCOPES)

    assert tokens.get_by_user("user_1") is None
    assert "api_token_revoked" in caplog.text
    assert "user_1" not in caplog.text
    assert "refresh" not in caplog.text.replace("api_token_revoked", "")


def test_missing_scopes_logged_as_scope_gap(caplog: pytest.LogCaptureFixture) -> None:
    service, tokens, crypto = _service()
    _seed(tokens, crypto, expiry=None, scopes=IDENTITY_SCOPES)

    with caplog.at_level(logging.INFO), pytest.raises(MissingRequiredScopes):
        service.credentials_for_user("user_1", required_scopes=FORM_SCOPES)

    assert "api_scope_gap" in caplog.text
    assert "user_1" not in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        _refresh_error("temporarily_unavailable", retryable=True),
        _refresh_error("internal_failure", retryable=True),
        _refresh_error("invalid_client"),
        TransportError("token endpoint unreachable"),
    ],
    ids=["temporarily_unavailable", "internal_failure", "invalid_client", "transport"],
)
def test_non_revocation_refresh_failures_keep_the_grant(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, error: Exception
) -> None:
    """A Google outage or a bad client secret must never wipe users' tokens."""
    service, tokens, crypto = _service()
    _seed(tokens, crypto, expiry=datetime.now(UTC) - timedelta(minutes=5))

    def refresh(self: Credentials, request: object) -> None:
        raise error

    monkeypatch.setattr(Credentials, "refresh", refresh)

    with caplog.at_level(logging.WARNING), pytest.raises(GoogleTokenRefreshFailed) as info:
        service.credentials_for_user("user_1", required_scopes=FORM_SCOPES)

    assert isinstance(info.value, GoogleApiTemporaryError)
    assert tokens.get_by_user("user_1") is not None
    assert "api_token_refresh_failed" in caplog.text
    assert "api_token_revoked" not in caplog.text


def test_token_inside_google_auth_skew_is_refreshed_and_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """google-auth treats a token as expired ~3m45s early; we must not fall behind it."""
    service, tokens, crypto = _service()
    _seed(tokens, crypto, expiry=datetime.now(UTC) + timedelta(seconds=120))
    calls: list[str] = []
    _fake_refresh(monkeypatch, calls)

    service.credentials_for_user("user_1", required_scopes=FORM_SCOPES)

    assert calls == ["old-access"]
    saved = tokens.get_by_user("user_1")
    assert saved is not None
    assert crypto.decrypt(saved.encrypted_access_token or "") == "new-access"


class _FakeHttp:
    """Stands in for the network under google_auth_httplib2.AuthorizedHttp."""

    def __init__(self, statuses: list[int]) -> None:
        self.statuses = statuses
        self.seen_tokens: list[str] = []

    def request(self, uri: str, method: str = "GET", body=None, headers=None, **kwargs):
        self.seen_tokens.append((headers or {}).get("authorization", ""))
        return httplib2.Response({"status": self.statuses.pop(0)}), b"{}"


def test_refresh_triggered_by_google_auth_on_401_is_persisted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Real AuthorizedHttp refreshes on 401 by itself, deep inside execute()."""
    service, tokens, crypto = _service()
    _seed(tokens, crypto, expiry=datetime.now(UTC) + timedelta(minutes=30))
    creds = service.credentials_for_user("user_1", required_scopes=FORM_SCOPES)
    calls: list[str] = []
    _fake_refresh(monkeypatch, calls)
    http = _FakeHttp([401, 200])

    with caplog.at_level(logging.INFO):
        response, _ = google_auth_httplib2.AuthorizedHttp(creds, http=http).request(
            "https://forms.googleapis.com/v1/forms/x"
        )

    assert response.status == 200
    assert calls == ["old-access"]
    assert http.seen_tokens[-1] == "Bearer new-access"
    saved = tokens.get_by_user("user_1")
    assert saved is not None
    assert crypto.decrypt(saved.encrypted_access_token or "") == "new-access"
    assert "api_token_refreshed" in caplog.text


def test_revoked_grant_discovered_during_a_call_raises_domain_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consent revoked while the access token is still unexpired (the common case)."""
    service, tokens, crypto = _service()
    _seed(tokens, crypto, expiry=datetime.now(UTC) + timedelta(minutes=30))
    creds = service.credentials_for_user("user_1", required_scopes=FORM_SCOPES)

    def refresh(self: Credentials, request: object) -> None:
        raise _refresh_error("invalid_grant")

    monkeypatch.setattr(Credentials, "refresh", refresh)
    http = _FakeHttp([401, 200])

    with pytest.raises(GoogleTokenRevoked):
        google_auth_httplib2.AuthorizedHttp(creds, http=http).request("https://example.com/x")

    assert tokens.get_by_user("user_1") is None


def test_grant_without_refresh_token_is_treated_as_revoked_when_google_rejects_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, tokens, crypto = _service()
    _seed(tokens, crypto, expiry=datetime.now(UTC) + timedelta(minutes=30))
    account = tokens.get_by_user("user_1")
    assert account is not None
    tokens.save(replace(account, encrypted_refresh_token=None))
    creds = service.credentials_for_user("user_1", required_scopes=FORM_SCOPES)

    def refresh(self: Credentials, request: object) -> None:
        raise RefreshError("The credentials do not contain the necessary fields.")

    monkeypatch.setattr(Credentials, "refresh", refresh)

    with pytest.raises(GoogleTokenRevoked):
        creds.refresh(None)

    assert tokens.get_by_user("user_1") is None
