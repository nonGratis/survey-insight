from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import pytest
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials

from core.saas.errors import GoogleTokenRevoked, MissingRequiredScopes
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
        raise RefreshError("invalid_grant")

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
