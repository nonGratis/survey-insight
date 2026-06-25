"""Server-side reconstruction of Google OAuth credentials.

The UI must never receive Google tokens. This service is the only SaaS-domain
place that decrypts stored Google OAuth tokens and refreshes access tokens.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from core.saas.errors import MissingRequiredScopes
from core.saas.models import OAuthAccount
from core.saas.ports import TokenCrypto, TokenRepository
from core.saas.security import utcnow


class GoogleCredentialService:
    """Build Google credentials from encrypted server-side token records."""

    def __init__(
        self,
        *,
        tokens: TokenRepository,
        token_crypto: TokenCrypto,
        client_config_json: str,
    ) -> None:
        self.tokens = tokens
        self.token_crypto = token_crypto
        self.client_config = _parse_google_client_config(client_config_json)

    def credentials_for_user(
        self,
        user_id: str,
        *,
        required_scopes: Sequence[str],
    ) -> Credentials:
        account = self.tokens.get_by_user(user_id)
        if account is None:
            raise MissingRequiredScopes("Google account is not connected.")

        missing = set(required_scopes) - set(account.scopes)
        if missing:
            raise MissingRequiredScopes(f"Missing Google OAuth scopes: {sorted(missing)}")

        creds = Credentials(
            token=(
                self.token_crypto.decrypt(account.encrypted_access_token)
                if account.encrypted_access_token
                else None
            ),
            refresh_token=(
                self.token_crypto.decrypt(account.encrypted_refresh_token)
                if account.encrypted_refresh_token
                else None
            ),
            token_uri=self.client_config["token_uri"],
            client_id=self.client_config["client_id"],
            client_secret=self.client_config["client_secret"],
            scopes=list(account.scopes),
        )
        creds.expiry = _naive_utc(account.token_expiry)

        if (not creds.token or creds.expired) and creds.refresh_token:
            creds.refresh(Request())
            self._save_refreshed(account, creds)

        if not creds.token:
            raise MissingRequiredScopes("Google access token is unavailable.")

        return creds

    def _save_refreshed(self, account: OAuthAccount, creds: Credentials) -> None:
        encrypted_refresh_token = account.encrypted_refresh_token
        if creds.refresh_token:
            encrypted_refresh_token = self.token_crypto.encrypt(creds.refresh_token)

        scopes = tuple(dict.fromkeys([*(creds.scopes or ()), *account.scopes]))
        self.tokens.save(
            replace(
                account,
                scopes=scopes,
                encrypted_access_token=(
                    self.token_crypto.encrypt(creds.token) if creds.token else None
                ),
                encrypted_refresh_token=encrypted_refresh_token,
                token_expiry=creds.expiry,
                updated_at=utcnow(),
            )
        )


def _parse_google_client_config(client_config_json: str) -> dict[str, str]:
    if not client_config_json:
        raise ValueError("Google OAuth client config JSON is required.")
    raw = json.loads(client_config_json)
    config: dict[str, Any] = raw.get("web") or raw.get("installed") or raw
    required = ("token_uri", "client_id", "client_secret")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Google OAuth client config is missing: {', '.join(missing)}")
    return {key: str(config[key]) for key in required}


def _naive_utc(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)
