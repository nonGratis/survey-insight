"""Server-side reconstruction of Google OAuth credentials.

The UI must never receive Google tokens. This service is the only SaaS-domain
place that decrypts stored Google OAuth tokens and refreshes access tokens.

google-auth refreshes on its own, not only when we ask: before a request when
the token is about to expire, and again after a 401 response (via
``AuthorizedHttp``), i.e. deep inside a Google client call. Every one of those
goes through ``Credentials.refresh``, so the credentials built here override it
to persist the new token and to turn failures into domain errors. Nothing else
in the API should have to catch ``google.auth`` exceptions.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

from google.auth.exceptions import RefreshError, TransportError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from core.logger import get_logger
from core.saas.errors import (
    GoogleTokenRefreshFailed,
    GoogleTokenRevoked,
    MissingRequiredScopes,
)
from core.saas.models import OAuthAccount
from core.saas.ports import TokenCrypto, TokenRepository
from core.saas.security import log_user_ref, utcnow

log = get_logger(__name__)

REFRESH_SKEW = timedelta(seconds=60)


class _ManagedCredentials(Credentials):
    """Credentials that report every refresh outcome to their owner."""

    def __init__(
        self,
        *args: Any,
        on_refreshed: Callable[[Credentials], None],
        on_refresh_error: Callable[[Exception], Exception],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._on_refreshed = on_refreshed
        self._on_refresh_error = on_refresh_error

    def refresh(self, request: Any) -> None:
        try:
            super().refresh(request)
        except (RefreshError, TransportError) as exc:
            raise self._on_refresh_error(exc) from exc
        self._on_refreshed(self)


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
            log.info(
                "api_scope_gap",
                extra={"user_id": log_user_ref(user_id), "missing": sorted(missing)},
            )
            raise MissingRequiredScopes(f"Missing Google OAuth scopes: {sorted(missing)}")

        creds = _ManagedCredentials(
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
            # No scopes here on purpose: they would be sent with every refresh, and
            # Google rejects (invalid_scope) any set wider than the refresh token's.
            on_refreshed=lambda refreshed: self._save_refreshed(account, refreshed),
            on_refresh_error=lambda exc: self._refresh_failure(account, exc),
        )
        creds.expiry = _naive_utc(account.token_expiry)

        if creds.refresh_token and (not creds.token or _needs_refresh(creds)):
            creds.refresh(Request())

        if not creds.token:
            raise MissingRequiredScopes("Google access token is unavailable.")

        return creds

    def _refresh_failure(self, account: OAuthAccount, exc: Exception) -> Exception:
        """Decide what a failed refresh means and return the domain error to raise.

        Only a revoked grant justifies dropping the stored record. An outage, a
        network failure or a bad client secret affects every user and says
        nothing about this grant, so it must leave the record alone.
        """
        user_ref = log_user_ref(account.user_id)
        if not account.encrypted_refresh_token or _is_invalid_grant(exc):
            self.tokens.delete_by_user(account.user_id)
            log.warning("api_token_revoked", extra={"user_id": user_ref})
            return GoogleTokenRevoked("Google grant was revoked.")
        log.error(
            "api_token_refresh_failed",
            extra={
                "user_id": user_ref,
                "error_code": _refresh_error_code(exc),
                "retryable": bool(getattr(exc, "retryable", False)),
            },
        )
        return GoogleTokenRefreshFailed("Google token refresh failed.")

    def _save_refreshed(self, account: OAuthAccount, creds: Credentials) -> None:
        encrypted_refresh_token = account.encrypted_refresh_token
        if creds.refresh_token:
            encrypted_refresh_token = self.token_crypto.encrypt(creds.refresh_token)

        self.tokens.save(
            replace(
                account,
                encrypted_access_token=(
                    self.token_crypto.encrypt(creds.token) if creds.token else None
                ),
                encrypted_refresh_token=encrypted_refresh_token,
                token_expiry=creds.expiry,
                updated_at=utcnow(),
            )
        )
        log.info(
            "api_token_refreshed",
            extra={"user_id": log_user_ref(account.user_id), "scopes": list(account.scopes)},
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


def _needs_refresh(creds: Credentials) -> bool:
    # google-auth's own ``expired`` already applies a ~3m45s skew; keep it as a
    # floor so this check is never laxer than the library's in-band refresh.
    if creds.expired:
        return True
    if creds.expiry is None:
        return False
    return creds.expiry <= datetime.now(UTC).replace(tzinfo=None) + REFRESH_SKEW


def _error_payload(exc: Exception) -> dict[str, Any]:
    data = exc.args[1] if len(exc.args) > 1 else None
    return data if isinstance(data, dict) else {}


def _is_invalid_grant(exc: Exception) -> bool:
    """True only when Google says the grant itself is gone (revoked or expired)."""
    if not isinstance(exc, RefreshError):
        return False
    payload = _error_payload(exc)
    if payload:
        return payload.get("error") == "invalid_grant"
    return str(exc.args[0] if exc.args else "").startswith("invalid_grant")


def _refresh_error_code(exc: Exception) -> str:
    return str(_error_payload(exc).get("error") or type(exc).__name__)


def _naive_utc(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)
