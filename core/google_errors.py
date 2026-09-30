"""Helpers for reading machine-readable details out of Google API errors."""

from __future__ import annotations

import json

from googleapiclient.errors import HttpError

# ACCESS_TOKEN_SCOPE_INSUFFICIENT is what current Google APIs (gRPC-style errors,
# machine reason in details[]) return for a token without the needed scope;
# insufficientPermissions is the older errors[] equivalent.
SCOPE_ERROR_REASONS = frozenset(
    {"insufficientPermissions", "accessNotConfigured", "ACCESS_TOKEN_SCOPE_INSUFFICIENT"}
)
_SCOPE_ERROR_MESSAGE = "insufficient authentication scopes"


def google_error_reason(exc: HttpError) -> str | None:
    """Return Google's machine reason (e.g. ``insufficientPermissions``), if present.

    A response can carry several reasons (errors[] and details[]); a scope-related
    one is preferred so it is never hidden behind a generic one listed first.
    """
    try:
        error = json.loads(exc.content.decode("utf-8")).get("error", {})
    except (ValueError, AttributeError, UnicodeDecodeError):
        return None
    if not isinstance(error, dict):
        return None
    reasons: list[str] = []
    for key in ("errors", "details"):
        for item in error.get(key) or []:
            reason = item.get("reason") if isinstance(item, dict) else None
            if reason:
                reasons.append(str(reason))
    for reason in reasons:
        if reason in SCOPE_ERROR_REASONS:
            return reason
    return reasons[0] if reasons else None


def is_scope_error(status: int | None, reason: str | None, message: str) -> bool:
    """True when Google rejected the call because the token lacks a needed scope."""
    if status != 403:
        return False
    return reason in SCOPE_ERROR_REASONS or _SCOPE_ERROR_MESSAGE in message.lower()
