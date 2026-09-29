"""Helpers for reading machine-readable details out of Google API errors."""

from __future__ import annotations

import json

from googleapiclient.errors import HttpError

SCOPE_ERROR_REASONS = frozenset({"insufficientPermissions", "accessNotConfigured"})


def google_error_reason(exc: HttpError) -> str | None:
    """Return Google's machine reason (e.g. ``insufficientPermissions``), if present."""
    try:
        error = json.loads(exc.content.decode("utf-8")).get("error", {})
    except (ValueError, AttributeError, UnicodeDecodeError):
        return None
    if not isinstance(error, dict):
        return None
    for key in ("errors", "details"):
        for item in error.get(key) or []:
            reason = item.get("reason") if isinstance(item, dict) else None
            if reason:
                return str(reason)
    return None
