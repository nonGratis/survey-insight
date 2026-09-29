"""Shared translation of Google API failures into HTTP responses."""

from __future__ import annotations

from typing import Protocol

from fastapi import HTTPException, status

from core.google_errors import SCOPE_ERROR_REASONS
from core.logger import get_logger

log = get_logger(__name__)


class GoogleApiFailure(Protocol):
    status: int | None
    reason: str | None


def google_http_exception(
    exc: GoogleApiFailure,
    *,
    purpose: str,
    error_code: str,
) -> HTTPException:
    if exc.status == 403 and exc.reason in SCOPE_ERROR_REASONS:
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "google_insufficient_scopes",
                "action": f"reconnect_{purpose}_required",
                "purpose": purpose,
            },
        )
    if exc.status in {401, 403}:
        code = status.HTTP_403_FORBIDDEN
    elif exc.status == 404:
        code = status.HTTP_404_NOT_FOUND
    else:
        code = status.HTTP_502_BAD_GATEWAY
        log.warning(
            "api_google_error",
            extra={"error_code": f"google_{exc.status}", "reason": exc.reason},
        )
    return HTTPException(
        status_code=code,
        detail={"code": error_code, "message": str(exc)},
    )
