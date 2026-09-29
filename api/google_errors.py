"""Shared translation of Google API failures into HTTP responses."""

from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from core.google_errors import SCOPE_ERROR_REASONS
from core.logger import get_logger
from core.saas.errors import GoogleTokenRefreshFailed, GoogleTokenRevoked

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


def register_google_error_handlers(app: FastAPI) -> None:
    """Translate credential-lifecycle errors raised anywhere in a request.

    google-auth can refresh a token deep inside a Google client call, so these
    errors are not confined to where credentials are first built. Handling them
    once here keeps every Google-backed endpoint consistent.
    """

    @app.exception_handler(GoogleTokenRevoked)
    async def _token_revoked(request: Request, exc: GoogleTokenRevoked) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": {"code": "google_token_revoked", "action": "reauth_required"}},
        )

    @app.exception_handler(GoogleTokenRefreshFailed)
    async def _token_refresh_failed(
        request: Request, exc: GoogleTokenRefreshFailed
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": {"code": "google_token_refresh_failed"}},
        )
