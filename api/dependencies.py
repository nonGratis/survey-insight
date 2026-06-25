"""Shared FastAPI dependencies for SaaS API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import Cookie, HTTPException, Request, status

from core.saas.container import SaaSContainer
from core.saas.errors import InvalidSession
from core.saas.models import Session, User, UserStatus

SESSION_COOKIE_NAME = "session_id"
SessionCookie = Annotated[str | None, Cookie(alias=SESSION_COOKIE_NAME)]


def get_container(request: Request) -> SaaSContainer:
    return request.app.state.container


def require_session(request: Request, session_id: SessionCookie = None) -> Session:
    if not session_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing_session")
    try:
        return get_container(request).session_service.validate(session_id)
    except InvalidSession as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_session",
        ) from exc


def require_user(container: SaaSContainer, user_id: str) -> User:
    user = container.users.get(user_id)
    if user is None or user.status != UserStatus.ACTIVE:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_user")
    return user
