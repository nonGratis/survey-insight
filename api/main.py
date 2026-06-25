"""FastAPI service for SaaS auth/session/report APIs."""

from __future__ import annotations

from contextlib import suppress
from datetime import timedelta
from typing import Annotated, Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from api.dependencies import (
    SESSION_COOKIE_NAME,
    SessionCookie,
)
from api.dependencies import (
    get_container as _container,
)
from api.dependencies import (
    require_session as _require_session,
)
from api.dependencies import (
    require_user as _require_user,
)
from api.routes.google_forms import router as google_forms_router
from core.logger import get_logger
from core.saas.adapters.google_forms import GoogleFormsApiClient
from core.saas.adapters.google_oauth import GoogleOAuthClient, GoogleOAuthWebClient
from core.saas.container import SaaSContainer
from core.saas.errors import AuthError, InvalidSession, QuotaExceeded
from core.saas.google_scopes import scopes_for_purpose
from core.saas.models import OAuthAccount, Plan, Quota, ReportJob, Session, User, UserStatus
from core.saas.ports import GoogleFormsClient
from core.saas.security import hash_secret, utcnow

log = get_logger(__name__)
DEFAULT_NEW_USER_QUOTA = Quota(monthly_report_limit=20, reports_used_this_month=0)


class HealthResponse(BaseModel):
    status: str
    service: str


class SessionResponse(BaseModel):
    authenticated: bool
    user_id: str | None = None
    email: str | None = None
    name: str | None = None
    plan: str | None = None


class LoginTicketExchangeResponse(SessionResponse):
    # Streamlit and API are different run.app origins before a custom domain is
    # attached, so the web client needs the opaque session id to store its own
    # browser cookie. Firestore still stores only the HMAC session hash.
    session_id: str


class ReportJobRequest(BaseModel):
    form_id: str = Field(min_length=1)
    form_title: str | None = Field(default=None, max_length=240)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)


class ReportJobResponse(BaseModel):
    job_id: str
    report_id: str
    status: str
    attempts: int


class LoginTicketExchangeRequest(BaseModel):
    ticket: str = Field(min_length=1)


def create_api_app(
    container: SaaSContainer | None = None,
    oauth_client: GoogleOAuthClient | None = None,
    google_forms_client: GoogleFormsClient | None = None,
) -> FastAPI:
    app = FastAPI(title="Survey Insight API", version="0.1.0")
    app.state.container = container or SaaSContainer.from_settings()
    app.state.oauth_client = oauth_client or GoogleOAuthWebClient(
        client_config_json=app.state.container.settings.google_oauth_client_config_json,
        api_base_url=app.state.container.settings.api_base_url,
    )
    app.state.google_forms_client = google_forms_client or GoogleFormsApiClient()
    app.include_router(google_forms_router)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service="survey-insight-api")

    @app.get("/v1/session", response_model=SessionResponse, response_model_exclude_none=True)
    def read_session(request: Request, session_id: SessionCookie = None) -> SessionResponse:
        if not session_id:
            return SessionResponse(authenticated=False)
        container = _container(request)
        try:
            session = container.session_service.validate(session_id)
        except InvalidSession:
            return SessionResponse(authenticated=False)
        user = container.users.get(session.user_id)
        if user is None or user.status != UserStatus.ACTIVE:
            return SessionResponse(authenticated=False)
        return _session_response(user)

    @app.post("/v1/auth/logout")
    def logout(
        request: Request,
        response: Response,
        session_id: SessionCookie = None,
    ) -> dict[str, bool]:
        if session_id:
            with suppress(InvalidSession):
                _container(request).session_service.revoke(session_id)
        response.delete_cookie(SESSION_COOKIE_NAME, path="/")
        return {"ok": True}

    @app.get("/v1/auth/google/start")
    def start_google_auth(
        request: Request,
        next_url: Annotated[str, Query(max_length=2048)] = "/",
        purpose: Annotated[str, Query(pattern="^(identity|forms|sheets)$")] = "identity",
    ) -> RedirectResponse:
        container = _container(request)
        scopes = scopes_for_purpose(purpose)
        state_secret = container.oauth_state_service.create(
            scopes=scopes,
            next_url=_safe_next_url(next_url, container.settings.app_base_url),
            ttl=timedelta(minutes=10),
        )
        authorization_url = _oauth_client(request).authorization_url(
            state=state_secret.state,
            code_verifier=state_secret.record.code_verifier,
            scopes=state_secret.record.scopes,
            include_granted_scopes=purpose != "identity",
        )
        return RedirectResponse(authorization_url)

    @app.get("/v1/auth/google/callback")
    def google_auth_callback(
        request: Request,
        state: Annotated[str | None, Query()] = None,
        code: Annotated[str | None, Query()] = None,
    ) -> RedirectResponse:
        if not state or not code:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="missing_oauth_params"
            )
        container = _container(request)
        try:
            state_record = container.oauth_state_service.consume(state)
        except AuthError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_oauth_state"
            ) from exc

        stage = "exchange_code"
        try:
            credentials = _oauth_client(request).exchange_code(
                code=code,
                state=state,
                code_verifier=state_record.code_verifier,
                scopes=state_record.scopes,
            )
            stage = "userinfo"
            user_info = _oauth_client(request).user_info(credentials)
            stage = "persist_user"
            user = _upsert_google_user(container, user_info)
            stage = "persist_tokens"
            _save_google_tokens(container, user=user, user_info=user_info, credentials=credentials)
            stage = "create_login_ticket"
            login_ticket = container.login_ticket_service.create(
                user_id=user.id,
                ttl=timedelta(minutes=5),
            )
        except Exception as exc:  # noqa: BLE001
            log.exception(
                "google_oauth_callback_failed",
                extra={"error_code": type(exc).__name__, "stage": stage},
            )
            return RedirectResponse(
                _with_auth_error(state_record.next_url, "oauth_callback_failed")
            )

        return RedirectResponse(_with_login_ticket(state_record.next_url, login_ticket.ticket))

    @app.post("/v1/auth/session/exchange", response_model=LoginTicketExchangeResponse)
    def exchange_login_ticket(
        body: LoginTicketExchangeRequest,
        request: Request,
        response: Response,
    ) -> LoginTicketExchangeResponse:
        container = _container(request)
        try:
            ticket = container.login_ticket_service.consume(body.ticket)
        except AuthError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_login_ticket"
            ) from exc
        user = _require_user(container, ticket.user_id)
        session = container.session_service.create(user_id=user.id)
        response.set_cookie(
            SESSION_COOKIE_NAME,
            session.session_id,
            httponly=True,
            secure=container.settings.is_production,
            samesite="lax",
            path="/",
            max_age=60 * 60 * 24 * 30,
        )
        return LoginTicketExchangeResponse(
            **_session_response(user).model_dump(),
            session_id=session.session_id,
        )

    @app.post(
        "/v1/reports/jobs", response_model=ReportJobResponse, status_code=status.HTTP_202_ACCEPTED
    )
    def create_report_job(
        body: ReportJobRequest,
        request: Request,
        session: Annotated[Session, Depends(_require_session)],
    ) -> ReportJobResponse:
        container = _container(request)
        user = _require_user(container, session.user_id)
        quota = container.quotas.get_for_user(user.id)
        if quota is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="quota_not_configured"
            )

        form_id_hash = hash_secret(body.form_id, container.settings.session_pepper or "form-pepper")
        try:
            _, job = container.report_job_service.create_report_job(
                user_id=user.id,
                form_id_hash=form_id_hash,
                form_title=body.form_title,
                config_snapshot=body.config_snapshot,
                quota=quota,
            )
        except QuotaExceeded as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)
            ) from exc
        return _job_response(job)

    @app.get("/v1/reports/jobs/{job_id}", response_model=ReportJobResponse)
    def read_report_job(
        job_id: str,
        request: Request,
        session: Annotated[Session, Depends(_require_session)],
    ) -> ReportJobResponse:
        container = _container(request)
        job = container.jobs.get(job_id)
        if job is None or job.user_id != session.user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job_not_found")
        return _job_response(job)

    return app


def _oauth_client(request: Request) -> GoogleOAuthClient:
    return request.app.state.oauth_client


def _job_response(job: ReportJob) -> ReportJobResponse:
    return ReportJobResponse(
        job_id=job.id,
        report_id=job.report_id,
        status=job.status.value,
        attempts=job.attempts,
    )


def _session_response(user: User) -> SessionResponse:
    return SessionResponse(
        authenticated=True,
        user_id=user.id,
        email=user.email,
        name=user.name,
        plan=user.plan.value,
    )


def _upsert_google_user(container: SaaSContainer, user_info: dict[str, Any]) -> User:
    google_sub = str(user_info["id"])
    user_id = f"google:{google_sub}"
    existing = container.users.get(user_id)
    current = utcnow()
    user = User(
        id=user_id,
        google_sub=google_sub,
        email=str(user_info.get("email") or ""),
        name=str(user_info.get("name") or user_info.get("email") or "Google user"),
        picture=user_info.get("picture"),
        plan=existing.plan if existing else Plan.PILOT,
        status=existing.status if existing else UserStatus.ACTIVE,
        created_at=existing.created_at if existing else current,
        last_seen_at=current,
    )
    container.users.save(user)
    if container.quotas.get_for_user(user.id) is None:
        container.quotas.save(user.id, DEFAULT_NEW_USER_QUOTA)
    return user


def _save_google_tokens(
    container: SaaSContainer,
    *,
    user: User,
    user_info: dict[str, Any],
    credentials: Any,
) -> None:
    existing = container.tokens.get_by_user(user.id)
    scopes = tuple(
        dict.fromkeys([*(existing.scopes if existing else ()), *(credentials.scopes or ())])
    )
    encrypted_refresh_token = existing.encrypted_refresh_token if existing else None
    if credentials.refresh_token:
        encrypted_refresh_token = container.token_crypto.encrypt(credentials.refresh_token)

    account = OAuthAccount(
        user_id=user.id,
        provider="google",
        google_sub=str(user_info["id"]),
        email=user.email,
        scopes=scopes,
        encrypted_access_token=(
            container.token_crypto.encrypt(credentials.token) if credentials.token else None
        ),
        encrypted_refresh_token=encrypted_refresh_token,
        token_expiry=credentials.expiry,
        updated_at=utcnow(),
    )
    container.tokens.save(account)


def _safe_next_url(next_url: str, app_base_url: str) -> str:
    if next_url.startswith("/"):
        return next_url
    if app_base_url and next_url.startswith(app_base_url.rstrip("/") + "/"):
        return next_url
    return "/"


def _with_login_ticket(next_url: str, ticket: str) -> str:
    return _with_query(next_url, {"login_ticket": ticket})


def _with_auth_error(next_url: str, error_code: str) -> str:
    return _with_query(next_url, {"auth_error": error_code})


def _with_query(next_url: str, values: dict[str, str]) -> str:
    parts = urlsplit(next_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(values)
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(query),
            parts.fragment,
        )
    )


app = create_api_app()
