"""Google Forms/Catalog routes owned by the SaaS API."""

from __future__ import annotations

from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

from api.dependencies import get_container, require_session
from core.forms_api import FormsApiError
from core.saas.container import SaaSContainer
from core.saas.errors import MissingRequiredScopes
from core.saas.google_credentials import GoogleCredentialService
from core.saas.google_scopes import scopes_for_purpose
from core.saas.models import Session
from core.saas.ports import GoogleFormsClient

router = APIRouter(prefix="/v1", tags=["google-forms"])


class GoogleAccessResponse(BaseModel):
    ok: bool
    purpose: str


class FormListItem(BaseModel):
    id: str
    name: str
    owner_email: str | None = None
    owner_name: str | None = None
    created_time: str | None = None
    modified_time: str | None = None
    edit_url: str | None = None


class FormSummaryResponse(BaseModel):
    title: str
    description: str
    sections_count: int
    questions_count: int
    linked_sheet_id: str | None = None
    is_published: bool | None = None
    accepting_responses: bool | None = None


class ResponseStatsResponse(BaseModel):
    total: int
    first_response: str | None = None
    second_response: str | None = None
    last_response: str | None = None


class ResponseTimestampsResponse(BaseModel):
    timestamps: list[str]


class CatalogFormRow(BaseModel):
    status: str
    error_code: str | None = None
    form: FormListItem
    summary: FormSummaryResponse | None = None
    response_stats: ResponseStatsResponse | None = None


@router.get("/google/access", response_model=GoogleAccessResponse)
def check_google_access(
    request: Request,
    session: Annotated[Session, Depends(require_session)],
    purpose: Annotated[str, Query(pattern="^(forms|sheets)$")] = "forms",
    next_url: Annotated[str, Query(max_length=2048)] = "/",
) -> GoogleAccessResponse:
    require_google_credentials(request, session, purpose=purpose, next_url=next_url)
    return GoogleAccessResponse(ok=True, purpose=purpose)


@router.get("/forms", response_model=list[FormListItem])
def list_forms(
    request: Request,
    session: Annotated[Session, Depends(require_session)],
) -> list[FormListItem]:
    creds = require_google_credentials(request, session, purpose="forms")
    try:
        return [
            FormListItem.model_validate(item) for item in _forms_client(request).list_forms(creds)
        ]
    except FormsApiError as exc:
        raise google_http_exception(exc) from exc


@router.get("/forms/catalog", response_model=list[CatalogFormRow])
def read_forms_catalog(
    request: Request,
    session: Annotated[Session, Depends(require_session)],
) -> list[CatalogFormRow]:
    creds = require_google_credentials(request, session, purpose="forms")
    try:
        forms = [
            FormListItem.model_validate(item) for item in _forms_client(request).list_forms(creds)
        ]
    except FormsApiError as exc:
        raise google_http_exception(exc) from exc

    rows: list[CatalogFormRow] = []
    for form in forms:
        rows.append(_catalog_row(request, creds, form))
    return rows


@router.get("/forms/{form_id}/summary", response_model=FormSummaryResponse)
def read_form_summary(
    form_id: str,
    request: Request,
    session: Annotated[Session, Depends(require_session)],
) -> FormSummaryResponse:
    creds = require_google_credentials(request, session, purpose="forms")
    try:
        return FormSummaryResponse.model_validate(
            _forms_client(request).get_form_summary(creds, form_id)
        )
    except FormsApiError as exc:
        raise google_http_exception(exc) from exc


@router.get("/forms/{form_id}/response-stats", response_model=ResponseStatsResponse)
def read_form_response_stats(
    form_id: str,
    request: Request,
    session: Annotated[Session, Depends(require_session)],
) -> ResponseStatsResponse:
    creds = require_google_credentials(request, session, purpose="forms")
    try:
        return ResponseStatsResponse.model_validate(
            _forms_client(request).get_response_stats(creds, form_id)
        )
    except FormsApiError as exc:
        raise google_http_exception(exc) from exc


@router.get("/forms/{form_id}/response-timestamps", response_model=ResponseTimestampsResponse)
def read_form_response_timestamps(
    form_id: str,
    request: Request,
    session: Annotated[Session, Depends(require_session)],
) -> ResponseTimestampsResponse:
    creds = require_google_credentials(request, session, purpose="forms")
    try:
        return ResponseTimestampsResponse(
            timestamps=list(_forms_client(request).list_response_timestamps(creds, form_id))
        )
    except FormsApiError as exc:
        raise google_http_exception(exc) from exc


@router.get("/forms/{form_id}/structure", response_model=dict[str, Any])
def read_form_structure(
    form_id: str,
    request: Request,
    session: Annotated[Session, Depends(require_session)],
) -> dict[str, Any]:
    creds = require_google_credentials(request, session, purpose="forms")
    try:
        return dict(_forms_client(request).get_form_structure(creds, form_id))
    except FormsApiError as exc:
        raise google_http_exception(exc) from exc


@router.get("/forms/{form_id}/responses", response_model=list[dict[str, Any]])
def read_form_responses(
    form_id: str,
    request: Request,
    session: Annotated[Session, Depends(require_session)],
) -> list[dict[str, Any]]:
    creds = require_google_credentials(request, session, purpose="forms")
    try:
        return [dict(item) for item in _forms_client(request).list_responses(creds, form_id)]
    except FormsApiError as exc:
        raise google_http_exception(exc) from exc


def require_google_credentials(
    request: Request,
    session: Session,
    *,
    purpose: str,
    next_url: str = "/",
) -> Any:
    container = get_container(request)
    try:
        return GoogleCredentialService(
            tokens=container.tokens,
            token_crypto=container.token_crypto,
            client_config_json=container.settings.google_oauth_client_config_json,
        ).credentials_for_user(
            session.user_id,
            required_scopes=scopes_for_purpose(purpose),
        )
    except MissingRequiredScopes as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "missing_required_scopes",
                "purpose": purpose,
                "missing_scopes": list(_missing_scopes(container, session.user_id, purpose)),
                "connect_url": _google_connect_url(request, purpose=purpose, next_url=next_url),
            },
        ) from exc


def google_http_exception(exc: FormsApiError) -> HTTPException:
    if exc.status in {401, 403}:
        code = status.HTTP_403_FORBIDDEN
    elif exc.status == 404:
        code = status.HTTP_404_NOT_FOUND
    elif exc.status in {429, 500, 502, 503, 504}:
        code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        code = status.HTTP_502_BAD_GATEWAY
    return HTTPException(
        status_code=code,
        detail={"code": "google_forms_error", "message": str(exc)},
    )


def _forms_client(request: Request) -> GoogleFormsClient:
    return request.app.state.google_forms_client


def _catalog_row(request: Request, creds: Any, form: FormListItem) -> CatalogFormRow:
    try:
        summary = FormSummaryResponse.model_validate(
            _forms_client(request).get_form_summary(creds, form.id)
        )
        stats = ResponseStatsResponse.model_validate(
            _forms_client(request).get_response_stats(creds, form.id)
        )
    except FormsApiError as exc:
        return CatalogFormRow(
            status=_catalog_status(exc),
            error_code="google_forms_error",
            form=form,
        )
    return CatalogFormRow(status="ok", form=form, summary=summary, response_stats=stats)


def _catalog_status(exc: FormsApiError) -> str:
    if exc.status in {401, 403}:
        return "no_access"
    if exc.status == 404:
        return "deleted"
    if exc.status == 429:
        return "rate_limited"
    if exc.status == 400:
        return "unsupported"
    return "api_error"


def _missing_scopes(container: SaaSContainer, user_id: str, purpose: str) -> tuple[str, ...]:
    account = container.tokens.get_by_user(user_id)
    granted = set(account.scopes if account else ())
    return tuple(scope for scope in scopes_for_purpose(purpose) if scope not in granted)


def _google_connect_url(request: Request, *, purpose: str, next_url: str) -> str:
    container = get_container(request)
    safe_next = _safe_next_url(next_url, container.settings.app_base_url)
    query = urlencode({"purpose": purpose, "next_url": safe_next})
    return f"{container.settings.api_base_url.rstrip('/')}/v1/auth/google/start?{query}"


def _safe_next_url(next_url: str, app_base_url: str) -> str:
    if next_url.startswith("/"):
        return next_url
    if app_base_url and next_url.startswith(app_base_url.rstrip("/") + "/"):
        return next_url
    return "/"
