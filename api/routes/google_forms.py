"""Google Forms/Catalog routes owned by the SaaS API."""

from __future__ import annotations

import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

from api.dependencies import get_container, require_session
from api.google_data_cache import ApiCacheKey, ApiCacheResult, get_or_load
from core.forms_api import FormsApiError
from core.logger import get_logger
from core.saas.container import SaaSContainer
from core.saas.errors import MissingRequiredScopes
from core.saas.google_credentials import GoogleCredentialService
from core.saas.google_scopes import scopes_for_purpose
from core.saas.models import Session
from core.saas.ports import GoogleFormsClient

router = APIRouter(prefix="/v1", tags=["google-forms"])
log = get_logger(__name__)

CATALOG_SUMMARY_TTL_SECONDS = int(os.getenv("SI_API_CATALOG_SUMMARY_TTL_SECONDS", "600"))
RESPONSE_STATS_TTL_SECONDS = int(os.getenv("SI_API_RESPONSE_STATS_TTL_SECONDS", "120"))
CATALOG_ENRICH_MAX_IDS = int(os.getenv("SI_CATALOG_ENRICH_MAX_IDS", "20"))
CATALOG_ENRICH_TIMEOUT_SECONDS = float(os.getenv("SI_CATALOG_ENRICH_TIMEOUT_SECONDS", "8"))
CATALOG_ENRICH_MAX_WORKERS = int(os.getenv("SI_CATALOG_ENRICH_MAX_WORKERS", "5"))


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


class CatalogEnrichRequest(BaseModel):
    form_ids: list[str]
    include_summary: bool = True
    include_stats: bool = True


class CatalogEnrichRow(BaseModel):
    form_id: str
    status: str
    error_code: str | None = None
    summary: FormSummaryResponse | None = None
    response_stats: ResponseStatsResponse | None = None
    fetched_at: str | None = None
    cache_hit: bool = False


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
        rows.append(_catalog_row(request, creds, session.user_id, form))
    return rows


@router.post("/forms/catalog/enrich", response_model=list[CatalogEnrichRow])
def enrich_forms_catalog(
    body: CatalogEnrichRequest,
    request: Request,
    session: Annotated[Session, Depends(require_session)],
) -> list[CatalogEnrichRow]:
    creds = require_google_credentials(request, session, purpose="forms")
    form_ids = _bounded_form_ids(body.form_ids)
    start = time.perf_counter()
    rows = _catalog_enrich_rows_with_budget(
        request,
        creds,
        user_id=session.user_id,
        form_ids=form_ids,
        include_summary=body.include_summary,
        include_stats=body.include_stats,
    )
    _log_catalog_enrich_telemetry(
        rows,
        chunk_size=len(form_ids),
        include_summary=body.include_summary,
        include_stats=body.include_stats,
        duration_ms=round((time.perf_counter() - start) * 1000, 1),
    )
    return rows


@router.get("/forms/{form_id}/summary", response_model=FormSummaryResponse)
def read_form_summary(
    form_id: str,
    request: Request,
    session: Annotated[Session, Depends(require_session)],
) -> FormSummaryResponse:
    creds = require_google_credentials(request, session, purpose="forms")
    try:
        return _cached_form_summary(request, creds, session.user_id, form_id).value
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
        return _cached_response_stats(request, creds, session.user_id, form_id).value
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


def _catalog_row(
    request: Request,
    creds: Any,
    user_id: str,
    form: FormListItem,
) -> CatalogFormRow:
    try:
        summary = _cached_form_summary(request, creds, user_id, form.id).value
        stats = _cached_response_stats(request, creds, user_id, form.id).value
    except FormsApiError as exc:
        return CatalogFormRow(
            status=_catalog_status(exc),
            error_code="google_forms_error",
            form=form,
        )
    return CatalogFormRow(status="ok", form=form, summary=summary, response_stats=stats)


def _catalog_enrich_rows_with_budget(
    request: Request,
    creds: Any,
    *,
    user_id: str,
    form_ids: list[str],
    include_summary: bool,
    include_stats: bool,
) -> list[CatalogEnrichRow]:
    if not form_ids:
        return []

    workers = max(1, min(CATALOG_ENRICH_MAX_WORKERS, len(form_ids)))
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {
        executor.submit(
            _catalog_enrich_row,
            request,
            creds,
            user_id,
            form_id,
            include_summary=include_summary,
            include_stats=include_stats,
        ): form_id
        for form_id in form_ids
    }
    done, pending = wait(futures, timeout=CATALOG_ENRICH_TIMEOUT_SECONDS)
    executor.shutdown(wait=False, cancel_futures=True)

    rows_by_id: dict[str, CatalogEnrichRow] = {}
    for future in done:
        form_id = futures[future]
        try:
            rows_by_id[form_id] = future.result()
        except Exception as exc:  # noqa: BLE001 - keep row-level failure contract.
            rows_by_id[form_id] = CatalogEnrichRow(
                form_id=form_id,
                status="api_error",
                error_code=type(exc).__name__,
            )

    for future in pending:
        form_id = futures[future]
        rows_by_id[form_id] = CatalogEnrichRow(
            form_id=form_id,
            status="timeout",
            error_code="catalog_enrich_timeout",
        )

    return [rows_by_id[form_id] for form_id in form_ids]


def _catalog_enrich_row(
    request: Request,
    creds: Any,
    user_id: str,
    form_id: str,
    *,
    include_summary: bool,
    include_stats: bool,
) -> CatalogEnrichRow:
    summary: FormSummaryResponse | None = None
    stats: ResponseStatsResponse | None = None
    cache_hits: list[bool] = []
    fetched_at_values: list[datetime] = []
    row_status = "ok"
    error_code: str | None = None

    if include_summary:
        try:
            summary_result = _cached_form_summary(request, creds, user_id, form_id)
            summary = summary_result.value
            cache_hits.append(summary_result.cache_hit)
            fetched_at_values.append(summary_result.fetched_at)
        except FormsApiError as exc:
            return CatalogEnrichRow(
                form_id=form_id,
                status=_catalog_status(exc),
                error_code="google_forms_summary_error",
            )

    if include_stats:
        try:
            stats_result = _cached_response_stats(request, creds, user_id, form_id)
            stats = stats_result.value
            cache_hits.append(stats_result.cache_hit)
            fetched_at_values.append(stats_result.fetched_at)
        except FormsApiError as exc:
            row_status = _catalog_status(exc)
            error_code = "google_forms_stats_error"

    return CatalogEnrichRow(
        form_id=form_id,
        status=row_status,
        error_code=error_code,
        summary=summary,
        response_stats=stats,
        fetched_at=_oldest_fetched_at(fetched_at_values),
        cache_hit=bool(cache_hits) and all(cache_hits),
    )


def _bounded_form_ids(form_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for form_id in form_ids:
        if form_id in seen:
            continue
        seen.add(form_id)
        unique.append(form_id)
    if len(unique) > CATALOG_ENRICH_MAX_IDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="too_many_form_ids",
        )
    return unique


def _cached_form_summary(
    request: Request,
    creds: Any,
    user_id: str,
    form_id: str,
) -> ApiCacheResult[FormSummaryResponse]:
    return get_or_load(
        ApiCacheKey(user_id=user_id, data_kind="form_summary", resource_id=form_id),
        ttl_seconds=CATALOG_SUMMARY_TTL_SECONDS,
        loader=lambda: FormSummaryResponse.model_validate(
            _forms_client(request).get_form_summary(creds, form_id)
        ),
    )


def _cached_response_stats(
    request: Request,
    creds: Any,
    user_id: str,
    form_id: str,
) -> ApiCacheResult[ResponseStatsResponse]:
    return get_or_load(
        ApiCacheKey(user_id=user_id, data_kind="response_stats", resource_id=form_id),
        ttl_seconds=RESPONSE_STATS_TTL_SECONDS,
        loader=lambda: ResponseStatsResponse.model_validate(
            _forms_client(request).get_response_stats(creds, form_id)
        ),
    )


def _oldest_fetched_at(values: list[datetime]) -> str | None:
    if not values:
        return None
    return min(values).isoformat()


def _log_catalog_enrich_telemetry(
    rows: list[CatalogEnrichRow],
    *,
    chunk_size: int,
    include_summary: bool,
    include_stats: bool,
    duration_ms: float,
) -> None:
    status_counts = Counter(row.status for row in rows)
    log.info(
        "forms_catalog_enrich_completed",
        extra={
            "chunk_size": chunk_size,
            "row_count": len(rows),
            "include_summary": include_summary,
            "include_stats": include_stats,
            "duration_ms": duration_ms,
            "cache_hit_count": sum(1 for row in rows if row.cache_hit),
            "timeout_count": status_counts.get("timeout", 0),
            "api_error_count": status_counts.get("api_error", 0),
            "rate_limited_count": status_counts.get("rate_limited", 0),
            "no_access_count": status_counts.get("no_access", 0),
            "deleted_count": status_counts.get("deleted", 0),
            "unsupported_count": status_counts.get("unsupported", 0),
            "ok_count": status_counts.get("ok", 0),
        },
    )


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
