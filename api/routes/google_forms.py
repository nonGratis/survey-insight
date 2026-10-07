"""Google Forms/Catalog routes owned by the SaaS API."""

from __future__ import annotations

import math
import os
import statistics
import threading
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime
from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

from api.dependencies import get_container, require_session
from api.google_data_cache import ApiCacheKey, ApiCacheResult, get_or_load
from api.google_errors import google_http_exception as map_google_error
from api.google_quota import FormsQuotaGuards, RollingWindowGuard
from api.urls import safe_next_url
from core.forms_api import FormsApiError
from core.logger import get_logger
from core.saas.container import SaaSContainer
from core.saas.errors import GoogleTokenRefreshFailed, GoogleTokenRevoked, MissingRequiredScopes
from core.saas.google_credentials import GoogleCredentialService
from core.saas.google_scopes import scopes_for_purpose
from core.saas.models import Session
from core.saas.ports import GoogleFormsClient
from core.saas.security import log_user_ref

router = APIRouter(prefix="/v1", tags=["google-forms"])
log = get_logger(__name__)

CATALOG_SUMMARY_TTL_SECONDS = int(os.getenv("SI_API_CATALOG_SUMMARY_TTL_SECONDS", "600"))
RESPONSE_STATS_TTL_SECONDS = int(os.getenv("SI_API_RESPONSE_STATS_TTL_SECONDS", "120"))
CATALOG_ENRICH_MAX_IDS = int(os.getenv("SI_CATALOG_ENRICH_MAX_IDS", "20"))
CATALOG_ENRICH_TIMEOUT_SECONDS = float(os.getenv("SI_CATALOG_ENRICH_TIMEOUT_SECONDS", "8"))
CATALOG_ENRICH_MAX_WORKERS = int(os.getenv("SI_CATALOG_ENRICH_MAX_WORKERS", "10"))
QUOTA_GUARD_REASON = "quota_guard"


class QuotaGuardHoldError(FormsApiError):
    """A call our quota guard held back, with the time its slot frees."""

    def __init__(self, retry_after_seconds: float) -> None:
        super().__init__(
            "Catalog paused to stay within the Google Forms API quota.",
            status=429,
            reason=QUOTA_GUARD_REASON,
        )
        self.retry_after_seconds = retry_after_seconds


class _CatalogTimings:
    """How long one catalog enrich request spent on what, for its telemetry line.

    Google calls are timed alone (no quota guard, no cache), by kind: ``get`` is
    forms.get, ``list`` is the response count (one or more responses.list pages). Rows
    are timed whole. Rows run in worker threads, hence the lock.
    """

    def __init__(self) -> None:
        self._ms: dict[str, list[float]] = {"get": [], "list": [], "row": []}
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, kind: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            with self._lock:
                self._ms[kind].append(elapsed)

    def fields(self) -> dict[str, float | int]:
        with self._lock:
            ms = {kind: list(values) for kind, values in self._ms.items()}
        out: dict[str, float | int] = {
            "google_ms_total": round(sum(ms["get"]) + sum(ms["list"]), 1),
        }
        for kind, prefix in (("get", "google_get"), ("list", "google_list"), ("row", "row")):
            values = ms[kind]
            if kind != "row":
                out[f"{prefix}_count"] = len(values)
            out[f"{prefix}_ms_p50"] = round(statistics.median(values), 1) if values else 0.0
            out[f"{prefix}_ms_max"] = round(max(values), 1) if values else 0.0
        return out


class GoogleAccessResponse(BaseModel):
    ok: bool
    has_access: bool
    purpose: str
    missing_scopes: list[str] = []
    connect_url: str | None = None


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
    # Rows held back by the quota guard: when to ask again (the guard keeps that slot).
    retry_after_seconds: float | None = None
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
    missing = _missing_scopes(get_container(request), session.user_id, purpose)
    if not missing:
        return GoogleAccessResponse(ok=True, has_access=True, purpose=purpose)
    log.info(
        "api_scope_gap",
        extra={"user_id": log_user_ref(session.user_id), "missing": list(missing)},
    )
    return GoogleAccessResponse(
        ok=False,
        has_access=False,
        purpose=purpose,
        missing_scopes=list(missing),
        connect_url=google_connect_url(request, purpose=purpose, next_url=next_url),
    )


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
        raise google_http_exception(exc, request) from exc


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
        raise google_http_exception(exc, request) from exc

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
    credentials_start = time.perf_counter()
    creds = require_google_credentials(request, session, purpose="forms")
    credentials_ms = round((time.perf_counter() - credentials_start) * 1000, 1)
    form_ids = _bounded_form_ids(body.form_ids)
    timings = _CatalogTimings()
    start = time.perf_counter()
    rows = _catalog_enrich_rows_with_budget(
        request,
        creds,
        user_id=session.user_id,
        form_ids=form_ids,
        include_summary=body.include_summary,
        include_stats=body.include_stats,
        timings=timings,
    )
    _log_catalog_enrich_telemetry(
        rows,
        chunk_size=len(form_ids),
        include_summary=body.include_summary,
        include_stats=body.include_stats,
        duration_ms=round((time.perf_counter() - start) * 1000, 1),
        timing_fields={"credentials_ms": credentials_ms, **timings.fields()},
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
        raise google_http_exception(exc, request) from exc


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
        raise google_http_exception(exc, request) from exc


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
        raise google_http_exception(exc, request) from exc


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
        raise google_http_exception(exc, request) from exc


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
        raise google_http_exception(exc, request) from exc


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
                "code": "google_insufficient_scopes",
                "action": f"reconnect_{purpose}_required",
                "purpose": purpose,
                "missing_scopes": list(_missing_scopes(container, session.user_id, purpose)),
                "connect_url": google_connect_url(request, purpose=purpose, next_url=next_url),
            },
        ) from exc


def google_http_exception(exc: FormsApiError, request: Request) -> HTTPException:
    return map_google_error(
        exc,
        purpose="forms",
        error_code="google_forms_error",
        connect_url=google_connect_url(request, purpose="forms", next_url="/"),
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
    timings: _CatalogTimings | None = None,
) -> list[CatalogEnrichRow]:
    if not form_ids:
        return []
    times = timings or _CatalogTimings()

    def timed_row(form_id: str) -> CatalogEnrichRow:
        with times.measure("row"):
            return _catalog_enrich_row(
                request,
                creds,
                user_id,
                form_id,
                include_summary=include_summary,
                include_stats=include_stats,
                timings=times,
            )

    workers = max(1, min(CATALOG_ENRICH_MAX_WORKERS, len(form_ids)))
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {executor.submit(timed_row, form_id): form_id for form_id in form_ids}
    done, pending = wait(futures, timeout=CATALOG_ENRICH_TIMEOUT_SECONDS)
    executor.shutdown(wait=False, cancel_futures=True)

    rows_by_id: dict[str, CatalogEnrichRow] = {}
    for future in done:
        form_id = futures[future]
        try:
            rows_by_id[form_id] = future.result()
        except (GoogleTokenRevoked, GoogleTokenRefreshFailed):
            # Account-level failure: every row would fail the same way, and the
            # user has to act (re-authenticate), so do not hide it as row errors.
            raise
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
    timings: _CatalogTimings | None = None,
) -> CatalogEnrichRow:
    summary: FormSummaryResponse | None = None
    stats: ResponseStatsResponse | None = None
    cache_hits: list[bool] = []
    fetched_at_values: list[datetime] = []
    row_status = "ok"
    error_code: str | None = None
    retry_after_seconds: float | None = None

    quota = _forms_quota(request)
    if include_summary:
        try:
            summary_result = _cached_form_summary(
                request, creds, user_id, form_id, guard=quota.reads, timings=timings
            )
            summary = summary_result.value
            cache_hits.append(summary_result.cache_hit)
            fetched_at_values.append(summary_result.fetched_at)
        except FormsApiError as exc:
            return CatalogEnrichRow(
                form_id=form_id,
                status=_catalog_status(exc),
                error_code=_enrich_error_code(exc, "google_forms_summary_error"),
                retry_after_seconds=_retry_after(exc),
            )

    if include_stats:
        try:
            stats_result = _cached_response_stats(
                request, creds, user_id, form_id, guard=quota.response_lists, timings=timings
            )
            stats = stats_result.value
            cache_hits.append(stats_result.cache_hit)
            fetched_at_values.append(stats_result.fetched_at)
        except FormsApiError as exc:
            row_status = _catalog_status(exc)
            error_code = _enrich_error_code(exc, "google_forms_stats_error")
            retry_after_seconds = _retry_after(exc)

    return CatalogEnrichRow(
        form_id=form_id,
        status=row_status,
        error_code=error_code,
        retry_after_seconds=retry_after_seconds,
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
    *,
    guard: RollingWindowGuard | None = None,
    timings: _CatalogTimings | None = None,
) -> ApiCacheResult[FormSummaryResponse]:
    def load() -> FormSummaryResponse:
        _admit(guard, user_id)
        with _measured(timings, "get"):
            raw = _forms_client(request).get_form_summary(creds, form_id)
        return FormSummaryResponse.model_validate(raw)

    return get_or_load(
        ApiCacheKey(user_id=user_id, data_kind="form_summary", resource_id=form_id),
        ttl_seconds=CATALOG_SUMMARY_TTL_SECONDS,
        loader=load,
    )


def _cached_response_stats(
    request: Request,
    creds: Any,
    user_id: str,
    form_id: str,
    *,
    guard: RollingWindowGuard | None = None,
    timings: _CatalogTimings | None = None,
) -> ApiCacheResult[ResponseStatsResponse]:
    def load() -> ResponseStatsResponse:
        _admit(guard, user_id)
        with _measured(timings, "list"):
            raw = _forms_client(request).get_response_stats(creds, form_id)
        return ResponseStatsResponse.model_validate(raw)

    return get_or_load(
        ApiCacheKey(user_id=user_id, data_kind="response_stats", resource_id=form_id),
        ttl_seconds=RESPONSE_STATS_TTL_SECONDS,
        loader=load,
    )


@contextmanager
def _measured(timings: _CatalogTimings | None, kind: str) -> Iterator[None]:
    if timings is None:
        yield
        return
    with timings.measure(kind):
        yield


def _admit(guard: RollingWindowGuard | None, user_id: str) -> None:
    """Spend one call of the user's quota, or fail like Google's 429 without calling Google.

    Runs only when the cache misses, so cached answers cost nothing.
    """
    if guard is None:
        return
    wait_seconds = guard.acquire_or_wait(user_id)
    if wait_seconds > 0:
        raise QuotaGuardHoldError(wait_seconds)


def _retry_after(exc: FormsApiError) -> float | None:
    if not isinstance(exc, QuotaGuardHoldError):
        return None
    # Rounded up: asked a little late the slot is free, a little early it is not.
    return math.ceil(exc.retry_after_seconds * 10) / 10


def _enrich_error_code(exc: FormsApiError, default: str) -> str:
    return QUOTA_GUARD_REASON if exc.reason == QUOTA_GUARD_REASON else default


def _forms_quota(request: Request) -> FormsQuotaGuards:
    return request.app.state.forms_quota


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
    timing_fields: dict[str, float | int] | None = None,
) -> None:
    """One line per enrich request. ``duration_ms`` is the rows' wall time; with
    ``workers`` parallel rows it compares to ``google_ms_total / workers`` (Google time)
    and ``row_ms_max`` (slowest row). ``credentials_ms`` is session, key and token work
    before the rows start. Timing fields are milliseconds."""
    status_counts = Counter(row.status for row in rows)
    log.info(
        "forms_catalog_enrich_completed",
        extra={
            **(timing_fields or {}),
            "workers": max(1, min(CATALOG_ENRICH_MAX_WORKERS, chunk_size)),
            "chunk_size": chunk_size,
            "row_count": len(rows),
            "include_summary": include_summary,
            "include_stats": include_stats,
            "duration_ms": duration_ms,
            "cache_hit_count": sum(1 for row in rows if row.cache_hit),
            "timeout_count": status_counts.get("timeout", 0),
            # Rows held back by our own quota guard; the rest of rate_limited is Google's 429.
            "quota_guard_count": sum(1 for row in rows if row.error_code == QUOTA_GUARD_REASON),
            "api_error_count": status_counts.get("api_error", 0),
            "rate_limited_count": status_counts.get("rate_limited", 0),
            "no_access_count": status_counts.get("no_access", 0),
            "deleted_count": status_counts.get("deleted", 0),
            "unsupported_count": status_counts.get("unsupported", 0),
            "ok_count": status_counts.get("ok", 0),
            # Loaded forms whose state Google does not report (legacy forms without
            # publishSettings): the catalog shows them as «Невідомо».
            "publish_state_unknown_count": sum(
                1 for row in rows if row.summary is not None and row.summary.is_published is None
            ),
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


def google_connect_url(request: Request, *, purpose: str, next_url: str) -> str:
    container = get_container(request)
    app_base_url = container.settings.app_base_url
    safe_next = safe_next_url(next_url, app_base_url)
    if safe_next.startswith("/") and app_base_url:
        # After consent the callback redirects to next_url; a bare path would
        # resolve against the API host instead of the web app.
        safe_next = app_base_url.rstrip("/") + safe_next
    query = urlencode({"purpose": purpose, "next_url": safe_next})
    return f"{container.settings.api_base_url.rstrip('/')}/v1/auth/google/start?{query}"
