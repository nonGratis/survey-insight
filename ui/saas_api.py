"""Small HTTP client used by the Streamlit UI to talk to the SaaS API."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

from core.logger import get_logger

SESSION_COOKIE_NAME = "session_id"
# Session checks must fail fast so a cold-starting API degrades gracefully;
# data endpoints do heavier Google round-trips and get a longer read window.
SESSION_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=2.0)
DATA_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=2.0)
# Statuses a proxy or Cloud Run itself returns while the API is starting, being
# replaced or out of capacity. They describe the API's state, not the session's.
_API_UNAVAILABLE_STATUSES = frozenset({429, 502, 503, 504})
log = get_logger(__name__)


@dataclass(frozen=True)
class SaaSSession:
    authenticated: bool
    user_id: str | None = None
    email: str | None = None
    name: str | None = None
    plan: str | None = None
    session_id: str | None = None


@dataclass(frozen=True)
class GoogleAccess:
    ok: bool
    purpose: str


class MissingGoogleScopesError(RuntimeError):
    def __init__(
        self,
        *,
        purpose: str,
        missing_scopes: list[str],
        connect_url: str,
    ) -> None:
        super().__init__("Missing Google OAuth scopes.")
        self.purpose = purpose
        self.missing_scopes = missing_scopes
        self.connect_url = connect_url


class SaaSApiError(httpx.HTTPError):
    """Base for API failures the UI knows how to present.

    Subclasses httpx.HTTPError so existing ``except httpx.HTTPError`` handlers
    (including per-row failure handling in worker threads) keep working.
    """


class SessionExpiredError(SaaSApiError):
    """The API rejected our session (401)."""


class GoogleTokenRevokedError(SaaSApiError):
    """Google no longer honours the stored grant; the user must sign in again."""


class GoogleUnavailableError(SaaSApiError):
    """Google answered with an upstream failure (API returned 502)."""


class ApiServerError(SaaSApiError):
    """The SaaS API itself failed with a 5xx other than 502."""

    def __init__(self, status_code: int, error_code: str = "") -> None:
        super().__init__(f"SaaS API returned {status_code}.")
        self.status_code = status_code
        self.error_code = error_code


class SaaSApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        session_timeout: httpx.Timeout = SESSION_TIMEOUT,
        data_timeout: httpx.Timeout = DATA_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_timeout = session_timeout
        self.data_timeout = data_timeout
        self.transport = transport

    def google_auth_start_url(self, next_url: str, *, purpose: str = "identity") -> str:
        query = urlencode({"next_url": next_url, "purpose": purpose})
        return f"{self.base_url}/v1/auth/google/start?{query}"

    def exchange_login_ticket(self, ticket: str) -> SaaSSession:
        with self._client(self.session_timeout) as client:
            response = client.post("/v1/auth/session/exchange", json={"ticket": ticket})
            response.raise_for_status()
            return _session_from_payload(response.json())

    def read_session(self, session_id: str | None) -> SaaSSession | None:
        """Return the session, or None when the API could not answer.

        None means "unknown" (cold start, restart, network trouble), which is
        different from an unauthenticated SaaSSession: callers must not drop the
        stored session on None.
        """
        try:
            with self._client(self.session_timeout) as client:
                if session_id:
                    client.cookies.set(SESSION_COOKIE_NAME, session_id)
                response = client.get("/v1/session")
                if response.status_code in _API_UNAVAILABLE_STATUSES:
                    log.warning("saas_session_unavailable", extra={"status": response.status_code})
                    return None
                response.raise_for_status()
                return _session_from_payload(response.json(), fallback_session_id=session_id)
        except httpx.TimeoutException as exc:
            log.warning("saas_session_timeout", extra={"error_code": type(exc).__name__})
            return None
        except (httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            log.warning("saas_session_unavailable", extra={"error_code": type(exc).__name__})
            return None

    def logout(self, session_id: str | None) -> None:
        with self._client(self.session_timeout) as client:
            if session_id:
                client.cookies.set(SESSION_COOKIE_NAME, session_id)
            response = client.post("/v1/auth/logout")
            response.raise_for_status()

    def check_google_access(
        self,
        session_id: str,
        *,
        purpose: str = "forms",
        next_url: str = "/",
    ) -> GoogleAccess:
        response = self._request_with_session(
            session_id,
            "GET",
            "/v1/google/access",
            params={"purpose": purpose, "next_url": next_url},
            timeout=self.session_timeout,
        )
        if not response.get("has_access", response.get("ok")):
            raise MissingGoogleScopesError(
                purpose=str(response.get("purpose") or purpose),
                missing_scopes=[str(scope) for scope in response.get("missing_scopes", [])],
                connect_url=str(response.get("connect_url") or ""),
            )
        return GoogleAccess(ok=True, purpose=str(response.get("purpose")))

    def list_forms(self, session_id: str) -> list[dict[str, Any]]:
        return list(self._request_with_session(session_id, "GET", "/v1/forms"))

    def list_forms_catalog(self, session_id: str) -> list[dict[str, Any]]:
        return list(self._request_with_session(session_id, "GET", "/v1/forms/catalog"))

    def enrich_forms_catalog(
        self,
        session_id: str,
        form_ids: list[str],
        *,
        include_summary: bool = True,
        include_stats: bool = True,
    ) -> list[dict[str, Any]]:
        return list(
            self._request_with_session(
                session_id,
                "POST",
                "/v1/forms/catalog/enrich",
                json={
                    "form_ids": form_ids,
                    "include_summary": include_summary,
                    "include_stats": include_stats,
                },
            )
        )

    def get_form_summary(self, session_id: str, form_id: str) -> dict[str, Any]:
        return dict(self._request_with_session(session_id, "GET", f"/v1/forms/{form_id}/summary"))

    def get_response_stats(self, session_id: str, form_id: str) -> dict[str, Any]:
        return dict(
            self._request_with_session(session_id, "GET", f"/v1/forms/{form_id}/response-stats")
        )

    def get_form_structure(self, session_id: str, form_id: str) -> dict[str, Any]:
        return dict(self._request_with_session(session_id, "GET", f"/v1/forms/{form_id}/structure"))

    def list_form_responses(self, session_id: str, form_id: str) -> list[dict[str, Any]]:
        return list(self._request_with_session(session_id, "GET", f"/v1/forms/{form_id}/responses"))

    def list_response_timestamps(self, session_id: str, form_id: str) -> list[str]:
        payload = self._request_with_session(
            session_id,
            "GET",
            f"/v1/forms/{form_id}/response-timestamps",
        )
        return [str(item) for item in payload.get("timestamps", [])]

    def list_population_tables(
        self,
        session_id: str,
        sheet_id: str,
        *,
        next_url: str = "/",
    ) -> list[dict[str, Any]]:
        return list(
            self._request_with_session(
                session_id,
                "GET",
                f"/v1/sheets/{sheet_id}/population-tables",
                params={"next_url": next_url},
            )
        )

    def _request_with_session(
        self,
        session_id: str,
        method: str,
        path: str,
        *,
        timeout: httpx.Timeout | None = None,
        **kwargs: Any,
    ) -> Any:
        start = time.perf_counter()
        status_code = 0
        error_code = ""
        with self._client(timeout or self.data_timeout) as client:
            client.cookies.set(SESSION_COOKIE_NAME, session_id)
            try:
                response = client.request(method, path, **kwargs)
                status_code = response.status_code
                error_code = _raise_typed_error(response)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                error_code = _error_code(exc.response) or type(exc).__name__
                raise
            except Exception as exc:
                error_code = error_code or _typed_error_code(exc)
                raise
            finally:
                duration_ms = round((time.perf_counter() - start) * 1000, 1)
                log.info(
                    "ui_saas_api_request",
                    extra={
                        "method": method,
                        "path": path,
                        "status": status_code,
                        "duration_ms": duration_ms,
                        "error_code": error_code,
                        "cache_hit": False,
                        "cache_layer": "none",
                    },
                )

    def _client(self, timeout: httpx.Timeout) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=self.transport,
            follow_redirects=False,
        )


def _session_from_payload(
    payload: dict[str, object],
    *,
    fallback_session_id: str | None = None,
) -> SaaSSession:
    session_id = payload.get("session_id")
    return SaaSSession(
        authenticated=bool(payload.get("authenticated", False)),
        user_id=_optional_str(payload.get("user_id")),
        email=_optional_str(payload.get("email")),
        name=_optional_str(payload.get("name")),
        plan=_optional_str(payload.get("plan")),
        session_id=_optional_str(session_id) or fallback_session_id,
    )


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _detail_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    if not isinstance(payload, dict):
        return {}
    detail = payload.get("detail")
    return detail if isinstance(detail, dict) else {}


def _raise_typed_error(response: httpx.Response) -> str:
    """Raise the typed error for statuses the UI handles centrally.

    Returns the error code for successful responses so callers can log it;
    other statuses fall through to ``raise_for_status``.
    """
    status = response.status_code
    if status < 400:
        return ""
    detail = _detail_payload(response)
    code = str(detail.get("code") or "")
    if status in (401, 403) and code == "google_token_revoked":
        raise GoogleTokenRevokedError("Google grant was revoked.")
    if status == 401:
        raise SessionExpiredError("SaaS session is not valid.")
    if status == 403 and code == "google_insufficient_scopes":
        raise MissingGoogleScopesError(
            purpose=str(detail.get("purpose") or ""),
            missing_scopes=[str(scope) for scope in detail.get("missing_scopes", [])],
            connect_url=str(detail.get("connect_url") or ""),
        )
    if status == 502:
        raise GoogleUnavailableError("Google API is temporarily unavailable.")
    if status >= 500:
        raise ApiServerError(status, code)
    return ""


def _typed_error_code(exc: Exception) -> str:
    code = getattr(exc, "error_code", "")
    return str(code) if code else type(exc).__name__


def _error_code(response: httpx.Response) -> str:
    detail = _detail_payload(response)
    code = detail.get("code")
    return str(code) if code else ""
