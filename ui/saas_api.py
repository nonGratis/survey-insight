"""Small HTTP client used by the Streamlit UI to talk to the SaaS API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

SESSION_COOKIE_NAME = "session_id"


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


class SaaSApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    def google_auth_start_url(self, next_url: str, *, purpose: str = "identity") -> str:
        query = urlencode({"next_url": next_url, "purpose": purpose})
        return f"{self.base_url}/v1/auth/google/start?{query}"

    def exchange_login_ticket(self, ticket: str) -> SaaSSession:
        with self._client() as client:
            response = client.post("/v1/auth/session/exchange", json={"ticket": ticket})
            response.raise_for_status()
            return _session_from_payload(response.json())

    def read_session(self, session_id: str | None) -> SaaSSession:
        with self._client() as client:
            if session_id:
                client.cookies.set(SESSION_COOKIE_NAME, session_id)
            response = client.get("/v1/session")
            response.raise_for_status()
            return _session_from_payload(response.json(), fallback_session_id=session_id)

    def logout(self, session_id: str | None) -> None:
        with self._client() as client:
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
        )
        return GoogleAccess(ok=bool(response.get("ok")), purpose=str(response.get("purpose")))

    def list_forms(self, session_id: str) -> list[dict[str, Any]]:
        return list(self._request_with_session(session_id, "GET", "/v1/forms"))

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
        **kwargs: Any,
    ) -> Any:
        with self._client() as client:
            client.cookies.set(SESSION_COOKIE_NAME, session_id)
            response = client.request(method, path, **kwargs)
            if response.status_code == 403:
                detail = _detail_payload(response)
                if detail.get("code") == "missing_required_scopes":
                    raise MissingGoogleScopesError(
                        purpose=str(detail.get("purpose") or ""),
                        missing_scopes=[str(scope) for scope in detail.get("missing_scopes", [])],
                        connect_url=str(detail.get("connect_url") or ""),
                    )
            response.raise_for_status()
            return response.json()

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
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
    detail = payload.get("detail")
    return detail if isinstance(detail, dict) else {}
