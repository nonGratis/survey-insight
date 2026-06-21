"""Small HTTP client used by the Streamlit UI to talk to the SaaS API."""

from __future__ import annotations

from dataclasses import dataclass
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

    def google_auth_start_url(self, next_url: str) -> str:
        query = urlencode({"next_url": next_url})
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
