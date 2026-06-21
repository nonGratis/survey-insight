from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import httpx

from ui.saas_api import SESSION_COOKIE_NAME, SaaSApiClient


def test_google_auth_start_url_preserves_next_url() -> None:
    client = SaaSApiClient("https://api.example.com/")

    url = client.google_auth_start_url("https://app.example.com/catalog?form_id=abc")

    parsed = urlsplit(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "api.example.com"
    assert parsed.path == "/v1/auth/google/start"
    assert parse_qs(parsed.query) == {"next_url": ["https://app.example.com/catalog?form_id=abc"]}


def test_exchange_login_ticket_returns_session_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/auth/session/exchange"
        assert request.read() == b'{"ticket":"ticket-1"}'
        return httpx.Response(
            200,
            json={
                "authenticated": True,
                "user_id": "user_1",
                "email": "owner@example.com",
                "name": "Owner",
                "plan": "pilot",
                "session_id": "raw-session-id",
            },
        )

    client = SaaSApiClient(
        "https://api.example.com",
        transport=httpx.MockTransport(handler),
    )

    session = client.exchange_login_ticket("ticket-1")

    assert session.authenticated is True
    assert session.user_id == "user_1"
    assert session.session_id == "raw-session-id"


def test_read_session_sends_cookie_and_keeps_fallback_session_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/session"
        assert request.headers["cookie"] == f"{SESSION_COOKIE_NAME}=raw-session-id"
        return httpx.Response(
            200,
            json={
                "authenticated": True,
                "user_id": "user_1",
                "email": "owner@example.com",
                "name": "Owner",
                "plan": "pilot",
            },
        )

    client = SaaSApiClient(
        "https://api.example.com",
        transport=httpx.MockTransport(handler),
    )

    session = client.read_session("raw-session-id")

    assert session.authenticated is True
    assert session.session_id == "raw-session-id"
