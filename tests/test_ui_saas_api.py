from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import httpx

from ui.saas_api import (
    SESSION_COOKIE_NAME,
    MissingGoogleScopesError,
    SaaSApiClient,
)


def test_google_auth_start_url_preserves_next_url() -> None:
    client = SaaSApiClient("https://api.example.com/")

    url = client.google_auth_start_url("https://app.example.com/catalog?form_id=abc")

    parsed = urlsplit(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "api.example.com"
    assert parsed.path == "/v1/auth/google/start"
    assert parse_qs(parsed.query) == {
        "next_url": ["https://app.example.com/catalog?form_id=abc"],
        "purpose": ["identity"],
    }


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


def test_check_google_access_sends_session_cookie() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/google/access"
        assert request.url.params["purpose"] == "forms"
        assert request.headers["cookie"] == f"{SESSION_COOKIE_NAME}=raw-session-id"
        return httpx.Response(200, json={"ok": True, "purpose": "forms"})

    client = SaaSApiClient(
        "https://api.example.com",
        transport=httpx.MockTransport(handler),
    )

    access = client.check_google_access(
        "raw-session-id",
        purpose="forms",
        next_url="https://app.example.com/",
    )

    assert access.ok is True
    assert access.purpose == "forms"


def test_check_google_access_raises_typed_missing_scope_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "detail": {
                    "code": "missing_required_scopes",
                    "purpose": "forms",
                    "missing_scopes": ["https://www.googleapis.com/auth/forms.body.readonly"],
                    "connect_url": "https://api.example.com/v1/auth/google/start?purpose=forms",
                }
            },
        )

    client = SaaSApiClient(
        "https://api.example.com",
        transport=httpx.MockTransport(handler),
    )

    try:
        client.check_google_access("raw-session-id", purpose="forms")
    except MissingGoogleScopesError as exc:
        assert exc.purpose == "forms"
        assert exc.missing_scopes == ["https://www.googleapis.com/auth/forms.body.readonly"]
        assert exc.connect_url.endswith("purpose=forms")
    else:  # pragma: no cover - assertion branch
        raise AssertionError("MissingGoogleScopesError was not raised")


def test_forms_client_methods_send_session_cookie() -> None:
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["cookie"] == f"{SESSION_COOKIE_NAME}=raw-session-id"
        seen_paths.append(request.url.path)
        payloads = {
            "/v1/forms": [{"id": "form_1", "name": "Survey"}],
            "/v1/forms/catalog": [{"status": "ok", "form": {"id": "form_1", "name": "Survey"}}],
            "/v1/forms/form_1/summary": {"title": "Survey", "questions_count": 1},
            "/v1/forms/form_1/response-stats": {"total": 2},
            "/v1/forms/form_1/response-timestamps": {"timestamps": ["2026-06-01T10:00:00"]},
            "/v1/forms/form_1/structure": {"formId": "form_1"},
            "/v1/forms/form_1/responses": [{"responseId": "r1"}],
            "/v1/sheets/sheet_1/population-tables": [
                {
                    "source": "Population",
                    "label_header": "Faculty",
                    "count_header": "N",
                    "population": {"FICT": 120},
                }
            ],
        }
        return httpx.Response(200, json=payloads[request.url.path])

    client = SaaSApiClient(
        "https://api.example.com",
        transport=httpx.MockTransport(handler),
    )

    assert client.list_forms("raw-session-id")[0]["id"] == "form_1"
    assert client.list_forms_catalog("raw-session-id")[0]["status"] == "ok"
    assert client.get_form_summary("raw-session-id", "form_1")["title"] == "Survey"
    assert client.get_response_stats("raw-session-id", "form_1")["total"] == 2
    assert client.list_response_timestamps("raw-session-id", "form_1") == ["2026-06-01T10:00:00"]
    assert client.get_form_structure("raw-session-id", "form_1")["formId"] == "form_1"
    assert client.list_form_responses("raw-session-id", "form_1")[0]["responseId"] == "r1"
    assert client.list_population_tables(
        "raw-session-id",
        "sheet_1",
        next_url="https://app.example.com/weighting",
    )[0]["population"] == {"FICT": 120}
    assert seen_paths == [
        "/v1/forms",
        "/v1/forms/catalog",
        "/v1/forms/form_1/summary",
        "/v1/forms/form_1/response-stats",
        "/v1/forms/form_1/response-timestamps",
        "/v1/forms/form_1/structure",
        "/v1/forms/form_1/responses",
        "/v1/sheets/sheet_1/population-tables",
    ]
