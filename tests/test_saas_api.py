from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient
from google.oauth2.credentials import Credentials

from api.main import SESSION_COOKIE_NAME, create_api_app
from core.saas.container import SaaSContainer
from core.saas.inmemory import InMemoryTaskQueue
from core.saas.models import Plan, Quota, User, UserStatus
from core.saas.settings import load_saas_settings

NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)


class _FakeOAuthClient:
    def __init__(self) -> None:
        self.last_state: str | None = None
        self.last_code_verifier: str | None = None
        self.last_scopes: tuple[str, ...] = ()

    def authorization_url(self, *, state: str, code_verifier: str, scopes) -> str:
        self.last_state = state
        self.last_code_verifier = code_verifier
        self.last_scopes = tuple(scopes)
        return f"https://accounts.example/auth?state={state}"

    def exchange_code(self, *, code: str, state: str, code_verifier: str, scopes) -> Credentials:
        assert code == "oauth-code"
        assert state == self.last_state
        assert code_verifier == self.last_code_verifier
        return Credentials(
            token="access-token",
            refresh_token="refresh-token",
            token_uri="https://oauth2.googleapis.com/token",
            client_id="client-id",
            client_secret="client-secret",
            scopes=list(scopes),
        )

    def user_info(self, credentials: Credentials) -> dict:
        assert credentials.token == "access-token"
        return {
            "id": "sub_1",
            "email": "owner@example.com",
            "name": "Owner",
            "picture": "https://example.com/avatar.png",
        }


def _test_container() -> SaaSContainer:
    return SaaSContainer.in_memory(
        load_saas_settings({"APP_ENV": "test", "SESSION_PEPPER": "test-pepper"})
    )


def _seed_user_session(container: SaaSContainer) -> str:
    user = User(
        id="user_1",
        google_sub="google-sub-1",
        email="owner@example.com",
        name="Owner",
        picture=None,
        plan=Plan.PILOT,
        status=UserStatus.ACTIVE,
        created_at=NOW,
    )
    container.users.save(user)
    container.quotas.save(user.id, Quota(monthly_report_limit=3, reports_used_this_month=0))
    return container.session_service.create(user_id=user.id, now=NOW).session_id


def test_api_session_restores_from_cookie_and_never_requires_streamlit_state() -> None:
    container = _test_container()
    session_id = _seed_user_session(container)
    client = TestClient(create_api_app(container))

    assert client.get("/v1/session").json() == {"authenticated": False}

    client.cookies.set(SESSION_COOKIE_NAME, session_id)
    response = client.get("/v1/session")

    assert response.status_code == 200
    assert response.json() == {
        "authenticated": True,
        "user_id": "user_1",
        "email": "owner@example.com",
        "name": "Owner",
        "plan": "pilot",
    }


def test_api_report_job_requires_session_and_enqueues_with_hashed_form_id() -> None:
    container = _test_container()
    session_id = _seed_user_session(container)
    client = TestClient(create_api_app(container))

    unauthorized = client.post("/v1/reports/jobs", json={"form_id": "form_raw"})
    assert unauthorized.status_code == 401

    client.cookies.set(SESSION_COOKIE_NAME, session_id)
    response = client.post(
        "/v1/reports/jobs",
        json={
            "form_id": "form_raw",
            "form_title": "Admissions poll",
            "config_snapshot": {"sections": ["overview"]},
        },
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    assert isinstance(container.tasks, InMemoryTaskQueue)
    assert container.tasks.report_job_ids == [payload["job_id"]]
    report = container.reports.get(payload["report_id"])
    assert report is not None
    assert report.form_id_hash != "form_raw"


def test_google_oauth_callback_creates_cookie_session_and_encrypted_tokens() -> None:
    container = _test_container()
    oauth = _FakeOAuthClient()
    client = TestClient(create_api_app(container, oauth_client=oauth))

    start = client.get(
        "/v1/auth/google/start",
        params={"next_url": "/catalog"},
        follow_redirects=False,
    )

    assert start.status_code == 307
    assert start.headers["location"].startswith("https://accounts.example/auth")
    assert oauth.last_state is not None

    callback = client.get(
        "/v1/auth/google/callback",
        params={"state": oauth.last_state, "code": "oauth-code"},
        follow_redirects=False,
    )

    assert callback.status_code == 307
    assert callback.headers["location"] == "/catalog"
    session_id = client.cookies.get(SESSION_COOKIE_NAME)
    assert session_id is not None
    session = container.session_service.validate(session_id)
    assert session.user_id == "google:sub_1"

    user = container.users.get("google:sub_1")
    assert user is not None
    assert user.email == "owner@example.com"
    assert container.quotas.get_for_user(user.id) is not None

    account = container.tokens.get_by_user(user.id)
    assert account is not None
    assert account.encrypted_refresh_token is not None
    assert "refresh-token" not in account.encrypted_refresh_token

    replay = client.get(
        "/v1/auth/google/callback",
        params={"state": oauth.last_state, "code": "oauth-code"},
        follow_redirects=False,
    )
    assert replay.status_code == 401


def test_google_oauth_start_rejects_open_redirect_next_url() -> None:
    container = _test_container()
    oauth = _FakeOAuthClient()
    client = TestClient(create_api_app(container, oauth_client=oauth))

    client.get(
        "/v1/auth/google/start",
        params={"next_url": "https://evil.example/callback"},
        follow_redirects=False,
    )

    callback = client.get(
        "/v1/auth/google/callback",
        params={"state": oauth.last_state, "code": "oauth-code"},
        follow_redirects=False,
    )

    assert callback.status_code == 307
    assert callback.headers["location"] == "/"
