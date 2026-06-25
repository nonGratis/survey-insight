from __future__ import annotations

from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlsplit

from fastapi.testclient import TestClient
from google.oauth2.credentials import Credentials

from api.main import SESSION_COOKIE_NAME, create_api_app
from core.saas.container import SaaSContainer
from core.saas.google_scopes import FORM_SCOPES
from core.saas.inmemory import InMemoryTaskQueue
from core.saas.models import OAuthAccount, Plan, Quota, User, UserStatus
from core.saas.settings import load_saas_settings

NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)


class _FakeOAuthClient:
    def __init__(self) -> None:
        self.last_state: str | None = None
        self.last_code_verifier: str | None = None
        self.last_scopes: tuple[str, ...] = ()
        self.last_include_granted_scopes = False

    def authorization_url(
        self,
        *,
        state: str,
        code_verifier: str,
        scopes,
        include_granted_scopes: bool = False,
    ) -> str:
        self.last_state = state
        self.last_code_verifier = code_verifier
        self.last_scopes = tuple(scopes)
        self.last_include_granted_scopes = include_granted_scopes
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


class _FailingUserInfoOAuthClient(_FakeOAuthClient):
    def user_info(self, credentials: Credentials) -> dict:
        raise RuntimeError("userinfo failed")


class _FakeGoogleFormsClient:
    def list_forms(self, creds: Credentials) -> list[dict]:
        assert creds.token == "access-token"
        return [
            {
                "id": "form_1",
                "name": "Admissions poll",
                "owner_email": "owner@example.com",
                "owner_name": "Owner",
                "created_time": "2026-06-01T10:00:00Z",
                "modified_time": "2026-06-02T10:00:00Z",
                "edit_url": "https://docs.google.com/forms/d/form_1/edit",
            }
        ]

    def get_form_summary(self, creds: Credentials, form_id: str) -> dict:
        assert form_id == "form_1"
        return {
            "title": "Admissions poll",
            "description": "desc",
            "sections_count": 2,
            "questions_count": 5,
            "linked_sheet_id": None,
            "is_published": True,
            "accepting_responses": True,
        }

    def get_response_stats(self, creds: Credentials, form_id: str) -> dict:
        assert form_id == "form_1"
        return {
            "total": 2,
            "first_response": "2026-06-01T10:00:00",
            "second_response": "2026-06-01T10:05:00",
            "last_response": "2026-06-01T10:05:00",
        }

    def get_form_structure(self, creds: Credentials, form_id: str) -> dict:
        assert form_id == "form_1"
        return {"formId": form_id, "info": {"title": "Admissions poll"}, "items": []}

    def list_responses(self, creds: Credentials, form_id: str) -> list[dict]:
        assert form_id == "form_1"
        return [{"responseId": "r1", "answers": {"q1": {"textAnswers": {"answers": []}}}}]


def _test_container() -> SaaSContainer:
    return SaaSContainer.in_memory(
        load_saas_settings(
            {
                "APP_ENV": "test",
                "API_BASE_URL": "https://api.example.com",
                "APP_BASE_URL": "https://app.example.com",
                "GOOGLE_OAUTH_CLIENT_CONFIG_JSON": (
                    '{"web":{"token_uri":"https://oauth2.googleapis.com/token",'
                    '"client_id":"client-id","client_secret":"client-secret"}}'
                ),
                "SESSION_PEPPER": "test-pepper",
            }
        )
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


def _seed_google_grant(container: SaaSContainer, user_id: str = "user_1") -> None:
    container.tokens.save(
        OAuthAccount(
            user_id=user_id,
            provider="google",
            google_sub="google-sub-1",
            email="owner@example.com",
            scopes=FORM_SCOPES,
            encrypted_access_token=container.token_crypto.encrypt("access-token"),
            encrypted_refresh_token=container.token_crypto.encrypt("refresh-token"),
            token_expiry=datetime.now(UTC) + timedelta(hours=1),
            updated_at=NOW,
        )
    )


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


def test_google_access_returns_connect_url_when_forms_scopes_missing() -> None:
    container = _test_container()
    session_id = _seed_user_session(container)
    client = TestClient(create_api_app(container))
    client.cookies.set(SESSION_COOKIE_NAME, session_id)

    response = client.get(
        "/v1/google/access",
        params={"purpose": "forms", "next_url": "https://app.example.com/catalog"},
    )

    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail["code"] == "missing_required_scopes"
    assert detail["purpose"] == "forms"
    assert "https://www.googleapis.com/auth/forms.responses.readonly" in detail["missing_scopes"]
    assert detail["connect_url"].startswith("https://api.example.com/v1/auth/google/start?")
    assert "purpose=forms" in detail["connect_url"]


def test_google_auth_start_supports_incremental_forms_purpose() -> None:
    container = _test_container()
    oauth = _FakeOAuthClient()
    client = TestClient(create_api_app(container, oauth_client=oauth))

    response = client.get(
        "/v1/auth/google/start",
        params={"purpose": "forms", "next_url": "https://app.example.com/catalog"},
        follow_redirects=False,
    )

    assert response.status_code == 307
    assert oauth.last_include_granted_scopes is True
    assert "https://www.googleapis.com/auth/forms.body.readonly" in oauth.last_scopes
    assert "https://www.googleapis.com/auth/drive.metadata.readonly" in oauth.last_scopes


def test_forms_api_routes_use_server_side_google_credentials_without_exposing_tokens() -> None:
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    client = TestClient(create_api_app(container, google_forms_client=_FakeGoogleFormsClient()))
    client.cookies.set(SESSION_COOKIE_NAME, session_id)

    access = client.get("/v1/google/access", params={"purpose": "forms"})
    forms = client.get("/v1/forms")
    summary = client.get("/v1/forms/form_1/summary")
    stats = client.get("/v1/forms/form_1/response-stats")
    structure = client.get("/v1/forms/form_1/structure")
    responses = client.get("/v1/forms/form_1/responses")

    assert access.json() == {"ok": True, "purpose": "forms"}
    assert forms.status_code == 200
    assert forms.json()[0]["id"] == "form_1"
    assert summary.json()["questions_count"] == 5
    assert stats.json()["total"] == 2
    assert structure.json()["formId"] == "form_1"
    assert responses.json()[0]["responseId"] == "r1"

    combined_payload = str(
        [forms.json(), summary.json(), stats.json(), structure.json(), responses.json()]
    )
    assert "access-token" not in combined_payload
    assert "refresh-token" not in combined_payload
    assert getattr(container.artifacts, "pdfs", {}) == {}


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
    redirect = urlsplit(callback.headers["location"])
    assert redirect.path == "/catalog"
    ticket = parse_qs(redirect.query)["login_ticket"][0]
    assert client.cookies.get(SESSION_COOKIE_NAME) is None

    exchanged = client.post("/v1/auth/session/exchange", json={"ticket": ticket})

    assert exchanged.status_code == 200
    assert exchanged.json()["authenticated"] is True
    session_id = client.cookies.get(SESSION_COOKIE_NAME)
    assert session_id is not None
    assert exchanged.json()["session_id"] == session_id
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

    replay_ticket = client.post("/v1/auth/session/exchange", json={"ticket": ticket})
    assert replay_ticket.status_code == 401


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
    redirect = urlsplit(callback.headers["location"])
    assert redirect.path == "/"
    assert "login_ticket" in parse_qs(redirect.query)


def test_google_oauth_callback_redirects_to_web_on_internal_failure() -> None:
    container = SaaSContainer.in_memory(
        load_saas_settings(
            {
                "APP_ENV": "test",
                "APP_BASE_URL": "https://app.example.com",
                "API_BASE_URL": "https://api.example.com",
                "SESSION_PEPPER": "test-pepper",
            }
        )
    )
    oauth = _FailingUserInfoOAuthClient()
    client = TestClient(create_api_app(container, oauth_client=oauth))

    client.get(
        "/v1/auth/google/start",
        params={"next_url": "https://app.example.com/"},
        follow_redirects=False,
    )

    callback = client.get(
        "/v1/auth/google/callback",
        params={"state": oauth.last_state, "code": "oauth-code"},
        follow_redirects=False,
    )

    assert callback.status_code == 307
    redirect = urlsplit(callback.headers["location"])
    assert redirect.scheme == "https"
    assert redirect.netloc == "app.example.com"
    assert parse_qs(redirect.query) == {"auth_error": ["oauth_callback_failed"]}
