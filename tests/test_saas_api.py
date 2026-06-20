from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from api.main import SESSION_COOKIE_NAME, create_api_app
from core.saas.container import SaaSContainer
from core.saas.models import Plan, Quota, User, UserStatus
from core.saas.settings import load_saas_settings

NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)


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
    assert container.tasks.report_job_ids == [payload["job_id"]]
    report = container.reports.get(payload["report_id"])
    assert report is not None
    assert report.form_id_hash != "form_raw"
