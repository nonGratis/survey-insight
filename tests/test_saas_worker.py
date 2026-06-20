from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from core.saas.container import SaaSContainer
from core.saas.models import Plan, Quota, User, UserStatus
from core.saas.settings import load_saas_settings
from worker.main import create_worker_app

NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)


def _test_container() -> SaaSContainer:
    return SaaSContainer.in_memory(
        load_saas_settings({"APP_ENV": "test", "SESSION_PEPPER": "test-pepper"})
    )


def _production_container() -> SaaSContainer:
    settings = load_saas_settings(
        {
            "APP_ENV": "production",
            "APP_BASE_URL": "https://app.example.com",
            "API_BASE_URL": "https://api.example.com",
            "GCP_PROJECT_ID": "project",
            "KMS_KEY_NAME": "kms",
            "GCS_BUCKET": "bucket",
            "TASKS_QUEUE_NAME": "reports",
            "GOOGLE_OAUTH_CLIENT_CONFIG_JSON": "{}",
            "SESSION_PEPPER": "prod-pepper",
        }
    )
    return SaaSContainer.in_memory(settings)


def _seed_user(container: SaaSContainer) -> None:
    container.users.save(
        User(
            id="user_1",
            google_sub="google-sub-1",
            email="owner@example.com",
            name="Owner",
            picture=None,
            plan=Plan.PILOT,
            status=UserStatus.ACTIVE,
            created_at=NOW,
        )
    )


def test_worker_runs_report_job_idempotently_without_raw_response_storage() -> None:
    container = _test_container()
    _seed_user(container)
    _, job = container.report_job_service.create_report_job(
        user_id="user_1",
        form_id_hash="form_hash",
        form_title="Admissions poll",
        config_snapshot={"sections": ["overview"]},
        quota=Quota(monthly_report_limit=-1),
        now=NOW,
    )
    client = TestClient(create_worker_app(container))

    first = client.post(f"/tasks/reports/{job.id}")
    second = client.post(f"/tasks/reports/{job.id}")

    assert first.status_code == 200
    assert first.json()["status"] == "succeeded"
    assert second.status_code == 200
    assert second.json()["status"] == "succeeded"
    assert len(container.artifacts.pdfs) == 1
    assert b"answers" not in next(iter(container.artifacts.pdfs.values()))
    assert b"responses" not in next(iter(container.artifacts.pdfs.values()))


def test_worker_rejects_production_calls_without_cloud_tasks_oidc_headers() -> None:
    container = _production_container()
    client = TestClient(create_worker_app(container))

    missing_headers = client.post("/tasks/reports/job_1")
    assert missing_headers.status_code == 403

    with_headers = client.post(
        "/tasks/reports/job_1",
        headers={
            "Authorization": "Bearer signed-google-oidc-token",
            "X-CloudTasks-TaskName": "queues/reports/tasks/job_1",
        },
    )
    assert with_headers.status_code == 404
