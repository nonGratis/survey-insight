from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from core.saas import (
    ExpiredLoginTicket,
    ExpiredOAuthState,
    InvalidSession,
    JobStatus,
    MissingRequiredScopes,
    Quota,
    QuotaExceeded,
    ReplayedLoginTicket,
    ReplayedOAuthState,
)
from core.saas.container import SaaSContainer
from core.saas.inmemory import (
    InMemoryJobRepository,
    InMemoryLoginTicketRepository,
    InMemoryOAuthStateRepository,
    InMemoryReportRepository,
    InMemorySessionRepository,
    InMemoryTaskQueue,
    InMemoryTokenCrypto,
)
from core.saas.security import hash_secret
from core.saas.services import (
    JobStateMachine,
    LoginTicketService,
    OAuthStateService,
    ReportJobService,
    SessionService,
)
from core.saas.settings import load_saas_settings

NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
PEPPER = "test-pepper"


def test_session_cookie_secret_is_never_stored_raw_and_validates_by_hmac() -> None:
    repo = InMemorySessionRepository()
    service = SessionService(repo, pepper=PEPPER)

    secret = service.create(user_id="user_1", now=NOW)

    assert secret.session_id not in repo.items
    assert secret.record.session_hash in repo.items
    assert secret.record.session_hash == hash_secret(secret.session_id, PEPPER)

    restored = service.validate(secret.session_id, now=NOW + timedelta(minutes=5))
    assert restored.user_id == "user_1"
    assert restored.last_seen_at == NOW + timedelta(minutes=5)


def test_session_rejects_tampered_expired_and_revoked_ids() -> None:
    repo = InMemorySessionRepository()
    service = SessionService(repo, pepper=PEPPER)
    secret = service.create(user_id="user_1", ttl=timedelta(minutes=1), now=NOW)

    with pytest.raises(InvalidSession):
        service.validate(secret.session_id + "x", now=NOW)
    with pytest.raises(InvalidSession):
        service.validate(secret.session_id, now=NOW + timedelta(minutes=2))

    fresh = service.create(user_id="user_2", ttl=timedelta(days=1), now=NOW)
    service.revoke(fresh.session_id, now=NOW + timedelta(minutes=1))
    with pytest.raises(InvalidSession):
        service.validate(fresh.session_id, now=NOW + timedelta(minutes=2))


def test_oauth_state_is_one_time_expiring_and_scope_checked() -> None:
    repo = InMemoryOAuthStateRepository()
    service = OAuthStateService(repo, pepper=PEPPER)
    secret = service.create(
        scopes=["openid", "email", "forms.readonly"],
        next_url="/catalog",
        ttl=timedelta(minutes=1),
        now=NOW,
    )

    with pytest.raises(MissingRequiredScopes):
        service.consume(secret.state, required_scopes=["sheets.readonly"], now=NOW)

    consumed = service.consume(secret.state, required_scopes=["openid"], now=NOW)
    assert consumed.used_at == NOW

    with pytest.raises(ReplayedOAuthState):
        service.consume(secret.state, now=NOW)

    expired = service.create(scopes=["openid"], next_url="/", ttl=timedelta(seconds=1), now=NOW)
    with pytest.raises(ExpiredOAuthState):
        service.consume(expired.state, now=NOW + timedelta(seconds=2))


def test_login_ticket_is_one_time_and_expiring() -> None:
    repo = InMemoryLoginTicketRepository()
    service = LoginTicketService(repo, pepper=PEPPER)
    secret = service.create(user_id="user_1", ttl=timedelta(minutes=1), now=NOW)

    consumed = service.consume(secret.ticket, now=NOW)
    assert consumed.user_id == "user_1"

    with pytest.raises(ReplayedLoginTicket):
        service.consume(secret.ticket, now=NOW)

    expired = service.create(user_id="user_1", ttl=timedelta(seconds=1), now=NOW)
    with pytest.raises(ExpiredLoginTicket):
        service.consume(expired.ticket, now=NOW + timedelta(seconds=2))


def test_report_job_creation_enforces_quota_and_enqueues_once() -> None:
    reports = InMemoryReportRepository()
    jobs = InMemoryJobRepository()
    queue = InMemoryTaskQueue()
    service = ReportJobService(reports, jobs, queue)

    report, job = service.create_report_job(
        user_id="user_1",
        form_id_hash="form_hash",
        form_title="Customer pulse",
        config_snapshot={"sections": ["overview"]},
        quota=Quota(monthly_report_limit=1, reports_used_this_month=0),
        now=NOW,
    )

    assert reports.get(report.id) == report
    assert jobs.get(job.id) == job
    assert queue.report_job_ids == [job.id]

    with pytest.raises(QuotaExceeded):
        service.create_report_job(
            user_id="user_1",
            form_id_hash="form_hash",
            form_title=None,
            config_snapshot={},
            quota=Quota(monthly_report_limit=1, reports_used_this_month=1),
            now=NOW,
        )


def test_job_state_machine_is_idempotent_for_running_and_succeeded_jobs() -> None:
    reports = InMemoryReportRepository()
    jobs = InMemoryJobRepository()
    queue = InMemoryTaskQueue()
    service = ReportJobService(reports, jobs, queue)
    _, job = service.create_report_job(
        user_id="user_1",
        form_id_hash="form_hash",
        form_title=None,
        config_snapshot={},
        quota=Quota(monthly_report_limit=-1),
        now=NOW,
    )
    machine = JobStateMachine(jobs)

    running = machine.start(job.id, now=NOW + timedelta(seconds=1))
    assert running.status == JobStatus.RUNNING
    assert running.attempts == 1
    assert machine.start(job.id, now=NOW + timedelta(seconds=2)) == running

    succeeded = machine.succeed(job.id, now=NOW + timedelta(seconds=3))
    assert succeeded.status == JobStatus.SUCCEEDED
    assert machine.succeed(job.id, now=NOW + timedelta(seconds=4)) == succeeded
    assert machine.start(job.id, now=NOW + timedelta(seconds=5)) == succeeded


def test_retryable_job_failure_can_be_started_again() -> None:
    reports = InMemoryReportRepository()
    jobs = InMemoryJobRepository()
    queue = InMemoryTaskQueue()
    service = ReportJobService(reports, jobs, queue)
    _, job = service.create_report_job(
        user_id="user_1",
        form_id_hash="form_hash",
        form_title=None,
        config_snapshot={},
        quota=Quota(monthly_report_limit=-1),
        now=NOW,
    )
    machine = JobStateMachine(jobs)

    machine.start(job.id, now=NOW)
    retrying = machine.fail(
        job.id,
        error_code="google_503",
        error_summary="temporary",
        retryable=True,
        now=NOW + timedelta(seconds=1),
    )
    assert retrying.status == JobStatus.RETRYING

    restarted = machine.start(job.id, now=NOW + timedelta(seconds=2))
    assert restarted.status == JobStatus.RUNNING
    assert restarted.attempts == 2


def test_fake_token_crypto_round_trips_without_exposing_plaintext() -> None:
    crypto = InMemoryTokenCrypto()
    ciphertext = crypto.encrypt("refresh-token")

    assert "refresh-token" not in ciphertext
    assert crypto.decrypt(ciphertext) == "refresh-token"


def test_production_settings_fail_closed_without_https_and_secrets() -> None:
    with pytest.raises(ValueError, match="APP_BASE_URL"):
        load_saas_settings(
            {
                "APP_ENV": "production",
                "APP_BASE_URL": "http://app.example.com",
                "API_BASE_URL": "https://api.example.com",
            }
        )

    with pytest.raises(ValueError, match="Missing production settings"):
        load_saas_settings(
            {
                "APP_ENV": "production",
                "APP_BASE_URL": "https://app.example.com",
                "API_BASE_URL": "https://api.example.com",
            }
        )


def test_production_settings_accept_complete_https_config() -> None:
    settings = load_saas_settings(
        {
            "APP_ENV": "production",
            "APP_BASE_URL": "https://app.example.com",
            "API_BASE_URL": "https://api.example.com",
            "GCP_PROJECT_ID": "project",
            "FIRESTORE_DATABASE": "(default)",
            "KMS_KEY_NAME": "kms",
            "GCS_BUCKET": "bucket",
            "CLOUD_TASKS_LOCATION": "europe-central2",
            "TASKS_QUEUE_NAME": "reports",
            "WORKER_BASE_URL": "https://worker.example.com",
            "CLOUD_TASKS_SERVICE_ACCOUNT_EMAIL": "tasks@example.iam.gserviceaccount.com",
            "GOOGLE_OAUTH_CLIENT_CONFIG_JSON": "{}",
            "SESSION_PEPPER": "pepper",
        }
    )

    assert settings.is_production


def test_production_settings_require_worker_and_cloud_tasks_oidc_config() -> None:
    with pytest.raises(ValueError, match="Missing production settings"):
        load_saas_settings(
            {
                "APP_ENV": "production",
                "APP_BASE_URL": "https://app.example.com",
                "API_BASE_URL": "https://api.example.com",
                "GCP_PROJECT_ID": "project",
                "FIRESTORE_DATABASE": "(default)",
                "KMS_KEY_NAME": "kms",
                "GCS_BUCKET": "bucket",
                "TASKS_QUEUE_NAME": "reports",
                "GOOGLE_OAUTH_CLIENT_CONFIG_JSON": "{}",
                "SESSION_PEPPER": "pepper",
            }
        )


def test_container_from_settings_uses_in_memory_outside_production() -> None:
    container = SaaSContainer.from_settings(
        load_saas_settings({"APP_ENV": "test", "SESSION_PEPPER": "pepper"})
    )

    assert isinstance(container.sessions, InMemorySessionRepository)


def test_raw_response_like_payloads_are_not_part_of_report_metadata() -> None:
    reports = InMemoryReportRepository()
    jobs = InMemoryJobRepository()
    queue = InMemoryTaskQueue()
    service = ReportJobService(reports, jobs, queue)

    report, _ = service.create_report_job(
        user_id="user_1",
        form_id_hash="form_hash",
        form_title=None,
        config_snapshot={"render_mode": "both"},
        quota=Quota(monthly_report_limit=-1),
        now=NOW,
    )

    forbidden = {"answers", "responses", "raw_payload"}
    assert forbidden.isdisjoint(report.config_snapshot)


def test_inmemory_session_repo_does_not_need_raw_secret_even_after_touch() -> None:
    repo = InMemorySessionRepository()
    service = SessionService(repo, pepper=PEPPER)
    secret = service.create(user_id="user_1", now=NOW)
    touched = service.validate(secret.session_id, now=NOW + timedelta(seconds=1))

    updated_repo = replace(repo.items[touched.session_hash], last_seen_at=touched.last_seen_at)
    assert updated_repo.session_hash == touched.session_hash
    assert secret.session_id not in repr(updated_repo)
