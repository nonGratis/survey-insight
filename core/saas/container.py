"""Composition root for SaaS services.

Production will swap the in-memory adapters for Firestore/KMS/GCS/Cloud Tasks
implementations. The rest of the API/worker code should depend on this object
instead of constructing infrastructure clients directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.saas.inmemory import (
    InMemoryArtifactStorage,
    InMemoryAuditLog,
    InMemoryJobRepository,
    InMemoryLoginTicketRepository,
    InMemoryOAuthStateRepository,
    InMemoryQuotaRepository,
    InMemoryReportRepository,
    InMemorySessionRepository,
    InMemoryTaskQueue,
    InMemoryTokenCrypto,
    InMemoryTokenRepository,
    InMemoryUserRepository,
)
from core.saas.ports import (
    ArtifactStorage,
    AuditLog,
    JobRepository,
    LoginTicketRepository,
    OAuthStateRepository,
    QuotaRepository,
    ReportRepository,
    SessionRepository,
    TaskQueue,
    TokenCrypto,
    TokenRepository,
    UserRepository,
)
from core.saas.services import (
    JobStateMachine,
    LoginTicketService,
    OAuthStateService,
    ReportJobService,
    SessionService,
)
from core.saas.settings import SaaSSettings, load_saas_settings


@dataclass(frozen=True)
class SaaSContainer:
    settings: SaaSSettings
    token_crypto: TokenCrypto
    users: UserRepository
    quotas: QuotaRepository
    oauth_states: OAuthStateRepository
    login_tickets: LoginTicketRepository
    sessions: SessionRepository
    tokens: TokenRepository
    reports: ReportRepository
    jobs: JobRepository
    tasks: TaskQueue
    artifacts: ArtifactStorage
    audit: AuditLog
    oauth_state_service: OAuthStateService
    login_ticket_service: LoginTicketService
    session_service: SessionService
    report_job_service: ReportJobService
    job_state_machine: JobStateMachine

    @classmethod
    def from_settings(cls, settings: SaaSSettings | None = None) -> SaaSContainer:
        resolved_settings = settings or load_saas_settings()
        if resolved_settings.is_production:
            return cls.production(resolved_settings)
        return cls.in_memory(resolved_settings)

    @classmethod
    def in_memory(cls, settings: SaaSSettings | None = None) -> SaaSContainer:
        resolved_settings = settings or load_saas_settings(
            {"APP_ENV": "test", "SESSION_PEPPER": "test-session-pepper"}
        )
        pepper = resolved_settings.session_pepper or "dev-session-pepper"

        users = InMemoryUserRepository()
        quotas = InMemoryQuotaRepository()
        oauth_states = InMemoryOAuthStateRepository()
        login_tickets = InMemoryLoginTicketRepository()
        sessions = InMemorySessionRepository()
        reports = InMemoryReportRepository()
        jobs = InMemoryJobRepository()
        tasks = InMemoryTaskQueue()

        return cls(
            settings=resolved_settings,
            token_crypto=InMemoryTokenCrypto(),
            users=users,
            quotas=quotas,
            oauth_states=oauth_states,
            login_tickets=login_tickets,
            sessions=sessions,
            tokens=InMemoryTokenRepository(),
            reports=reports,
            jobs=jobs,
            tasks=tasks,
            artifacts=InMemoryArtifactStorage(),
            audit=InMemoryAuditLog(),
            oauth_state_service=OAuthStateService(oauth_states, pepper=pepper),
            login_ticket_service=LoginTicketService(login_tickets, pepper=pepper),
            session_service=SessionService(sessions, pepper=pepper),
            report_job_service=ReportJobService(reports, jobs, tasks),
            job_state_machine=JobStateMachine(jobs),
        )

    @classmethod
    def production(cls, settings: SaaSSettings) -> SaaSContainer:
        from core.saas.adapters.firestore import (
            FirestoreAuditLog,
            FirestoreJobRepository,
            FirestoreLoginTicketRepository,
            FirestoreOAuthStateRepository,
            FirestoreQuotaRepository,
            FirestoreReportRepository,
            FirestoreSessionRepository,
            FirestoreTokenRepository,
            FirestoreUserRepository,
            create_firestore_client,
        )
        from core.saas.adapters.kms import KmsTokenCrypto
        from core.saas.adapters.storage import GcsArtifactStorage
        from core.saas.adapters.tasks import CloudTasksQueue

        settings.validate()
        firestore_client = create_firestore_client(
            settings.gcp_project_id,
            database=settings.firestore_database,
        )
        users = FirestoreUserRepository(firestore_client)
        quotas = FirestoreQuotaRepository(firestore_client)
        oauth_states = FirestoreOAuthStateRepository(firestore_client)
        login_tickets = FirestoreLoginTicketRepository(firestore_client)
        sessions = FirestoreSessionRepository(firestore_client)
        reports = FirestoreReportRepository(firestore_client)
        jobs = FirestoreJobRepository(firestore_client)
        tasks = CloudTasksQueue(
            project_id=settings.gcp_project_id,
            location=settings.cloud_tasks_location,
            queue_name=settings.tasks_queue_name,
            worker_base_url=settings.worker_base_url,
            service_account_email=settings.cloud_tasks_service_account_email,
        )

        return cls(
            settings=settings,
            token_crypto=KmsTokenCrypto(settings.kms_key_name),
            users=users,
            quotas=quotas,
            oauth_states=oauth_states,
            login_tickets=login_tickets,
            sessions=sessions,
            tokens=FirestoreTokenRepository(firestore_client),
            reports=reports,
            jobs=jobs,
            tasks=tasks,
            artifacts=GcsArtifactStorage(settings.gcs_bucket),
            audit=FirestoreAuditLog(firestore_client),
            oauth_state_service=OAuthStateService(oauth_states, pepper=settings.session_pepper),
            login_ticket_service=LoginTicketService(login_tickets, pepper=settings.session_pepper),
            session_service=SessionService(sessions, pepper=settings.session_pepper),
            report_job_service=ReportJobService(reports, jobs, tasks),
            job_state_machine=JobStateMachine(jobs),
        )
