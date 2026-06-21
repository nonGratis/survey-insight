"""In-memory adapters for SaaS service tests and local development."""

from __future__ import annotations

import base64
from dataclasses import replace
from datetime import datetime, timedelta

from core.saas.errors import JobConflict
from core.saas.models import (
    Artifact,
    AuditEvent,
    JobStatus,
    LoginTicket,
    OAuthAccount,
    OAuthState,
    Quota,
    Report,
    ReportJob,
    Session,
    User,
)
from core.saas.security import utcnow


class InMemoryTokenCrypto:
    """Non-secure fake crypto for tests. Production must use Cloud KMS."""

    prefix = "fake-kms:"

    def encrypt(self, plaintext: str) -> str:
        return self.prefix + base64.urlsafe_b64encode(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, ciphertext: str) -> str:
        if not ciphertext.startswith(self.prefix):
            raise ValueError("Invalid fake ciphertext.")
        encoded = ciphertext.removeprefix(self.prefix)
        return base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")


class InMemoryUserRepository:
    def __init__(self) -> None:
        self.items: dict[str, User] = {}

    def save(self, user: User) -> None:
        self.items[user.id] = user

    def get(self, user_id: str) -> User | None:
        return self.items.get(user_id)


class InMemoryQuotaRepository:
    def __init__(self) -> None:
        self.items: dict[str, Quota] = {}

    def save(self, user_id: str, quota: Quota) -> None:
        self.items[user_id] = quota

    def get_for_user(self, user_id: str) -> Quota | None:
        return self.items.get(user_id)


class InMemoryOAuthStateRepository:
    def __init__(self) -> None:
        self.items: dict[str, OAuthState] = {}

    def save(self, state: OAuthState) -> None:
        self.items[state.state_hash] = state

    def get(self, state_hash: str) -> OAuthState | None:
        return self.items.get(state_hash)

    def mark_used(self, state_hash: str, used_at: datetime) -> OAuthState:
        state = self.items[state_hash]
        updated = replace(state, used_at=used_at)
        self.items[state_hash] = updated
        return updated


class InMemoryLoginTicketRepository:
    def __init__(self) -> None:
        self.items: dict[str, LoginTicket] = {}

    def save(self, ticket: LoginTicket) -> None:
        self.items[ticket.ticket_hash] = ticket

    def get(self, ticket_hash: str) -> LoginTicket | None:
        return self.items.get(ticket_hash)

    def mark_used(self, ticket_hash: str, used_at: datetime) -> LoginTicket:
        ticket = self.items[ticket_hash]
        updated = replace(ticket, used_at=used_at)
        self.items[ticket_hash] = updated
        return updated


class InMemorySessionRepository:
    def __init__(self) -> None:
        self.items: dict[str, Session] = {}

    def save(self, session: Session) -> None:
        self.items[session.session_hash] = session

    def get(self, session_hash: str) -> Session | None:
        return self.items.get(session_hash)

    def touch(self, session_hash: str, seen_at: datetime) -> Session:
        session = self.items[session_hash]
        updated = replace(session, last_seen_at=seen_at)
        self.items[session_hash] = updated
        return updated

    def revoke(self, session_hash: str, revoked_at: datetime) -> Session:
        session = self.items[session_hash]
        updated = replace(session, revoked_at=revoked_at)
        self.items[session_hash] = updated
        return updated


class InMemoryTokenRepository:
    def __init__(self) -> None:
        self.by_user: dict[str, OAuthAccount] = {}

    def save(self, account: OAuthAccount) -> None:
        self.by_user[account.user_id] = account

    def get_by_user(self, user_id: str) -> OAuthAccount | None:
        return self.by_user.get(user_id)


class InMemoryReportRepository:
    def __init__(self) -> None:
        self.items: dict[str, Report] = {}

    def save(self, report: Report) -> None:
        self.items[report.id] = report

    def get(self, report_id: str) -> Report | None:
        return self.items.get(report_id)

    def update(self, report: Report) -> None:
        self.items[report.id] = report


class InMemoryJobRepository:
    def __init__(self) -> None:
        self.items: dict[str, ReportJob] = {}

    def save(self, job: ReportJob) -> None:
        self.items[job.id] = job

    def get(self, job_id: str) -> ReportJob | None:
        return self.items.get(job_id)

    def transition(self, job_id: str, allowed: set[JobStatus], target: ReportJob) -> ReportJob:
        current = self.items[job_id]
        if current.status not in allowed:
            raise JobConflict(f"Cannot move job {job_id} from {current.status} to {target.status}.")
        self.items[job_id] = target
        return target


class InMemoryTaskQueue:
    def __init__(self) -> None:
        self.report_job_ids: list[str] = []

    def enqueue_report_job(self, job_id: str) -> None:
        self.report_job_ids.append(job_id)


class InMemoryArtifactStorage:
    def __init__(self) -> None:
        self.pdfs: dict[str, bytes] = {}

    def save_pdf(self, report_id: str, user_id: str, content: bytes) -> Artifact:
        artifact = Artifact(
            id=f"artifact_{report_id}",
            report_id=report_id,
            user_id=user_id,
            gcs_uri=f"memory://reports/{report_id}.pdf",
            content_type="application/pdf",
            created_at=utcnow(),
            expires_at=utcnow() + timedelta(days=30),
        )
        self.pdfs[artifact.gcs_uri] = content
        return artifact

    def signed_download_url(self, artifact: Artifact) -> str:
        return f"https://download.local/{artifact.id}"


class InMemoryAuditLog:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)
