"""Pure SaaS domain services for auth/session/job state machines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from uuid import uuid4

from core.saas.errors import (
    ExpiredLoginTicket,
    ExpiredOAuthState,
    InvalidLoginTicket,
    InvalidOAuthState,
    InvalidSession,
    JobConflict,
    MissingRequiredScopes,
    QuotaExceeded,
    ReplayedLoginTicket,
    ReplayedOAuthState,
)
from core.saas.models import (
    JobStatus,
    LoginTicket,
    OAuthState,
    Quota,
    Report,
    ReportJob,
    ReportStatus,
    Session,
)
from core.saas.ports import (
    JobRepository,
    LoginTicketRepository,
    OAuthStateRepository,
    ReportRepository,
    SessionRepository,
    TaskQueue,
)
from core.saas.security import generate_opaque_token, hash_secret, is_expired, utcnow


@dataclass(frozen=True)
class OAuthStateSecret:
    state: str
    record: OAuthState


@dataclass(frozen=True)
class LoginTicketSecret:
    ticket: str
    record: LoginTicket


@dataclass(frozen=True)
class SessionSecret:
    session_id: str
    record: Session


class OAuthStateService:
    def __init__(self, repo: OAuthStateRepository, *, pepper: str) -> None:
        self.repo = repo
        self.pepper = pepper

    def create(
        self,
        *,
        scopes: Sequence[str],
        next_url: str,
        ttl: timedelta = timedelta(minutes=10),
        now: datetime | None = None,
    ) -> OAuthStateSecret:
        current = now or utcnow()
        raw_state = generate_opaque_token()
        record = OAuthState(
            state_hash=hash_secret(raw_state, self.pepper),
            code_verifier=generate_opaque_token(64),
            scopes=tuple(scopes),
            next_url=next_url,
            created_at=current,
            expires_at=current + ttl,
        )
        self.repo.save(record)
        return OAuthStateSecret(state=raw_state, record=record)

    def consume(
        self,
        raw_state: str,
        *,
        now: datetime | None = None,
        required_scopes: Sequence[str] = (),
    ) -> OAuthState:
        current = now or utcnow()
        state_hash = hash_secret(raw_state, self.pepper)
        record = self.repo.get(state_hash)
        if record is None:
            raise InvalidOAuthState("OAuth state is unknown.")
        if record.used_at is not None:
            raise ReplayedOAuthState("OAuth state was already used.")
        if is_expired(record.expires_at, now=current):
            raise ExpiredOAuthState("OAuth state expired.")
        missing = set(required_scopes) - set(record.scopes)
        if missing:
            raise MissingRequiredScopes(f"OAuth state is missing scopes: {sorted(missing)}")
        return self.repo.mark_used(state_hash, current)


class LoginTicketService:
    def __init__(self, repo: LoginTicketRepository, *, pepper: str) -> None:
        self.repo = repo
        self.pepper = pepper

    def create(
        self,
        *,
        user_id: str,
        ttl: timedelta = timedelta(minutes=5),
        now: datetime | None = None,
    ) -> LoginTicketSecret:
        current = now or utcnow()
        raw_ticket = generate_opaque_token()
        record = LoginTicket(
            ticket_hash=hash_secret(raw_ticket, self.pepper),
            user_id=user_id,
            created_at=current,
            expires_at=current + ttl,
        )
        self.repo.save(record)
        return LoginTicketSecret(ticket=raw_ticket, record=record)

    def consume(self, raw_ticket: str, *, now: datetime | None = None) -> LoginTicket:
        current = now or utcnow()
        ticket_hash = hash_secret(raw_ticket, self.pepper)
        record = self.repo.get(ticket_hash)
        if record is None:
            raise InvalidLoginTicket("Login ticket is unknown.")
        if record.used_at is not None:
            raise ReplayedLoginTicket("Login ticket was already used.")
        if is_expired(record.expires_at, now=current):
            raise ExpiredLoginTicket("Login ticket expired.")
        return self.repo.mark_used(ticket_hash, current)


class SessionService:
    def __init__(self, repo: SessionRepository, *, pepper: str) -> None:
        self.repo = repo
        self.pepper = pepper

    def create(
        self,
        *,
        user_id: str,
        ttl: timedelta = timedelta(days=30),
        now: datetime | None = None,
    ) -> SessionSecret:
        current = now or utcnow()
        raw_session_id = generate_opaque_token()
        record = Session(
            session_hash=hash_secret(raw_session_id, self.pepper),
            user_id=user_id,
            created_at=current,
            expires_at=current + ttl,
            last_seen_at=current,
        )
        self.repo.save(record)
        return SessionSecret(session_id=raw_session_id, record=record)

    def validate(self, raw_session_id: str, *, now: datetime | None = None) -> Session:
        current = now or utcnow()
        session_hash = hash_secret(raw_session_id, self.pepper)
        record = self.repo.get(session_hash)
        if record is None or record.revoked_at is not None:
            raise InvalidSession("Session is invalid.")
        if is_expired(record.expires_at, now=current):
            raise InvalidSession("Session expired.")
        return self.repo.touch(session_hash, current)

    def revoke(self, raw_session_id: str, *, now: datetime | None = None) -> Session:
        current = now or utcnow()
        session_hash = hash_secret(raw_session_id, self.pepper)
        record = self.repo.get(session_hash)
        if record is None:
            raise InvalidSession("Session is invalid.")
        return self.repo.revoke(session_hash, current)


class ReportJobService:
    def __init__(
        self,
        reports: ReportRepository,
        jobs: JobRepository,
        queue: TaskQueue,
    ) -> None:
        self.reports = reports
        self.jobs = jobs
        self.queue = queue

    def create_report_job(
        self,
        *,
        user_id: str,
        form_id_hash: str,
        form_title: str | None,
        config_snapshot: Mapping[str, object],
        quota: Quota,
        now: datetime | None = None,
    ) -> tuple[Report, ReportJob]:
        if not quota.can_create_report():
            raise QuotaExceeded("Monthly report quota exceeded.")
        current = now or utcnow()
        report = Report(
            id=f"report_{uuid4().hex}",
            user_id=user_id,
            form_id_hash=form_id_hash,
            form_title=form_title,
            config_snapshot=dict(config_snapshot),
            status=ReportStatus.QUEUED,
            created_at=current,
            updated_at=current,
        )
        job = ReportJob(
            id=f"job_{uuid4().hex}",
            report_id=report.id,
            user_id=user_id,
            status=JobStatus.QUEUED,
            created_at=current,
            updated_at=current,
        )
        self.reports.save(report)
        self.jobs.save(job)
        self.queue.enqueue_report_job(job.id)
        return report, job


class JobStateMachine:
    def __init__(self, jobs: JobRepository) -> None:
        self.jobs = jobs

    def start(self, job_id: str, *, now: datetime | None = None) -> ReportJob:
        current = now or utcnow()
        job = self._get(job_id)
        if job.status in {JobStatus.RUNNING, JobStatus.SUCCEEDED}:
            return job
        if job.status not in {JobStatus.QUEUED, JobStatus.RETRYING, JobStatus.FAILED}:
            raise JobConflict(f"Cannot start job in state {job.status}.")
        target = replace(
            job,
            status=JobStatus.RUNNING,
            attempts=job.attempts + 1,
            updated_at=current,
            error_code=None,
            error_summary=None,
        )
        return self.jobs.transition(
            job_id,
            {JobStatus.QUEUED, JobStatus.RETRYING, JobStatus.FAILED},
            target,
        )

    def succeed(self, job_id: str, *, now: datetime | None = None) -> ReportJob:
        current = now or utcnow()
        job = self._get(job_id)
        if job.status == JobStatus.SUCCEEDED:
            return job
        target = replace(job, status=JobStatus.SUCCEEDED, updated_at=current)
        return self.jobs.transition(job_id, {JobStatus.RUNNING}, target)

    def fail(
        self,
        job_id: str,
        *,
        error_code: str,
        error_summary: str,
        retryable: bool,
        now: datetime | None = None,
    ) -> ReportJob:
        current = now or utcnow()
        job = self._get(job_id)
        target_status = JobStatus.RETRYING if retryable else JobStatus.FAILED
        target = replace(
            job,
            status=target_status,
            updated_at=current,
            error_code=error_code,
            error_summary=error_summary,
        )
        return self.jobs.transition(job_id, {JobStatus.RUNNING}, target)

    def _get(self, job_id: str) -> ReportJob:
        job = self.jobs.get(job_id)
        if job is None:
            raise JobConflict(f"Unknown job: {job_id}.")
        return job
