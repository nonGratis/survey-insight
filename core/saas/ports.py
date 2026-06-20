"""Ports for external SaaS dependencies.

Production adapters will implement these interfaces with Firestore, Cloud KMS,
GCS, Cloud Tasks, and Google APIs. Tests and local development can use fakes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from google.oauth2.credentials import Credentials

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


class TokenCrypto(Protocol):
    def encrypt(self, plaintext: str) -> str: ...

    def decrypt(self, ciphertext: str) -> str: ...


class UserRepository(Protocol):
    def save(self, user: User) -> None: ...

    def get(self, user_id: str) -> User | None: ...


class QuotaRepository(Protocol):
    def save(self, user_id: str, quota: Quota) -> None: ...

    def get_for_user(self, user_id: str) -> Quota | None: ...


class OAuthStateRepository(Protocol):
    def save(self, state: OAuthState) -> None: ...

    def get(self, state_hash: str) -> OAuthState | None: ...

    def mark_used(self, state_hash: str, used_at: datetime) -> OAuthState: ...


class LoginTicketRepository(Protocol):
    def save(self, ticket: LoginTicket) -> None: ...

    def get(self, ticket_hash: str) -> LoginTicket | None: ...

    def mark_used(self, ticket_hash: str, used_at: datetime) -> LoginTicket: ...


class SessionRepository(Protocol):
    def save(self, session: Session) -> None: ...

    def get(self, session_hash: str) -> Session | None: ...

    def touch(self, session_hash: str, seen_at: datetime) -> Session: ...

    def revoke(self, session_hash: str, revoked_at: datetime) -> Session: ...


class TokenRepository(Protocol):
    def save(self, account: OAuthAccount) -> None: ...

    def get_by_user(self, user_id: str) -> OAuthAccount | None: ...


class ReportRepository(Protocol):
    def save(self, report: Report) -> None: ...

    def get(self, report_id: str) -> Report | None: ...

    def update(self, report: Report) -> None: ...


class JobRepository(Protocol):
    def save(self, job: ReportJob) -> None: ...

    def get(self, job_id: str) -> ReportJob | None: ...

    def transition(self, job_id: str, allowed: set[JobStatus], target: ReportJob) -> ReportJob: ...


class ArtifactStorage(Protocol):
    def save_pdf(self, report_id: str, user_id: str, content: bytes) -> Artifact: ...

    def signed_download_url(self, artifact: Artifact) -> str: ...


class TaskQueue(Protocol):
    def enqueue_report_job(self, job_id: str) -> None: ...


class GoogleFormsClient(Protocol):
    def list_forms(self, creds: Credentials) -> Sequence[Mapping[str, object]]: ...

    def get_form_structure(self, creds: Credentials, form_id: str) -> Mapping[str, object]: ...

    def list_responses(
        self, creds: Credentials, form_id: str
    ) -> Sequence[Mapping[str, object]]: ...


class AuditLog(Protocol):
    def record(self, event: AuditEvent) -> None: ...
