"""Typed SaaS domain models.

These dataclasses intentionally keep external-provider payloads out of the
domain surface. Tokens are stored only as encrypted strings, raw survey
responses are not represented here, and state-machine timestamps are explicit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class Plan(StrEnum):
    FREE = "free"
    PILOT = "pilot"
    PRO = "pro"
    ADMIN = "admin"


class UserStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"


class ReportStatus(StrEnum):
    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DELETED = "deleted"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass(frozen=True)
class Quota:
    monthly_report_limit: int
    reports_used_this_month: int = 0

    def can_create_report(self) -> bool:
        return self.monthly_report_limit < 0 or (
            self.reports_used_this_month < self.monthly_report_limit
        )


@dataclass(frozen=True)
class User:
    id: str
    google_sub: str
    email: str
    name: str
    picture: str | None
    plan: Plan
    status: UserStatus
    created_at: datetime
    last_seen_at: datetime | None = None


@dataclass(frozen=True)
class OAuthAccount:
    user_id: str
    provider: str
    google_sub: str
    email: str
    scopes: tuple[str, ...]
    encrypted_access_token: str | None
    encrypted_refresh_token: str | None
    token_expiry: datetime | None
    updated_at: datetime


@dataclass(frozen=True)
class OAuthState:
    state_hash: str
    code_verifier: str
    scopes: tuple[str, ...]
    next_url: str
    created_at: datetime
    expires_at: datetime
    used_at: datetime | None = None


@dataclass(frozen=True)
class LoginTicket:
    ticket_hash: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    used_at: datetime | None = None


@dataclass(frozen=True)
class Session:
    session_hash: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    last_seen_at: datetime
    revoked_at: datetime | None = None
    user_agent_hash: str | None = None
    ip_prefix_hash: str | None = None


@dataclass(frozen=True)
class Report:
    id: str
    user_id: str
    form_id_hash: str
    form_title: str | None
    config_snapshot: Mapping[str, object]
    status: ReportStatus
    created_at: datetime
    updated_at: datetime
    artifact_uri: str | None = None
    error_code: str | None = None
    error_summary: str | None = None


@dataclass(frozen=True)
class ReportJob:
    id: str
    report_id: str
    user_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    attempts: int = 0
    error_code: str | None = None
    error_summary: str | None = None


@dataclass(frozen=True)
class Artifact:
    id: str
    report_id: str
    user_id: str
    gcs_uri: str
    content_type: str
    created_at: datetime
    expires_at: datetime | None = None


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    user_id: str | None
    created_at: datetime
    job_id: str | None = None
    report_id: str | None = None
    form_id_hash: str | None = None
    duration_ms: int | None = None
    status: str | None = None
    error_code: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
