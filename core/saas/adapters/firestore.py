"""Firestore-backed SaaS repositories.

These adapters persist only SaaS metadata: users, sessions, OAuth states,
encrypted token records, report metadata, jobs, quotas, and audit events.
Raw Google Forms responses are intentionally not represented here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from typing import Any, cast

from google.cloud import firestore

from core.saas.errors import JobConflict
from core.saas.models import (
    AuditEvent,
    JobStatus,
    LoginTicket,
    OAuthAccount,
    OAuthState,
    Plan,
    Quota,
    Report,
    ReportJob,
    ReportStatus,
    Session,
    User,
    UserStatus,
)

USERS = "users"
QUOTAS = "quotas"
OAUTH_STATES = "oauth_states"
LOGIN_TICKETS = "login_tickets"
SESSIONS = "sessions"
OAUTH_ACCOUNTS = "oauth_accounts"
REPORTS = "reports"
JOBS = "jobs"
AUDIT_EVENTS = "audit_events"


class FirestoreUserRepository:
    def __init__(self, client: firestore.Client) -> None:
        self.client = client

    def save(self, user: User) -> None:
        self.client.collection(USERS).document(user.id).set(_user_to_doc(user))

    def get(self, user_id: str) -> User | None:
        data = _get_doc(self.client.collection(USERS).document(user_id))
        return _user_from_doc(data) if data else None


class FirestoreQuotaRepository:
    def __init__(self, client: firestore.Client) -> None:
        self.client = client

    def save(self, user_id: str, quota: Quota) -> None:
        self.client.collection(QUOTAS).document(user_id).set(_quota_to_doc(quota))

    def get_for_user(self, user_id: str) -> Quota | None:
        data = _get_doc(self.client.collection(QUOTAS).document(user_id))
        return _quota_from_doc(data) if data else None


class FirestoreOAuthStateRepository:
    def __init__(self, client: firestore.Client) -> None:
        self.client = client

    def save(self, state: OAuthState) -> None:
        self.client.collection(OAUTH_STATES).document(state.state_hash).set(
            _oauth_state_to_doc(state)
        )

    def get(self, state_hash: str) -> OAuthState | None:
        data = _get_doc(self.client.collection(OAUTH_STATES).document(state_hash))
        return _oauth_state_from_doc(data) if data else None

    def mark_used(self, state_hash: str, used_at: datetime) -> OAuthState:
        ref = self.client.collection(OAUTH_STATES).document(state_hash)
        data = _get_doc(ref)
        if data is None:
            raise KeyError(state_hash)
        updated = _oauth_state_from_doc({**data, "used_at": used_at})
        ref.set(_oauth_state_to_doc(updated))
        return updated


class FirestoreLoginTicketRepository:
    def __init__(self, client: firestore.Client) -> None:
        self.client = client

    def save(self, ticket: LoginTicket) -> None:
        self.client.collection(LOGIN_TICKETS).document(ticket.ticket_hash).set(
            _login_ticket_to_doc(ticket)
        )

    def get(self, ticket_hash: str) -> LoginTicket | None:
        data = _get_doc(self.client.collection(LOGIN_TICKETS).document(ticket_hash))
        return _login_ticket_from_doc(data) if data else None

    def mark_used(self, ticket_hash: str, used_at: datetime) -> LoginTicket:
        ref = self.client.collection(LOGIN_TICKETS).document(ticket_hash)
        data = _get_doc(ref)
        if data is None:
            raise KeyError(ticket_hash)
        updated = _login_ticket_from_doc({**data, "used_at": used_at})
        ref.set(_login_ticket_to_doc(updated))
        return updated


class FirestoreSessionRepository:
    def __init__(self, client: firestore.Client) -> None:
        self.client = client

    def save(self, session: Session) -> None:
        self.client.collection(SESSIONS).document(session.session_hash).set(
            _session_to_doc(session)
        )

    def get(self, session_hash: str) -> Session | None:
        data = _get_doc(self.client.collection(SESSIONS).document(session_hash))
        return _session_from_doc(data) if data else None

    def touch(self, session_hash: str, seen_at: datetime) -> Session:
        ref = self.client.collection(SESSIONS).document(session_hash)
        data = _get_doc(ref)
        if data is None:
            raise KeyError(session_hash)
        updated = _session_from_doc({**data, "last_seen_at": seen_at})
        ref.set(_session_to_doc(updated))
        return updated

    def revoke(self, session_hash: str, revoked_at: datetime) -> Session:
        ref = self.client.collection(SESSIONS).document(session_hash)
        data = _get_doc(ref)
        if data is None:
            raise KeyError(session_hash)
        updated = _session_from_doc({**data, "revoked_at": revoked_at})
        ref.set(_session_to_doc(updated))
        return updated


class FirestoreTokenRepository:
    def __init__(self, client: firestore.Client) -> None:
        self.client = client

    def save(self, account: OAuthAccount) -> None:
        self.client.collection(OAUTH_ACCOUNTS).document(account.user_id).set(
            _oauth_account_to_doc(account)
        )

    def get_by_user(self, user_id: str) -> OAuthAccount | None:
        data = _get_doc(self.client.collection(OAUTH_ACCOUNTS).document(user_id))
        return _oauth_account_from_doc(data) if data else None


class FirestoreReportRepository:
    def __init__(self, client: firestore.Client) -> None:
        self.client = client

    def save(self, report: Report) -> None:
        self.client.collection(REPORTS).document(report.id).set(_report_to_doc(report))

    def get(self, report_id: str) -> Report | None:
        data = _get_doc(self.client.collection(REPORTS).document(report_id))
        return _report_from_doc(data) if data else None

    def update(self, report: Report) -> None:
        self.save(report)


class FirestoreJobRepository:
    def __init__(self, client: firestore.Client) -> None:
        self.client = client

    def save(self, job: ReportJob) -> None:
        self.client.collection(JOBS).document(job.id).set(_job_to_doc(job))

    def get(self, job_id: str) -> ReportJob | None:
        data = _get_doc(self.client.collection(JOBS).document(job_id))
        return _job_from_doc(data) if data else None

    def transition(self, job_id: str, allowed: set[JobStatus], target: ReportJob) -> ReportJob:
        ref = self.client.collection(JOBS).document(job_id)
        transaction = self.client.transaction()

        @firestore.transactional
        def _transition(current_transaction: firestore.Transaction) -> ReportJob:
            snapshot = cast(Any, ref.get(transaction=current_transaction))
            if not snapshot.exists:
                raise JobConflict(f"Unknown job: {job_id}.")
            current = _job_from_doc(snapshot.to_dict() or {})
            if current.status not in allowed:
                raise JobConflict(
                    f"Cannot move job {job_id} from {current.status} to {target.status}."
                )
            current_transaction.set(ref, _job_to_doc(target))
            return target

        return _transition(transaction)


class FirestoreAuditLog:
    def __init__(self, client: firestore.Client) -> None:
        self.client = client

    def record(self, event: AuditEvent) -> None:
        self.client.collection(AUDIT_EVENTS).document().set(_audit_event_to_doc(event))


def create_firestore_client(project_id: str, database: str = "(default)") -> firestore.Client:
    return firestore.Client(project=project_id, database=database)


def _get_doc(ref: Any) -> dict[str, Any] | None:
    snapshot = ref.get()
    if not snapshot.exists:
        return None
    return dict(snapshot.to_dict() or {})


def _base_doc(model: Any) -> dict[str, Any]:
    data = asdict(model)
    return {key: _serialise_value(value) for key, value in data.items()}


def _serialise_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        return {key: _serialise_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialise_value(item) for item in value]
    return value


def _required_datetime(data: Mapping[str, Any], key: str) -> datetime:
    value = data[key]
    if not isinstance(value, datetime):
        raise TypeError(f"{key} must be datetime, got {type(value).__name__}.")
    return value


def _optional_datetime(data: Mapping[str, Any], key: str) -> datetime | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise TypeError(f"{key} must be datetime, got {type(value).__name__}.")
    return value


def _user_to_doc(user: User) -> dict[str, Any]:
    data = _base_doc(user)
    data["plan"] = user.plan.value
    data["status"] = user.status.value
    return data


def _user_from_doc(data: Mapping[str, Any]) -> User:
    return User(
        id=str(data["id"]),
        google_sub=str(data["google_sub"]),
        email=str(data["email"]),
        name=str(data["name"]),
        picture=data.get("picture"),
        plan=Plan(str(data["plan"])),
        status=UserStatus(str(data["status"])),
        created_at=_required_datetime(data, "created_at"),
        last_seen_at=_optional_datetime(data, "last_seen_at"),
    )


def _quota_to_doc(quota: Quota) -> dict[str, Any]:
    return _base_doc(quota)


def _quota_from_doc(data: Mapping[str, Any]) -> Quota:
    return Quota(
        monthly_report_limit=int(data["monthly_report_limit"]),
        reports_used_this_month=int(data.get("reports_used_this_month", 0)),
    )


def _oauth_state_to_doc(state: OAuthState) -> dict[str, Any]:
    return _base_doc(state)


def _oauth_state_from_doc(data: Mapping[str, Any]) -> OAuthState:
    return OAuthState(
        state_hash=str(data["state_hash"]),
        code_verifier=str(data["code_verifier"]),
        scopes=tuple(str(scope) for scope in data.get("scopes", [])),
        next_url=str(data["next_url"]),
        created_at=_required_datetime(data, "created_at"),
        expires_at=_required_datetime(data, "expires_at"),
        used_at=_optional_datetime(data, "used_at"),
    )


def _login_ticket_to_doc(ticket: LoginTicket) -> dict[str, Any]:
    return _base_doc(ticket)


def _login_ticket_from_doc(data: Mapping[str, Any]) -> LoginTicket:
    return LoginTicket(
        ticket_hash=str(data["ticket_hash"]),
        user_id=str(data["user_id"]),
        created_at=_required_datetime(data, "created_at"),
        expires_at=_required_datetime(data, "expires_at"),
        used_at=_optional_datetime(data, "used_at"),
    )


def _session_to_doc(session: Session) -> dict[str, Any]:
    return _base_doc(session)


def _session_from_doc(data: Mapping[str, Any]) -> Session:
    return Session(
        session_hash=str(data["session_hash"]),
        user_id=str(data["user_id"]),
        created_at=_required_datetime(data, "created_at"),
        expires_at=_required_datetime(data, "expires_at"),
        last_seen_at=_required_datetime(data, "last_seen_at"),
        revoked_at=_optional_datetime(data, "revoked_at"),
        user_agent_hash=data.get("user_agent_hash"),
        ip_prefix_hash=data.get("ip_prefix_hash"),
    )


def _oauth_account_to_doc(account: OAuthAccount) -> dict[str, Any]:
    return _base_doc(account)


def _oauth_account_from_doc(data: Mapping[str, Any]) -> OAuthAccount:
    return OAuthAccount(
        user_id=str(data["user_id"]),
        provider=str(data["provider"]),
        google_sub=str(data["google_sub"]),
        email=str(data["email"]),
        scopes=tuple(str(scope) for scope in data.get("scopes", [])),
        encrypted_access_token=data.get("encrypted_access_token"),
        encrypted_refresh_token=data.get("encrypted_refresh_token"),
        token_expiry=_optional_datetime(data, "token_expiry"),
        updated_at=_required_datetime(data, "updated_at"),
    )


def _report_to_doc(report: Report) -> dict[str, Any]:
    data = _base_doc(report)
    data["status"] = report.status.value
    return data


def _report_from_doc(data: Mapping[str, Any]) -> Report:
    return Report(
        id=str(data["id"]),
        user_id=str(data["user_id"]),
        form_id_hash=str(data["form_id_hash"]),
        form_title=data.get("form_title"),
        config_snapshot=dict(data.get("config_snapshot", {})),
        status=ReportStatus(str(data["status"])),
        created_at=_required_datetime(data, "created_at"),
        updated_at=_required_datetime(data, "updated_at"),
        artifact_uri=data.get("artifact_uri"),
        error_code=data.get("error_code"),
        error_summary=data.get("error_summary"),
    )


def _job_to_doc(job: ReportJob) -> dict[str, Any]:
    data = _base_doc(job)
    data["status"] = job.status.value
    return data


def _job_from_doc(data: Mapping[str, Any]) -> ReportJob:
    return ReportJob(
        id=str(data["id"]),
        report_id=str(data["report_id"]),
        user_id=str(data["user_id"]),
        status=JobStatus(str(data["status"])),
        created_at=_required_datetime(data, "created_at"),
        updated_at=_required_datetime(data, "updated_at"),
        attempts=int(data.get("attempts", 0)),
        error_code=data.get("error_code"),
        error_summary=data.get("error_summary"),
    )


def _audit_event_to_doc(event: AuditEvent) -> dict[str, Any]:
    return _base_doc(event)
