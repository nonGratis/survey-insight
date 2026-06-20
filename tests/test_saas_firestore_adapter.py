from __future__ import annotations

from datetime import UTC, datetime, timedelta

from core.saas.adapters.firestore import (
    _job_from_doc,
    _job_to_doc,
    _oauth_account_from_doc,
    _oauth_account_to_doc,
    _report_from_doc,
    _report_to_doc,
    _session_from_doc,
    _session_to_doc,
    _user_from_doc,
    _user_to_doc,
)
from core.saas.models import (
    JobStatus,
    OAuthAccount,
    Plan,
    Report,
    ReportJob,
    ReportStatus,
    Session,
    User,
    UserStatus,
)

NOW = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)


def test_firestore_user_round_trip_preserves_enums_and_datetimes() -> None:
    user = User(
        id="user_1",
        google_sub="sub",
        email="owner@example.com",
        name="Owner",
        picture=None,
        plan=Plan.PILOT,
        status=UserStatus.ACTIVE,
        created_at=NOW,
        last_seen_at=NOW + timedelta(minutes=1),
    )

    doc = _user_to_doc(user)

    assert doc["plan"] == "pilot"
    assert doc["status"] == "active"
    assert _user_from_doc(doc) == user


def test_firestore_session_round_trip_never_needs_raw_session_id() -> None:
    session = Session(
        session_hash="hmac-sha256:abc",
        user_id="user_1",
        created_at=NOW,
        expires_at=NOW + timedelta(days=30),
        last_seen_at=NOW,
    )

    doc = _session_to_doc(session)

    assert "session_id" not in doc
    assert _session_from_doc(doc) == session


def test_firestore_oauth_account_round_trip_keeps_only_encrypted_tokens() -> None:
    account = OAuthAccount(
        user_id="user_1",
        provider="google",
        google_sub="sub",
        email="owner@example.com",
        scopes=("openid", "forms.responses.readonly"),
        encrypted_access_token="kms:access",
        encrypted_refresh_token="kms:refresh",
        token_expiry=NOW + timedelta(hours=1),
        updated_at=NOW,
    )

    doc = _oauth_account_to_doc(account)

    assert doc["scopes"] == ["openid", "forms.responses.readonly"]
    assert "refresh-token" not in repr(doc)
    assert _oauth_account_from_doc(doc) == account


def test_firestore_report_round_trip_keeps_config_snapshot_not_raw_payload() -> None:
    report = Report(
        id="report_1",
        user_id="user_1",
        form_id_hash="form_hash",
        form_title="Pulse",
        config_snapshot={"sections": ["overview"]},
        status=ReportStatus.QUEUED,
        created_at=NOW,
        updated_at=NOW,
    )

    doc = _report_to_doc(report)

    assert doc["status"] == "queued"
    assert "responses" not in doc["config_snapshot"]
    assert _report_from_doc(doc) == report


def test_firestore_job_round_trip_preserves_state_machine_status() -> None:
    job = ReportJob(
        id="job_1",
        report_id="report_1",
        user_id="user_1",
        status=JobStatus.RETRYING,
        created_at=NOW,
        updated_at=NOW + timedelta(seconds=5),
        attempts=2,
        error_code="google_503",
        error_summary="temporary",
    )

    doc = _job_to_doc(job)

    assert doc["status"] == "retrying"
    assert _job_from_doc(doc) == job
