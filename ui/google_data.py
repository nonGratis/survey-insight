"""Google data facade for Streamlit pages.

Production uses the SaaS API. Local/demo mode keeps the previous direct Google
credential path for developer convenience.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import streamlit as st

from core.auth import credentials_from_dict
from core.context_tables import ContextTable, scan_sheets_for_tables
from core.forms_api import (
    get_form_structure as local_get_form_structure,
)
from core.forms_api import (
    list_form_responses as local_list_form_responses,
)
from core.forms_api import (
    list_response_timestamps as local_list_response_timestamps,
)
from core.forms_api import (
    list_user_forms as local_list_user_forms,
)
from core.forms_catalog import (
    FormDriveMeta,
    FormEnrichment,
    ResponseStats,
)
from core.forms_catalog import (
    enrich_form as local_enrich_form,
)
from core.forms_catalog import (
    list_forms_with_drive_meta as local_list_catalog_forms,
)
from core.sheets_api import fetch_all_grids as local_fetch_all_grids
from ui.saas_api import SaaSApiClient


def is_saas_mode() -> bool:
    return os.environ.get("APP_ENV") == "production" and bool(os.environ.get("API_BASE_URL"))


def cache_token() -> str:
    if is_saas_mode():
        session_id = st.session_state.get("saas_session_id")
        return session_id if isinstance(session_id, str) else ""
    creds = _local_credentials()
    return creds.token or ""


def list_forms_for_picker() -> list[dict[str, Any]]:
    if is_saas_mode():
        return _client().list_forms(_session_id())
    return local_list_user_forms(_local_credentials())


def list_catalog_forms() -> list[FormDriveMeta]:
    if is_saas_mode():
        return [FormDriveMeta(**item) for item in _client().list_forms(_session_id())]
    return local_list_catalog_forms(_local_credentials())


def get_form_summary(form_id: str) -> FormEnrichment:
    if is_saas_mode():
        return FormEnrichment(**_client().get_form_summary(_session_id(), form_id))
    return local_enrich_form(_local_credentials(), form_id)


def get_response_stats(form_id: str) -> ResponseStats:
    if is_saas_mode():
        return ResponseStats(**_client().get_response_stats(_session_id(), form_id))
    timestamps = local_list_response_timestamps(_local_credentials(), form_id)
    return ResponseStats(
        total=len(timestamps),
        first_response=_format_timestamp(timestamps[0]) if timestamps else None,
        second_response=_format_timestamp(timestamps[1]) if len(timestamps) >= 2 else None,
        last_response=_format_timestamp(timestamps[-1]) if timestamps else None,
    )


def get_form_structure(form_id: str) -> dict[str, Any]:
    if is_saas_mode():
        return _client().get_form_structure(_session_id(), form_id)
    return local_get_form_structure(_local_credentials(), form_id)


def list_form_responses(form_id: str) -> list[dict[str, Any]]:
    if is_saas_mode():
        return _client().list_form_responses(_session_id(), form_id)
    return local_list_form_responses(_local_credentials(), form_id)


def list_response_timestamps(form_id: str) -> list[datetime]:
    if not is_saas_mode():
        return local_list_response_timestamps(_local_credentials(), form_id)

    timestamps: list[datetime] = []
    for response in _client().list_form_responses(_session_id(), form_id):
        created = response.get("createTime")
        if isinstance(created, str) and created:
            timestamps.append(
                datetime.fromisoformat(created.replace("Z", "+00:00"))
                .astimezone(UTC)
                .replace(tzinfo=None)
            )
    timestamps.sort()
    return timestamps


def scan_population_tables(sheet_id: str) -> list[ContextTable]:
    if is_saas_mode():
        return [
            ContextTable(
                source=str(item.get("source") or ""),
                label_header=str(item.get("label_header") or ""),
                count_header=str(item.get("count_header") or ""),
                population={str(k): int(v) for k, v in dict(item.get("population") or {}).items()},
            )
            for item in _client().list_population_tables(
                _session_id(),
                sheet_id,
                next_url=_next_url(),
            )
        ]
    return scan_sheets_for_tables(local_fetch_all_grids(_local_credentials(), sheet_id))


@st.cache_resource
def _client() -> SaaSApiClient:
    return SaaSApiClient(os.environ.get("API_BASE_URL", "http://localhost:8000"))


def _session_id() -> str:
    session_id = st.session_state.get("saas_session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError("SaaS session is missing.")
    return session_id


def _next_url() -> str:
    app_base = os.environ.get("APP_BASE_URL", "").rstrip("/")
    return f"{app_base}/" if app_base else "/"


def _local_credentials():
    creds_dict = st.session_state.get("credentials")
    if not creds_dict:
        raise RuntimeError("Local Google credentials are missing.")
    return credentials_from_dict(creds_dict)


def _format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")
