"""Google data facade for Streamlit pages.

Production uses the SaaS API. Local/demo mode keeps the previous direct Google
credential path for developer convenience.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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
from ui.data_access_cache import (
    CATALOG_TTL_SECONDS,
    FORM_STRUCTURE_TTL_SECONDS,
    FORMS_LIST_TTL_SECONDS,
    POPULATION_TABLES_TTL_SECONDS,
    RAW_RESPONSES_MAX_BYTES,
    RAW_RESPONSES_MAX_ROWS,
    RAW_RESPONSES_TTL_SECONDS,
    RESPONSE_STATS_TTL_SECONDS,
    TIMESTAMPS_TTL_SECONDS,
    CacheKey,
    clear_cache,
    get_or_load,
    session_cache_key,
)
from ui.saas_api import SaaSApiClient


@dataclass(frozen=True)
class GoogleDataClient:
    """Session-bound Google data facade for Streamlit pages.

    The factory captures Streamlit session state once in the main script run.
    Methods can then be safely used inside worker threads, for example catalog
    enrichment via `parallel_map`, without reading `st.session_state` there.
    """

    session_id: str | None = None
    local_credentials: Any | None = None

    def list_forms_for_picker(self) -> list[dict[str, Any]]:
        if is_saas_mode():
            session_id = _require_session_id(self.session_id)
            return get_or_load(
                _cache_key(session_id, "forms_list"),
                ttl_seconds=FORMS_LIST_TTL_SECONDS,
                loader=lambda: _client().list_forms(session_id),
            )
        return local_list_user_forms(self._local_credentials())

    def list_catalog_forms(self) -> list[FormDriveMeta]:
        if is_saas_mode():
            session_id = _require_session_id(self.session_id)
            return [
                FormDriveMeta(**item)
                for item in get_or_load(
                    _cache_key(session_id, "catalog_metadata"),
                    ttl_seconds=CATALOG_TTL_SECONDS,
                    loader=lambda: _client().list_forms(session_id),
                )
            ]
        return local_list_catalog_forms(self._local_credentials())

    def list_catalog_snapshot(
        self,
    ) -> tuple[list[FormDriveMeta], dict[str, FormEnrichment | None], dict[str, ResponseStats]]:
        # Catalog initial render must stay fast: show Drive metadata first and let
        # the page progressively enrich rows in chunks. The aggregate catalog API
        # remains available for future bounded/server-side use, but it must not
        # block the first Streamlit render.
        return self.list_catalog_forms(), {}, {}

    def get_form_summary(self, form_id: str) -> FormEnrichment:
        if is_saas_mode():
            session_id = _require_session_id(self.session_id)
            return FormEnrichment(
                **get_or_load(
                    _cache_key(session_id, "form_summary", form_id),
                    ttl_seconds=FORM_STRUCTURE_TTL_SECONDS,
                    loader=lambda: _client().get_form_summary(session_id, form_id),
                )
            )
        return local_enrich_form(self._local_credentials(), form_id)

    def get_response_stats(self, form_id: str) -> ResponseStats:
        if is_saas_mode():
            session_id = _require_session_id(self.session_id)
            return ResponseStats(
                **get_or_load(
                    _cache_key(session_id, "response_stats", form_id),
                    ttl_seconds=RESPONSE_STATS_TTL_SECONDS,
                    loader=lambda: _client().get_response_stats(session_id, form_id),
                )
            )
        timestamps = local_list_response_timestamps(self._local_credentials(), form_id)
        return ResponseStats(
            total=len(timestamps),
            first_response=_format_timestamp(timestamps[0]) if timestamps else None,
            second_response=_format_timestamp(timestamps[1]) if len(timestamps) >= 2 else None,
            last_response=_format_timestamp(timestamps[-1]) if timestamps else None,
        )

    def get_form_structure(self, form_id: str) -> dict[str, Any]:
        if is_saas_mode():
            session_id = _require_session_id(self.session_id)
            return get_or_load(
                _cache_key(session_id, "form_structure", form_id),
                ttl_seconds=FORM_STRUCTURE_TTL_SECONDS,
                loader=lambda: _client().get_form_structure(session_id, form_id),
            )
        return local_get_form_structure(self._local_credentials(), form_id)

    def list_form_responses(self, form_id: str) -> list[dict[str, Any]]:
        if is_saas_mode():
            session_id = _require_session_id(self.session_id)
            return get_or_load(
                _cache_key(session_id, "raw_responses", form_id),
                ttl_seconds=RAW_RESPONSES_TTL_SECONDS,
                loader=lambda: _client().list_form_responses(session_id, form_id),
                max_rows=RAW_RESPONSES_MAX_ROWS,
                max_bytes=RAW_RESPONSES_MAX_BYTES,
            )
        return local_list_form_responses(self._local_credentials(), form_id)

    def list_response_timestamps(self, form_id: str) -> list[datetime]:
        if not is_saas_mode():
            return local_list_response_timestamps(self._local_credentials(), form_id)

        session_id = _require_session_id(self.session_id)
        timestamp_values = get_or_load(
            _cache_key(session_id, "response_timestamps", form_id),
            ttl_seconds=TIMESTAMPS_TTL_SECONDS,
            loader=lambda: _client().list_response_timestamps(session_id, form_id),
        )
        timestamps: list[datetime] = []
        for created in timestamp_values:
            if isinstance(created, str) and created:
                timestamps.append(
                    datetime.fromisoformat(created.replace("Z", "+00:00"))
                    .astimezone(UTC)
                    .replace(tzinfo=None)
                )
        timestamps.sort()
        return timestamps

    def scan_population_tables(self, sheet_id: str) -> list[ContextTable]:
        if is_saas_mode():
            session_id = _require_session_id(self.session_id)
            return [
                ContextTable(
                    source=str(item.get("source") or ""),
                    label_header=str(item.get("label_header") or ""),
                    count_header=str(item.get("count_header") or ""),
                    population={
                        str(k): int(v) for k, v in dict(item.get("population") or {}).items()
                    },
                )
                for item in get_or_load(
                    _cache_key(session_id, "population_tables", sheet_id, purpose="sheets"),
                    ttl_seconds=POPULATION_TABLES_TTL_SECONDS,
                    loader=lambda: _client().list_population_tables(
                        session_id,
                        sheet_id,
                        next_url=_next_url(),
                    ),
                )
            ]
        return scan_sheets_for_tables(local_fetch_all_grids(self._local_credentials(), sheet_id))

    def _local_credentials(self) -> Any:
        return self.local_credentials or _local_credentials()


def is_saas_mode() -> bool:
    return os.environ.get("APP_ENV") == "production" and bool(os.environ.get("API_BASE_URL"))


def google_data_client() -> GoogleDataClient:
    if is_saas_mode():
        return GoogleDataClient(session_id=_session_id_from_state())
    return GoogleDataClient(local_credentials=_local_credentials())


def google_data_client_for_session(session_id: str) -> GoogleDataClient:
    return GoogleDataClient(session_id=session_id)


def cache_token() -> str:
    if is_saas_mode():
        return session_cache_key(_session_id_from_state())
    creds = _local_credentials()
    return session_cache_key(creds.token or "")


def list_forms_for_picker() -> list[dict[str, Any]]:
    return google_data_client().list_forms_for_picker()


def list_catalog_forms() -> list[FormDriveMeta]:
    return google_data_client().list_catalog_forms()


def list_catalog_snapshot() -> tuple[
    list[FormDriveMeta], dict[str, FormEnrichment | None], dict[str, ResponseStats]
]:
    return google_data_client().list_catalog_snapshot()


def get_form_summary(form_id: str) -> FormEnrichment:
    return google_data_client().get_form_summary(form_id)


def get_response_stats(form_id: str) -> ResponseStats:
    return google_data_client().get_response_stats(form_id)


def get_form_structure(form_id: str) -> dict[str, Any]:
    return google_data_client().get_form_structure(form_id)


def list_form_responses(form_id: str) -> list[dict[str, Any]]:
    return google_data_client().list_form_responses(form_id)


def list_response_timestamps(form_id: str) -> list[datetime]:
    return google_data_client().list_response_timestamps(form_id)


def scan_population_tables(sheet_id: str) -> list[ContextTable]:
    return google_data_client().scan_population_tables(sheet_id)


def clear_google_data_cache(session_id: str | None = None) -> None:
    clear_cache(session_id=_session_for_clear(session_id))


def clear_forms_list_cache(session_id: str | None = None) -> None:
    scoped_session = _session_for_clear(session_id)
    clear_cache(session_id=scoped_session, data_kind="forms_list")
    clear_cache(session_id=scoped_session, data_kind="catalog_metadata")


def clear_catalog_cache(session_id: str | None = None) -> None:
    scoped_session = _session_for_clear(session_id)
    for kind in (
        "forms_list",
        "catalog_metadata",
        "form_summary",
        "response_stats",
    ):
        clear_cache(session_id=scoped_session, data_kind=kind)


def clear_form_cache(form_id: str, session_id: str | None = None) -> None:
    scoped_session = _session_for_clear(session_id)
    for kind in ("form_summary", "form_structure", "response_stats", "response_timestamps"):
        clear_cache(session_id=scoped_session, data_kind=kind, resource_id=form_id)


def clear_responses_cache(form_id: str, session_id: str | None = None) -> None:
    clear_cache(
        session_id=_session_for_clear(session_id), data_kind="raw_responses", resource_id=form_id
    )


def clear_timestamps_cache(form_id: str, session_id: str | None = None) -> None:
    clear_cache(
        session_id=_session_for_clear(session_id),
        data_kind="response_timestamps",
        resource_id=form_id,
    )


@st.cache_resource
def _client() -> SaaSApiClient:
    return SaaSApiClient(os.environ.get("API_BASE_URL", "http://localhost:8000"))


def _session_id_from_state() -> str:
    session_id = st.session_state.get("saas_session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError("SaaS session is missing.")
    return session_id


def _require_session_id(session_id: str | None) -> str:
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError("SaaS session is missing.")
    return session_id


def _cache_key(
    session_id: str,
    data_kind: str,
    resource_id: str = "",
    *,
    purpose: str = "forms",
) -> CacheKey:
    return CacheKey(
        session_key=session_cache_key(session_id),
        data_kind=data_kind,
        resource_id=resource_id,
        purpose=purpose,
    )


def _session_for_clear(session_id: str | None) -> str | None:
    if session_id:
        return session_id
    if is_saas_mode():
        value = st.session_state.get("saas_session_id")
        return value if isinstance(value, str) and value else None
    return None


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
