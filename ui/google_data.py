"""Google data facade for Streamlit pages.

Production uses the SaaS API. Local/demo mode keeps the previous direct Google
credential path for developer convenience.
"""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import streamlit as st

from core.auth import credentials_from_dict
from core.catalog_stream import Loaded, load_catalog
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
)
from core.forms_catalog import (
    enrich_form as local_enrich_form,
)
from core.forms_catalog import (
    list_forms_with_drive_meta as local_list_catalog_forms,
)
from core.forms_catalog import (
    response_stats as local_response_stats,
)
from core.saas.settings import load_web_settings
from core.sheets_api import fetch_all_grids as local_fetch_all_grids
from ui.api_boundary import handle_api_errors
from ui.data_access_cache import (
    CATALOG_TTL_SECONDS,
    FORM_STRUCTURE_TTL_SECONDS,
    POPULATION_TABLES_TTL_SECONDS,
    RAW_RESPONSES_MAX_BYTES,
    RAW_RESPONSES_MAX_ROWS,
    RAW_RESPONSES_TTL_SECONDS,
    TIMESTAMPS_TTL_SECONDS,
    CacheKey,
    clear_cache,
    get_or_load,
    session_cache_key,
)
from ui.saas_api import SaaSApiClient

# The local demo loads the catalog in process, with fewer parallel Google calls.
LOCAL_CATALOG_WORKERS = 4


@dataclass(frozen=True)
class GoogleDataClient:
    """Session-bound Google data facade for Streamlit pages.

    The factory captures Streamlit session state once in the main script run.
    Methods can then be safely used inside worker threads, for example catalog
    enrichment, without reading `st.session_state` there.
    """

    session_id: str | None = None
    local_credentials: Any | None = None
    # Resolved by the factory in the script run: `_client()` is a Streamlit cache, which a
    # worker thread should not touch.
    api: SaaSApiClient | None = None

    def list_forms_for_picker(self) -> list[dict[str, Any]]:
        if is_saas_mode():
            # The same Drive list as the catalog, from the same cache entry: a list of its
            # own with a shorter TTL cost a 2.5-3.5 s Drive call every couple of minutes.
            return [
                {"id": form.id, "name": form.name, "modifiedTime": form.modified_time}
                for form in self.list_catalog_forms()
            ]
        return local_list_user_forms(self._local_credentials())

    def list_catalog_forms(self) -> list[FormDriveMeta]:
        if is_saas_mode():
            session_id = _require_session_id(self.session_id)
            return [
                _drive_meta_from_payload(item)
                for item in get_or_load(
                    _cache_key(session_id, "catalog_metadata"),
                    ttl_seconds=CATALOG_TTL_SECONDS,
                    loader=lambda: _client().list_forms(session_id),
                )
            ]
        return local_list_catalog_forms(self._local_credentials())

    def stream_catalog(self, form_ids: list[str]) -> Iterator[dict[str, Any]]:
        """Events of one catalog load: from the API, or computed here in local mode.

        Runs in the page's background thread: it reads nothing from Streamlit.
        """
        if is_saas_mode():
            session_id = _require_session_id(self.session_id)
            yield from (self.api or _client()).stream_catalog(session_id, form_ids)
            return
        creds = self._local_credentials()
        for event in load_catalog(
            form_ids,
            load_summary=lambda form_id: Loaded(asdict(local_enrich_form(creds, form_id))),
            load_stats=lambda form_id: Loaded(asdict(local_response_stats(creds, form_id))),
            workers=LOCAL_CATALOG_WORKERS,
        ):
            yield event.to_dict()

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
    """True in production; raises there if the web settings are missing."""
    return load_web_settings().signs_in_through_api


def google_data_client() -> GoogleDataClient:
    if is_saas_mode():
        return GoogleDataClient(session_id=_session_id_from_state(), api=_client())
    return GoogleDataClient(local_credentials=_local_credentials())


def google_data_client_for_session(session_id: str) -> GoogleDataClient:
    return GoogleDataClient(session_id=session_id)


def cache_token() -> str:
    """Return the per-user, per-login key that partitions Streamlit caches.

    Unauthenticated state gets a fresh key on every call, so nothing is ever
    served from cache before the user identity is known.
    """
    if is_saas_mode():
        session_id = st.session_state.get("saas_session_id")
        user = st.session_state.get("user")
        user_id = user.get("id") if isinstance(user, dict) else None
        if not (
            isinstance(session_id, str) and session_id and isinstance(user_id, str) and user_id
        ):
            return session_cache_key(f"anonymous:{uuid.uuid4().hex}")
        return session_cache_key(f"{session_id}:{user_id}")
    creds = _local_credentials()
    return session_cache_key(creds.token or "")


@handle_api_errors
def list_forms_for_picker() -> list[dict[str, Any]]:
    return google_data_client().list_forms_for_picker()


@handle_api_errors
def list_catalog_forms() -> list[FormDriveMeta]:
    return google_data_client().list_catalog_forms()


@handle_api_errors
def get_form_structure(form_id: str) -> dict[str, Any]:
    return google_data_client().get_form_structure(form_id)


@handle_api_errors
def list_form_responses(form_id: str) -> list[dict[str, Any]]:
    return google_data_client().list_form_responses(form_id)


@handle_api_errors
def list_response_timestamps(form_id: str) -> list[datetime]:
    return google_data_client().list_response_timestamps(form_id)


@handle_api_errors
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
    for kind in ("forms_list", "catalog_metadata"):
        clear_cache(session_id=scoped_session, data_kind=kind)


def clear_form_cache(form_id: str, session_id: str | None = None) -> None:
    scoped_session = _session_for_clear(session_id)
    for kind in ("form_structure", "response_timestamps"):
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
    return SaaSApiClient(load_web_settings().api_base_url)


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
    return f"{load_web_settings().app_base_url}/"


def _local_credentials():
    creds_dict = st.session_state.get("credentials")
    if not creds_dict:
        raise RuntimeError("Local Google credentials are missing.")
    return credentials_from_dict(creds_dict)


def _drive_meta_from_payload(payload: dict[str, Any]) -> FormDriveMeta:
    """Fields this web knows; a field the API adds later must not break the catalog."""
    known = {field.name for field in dataclasses.fields(FormDriveMeta)}
    return FormDriveMeta(**{key: value for key, value in payload.items() if key in known})
