"""Small session-scoped TTL cache for Streamlit SaaS data access.

This cache is intentionally process-local and volatile. It reduces repeated
Streamlit -> API calls during reruns/navigation, but it is not a security
boundary and never persists raw Google Forms responses.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from core.logger import get_logger

log = get_logger(__name__)

T = TypeVar("T")

ACCESS_TTL_SECONDS = 180
FORMS_LIST_TTL_SECONDS = 120
CATALOG_TTL_SECONDS = 300
FORM_STRUCTURE_TTL_SECONDS = 600
RESPONSE_STATS_TTL_SECONDS = 120
TIMESTAMPS_TTL_SECONDS = 60
RAW_RESPONSES_TTL_SECONDS = 120
POPULATION_TABLES_TTL_SECONDS = 300

RAW_RESPONSES_MAX_ROWS = int(os.getenv("SI_RAW_RESPONSES_CACHE_MAX_ROWS", "10000"))
RAW_RESPONSES_MAX_BYTES = int(os.getenv("SI_RAW_RESPONSES_CACHE_MAX_BYTES", "8000000"))


@dataclass(frozen=True)
class CacheKey:
    session_key: str
    data_kind: str
    resource_id: str = ""
    purpose: str = ""

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.session_key, self.data_kind, _hash_resource(self.resource_id), self.purpose)


@dataclass
class _Entry:
    value: Any
    expires_at: float


_CACHE: dict[tuple[str, str, str, str], _Entry] = {}
_LOCK = threading.RLock()


def session_cache_key(session_id: str) -> str:
    """Return a stable non-secret key for cache partitioning."""
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]


def get_or_load(
    key: CacheKey,
    *,
    ttl_seconds: int,
    loader: Callable[[], T],
    max_rows: int | None = None,
    max_bytes: int | None = None,
) -> T:
    """Return cached value or load it, with optional size guard for large payloads."""
    now = time.monotonic()
    storage_key = key.as_tuple()
    with _LOCK:
        entry = _CACHE.get(storage_key)
        if entry and entry.expires_at > now:
            _log_cache_event(key, cache_hit=True)
            return entry.value
        if entry:
            _CACHE.pop(storage_key, None)

    _log_cache_event(key, cache_hit=False)
    value = loader()

    skipped_reason = _size_guard_skip_reason(value, max_rows=max_rows, max_bytes=max_bytes)
    if skipped_reason:
        _log_cache_event(key, cache_hit=False, cache_skipped_reason=skipped_reason)
        return value

    with _LOCK:
        _CACHE[storage_key] = _Entry(value=value, expires_at=now + ttl_seconds)
    return value


def clear_cache(
    *,
    session_id: str | None = None,
    data_kind: str | None = None,
    resource_id: str | None = None,
) -> None:
    """Clear matching cache entries."""
    session_key = session_cache_key(session_id) if session_id else None
    resource_hash = _hash_resource(resource_id or "") if resource_id is not None else None
    with _LOCK:
        for storage_key in list(_CACHE):
            key_session, key_kind, key_resource, _purpose = storage_key
            if session_key is not None and key_session != session_key:
                continue
            if data_kind is not None and key_kind != data_kind:
                continue
            if resource_hash is not None and key_resource != resource_hash:
                continue
            _CACHE.pop(storage_key, None)


def _size_guard_skip_reason(
    value: Any,
    *,
    max_rows: int | None,
    max_bytes: int | None,
) -> str | None:
    if max_rows is not None and isinstance(value, list) and len(value) > max_rows:
        return "max_rows"
    if max_bytes is None:
        return None
    try:
        payload = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return None
    if len(payload.encode("utf-8")) > max_bytes:
        return "max_bytes"
    return None


def _hash_resource(resource_id: str) -> str:
    if not resource_id:
        return ""
    return hashlib.sha256(resource_id.encode("utf-8")).hexdigest()[:16]


def _log_cache_event(
    key: CacheKey,
    *,
    cache_hit: bool,
    cache_skipped_reason: str | None = None,
) -> None:
    log.info(
        "ui_data_cache_access",
        extra={
            "cache_hit": cache_hit,
            "cache_layer": "ui_data",
            "data_kind": key.data_kind,
            "purpose": key.purpose,
            "resource_hash": _hash_resource(key.resource_id),
            "cache_skipped_reason": cache_skipped_reason or "",
        },
    )
