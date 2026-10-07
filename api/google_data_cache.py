"""Small process-local TTL cache for SaaS API Google-derived metadata."""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from core.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ApiCacheKey:
    user_id: str
    data_kind: str
    resource_id: str


@dataclass(frozen=True)
class ApiCacheResult[T]:
    value: T
    fetched_at: datetime
    cache_hit: bool


@dataclass
class _Entry:
    value: object
    fetched_at: datetime
    expires_at: float


_CACHE: dict[tuple[str, str, str], _Entry] = {}
# Loads in progress, so that the same data asked again meanwhile waits for them.
_LOADING: dict[tuple[str, str, str], threading.Event] = {}
_LOCK = threading.RLock()


def get_or_load[T](
    key: ApiCacheKey,
    *,
    ttl_seconds: int,
    loader: Callable[[], T],
) -> ApiCacheResult[T]:
    """Return cached API metadata or load it.

    One load per key at a time: a caller asking for data that another request is loading
    right now (a second tab, a page reloaded mid-load) waits for that load and gets its
    result, instead of calling Google again. If that load fails, the caller loads itself.

    Keys use user/domain identifiers only; no raw session ids, tokens, form titles,
    question text, responses, or query payload fragments are stored here.
    """
    storage_key = (key.user_id, key.data_kind, key.resource_id)
    waited = False
    while True:
        now = time.monotonic()
        with _LOCK:
            entry = _CACHE.get(storage_key)
            if entry and entry.expires_at > now:
                _log_cache_event(key, cache_hit=True, shared_load=waited)
                return ApiCacheResult(entry.value, entry.fetched_at, cache_hit=True)  # type: ignore[arg-type]
            if entry:
                _CACHE.pop(storage_key, None)
            loading = _LOADING.get(storage_key)
            if loading is None:
                loading = _LOADING[storage_key] = threading.Event()
                break
        loading.wait()
        waited = True

    _log_cache_event(key, cache_hit=False)
    try:
        value = loader()
        fetched_at = datetime.now(UTC)
        with _LOCK:
            _CACHE[storage_key] = _Entry(
                value=value,
                fetched_at=fetched_at,
                expires_at=now + ttl_seconds,
            )
    finally:
        with _LOCK:
            del _LOADING[storage_key]
        loading.set()
    return ApiCacheResult(value, fetched_at, cache_hit=False)


def clear_api_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def _hash_resource(resource_id: str) -> str:
    if not resource_id:
        return ""
    return hashlib.sha256(resource_id.encode("utf-8")).hexdigest()[:16]


def _log_cache_event(key: ApiCacheKey, *, cache_hit: bool, shared_load: bool = False) -> None:
    log.info(
        "api_google_data_cache_access",
        extra={
            "cache_hit": cache_hit,
            # Got the result of another request's load of the same data.
            "shared_load": shared_load,
            "cache_layer": "api_google_data",
            "data_kind": key.data_kind,
            "resource_hash": _hash_resource(key.resource_id),
        },
    )
