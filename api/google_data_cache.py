"""Small process-local TTL cache for SaaS API Google-derived metadata."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Generic, TypeVar

from core.logger import get_logger

T = TypeVar("T")

log = get_logger(__name__)


@dataclass(frozen=True)
class ApiCacheKey:
    user_id: str
    data_kind: str
    resource_id: str


@dataclass(frozen=True)
class ApiCacheResult(Generic[T]):
    value: T
    fetched_at: datetime
    cache_hit: bool


@dataclass
class _Entry:
    value: object
    fetched_at: datetime
    expires_at: float


_CACHE: dict[tuple[str, str, str], _Entry] = {}
_LOCK = threading.RLock()


def get_or_load(
    key: ApiCacheKey,
    *,
    ttl_seconds: int,
    loader: Callable[[], T],
) -> ApiCacheResult[T]:
    """Return cached API metadata or load it.

    Keys use user/domain identifiers only; no raw session ids, tokens, form titles,
    question text, responses, or query payload fragments are stored here.
    """
    now = time.monotonic()
    storage_key = (key.user_id, key.data_kind, key.resource_id)
    with _LOCK:
        entry = _CACHE.get(storage_key)
        if entry and entry.expires_at > now:
            _log_cache_event(key, cache_hit=True)
            return ApiCacheResult(entry.value, entry.fetched_at, cache_hit=True)  # type: ignore[arg-type]
        if entry:
            _CACHE.pop(storage_key, None)

    _log_cache_event(key, cache_hit=False)
    value = loader()
    fetched_at = datetime.now(UTC)
    with _LOCK:
        _CACHE[storage_key] = _Entry(
            value=value,
            fetched_at=fetched_at,
            expires_at=now + ttl_seconds,
        )
    return ApiCacheResult(value, fetched_at, cache_hit=False)


def clear_api_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def _log_cache_event(key: ApiCacheKey, *, cache_hit: bool) -> None:
    log.info(
        "api_google_data_cache_access",
        extra={
            "cache_hit": cache_hit,
            "cache_layer": "api_google_data",
            "data_kind": key.data_kind,
            "resource_id": key.resource_id,
        },
    )
