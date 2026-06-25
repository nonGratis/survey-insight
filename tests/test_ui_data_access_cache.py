from __future__ import annotations

from ui.data_access_cache import CacheKey, clear_cache, get_or_load, session_cache_key


def test_session_cache_key_does_not_expose_raw_session_id() -> None:
    raw = "raw-session-id"

    key = session_cache_key(raw)

    assert key != raw
    assert raw not in key


def test_cache_hit_reuses_loaded_value() -> None:
    clear_cache()
    calls = 0

    def load() -> list[int]:
        nonlocal calls
        calls += 1
        return [1]

    key = CacheKey(session_key=session_cache_key("s1"), data_kind="forms_list")

    assert get_or_load(key, ttl_seconds=60, loader=load) == [1]
    assert get_or_load(key, ttl_seconds=60, loader=load) == [1]
    assert calls == 1


def test_size_guard_skips_large_raw_response_cache() -> None:
    clear_cache()
    calls = 0

    def load() -> list[dict[str, str]]:
        nonlocal calls
        calls += 1
        return [{"responseId": "r1"}, {"responseId": "r2"}]

    key = CacheKey(session_key=session_cache_key("s1"), data_kind="raw_responses", resource_id="f1")

    assert get_or_load(key, ttl_seconds=60, loader=load, max_rows=1) == [
        {"responseId": "r1"},
        {"responseId": "r2"},
    ]
    assert get_or_load(key, ttl_seconds=60, loader=load, max_rows=1) == [
        {"responseId": "r1"},
        {"responseId": "r2"},
    ]
    assert calls == 2


def test_granular_clear_removes_only_matching_kind_and_resource() -> None:
    clear_cache()
    calls = {"a": 0, "b": 0}
    session_key = session_cache_key("s1")
    key_a = CacheKey(session_key=session_key, data_kind="form_structure", resource_id="f1")
    key_b = CacheKey(session_key=session_key, data_kind="raw_responses", resource_id="f1")

    def load_a() -> str:
        calls["a"] += 1
        return "a"

    def load_b() -> str:
        calls["b"] += 1
        return "b"

    get_or_load(key_a, ttl_seconds=60, loader=load_a)
    get_or_load(key_b, ttl_seconds=60, loader=load_b)
    clear_cache(session_id="s1", data_kind="form_structure", resource_id="f1")
    get_or_load(key_a, ttl_seconds=60, loader=load_a)
    get_or_load(key_b, ttl_seconds=60, loader=load_b)

    assert calls == {"a": 2, "b": 1}
