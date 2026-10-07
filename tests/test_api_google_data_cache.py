"""api.google_data_cache: one Google call for data asked by two requests at once."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from api.google_data_cache import ApiCacheKey, clear_api_cache, get_or_load

KEY = ApiCacheKey(user_id="user_1", data_kind="form_summary", resource_id="form_1")


@pytest.fixture(autouse=True)
def _empty_cache() -> None:
    clear_api_cache()


def test_the_same_data_asked_twice_at_once_is_loaded_once() -> None:
    started, release = threading.Event(), threading.Event()
    calls: list[str] = []

    def slow_google() -> dict:
        calls.append("google")
        started.set()
        release.wait(5)
        return {"title": "Survey"}

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(get_or_load, KEY, ttl_seconds=60, loader=slow_google)
        started.wait(5)
        # A page reloaded mid-load asks for the same form while Google is still answering.
        second = pool.submit(get_or_load, KEY, ttl_seconds=60, loader=slow_google)
        time.sleep(0.05)
        release.set()
        results = [first.result(5), second.result(5)]

    assert calls == ["google"]
    assert [r.value for r in results] == [{"title": "Survey"}, {"title": "Survey"}]
    assert [r.cache_hit for r in results] == [False, True]


def test_when_the_shared_load_fails_the_waiting_caller_loads_itself() -> None:
    started, release = threading.Event(), threading.Event()
    calls: list[str] = []

    def failing_google() -> dict:
        calls.append("failed")
        started.set()
        release.wait(5)
        raise RuntimeError("Google is down")

    def google() -> dict:
        calls.append("google")
        return {"title": "Survey"}

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(get_or_load, KEY, ttl_seconds=60, loader=failing_google)
        started.wait(5)
        second = pool.submit(get_or_load, KEY, ttl_seconds=60, loader=google)
        time.sleep(0.05)
        release.set()
        with pytest.raises(RuntimeError):
            first.result(5)
        result = second.result(5)

    assert calls == ["failed", "google"]
    assert result.value == {"title": "Survey"} and not result.cache_hit


def test_other_users_do_not_wait_for_each_other() -> None:
    started, release = threading.Event(), threading.Event()

    def slow_google() -> dict:
        started.set()
        release.wait(5)
        return {"title": "Theirs"}

    other = ApiCacheKey(user_id="user_2", data_kind=KEY.data_kind, resource_id=KEY.resource_id)
    with ThreadPoolExecutor(1) as pool:
        first = pool.submit(get_or_load, KEY, ttl_seconds=60, loader=slow_google)
        started.wait(5)
        mine = get_or_load(other, ttl_seconds=60, loader=lambda: {"title": "Mine"})
        release.set()
        first.result(5)

    assert mine.value == {"title": "Mine"} and not mine.cache_hit
