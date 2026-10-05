from __future__ import annotations

import pytest

from api.google_quota import FormsQuotaGuards, RollingWindowGuard


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_guard_admits_up_to_the_limit_within_the_window() -> None:
    guard = RollingWindowGuard(3, window_seconds=60, clock=_Clock())

    assert [guard.acquire_or_wait("user_1") == 0.0 for _ in range(4)] == [True, True, True, False]


def test_guard_admits_again_as_old_calls_leave_the_window() -> None:
    clock = _Clock()
    guard = RollingWindowGuard(2, window_seconds=60, clock=clock)
    assert guard.acquire_or_wait("user_1") == 0.0
    clock.now += 30
    assert guard.acquire_or_wait("user_1") == 0.0
    assert guard.acquire_or_wait("user_1") > 0

    clock.now += 30  # the first call is now 60 s old
    assert guard.acquire_or_wait("user_1") == 0.0
    assert guard.acquire_or_wait("user_1") > 0


def test_a_refused_call_is_told_when_its_slot_frees() -> None:
    clock = _Clock()
    guard = RollingWindowGuard(2, window_seconds=60, clock=clock)
    assert guard.acquire_or_wait("user_1") == 0.0  # at 1000
    clock.now += 10
    assert guard.acquire_or_wait("user_1") == 0.0  # at 1010
    clock.now += 10

    waits = [guard.acquire_or_wait("user_1") for _ in range(3)]

    # Each refused call gets its own slot: as the two calls leave the window, then as the
    # call promised the first slot (made at 1060) leaves it in its turn.
    assert waits == [40.0, 50.0, 100.0]


def test_a_call_that_comes_back_on_time_gets_its_slot() -> None:
    clock = _Clock()
    guard = RollingWindowGuard(1, window_seconds=60, clock=clock)
    assert guard.acquire_or_wait("user_1") == 0.0
    clock.now += 20
    wait = guard.acquire_or_wait("user_1")

    clock.now += wait
    assert guard.acquire_or_wait("user_1") == 0.0
    assert guard.acquire_or_wait("user_1") == 60.0  # the next slot: a full window later


def test_guard_counts_each_user_separately() -> None:
    guard = RollingWindowGuard(1, window_seconds=60, clock=_Clock())

    assert guard.acquire_or_wait("user_1") == 0.0
    assert guard.acquire_or_wait("user_2") == 0.0
    assert guard.acquire_or_wait("user_1") > 0


def test_quota_guards_stay_below_google_limits_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SI_FORMS_READS_PER_MINUTE", raising=False)
    monkeypatch.delenv("SI_FORMS_RESPONSE_LISTS_PER_MINUTE", raising=False)

    guards = FormsQuotaGuards.from_env()

    # Google: 390 reads and 180 response lists per minute per user.
    assert guards.reads.limit < 390
    assert guards.response_lists.limit < 180


def test_quota_guards_read_limits_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SI_FORMS_READS_PER_MINUTE", "50")
    monkeypatch.setenv("SI_FORMS_RESPONSE_LISTS_PER_MINUTE", "not-a-number")

    guards = FormsQuotaGuards.from_env()

    assert guards.reads.limit == 50
    assert guards.response_lists.limit == 140
