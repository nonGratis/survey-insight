from __future__ import annotations

import pytest

from core.catalog_loading import (
    MAX_ATTEMPTS,
    OTHER_RETRY_SECONDS,
    RATE_LIMITED_RETRY_SECONDS,
    EnrichBatch,
    Retry,
    is_loading,
    next_form_batches,
    schedule_retry,
    seconds_until_retried,
)

FORMS = [f"form_{index}" for index in range(7)]


def _batches(**state) -> list[EnrichBatch]:
    defaults = {"done": set(), "retries": {}, "have_summary": set(), "now": 0.0}
    return next_form_batches(FORMS, **{**defaults, **state})


def test_new_forms_come_in_catalog_order_split_into_batches() -> None:
    batches = _batches(batch_size=3, max_batches=2)

    assert batches == [
        EnrichBatch(["form_0", "form_1", "form_2"], include_summary=True),
        EnrichBatch(["form_3", "form_4", "form_5"], include_summary=True),
    ]


def test_forms_due_for_a_retry_follow_the_new_ones() -> None:
    retries = {
        "form_0": Retry(attempts=1, not_before=10.0),
        "form_1": Retry(attempts=1, not_before=30.0),
    }

    batches = _batches(
        done={"form_0", "form_1", "form_2", "form_3", "form_4"},
        retries=retries,
        now=20.0,
        batch_size=10,
        max_batches=1,
    )

    # form_1 waits till 30 s; form_0 is due and still has no details.
    assert batches == [EnrichBatch(["form_5", "form_6", "form_0"], include_summary=True)]


def test_a_form_with_details_retries_only_its_responses() -> None:
    retries = {
        "form_0": Retry(attempts=1, not_before=0.0),
        "form_1": Retry(attempts=1, not_before=0.0),
    }

    batches = _batches(
        done=set(FORMS),
        retries=retries,
        have_summary={"form_1"},
        now=5.0,
        batch_size=10,
        max_batches=3,
    )

    assert batches == [
        EnrichBatch(["form_0"], include_summary=True),
        EnrichBatch(["form_1"], include_summary=False),
    ]


def test_nothing_to_request_while_retries_wait() -> None:
    retries = {"form_0": Retry(attempts=1, not_before=30.0)}

    assert _batches(done=set(FORMS), retries=retries, now=20.0, batch_size=10, max_batches=3) == []
    assert is_loading(FORMS, done=set(FORMS), retries=retries)


def test_loading_ends_when_every_form_has_a_result_and_no_retry_waits() -> None:
    assert is_loading(FORMS, done=set(FORMS[:-1]), retries={})
    assert not is_loading(FORMS, done=set(FORMS), retries={})


@pytest.mark.parametrize("status", ["ok", "no_access", "deleted", "unsupported"])
def test_final_statuses_are_not_retried(status: str) -> None:
    assert schedule_retry(status, None, now=0.0) is None


def test_a_rate_limited_row_waits_for_the_quota_window_to_free_up() -> None:
    retry = schedule_retry("rate_limited", None, now=100.0)

    assert retry == Retry(attempts=1, not_before=100.0 + RATE_LIMITED_RETRY_SECONDS)


def test_a_row_held_by_the_api_guard_comes_back_when_its_slot_frees() -> None:
    previous = Retry(attempts=2, not_before=0.0)

    retry = schedule_retry("rate_limited", previous, now=100.0, retry_after=37.5)

    # The API kept a slot for the row: waiting for it is not a failed attempt.
    assert retry == Retry(attempts=2, not_before=137.5)


def test_the_wait_is_until_the_last_held_row_is_retried() -> None:
    retries = {
        "form_0": Retry(attempts=0, not_before=130.0),
        "form_1": Retry(attempts=0, not_before=145.0),
        "form_2": Retry(attempts=1, not_before=500.0),  # not one of the rows asked about
    }

    assert seconds_until_retried(retries, ["form_0", "form_1"], now=100.0) == 45.0
    assert seconds_until_retried(retries, ["form_0"], now=200.0) == 0.0
    assert seconds_until_retried(retries, ["form_3"], now=100.0) is None


def test_a_timeout_is_retried_sooner_and_attempts_add_up() -> None:
    first = schedule_retry("timeout", None, now=0.0)
    second = schedule_retry("api_error", first, now=10.0)

    assert first == Retry(attempts=1, not_before=OTHER_RETRY_SECONDS)
    assert second == Retry(attempts=2, not_before=10.0 + OTHER_RETRY_SECONDS)


def test_retries_stop_after_the_last_attempt() -> None:
    previous = Retry(attempts=MAX_ATTEMPTS - 1, not_before=0.0)

    assert schedule_retry("rate_limited", previous, now=50.0) is None
