"""Helpers для роботи з Google API лімітами: backoff + паралельні виклики.

Google API дозволяє паралельні запити — лімітує тільки requests/second.
ThreadPoolExecutor дає нам speedup проти послідовного fetch без потреби
sleep-між-викликами. На 429/5xx — експоненційний backoff per task.
"""
from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

from googleapiclient.errors import HttpError

T = TypeVar("T")
R = TypeVar("R")

# Forms API має квоту 100 req/100s, Sheets — теж 100/100s. 5 паралельних
# воркерів дають вистачає швидкості, при цьому backoff на 429 не дає
# залишити нас без квоти.
DEFAULT_MAX_WORKERS = 5
RETRY_HTTP_CODES = {429, 500, 502, 503, 504}


def call_with_backoff(
    fn: Callable[..., R],
    *args,
    max_retries: int = 5,
    base_delay: float = 1.0,
    **kwargs,
) -> R:
    """Викликати Google API з експоненційним backoff на 429/5xx.

    Args:
        fn: функція Google API клієнта (зазвичай method().execute()).
        max_retries: скільки разів пробуємо при ретраєбл-помилках.
        base_delay: стартова затримка в секундах; кожен ретрай ×2 + jitter.
    """
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except HttpError as exc:
            retryable = exc.resp.status in RETRY_HTTP_CODES
            if retryable and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
                continue
            raise
    raise RuntimeError("unreachable: max_retries should have raised")


def parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> list[tuple[T, R | Exception]]:
    """Виконати fn(item) паралельно для всіх items.

    Returns:
        Список (item, result_or_exception) у порядку завершення тасків.
        Помилки не валять весь batch — callee вирішує як їх відображати
        (наприклад, маркер ERROR + st.toast у UI).
    """
    results: list[tuple[T, R | Exception]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(call_with_backoff, fn, item): item for item in items
        }
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                results.append((item, future.result()))
            except Exception as exc:  # noqa: BLE001
                results.append((item, exc))
    return results
