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

from core.logger import get_logger

log = get_logger(__name__)

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

    Логування:
    - Success-path не логується ТУТ — кожен виклик уже обгорнутий log_call
      у відповідному API-helper'і (анти-дубль).
    - WARNING `api_op_retry` на кожній спробі-після-першої.
    - ERROR `api_op_failed` із traceback на остаточному провалі.

    Args:
        fn: функція Google API клієнта (зазвичай method().execute()).
        max_retries: скільки разів пробуємо при ретраєбл-помилках.
        base_delay: стартова затримка в секундах; кожен ретрай ×2 + jitter.
    """
    label = getattr(fn, "__name__", repr(fn))
    overall_start = time.perf_counter()
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except HttpError as exc:
            retryable = exc.resp.status in RETRY_HTTP_CODES
            if retryable and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                log.warning(
                    "api_op_retry",
                    extra={
                        "op": label,
                        "status": exc.resp.status,
                        "attempt": attempt + 1,
                        "delay_s": round(delay, 2),
                    },
                )
                time.sleep(delay)
                continue
            duration_ms = round((time.perf_counter() - overall_start) * 1000, 1)
            log.error(
                "api_op_failed",
                extra={
                    "op": label,
                    "duration_ms": duration_ms,
                    "status": exc.resp.status,
                },
                exc_info=True,
            )
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
                # exc_info=True не дасть traceback'у з future.result(), бо
                # sys.exc_info() тут порожній. Будуємо tuple явно зі
                # збереженим __traceback__.
                log.warning(
                    "parallel_task_failed",
                    extra={"op": getattr(fn, "__name__", repr(fn))},
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                results.append((item, exc))
    return results
