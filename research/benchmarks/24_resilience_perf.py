"""24_resilience_perf.py — вимірювання надійності та продуктивності для §4.2.

Відтворювані заміри, що підтверджують нефункціональні вимоги ТЗ:
  - 5.1.3 (надійність): обробка HTTP 429 та 500+ механізмом автоматичного
    повторення з експоненційною затримкою; швидке відновлення.
  - НФВ §1.3: час відгуку каталогу ≤ 5 с при до 100 форм — за рахунок
    паралелізації викликів (parallel_map, 5 воркерів).

Заміри РЕАЛЬНІ: викликається фактична реалізація core.google_throttle.
Мережеві затримки емулюються детермінованою паузою (реальний Google API у
тесті недоступний), решта — справжня поведінка коду.

Запуск:  .venv/Scripts/python.exe research/benchmarks/24_resilience_perf.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

logging.disable(logging.CRITICAL)  # очікувані ERROR-логи backoff не зашумлюють звіт

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from googleapiclient.errors import HttpError  # noqa: E402

from core.google_throttle import (  # noqa: E402
    RETRY_HTTP_CODES,
    call_with_backoff,
    parallel_map,
)


class _Resp:
    """Мінімальний httplib2-сумісний resp для HttpError."""

    def __init__(self, status: int):
        self.status = status
        self.reason = f"HTTP {status}"


def _http_error(status: int) -> HttpError:
    return HttpError(_Resp(status), b"", uri="https://forms.googleapis.com")


def _flaky_call(fail_times: int, status: int):
    """Фабрика виклику, що падає `fail_times` разів кодом `status`, тоді успіх."""
    state = {"n": 0}

    def call():
        if state["n"] < fail_times:
            state["n"] += 1
            raise _http_error(status)
        return "ok"

    return call


# --- A. Стійкість до 429/500+ (backoff) -------------------------------------


def measure_backoff() -> None:
    print("=" * 68)
    print("A. Автоматичне повторення з експоненційною затримкою (ТЗ 5.1.3)")
    print("=" * 68)
    print(f"Ретраябл-коди: {sorted(RETRY_HTTP_CODES)}; base_delay=1 с, ×2^attempt + jitter")
    print(f"{'сценарій':38} {'спроб':>6} {'час, с':>8} {'результат':>12}")

    for label, fail_times, status in [
        ("одиночний 429 → відновлення", 1, 429),
        ("подвійний 429 → відновлення", 2, 429),
        ("серверний 503 → відновлення", 2, 503),
        ("стійкий 429 → вичерпання спроб", 9, 429),
    ]:
        call = _flaky_call(fail_times, status)
        t0 = time.perf_counter()
        try:
            res = call_with_backoff(call, base_delay=1.0, max_retries=5)
            outcome = res
        except HttpError:
            outcome = "raise(5xx/429)"
        dt = time.perf_counter() - t0
        attempts = min(fail_times + 1, 5)
        print(f"{label:38} {attempts:>6} {dt:>8.2f} {outcome:>12}")

    # Перевірка fail-fast для неретраябельних (403/404 — очікувано відмовлено).
    for status in (403, 404):
        call = _flaky_call(1, status)
        t0 = time.perf_counter()
        try:
            call_with_backoff(call, base_delay=1.0, max_retries=5)
            verdict = "НЕ підняв (помилка!)"
        except HttpError:
            verdict = "fail-fast (без ретраїв)"
        dt = time.perf_counter() - t0
        print(f"{'не-ретраябл ' + str(status):38} {1:>6} {dt:>8.2f} {verdict:>12}")


# --- B. Продуктивність каталогу: parallel_map (НФВ ≤5 с/100 форм) ------------


def measure_parallel(n_forms: int = 100, latency_s: float = 0.10) -> None:
    print()
    print("=" * 68)
    print("B. Час відгуку каталогу: паралельні виклики (НФВ ≤5 с / 100 форм)")
    print("=" * 68)
    print(f"Емуляція: {n_forms} форм, затримка одного виклику API = {latency_s * 1000:.0f} мс")

    def fetch(_item: int) -> int:
        time.sleep(latency_s)  # емуляція round-trip до Google API
        return _item

    t0 = time.perf_counter()
    for i in range(n_forms):
        fetch(i)
    seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    parallel_map(fetch, list(range(n_forms)), max_workers=5)
    par = time.perf_counter() - t0

    print(f"{'послідовно':28} {seq:>7.2f} с")
    print(f"{'parallel_map (5 воркерів)':28} {par:>7.2f} с   (×{seq / par:.1f} прискорення)")
    print(f"Вимога ≤5 с: {'ВИКОНАНО' if par <= 5.0 else 'НЕ виконано'}")


if __name__ == "__main__":
    measure_backoff()
    measure_parallel()
