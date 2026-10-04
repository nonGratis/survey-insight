"""Черга довантаження деталей каталогу: що просити в API на наступному кроці.

Сторінка каталогу бере деталі форм порціями. Тут лише рішення «що далі», без Streamlit і
мережі. Спершу нові форми в порядку каталогу, потім ті, що чекають повтору. Рядок, який
притримав ліміт (наш обмежувач у API чи 429 від Google) або який не встиг (timeout,
api_error), сторінка повторює сама через паузу, поки не вичерпає спроби; лише тоді його
показує кнопка «Повторити проблемні рядки».
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

RETRYABLE_STATUSES = frozenset({"timeout", "api_error", "rate_limited"})
# Обмежувач API рахує виклики за останні 60 с, і місце в ньому звільняється поступово.
RATE_LIMITED_RETRY_SECONDS = 15.0
OTHER_RETRY_SECONDS = 5.0
MAX_ATTEMPTS = 8


@dataclass(frozen=True)
class Retry:
    """Автоматичний повтор рядка: скільки спроб уже було і коли можна наступну."""

    attempts: int
    not_before: float


@dataclass(frozen=True)
class EnrichBatch:
    """Одна порція для API. Без деталей форми — коли вони вже є і бракує лише відповідей."""

    form_ids: list[str]
    include_summary: bool


def next_form_batches(
    form_ids: Sequence[str],
    *,
    done: Collection[str],
    retries: Mapping[str, Retry],
    have_summary: Collection[str],
    now: float,
    batch_size: int,
    max_batches: int,
) -> list[EnrichBatch]:
    """Порції форм для наступного кроку: спершу нові, потім ті, кому настав час повтору.

    ``done`` — форми, для яких уже є результат; ``retries`` — ті з них, що чекають повтору;
    ``have_summary`` — форми, деталі яких уже прийшли: їм повторюємо лише відповіді і не
    витрачаємо на них ліміт читань.
    """
    due = [
        form_id for form_id in form_ids if form_id in retries and retries[form_id].not_before <= now
    ]
    full = [form_id for form_id in form_ids if form_id not in done]
    full += [form_id for form_id in due if form_id not in have_summary]
    responses_only = [form_id for form_id in due if form_id in have_summary]
    batches = [EnrichBatch(chunk, include_summary=True) for chunk in _chunks(full, batch_size)]
    batches += [
        EnrichBatch(chunk, include_summary=False) for chunk in _chunks(responses_only, batch_size)
    ]
    return batches[:max_batches]


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def schedule_retry(status: str, previous: Retry | None, now: float) -> Retry | None:
    """Наступна спроба для рядка, що отримав ``status``, або None, якщо повтору не буде."""
    if status not in RETRYABLE_STATUSES:
        return None
    attempts = (previous.attempts if previous else 0) + 1
    if attempts >= MAX_ATTEMPTS:
        return None
    delay = RATE_LIMITED_RETRY_SECONDS if status == "rate_limited" else OTHER_RETRY_SECONDS
    return Retry(attempts=attempts, not_before=now + delay)


def is_loading(
    form_ids: Sequence[str], *, done: Collection[str], retries: Mapping[str, Retry]
) -> bool:
    """Чи лишилось що довантажувати: нові форми або рядки, що чекають повтору."""
    return any(form_id not in done or form_id in retries for form_id in form_ids)
