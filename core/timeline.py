"""Часові серії відповідей: timestamp → daily counts → cumulative.

Працює з DataFrame, який повертає `core.sheets_api.fetch_responses`.
Перша колонка Forms-linked Sheet'у — це Timestamp подання (Google
Forms так записує завжди). Парсимо її, групуємо по доб, рахуємо
cumulative — це є вхід для прогнозних моделей у `core.forecast`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class TimelineSeries:
    """Снапшот часової динаміки відповідей форми.

    Attributes:
        timestamps: pd.Series[datetime64] усіх submit-timestamps,
            відсортовані за зростанням. Може містити дублікати (одна доба,
            різні секунди — це нормально).
        daily_counts: pd.Series, індекс = date, значення = к-сть відповідей
            тієї доби. Заповнено для **усіх** діб у діапазоні [first..last]
            включно з нулями, щоб модель прогнозу бачила "тихі" дні теж.
        cumulative: pd.Series, індекс = date, значення = cum-sum daily_counts.
    """

    timestamps: pd.Series
    daily_counts: pd.Series
    cumulative: pd.Series


def build_timeline(df: pd.DataFrame) -> TimelineSeries:
    """Побудувати TimelineSeries з DataFrame відповідей.

    Очікує, що перша колонка `df` містить submit-timestamps у форматі,
    придатному для `pd.to_datetime` (Google Forms-linked Sheets завжди
    кладуть Timestamp туди, локалізована назва не критична).

    Невпарсовані рядки мовчки відкидаються — це може бути header-рядок
    або порожні значення наприкінці.

    Args:
        df: результат `fetch_responses`.

    Returns:
        TimelineSeries. Якщо немає валідних timestamps — усі поля порожні.
    """
    if df.empty or df.shape[1] == 0:
        return _empty_timeline()

    first_col = df.iloc[:, 0]
    parsed = pd.to_datetime(first_col, errors="coerce", dayfirst=True)
    parsed = parsed.dropna().sort_values().reset_index(drop=True)

    if parsed.empty:
        log.warning(
            "timeline_no_valid_timestamps",
            extra={"rows": len(df), "col_name": str(df.columns[0])},
        )
        return _empty_timeline()

    # Daily counts: групуємо по даті, потім reindex на повний діапазон,
    # щоб тихі дні не зникали з серії.
    dates = parsed.dt.normalize()
    counts_by_day = dates.value_counts().sort_index()
    full_range = pd.date_range(
        start=counts_by_day.index.min(),
        end=counts_by_day.index.max(),
        freq="D",
    )
    daily_counts = counts_by_day.reindex(full_range, fill_value=0)
    daily_counts.index.name = "date"
    daily_counts.name = "count"

    cumulative = daily_counts.cumsum()
    cumulative.name = "cumulative"

    return TimelineSeries(
        timestamps=parsed,
        daily_counts=daily_counts,
        cumulative=cumulative,
    )


def _empty_timeline() -> TimelineSeries:
    """Порожня TimelineSeries для випадків df.empty або no valid timestamps."""
    empty_ts = pd.Series([], dtype="datetime64[ns]")
    empty_daily = pd.Series([], dtype="int64", name="count")
    empty_daily.index.name = "date"
    empty_cum = pd.Series([], dtype="int64", name="cumulative")
    empty_cum.index.name = "date"
    return TimelineSeries(
        timestamps=empty_ts,
        daily_counts=empty_daily,
        cumulative=empty_cum,
    )
