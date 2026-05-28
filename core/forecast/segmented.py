"""CP-aware wrapper around forecast_responses.

research/04_diagnostics показав 87% Ljung-Box rejection rate — single-curve
fit систематично пропускає temporal structure (хвилі агітації). Цей wrapper
вирішує проблему через PELT changepoint detection:

  1. Виявляє хвилі (CP) у rate-серії повного timeline.
  2. Якщо знайшов хоча б одну → бере timestamps ПІСЛЯ останнього CP як
     тренувальний subset.
  3. Викликає існуючий forecast_responses на цьому subset'і.
  4. Повертає (ForecastResult, list[Changepoint]) — щоб UI міг рендерити
     маркери CP на графіку.

Принцип: ми не торкаємо core/forecast/service.py. Single responsibility —
службовий шар фітить timeline, який йому дають. Segmentation logic окремо.

Якщо CP не знайдено або post-CP сегмент закороткий — fallback на повний
timeline. Список CP завжди повертаємо (навіть при fallback) — для візуалізації.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.detection import (
    Changepoint,
    InsufficientDataError,
    detect_changepoints,
    median_smooth,
    to_rate_series,
)
from core.timeline import TimelineSeries, build_timeline_from_timestamps

from .service import MIN_TRAIN_POINTS, forecast_responses
from .types import ForecastResult

DEFAULT_CP_PENALTY = 10.0
DEFAULT_CP_MIN_SEGMENT = 5
DEFAULT_RATE_FREQ = "1h"
DEFAULT_SMOOTH_WINDOW = 3


def forecast_with_segmentation(
    timeline: TimelineSeries,
    *,
    target: int | None = None,
    horizon_until: pd.Timestamp | None = None,
    cp_penalty: float = DEFAULT_CP_PENALTY,
    cp_min_segment: int = DEFAULT_CP_MIN_SEGMENT,
    rate_freq: str = DEFAULT_RATE_FREQ,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    auto_segment: bool = True,
    **forecast_kwargs: Any,
) -> tuple[ForecastResult, list[Changepoint]]:
    """Прогноз з автоматичною сегментацією на знайдених CP.

    Args:
        timeline: повний timeline з усіма timestamps.
        target: цільова кількість відповідей (soft prior).
        horizon_until: explicit горизонт прогнозу.
        cp_penalty: штраф для PELT (більше — менше CP).
        cp_min_segment: мінімальний розмір сегменту в bucket'ах.
        rate_freq: pandas-офсет для resample у rate-серію ("1h", "15min", ...).
        smooth_window: вікно median-фільтра проти разових spike'ів.
        auto_segment: вимкнути CP detection, тренувати на повному timeline.
        **forecast_kwargs: інші аргументи forecast_responses (n_simulations, ...).

    Returns:
        (forecast_result, changepoints). Список CP завжди валідний (може
        бути порожній), навіть коли fallback на повний timeline.
    """
    # Якщо segmentation вимкнено або серія закоротка для CP detection —
    # одразу повний timeline + порожній CP-список.
    if not auto_segment or len(timeline.timestamps) < 2 * cp_min_segment:
        result = forecast_responses(
            timeline,
            target=target,
            horizon_until=horizon_until,
            **forecast_kwargs,
        )
        return result, []

    # CP detection: timestamps → rate → smooth → PELT.
    try:
        rate = to_rate_series(timeline.timestamps, freq=rate_freq)
    except InsufficientDataError:
        result = forecast_responses(
            timeline,
            target=target,
            horizon_until=horizon_until,
            **forecast_kwargs,
        )
        return result, []

    smoothed = median_smooth(rate, window=smooth_window)
    changepoints = detect_changepoints(
        smoothed,
        penalty=cp_penalty,
        min_segment=cp_min_segment,
    )

    if not changepoints:
        result = forecast_responses(
            timeline,
            target=target,
            horizon_until=horizon_until,
            **forecast_kwargs,
        )
        return result, []

    # Беремо timestamps після ОСТАННЬОГО CP як training subset.
    last_cp_ts = changepoints[-1].timestamp
    ts_full = pd.to_datetime(timeline.timestamps).sort_values()
    post_cp_ts = [t.to_pydatetime() for t in ts_full if t >= last_cp_ts]

    if len(post_cp_ts) < MIN_TRAIN_POINTS:
        # Останній сегмент закороткий — fallback, але CP-список повертаємо.
        result = forecast_responses(
            timeline,
            target=target,
            horizon_until=horizon_until,
            **forecast_kwargs,
        )
        return result, changepoints

    seg_timeline = build_timeline_from_timestamps(post_cp_ts)
    result = forecast_responses(
        seg_timeline,
        target=target,
        horizon_until=horizon_until,
        **forecast_kwargs,
    )
    return result, changepoints
