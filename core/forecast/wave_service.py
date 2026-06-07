"""Current-wave forecast — prod entry point.

Питання продукту: «якщо без нової агітації, скільки набере ПОТОЧНА хвиля?»
(не «фінал форми» — майбутня агітація це рішення оператора, не подія).

Замінює `forecast_with_segmentation` у проді. Стара логіка фітила одну
saturation-криву на весь багатохвильовий timeline → абсурдні числа на
складних формах (47→±1784, 7433→±52333). Тут:

  1. CUSUM-детектор (per-type) знаходить старти хвиль агітації.
  2. Поточна хвиля = відповіді від останнього старту.
  3. Within-wave saturation fit, проєкція на надійний горизонт
     (wave_start + HORIZON_SPAN_FACTOR × wave_span), Mondrian-conformal CI.
  4. final_estimate = посадка хвилі; CI cap≤2× (ніколи не абсурдний).

Валідація (benchmark 18_/19_, multiseed): 15% MAPE (vs prod 20%, naive 25%),
87% coverage, detection penalty ≈ 0, стабільно на 5 сідах.

Fallback: якщо поточна хвиля < MIN_TRAIN_POINTS (5) → forecast_responses
(P17-версія: delta+cap_width, half-width ≤ 2× point за побудовою, тобто
структурно НЕ дає абсурду — доведено у tests/test_wave_service.py).
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from core.detection import Changepoint
from core.timeline import TimelineSeries

from .service import HORIZON_SPAN_FACTOR, MIN_HORIZON_DAYS, MIN_TRAIN_POINTS, forecast_responses
from .types import ForecastResult
from .wave_detector import detect_wave_starts
from .wave_estimator import wave_forecast_curve

_GRID_POINTS = 24  # точок у future-grid (гладка крива/CI band)


def _wave_changepoints(wave_starts: list, timestamps: pd.Series) -> list[Changepoint]:
    """Старти хвиль агітації (окрім найпершої) → маркери для графіка.

    Перша хвиля = початок форми, не «ре-агітація» — її не маркуємо. Решта
    стартів — це додаткові хвилі агітації (помаранчеві лінії в UI).
    """
    cps: list[Changepoint] = []
    for w in wave_starts[1:]:
        idx = int((timestamps < w.timestamp).sum())
        cps.append(Changepoint(timestamp=pd.Timestamp(w.timestamp), index=idx))
    return cps


def forecast_current_wave(
    timeline: TimelineSeries,
    *,
    form_type: str | None = None,
    horizon_until: pd.Timestamp | None = None,
) -> tuple[ForecastResult, list[Changepoint]]:
    """Прогноз посадки ПОТОЧНОЇ хвилі (prod entry).

    Args:
        timeline: timeline відповідей (повний або subset зі слайдера UI).
        form_type: категорія форми (per-type пороги детектора). None → дефолт.
        horizon_until: використовується ЛИШЕ fallback-шляхом; wave-шлях завжди
            проєктує на надійний горизонт 3× wave_span.

    Returns:
        (ForecastResult, changepoints). ForecastResult сумісний з UI/чартом
        (та сама форма що у forecast_responses): final_estimate = посадка хвилі.
        changepoints = старти хвиль агітації (окрім першої) для маркерів.
    """
    timestamps = pd.Series(pd.to_datetime(timeline.timestamps)).sort_values().reset_index(drop=True)
    if timestamps.empty:
        return forecast_responses(timeline, horizon_until=horizon_until), []

    waves = detect_wave_starts(timestamps, form_type=form_type, test_skip=0)
    changepoints = _wave_changepoints(waves, timestamps)

    ws_ts = pd.Timestamp(waves[-1].timestamp) if waves else timestamps.iloc[0]
    pre_count = int((timestamps < ws_ts).sum())
    current_wave = timestamps[timestamps >= ws_ts].reset_index(drop=True)

    # Fallback: поточна хвиля закоротка для within-wave fit → P17 capped forecast.
    if len(current_wave) < MIN_TRAIN_POINTS:
        return forecast_responses(timeline, horizon_until=horizon_until), changepoints

    # Надійний горизонт: 3× span поточної хвилі (без екстраполяції в дні).
    wave_span_h = (current_wave.iloc[-1] - ws_ts).total_seconds() / 3600.0
    min_ahead_h = MIN_HORIZON_DAYS * 24.0
    reliable_end_h = max(HORIZON_SPAN_FACTOR * wave_span_h, wave_span_h + min_ahead_h)

    # Future-grid у годинах від старту хвилі: від (трохи після) останнього факту.
    start_h = wave_span_h + (reliable_end_h - wave_span_h) / _GRID_POINTS
    t_future_h = np.linspace(start_h, reliable_end_h, _GRID_POINTS)

    curve = wave_forecast_curve(current_wave, t_future_h, form_type=form_type)

    # Години → дати; cumulative offset на pre_count (нумерація в межах timeline).
    future_dates = pd.DatetimeIndex([ws_ts + timedelta(hours=float(h)) for h in t_future_h])
    future_cum = pd.Series(curve.cum + pre_count, index=future_dates, name="future_cum")
    ci_lower = pd.Series(curve.lower + pre_count, index=future_dates, name="ci_lower")
    ci_upper = pd.Series(curve.upper + pre_count, index=future_dates, name="ci_upper")

    result = ForecastResult(
        model=curve.model_name,
        aicc=curve.aicc,
        future_dates=future_dates,
        future_cum=future_cum,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        final_estimate=int(round(float(future_cum.iloc[-1]))),
        final_ci=(int(round(float(ci_lower.iloc[-1]))), int(round(float(ci_upper.iloc[-1])))),
        rmse=curve.rmse,
        r_squared=curve.r_squared,
    )
    return result, changepoints
