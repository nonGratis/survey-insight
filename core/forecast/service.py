"""Оркестратор прогнозу: timeline → fit → delta-CI → пакування.

**Архітектура (P12 + P17):**
- Point estimate = `model.predict(t_future, *params)`, monotonic + floor.
- CI = `delta_method_ci(pcov)` + `cap_width` (half ≤ point). Без conformal,
  без NHPP, без multipliers — width відображає справжню pcov-uncertainty,
  обмежену фізично-розумною стелею.
- **Reliable-horizon clamp (P17):** прогноз НЕ екстраполюється на горизонт,
  довший за `HORIZON_SPAN_FACTOR × train_span`. Інакше fit early-burst даних
  (напр. 49 хв спостережень) проектується на 51 день → асимптота моделі →
  абсурд. Замість цього проектуємо лише на надійний горизонт (~3× span).

Continuous-time fit: модель навчається на парах `(t_i, i+1)` для кожної
відповіді. Future-grid адаптивна: щодоби для багатоденних горизонтів,
щогодини для sub-day (бо форми-події тривають годинами, не днями).

Legacy NHPP-симуляція + multipliers (P7/P10/P11) лишається у
`intervals.py`/`calibration.py` для відтворюваності research 03/10/11/13.
Не використовується в прод-flow.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from core.timeline import TimelineSeries

from .delta_ci import cap_width, delta_method_ci
from .models import models_for_n_points
from .priors import load_priors
from .selector import select_best_model
from .shape_classifier import classify_timeline
from .types import ForecastError, ForecastResult

DEFAULT_HORIZON_FRACTION = 0.25
MIN_TRAIN_POINTS = 5
MIN_DURATION_DAYS = 1.0 / 24.0  # 1 година — захист від span=0

# P17 reliable-horizon clamp: не екстраполюємо далі ніж 3× тривалості даних.
# Обґрунтування: fit на 49 хв даних не може передбачити 51 день — модель
# просто повертає свою асимптоту (capacity bound). Надійний горизонт
# пропорційний обсягу спостережень.
HORIZON_SPAN_FACTOR = 3.0
MIN_HORIZON_DAYS = 1.0 / 24.0  # завжди проектуємо хоча б 1 годину вперед

# cap_width: CI half-width ≤ point (max_relative=2.0 бо cap на width=2×half).
# Захист від ill-conditioned pcov (gompertz на multi-wave формах де delta
# вибухає). На добре-обумовлених fit-ах не впливає.
DELTA_CI_MAX_RELATIVE = 2.0  # width ≤ 2×point  ⟺  half ≤ point
DELTA_CI_MAX_ABSOLUTE = 1e9  # практично без абсолютної стелі — relative керує
DELTA_CI_MIN_ABSOLUTE = 10.0  # мінімальна width на крихітних формах


def forecast_responses(
    timeline: TimelineSeries,
    target: int | None = None,
    horizon_fraction: float = DEFAULT_HORIZON_FRACTION,
    horizon_until: pd.Timestamp | None = None,
    use_priors: bool = False,
    random_seed: int = 42,  # noqa: ARG001 — no-op, kept для backward compat
) -> ForecastResult:
    """Прогнозувати cumulative responses до надійного горизонту.

    Algorithm:
    1. Валідація: ≥ MIN_TRAIN_POINTS (5) timestamps.
    2. Continuous-time t_train, y_train = 1..N.
    3. AICc-селектор обирає кращу модель.
    4. Reliable-horizon clamp: horizon = min(requested, 3× train_span).
    5. Adaptive future-grid (hourly для sub-day, daily інакше).
    6. Point = model.predict, monotonic + floor на last_observed.
    7. CI = delta-method (Bates-Watts) + cap_width (half ≤ point).

    Args:
        timeline: побудована `build_timeline_from_timestamps`.
        target: цільова кількість відповідей; впливає на bounds моделі.
        horizon_fraction: частка тривалості як прогноз (якщо `horizon_until` None).
        horizon_until: бажаний кінцевий час; буде clamp'нутий до надійного.
        use_priors: вмикає emp. Bayes priors (P9) на bounds (opt-in).
        random_seed: no-op (deterministic flow). Збережено для backward compat.

    Raises:
        ForecastError: якщо < MIN_TRAIN_POINTS точок або жодна модель не зійшлася.
    """
    if timeline.timestamps.empty:
        raise ForecastError("Немає даних: timeline порожня.")

    n_points = len(timeline.timestamps)
    if n_points < MIN_TRAIN_POINTS:
        need = MIN_TRAIN_POINTS - n_points
        raise ForecastError(
            f"Замало точок для прогнозу: маємо {n_points}, мінімум {MIN_TRAIN_POINTS} "
            f"(потрібно ще {need})."
        )

    timestamps = pd.to_datetime(timeline.timestamps).sort_values().reset_index(drop=True)
    first_ts = timestamps.iloc[0].to_pydatetime()
    last_ts = timestamps.iloc[-1].to_pydatetime()

    t_train = _to_days_from(timestamps, first_ts)
    y_train = np.arange(1, n_points + 1, dtype=float)
    last_observed = n_points

    train_span_days = max(float(t_train[-1]), MIN_DURATION_DAYS)

    # Requested horizon end (timestamp).
    if horizon_until is not None:
        requested_end = pd.Timestamp(horizon_until).to_pydatetime()
    else:
        requested_end = last_ts + timedelta(days=train_span_days * horizon_fraction)

    # P17: clamp до надійного горизонту (≤ 3× train_span від last_ts).
    reliable_end = last_ts + timedelta(days=HORIZON_SPAN_FACTOR * train_span_days)
    min_end = last_ts + timedelta(days=MIN_HORIZON_DAYS)
    horizon_end = min(requested_end, reliable_end)
    horizon_end = max(horizon_end, min_end)

    models = models_for_n_points(n_points)
    priors = load_priors() if use_priors else None
    shape = classify_timeline(timeline.timestamps) if (use_priors and priors) else None

    fitted = select_best_model(
        t_train, y_train, target=target, models=models, priors=priors, shape=shape
    )

    # Granularity за train_span: sub-day форми (події) → погодинна сітка,
    # багатоденні → щоденна (історична поведінка).
    sub_daily = train_span_days < 2.0
    future_dates = _build_future_grid(last_ts, horizon_end, sub_daily=sub_daily)
    t_future = _to_days_from(pd.Series(future_dates), first_ts)

    # Point: deterministic model.predict + monotonic + floor на last_observed.
    point_arr = fitted.model.predict(t_future, *fitted.params)
    point_arr = np.maximum.accumulate(np.maximum(point_arr, float(last_observed)))

    # CI: delta-method + cap_width (half ≤ point).
    try:
        ci_lower_arr, ci_upper_arr = delta_method_ci(fitted, t_future, n_train=n_points)
    except ValueError:
        # pcov degenerate → degenerate CI = point ± 0. cap_width підніме до floor.
        ci_lower_arr = point_arr.copy()
        ci_upper_arr = point_arr.copy()
    ci_lower_arr, ci_upper_arr = cap_width(
        point_arr,
        ci_lower_arr,
        ci_upper_arr,
        max_relative=DELTA_CI_MAX_RELATIVE,
        max_absolute=DELTA_CI_MAX_ABSOLUTE,
        min_absolute=DELTA_CI_MIN_ABSOLUTE,
    )

    # Cumulative floor + point ∈ [lower, upper] guards.
    ci_lower_arr = np.maximum(ci_lower_arr, float(last_observed))
    ci_lower_arr = np.minimum(ci_lower_arr, point_arr)
    ci_upper_arr = np.maximum(ci_upper_arr, point_arr)

    future_cum = pd.Series(point_arr, index=future_dates, name="future_cum")
    ci_lower = pd.Series(ci_lower_arr, index=future_dates, name="ci_lower")
    ci_upper = pd.Series(ci_upper_arr, index=future_dates, name="ci_upper")

    return ForecastResult(
        model=fitted.model.name,
        aicc=fitted.aicc,
        future_dates=future_dates,
        future_cum=future_cum,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        final_estimate=int(round(point_arr[-1])),
        final_ci=(int(round(ci_lower_arr[-1])), int(round(ci_upper_arr[-1]))),
        rmse=fitted.rmse,
        r_squared=fitted.r_squared,
    )


def _build_future_grid(last_ts, horizon_end, sub_daily: bool) -> pd.DatetimeIndex:
    """Future-grid: погодинна для sub-day форм (події), щоденна для довгих.

    Granularity визначається тривалістю ДАНИХ (`sub_daily`), а не горизонту:
    форма-подія на кілька годин потребує погодинної сітки (daily = мінімум
    +1 день, уже за надійним горизонтом), а багатоденна форма — щоденної.
    """
    total_seconds = (pd.Timestamp(horizon_end) - pd.Timestamp(last_ts)).total_seconds()
    total_seconds = max(total_seconds, 3600.0)  # ≥ 1 година

    if sub_daily:
        # Погодинна сітка від last_ts + 1 год.
        periods = max(int(np.ceil(total_seconds / 3600.0)), 1)
        start = pd.Timestamp(last_ts) + timedelta(hours=1)
        return pd.date_range(start=start, periods=periods, freq="h")

    # Daily grid від наступного календарного дня (історична поведінка).
    start = pd.Timestamp(last_ts).normalize() + timedelta(days=1)
    periods = max(int(np.ceil(total_seconds / 86400.0)), 1)
    return pd.date_range(start=start, periods=periods, freq="D")


def _to_days_from(ts: pd.Series, anchor) -> np.ndarray:
    """Перевести Series datetime'ів у float-дні від anchor."""
    deltas = pd.to_datetime(ts) - pd.Timestamp(anchor)
    return (deltas.dt.total_seconds() / 86400.0).to_numpy(dtype=float)
