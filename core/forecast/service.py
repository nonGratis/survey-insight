"""Оркестратор прогнозу: timeline → fit → delta-CI → conformal → пакування.

**γ-architecture (P12 + P14):** point estimate = `model.predict(t_future, *params)`.
CI = `delta_method_ci(pcov)` + `cap_width` + `apply_conformal_adjustment`.
Жодних NHPP simulation, multipliers або post-hoc widening.

Continuous-time fit: модель навчається на парах `(t_i, i+1)` для кожної
окремої відповіді, де `t_i` — час у частках доби від першого спостереження.
Жодної агрегації по добі: 47 відповідей за 15 хв = 47 точок з малими
інтервалами, працює так само як 47 відповідей за 14 днів.

Legacy NHPP-симуляція + multipliers (P7/P10/P11) лишається у
`core/forecast/intervals.py` та `calibration.py` для відтворюваності
research-прогонів 03/10/11/13. Не використовується в прод-flow.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from core.timeline import TimelineSeries

from .conformal import apply_conformal_adjustment
from .delta_ci import cap_width, delta_method_ci
from .models import models_for_n_points
from .priors import load_priors
from .selector import select_best_model
from .shape_classifier import classify_timeline
from .types import ForecastError, ForecastResult

DEFAULT_HORIZON_FRACTION = 0.25
MIN_HORIZON_DAYS = 1
MIN_TRAIN_POINTS = 5
MIN_DURATION_DAYS = 1.0 / 24.0

# cap_width захист від ill-conditioned pcov.
DELTA_CI_MAX_RELATIVE = 5.0
DELTA_CI_MAX_ABSOLUTE = 5000.0
DELTA_CI_MIN_ABSOLUTE = 20.0


def forecast_responses(
    timeline: TimelineSeries,
    target: int | None = None,
    horizon_fraction: float = DEFAULT_HORIZON_FRACTION,
    horizon_until: pd.Timestamp | None = None,
    use_priors: bool = False,
    apply_conformal: bool = True,
    random_seed: int = 42,  # noqa: ARG001 — no-op, kept для backward compat
) -> ForecastResult:
    """Прогнозувати cumulative responses до `horizon_until` або +25% від тривалості.

    Algorithm:
    1. Валідація: ≥ MIN_TRAIN_POINTS (5) timestamps.
    2. Continuous-time t_train, y_train = 1..N.
    3. AICc-селектор обирає кращу модель (Logistic/Gompertz/AsympExp).
    4. Future-grid щодоби, від наступного дня після last_ts.
    5. Point = model.predict, monotonic + floor на last_observed.
    6. CI = delta-method (Bates-Watts) + cap_width + (optional) conformal.

    Args:
        timeline: побудована `build_timeline_from_timestamps`.
        target: цільова кількість відповідей; впливає на bounds моделі.
        horizon_fraction: частка тривалості як прогноз (якщо `horizon_until` None).
        horizon_until: явний кінцевий час горизонту (alt to `horizon_fraction`).
        use_priors: вмикає emp. Bayes priors (P9) на bounds (opt-in).
        apply_conformal: True (default) — застосовує conformal adjustment
            на CI. False — raw delta-method CI (для калібровочних прогонів).
        random_seed: no-op у γ-flow (deterministic). Збережено для backward compat.

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

    duration_days = max((last_ts - first_ts).total_seconds() / 86400.0, MIN_DURATION_DAYS)
    if horizon_until is not None:
        until_ts = pd.Timestamp(horizon_until)
        horizon_days = max(
            int(np.ceil((until_ts - pd.Timestamp(last_ts)).total_seconds() / 86400.0)),
            MIN_HORIZON_DAYS,
        )
    else:
        horizon_days = max(int(np.ceil(duration_days * horizon_fraction)), MIN_HORIZON_DAYS)

    models = models_for_n_points(n_points)
    priors = load_priors() if use_priors else None
    shape = classify_timeline(timeline.timestamps) if (use_priors and priors) else None

    fitted = select_best_model(
        t_train, y_train, target=target, models=models, priors=priors, shape=shape
    )

    last_known_day = pd.Timestamp(last_ts.date())
    future_dates = pd.date_range(
        start=last_known_day + timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )
    t_future = _to_days_from(pd.Series(future_dates), first_ts)

    # Point: deterministic model.predict + monotonic + floor на last_observed.
    point_arr = fitted.model.predict(t_future, *fitted.params)
    point_arr = np.maximum.accumulate(np.maximum(point_arr, float(last_observed)))

    # CI: delta-method + cap_width (захист від ill-conditioned pcov).
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

    # P14 conformal calibration: empirical residual quantile per-cell.
    if apply_conformal:
        last_observed_day = (last_ts - first_ts).total_seconds() / 86400.0
        horizon_days_arr = t_future - last_observed_day
        ci_lower_arr, ci_upper_arr = apply_conformal_adjustment(
            point_arr,
            ci_lower_arr,
            ci_upper_arr,
            n_train=n_points,
            horizon_days_arr=horizon_days_arr,
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


def _to_days_from(ts: pd.Series, anchor) -> np.ndarray:
    """Перевести Series datetime'ів у float-дні від anchor."""
    deltas = pd.to_datetime(ts) - pd.Timestamp(anchor)
    return (deltas.dt.total_seconds() / 86400.0).to_numpy(dtype=float)
