"""Оркестратор прогнозу: валідація, фіт, CI, пакування результату."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from core.timeline import TimelineSeries

from .intervals import bootstrap_ci
from .metrics import r_squared, rmse
from .models import asymptotic_exp, fit_asymptotic_exp
from .types import ForecastError, ForecastResult

DEFAULT_N_BOOTSTRAP = 200
DEFAULT_RANDOM_SEED = 42
DEFAULT_HORIZON_FRACTION = 0.25
MIN_HORIZON_DAYS = 1


def asymptotic_exp_forecast(
    timeline: TimelineSeries,
    horizon_fraction: float = DEFAULT_HORIZON_FRACTION,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> ForecastResult:
    """Спрогнозувати cumulative на ~`horizon_fraction` від тривалості опитування."""
    if timeline.daily_counts.empty:
        raise ForecastError("Немає даних: timeline порожня.")
    if len(timeline.daily_counts) < 3:
        raise ForecastError("Замало точок для прогнозу: потрібно мінімум 3 дні з даними.")

    first_known = timeline.daily_counts.index[0].to_pydatetime()
    last_known_day = timeline.daily_counts.index[-1].to_pydatetime()

    duration_days = (last_known_day.date() - first_known.date()).days
    horizon_days = max(int(round(duration_days * horizon_fraction)), MIN_HORIZON_DAYS)

    cum_array = timeline.cumulative.to_numpy(dtype=float)
    n_days = len(cum_array)
    t_train = np.arange(n_days, dtype=float)

    a, b, c = fit_asymptotic_exp(t_train, cum_array)
    fitted_cum = asymptotic_exp(t_train, a, b, c)

    future_dates = pd.date_range(
        start=last_known_day + timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )
    t_future = np.array(
        [(d.to_pydatetime() - first_known).days for d in future_dates],
        dtype=float,
    )
    future_cum_arr = asymptotic_exp(t_future, a, b, c)

    rng = np.random.default_rng(random_seed)
    ci_lower_arr, ci_upper_arr = bootstrap_ci(timeline.daily_counts, t_future, n_bootstrap, rng)

    future_cum = pd.Series(future_cum_arr, index=future_dates, name="future_cum")
    ci_lower = pd.Series(ci_lower_arr, index=future_dates, name="ci_lower")
    ci_upper = pd.Series(ci_upper_arr, index=future_dates, name="ci_upper")

    return ForecastResult(
        model="asymptotic_exp",
        future_dates=future_dates,
        future_cum=future_cum,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        final_estimate=int(round(future_cum_arr[-1])),
        final_ci=(int(round(ci_lower_arr[-1])), int(round(ci_upper_arr[-1]))),
        rmse=rmse(cum_array, fitted_cum),
        r_squared=r_squared(cum_array, fitted_cum),
    )
