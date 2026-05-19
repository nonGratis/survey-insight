"""Оркестратор прогнозу: валідація → селектор → NHPP CI → пакування."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from core.timeline import TimelineSeries

from .intervals import nhpp_prediction_interval
from .selector import select_best_model
from .types import ForecastError, ForecastResult

DEFAULT_N_SIMULATIONS = 2000
DEFAULT_RANDOM_SEED = 42
DEFAULT_HORIZON_FRACTION = 0.25
MIN_HORIZON_DAYS = 1
MIN_TRAIN_DAYS = 3


def forecast_responses(
    timeline: TimelineSeries,
    target: int | None = None,
    horizon_fraction: float = DEFAULT_HORIZON_FRACTION,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> ForecastResult:
    """Спрогнозувати cumulative на ~`horizon_fraction` від тривалості опитування.

    Алгоритм:
    1. Валідація timeline (мінімум 3 точки).
    2. Підрахунок горизонту в днях.
    3. Вибір кращої параметричної моделі (Logistic / Gompertz / AsymptoticExp)
       селектором за AICc; `target` — soft prior на асимптоту.
    4. NHPP-симуляція майбутніх Poisson-приходів для 95% CI з
       гарантованою монотонністю і floor'ом на last_observed.
    5. Точкова оцінка — predict кривої на майбутнє з floor'ом на last_observed
       (бо для S-кривих в ідеальній теорії, але cumsum-факт може ушкоджуватись).

    Args:
        timeline: побудована `build_timeline_from_timestamps`.
        target: цільова кількість відповідей; впливає на bounds моделі.
        horizon_fraction: частка тривалості, яку додаємо як прогноз.
        n_simulations: кількість Poisson-траєкторій для CI.
        random_seed: для відтворюваності NHPP-симуляції.

    Raises:
        ForecastError: якщо замало даних або всі моделі не зійшлися.
    """
    if timeline.daily_counts.empty:
        raise ForecastError("Немає даних: timeline порожня.")
    if len(timeline.daily_counts) < MIN_TRAIN_DAYS:
        raise ForecastError(
            f"Замало точок для прогнозу: потрібно мінімум {MIN_TRAIN_DAYS} дні з даними."
        )

    first_known = timeline.daily_counts.index[0].to_pydatetime()
    last_known_day = timeline.daily_counts.index[-1].to_pydatetime()
    duration_days = (last_known_day.date() - first_known.date()).days
    horizon_days = max(int(round(duration_days * horizon_fraction)), MIN_HORIZON_DAYS)

    cum_array = timeline.cumulative.to_numpy(dtype=float)
    last_observed = int(cum_array[-1])
    t_train = np.arange(len(cum_array), dtype=float)

    fitted = select_best_model(t_train, cum_array, target=target)

    future_dates = pd.date_range(
        start=last_known_day + timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )
    t_future = np.array(
        [(d.to_pydatetime() - first_known).days for d in future_dates],
        dtype=float,
    )

    rng = np.random.default_rng(random_seed)
    model_mean, ci_lower_arr, ci_upper_arr = nhpp_prediction_interval(
        fitted, t_future, last_observed=last_observed, n_sims=n_simulations, rng=rng
    )

    # Точкова оцінка — model.predict з floor'ом на last_observed (cumulative
    # не може зменшуватись; для коротких горизонтів S-крива може на 1-2
    # відсотки занижувати поточний рівень).
    future_cum_arr = np.maximum.accumulate(np.maximum(model_mean, float(last_observed)))

    future_cum = pd.Series(future_cum_arr, index=future_dates, name="future_cum")
    ci_lower = pd.Series(ci_lower_arr, index=future_dates, name="ci_lower")
    ci_upper = pd.Series(ci_upper_arr, index=future_dates, name="ci_upper")

    return ForecastResult(
        model=fitted.model.name,
        aicc=fitted.aicc,
        future_dates=future_dates,
        future_cum=future_cum,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        final_estimate=int(round(future_cum_arr[-1])),
        final_ci=(int(round(ci_lower_arr[-1])), int(round(ci_upper_arr[-1]))),
        rmse=fitted.rmse,
        r_squared=fitted.r_squared,
    )
