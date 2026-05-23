"""Оркестратор прогнозу: валідація → селектор → NHPP CI → пакування.

**Continuous-time fit**: модель навчається на парах
`(t_i, i+1)` для кожної окремої відповіді, де `t_i` — це час
у частках доби від першого спостереження (float). Жодної агрегації
по добі: 47 відповідей за 15 хвилин = 47 точок з малими інтервалами,
працює так само добре, як 47 відповідей за 14 днів.

Це прибирає обмеження "потрібно мінімум 3 дні даних" — тепер працює
з 5 точками будь-якого span'у.
"""

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
MIN_TRAIN_POINTS = 5  # curve_fit потребує ≥3 для 3-парам моделі, AICc — ≥5
MIN_DURATION_DAYS = 1.0 / 24.0  # 1 година: захист від span=0 (усі timestamps однакові)


def forecast_responses(
    timeline: TimelineSeries,
    target: int | None = None,
    horizon_fraction: float = DEFAULT_HORIZON_FRACTION,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> ForecastResult:
    """Спрогнозувати cumulative на ~`horizon_fraction` від тривалості опитування.

    Алгоритм:
    1. Валідація: мінімум `MIN_TRAIN_POINTS` (5) timestamps.
    2. Continuous-time t_train: float-дні від першого timestamp'у для кожної
       відповіді; y_train = 1..N (per-response cumulative).
    3. Selector обирає кращу модель за AICc; `target` — soft prior на K.
    4. Horizon: `max(duration_days * fraction, MIN_HORIZON_DAYS)`. Future-grid —
       щодоби, починаючи з наступного дня після останнього спостереження.
    5. NHPP-симуляція майбутніх Poisson-приходів для 95% CI з гарантованою
       монотонністю і floor'ом на N (last_observed = кількість відповідей).
    6. Точкова оцінка — predict з floor'ом на N + cumulative-monotonic.

    Args:
        timeline: побудована `build_timeline_from_timestamps`.
        target: цільова кількість відповідей; впливає на bounds моделі.
        horizon_fraction: частка тривалості, яку додаємо як прогноз.
        n_simulations: кількість Poisson-траєкторій для CI.
        random_seed: для відтворюваності NHPP-симуляції.

    Raises:
        ForecastError: якщо <`MIN_TRAIN_POINTS` точок або всі моделі не зійшлися.
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

    # Continuous-time training data: per-response.
    t_train = _to_days_from(timestamps, first_ts)
    y_train = np.arange(1, n_points + 1, dtype=float)
    last_observed = n_points

    # Span може бути 0 (усі однакові) → захист.
    duration_days = max((last_ts - first_ts).total_seconds() / 86400.0, MIN_DURATION_DAYS)
    horizon_days = max(int(np.ceil(duration_days * horizon_fraction)), MIN_HORIZON_DAYS)

    fitted = select_best_model(t_train, y_train, target=target)

    # Future grid: щодоби, від наступного дня після last_ts.
    last_known_day = pd.Timestamp(last_ts.date())
    future_dates = pd.date_range(
        start=last_known_day + timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )
    t_future = _to_days_from(pd.Series(future_dates), first_ts)

    rng = np.random.default_rng(random_seed)
    model_mean, ci_lower_arr, ci_upper_arr = nhpp_prediction_interval(
        fitted, t_future, last_observed=last_observed, n_sims=n_simulations, rng=rng
    )

    # Точкова оцінка — model.predict з floor'ом на last_observed.
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


def _to_days_from(ts: pd.Series, anchor) -> np.ndarray:
    """Перевести Series datetime'ів у float-дні від anchor."""
    deltas = pd.to_datetime(ts) - pd.Timestamp(anchor)
    return (deltas.dt.total_seconds() / 86400.0).to_numpy(dtype=float)
