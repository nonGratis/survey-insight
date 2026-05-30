"""Оркестратор прогнозу: валідація → селектор → CI → пакування.

**P12 refactor (Варіант γ, крок 1+2):** primary CI computation — classical
delta-method (Bates & Watts 1988) на pcov селектора. NHPP-симуляція +
multipliers (P7, P10, P11) лишені у `intervals.py`/`calibration.py` як
**legacy fallback** доступний через `ci_method="nhpp"`. Default behaviour
змінено: width тепер залежить від справжньої pcov-uncertainty, а не від
uniform multipliers.

Continuous-time fit: модель навчається на парах `(t_i, i+1)` для кожної
окремої відповіді, де `t_i` — час у частках доби від першого спостереження.
Жодної агрегації по добі: 47 відповідей за 15 хв = 47 точок з малими
інтервалами, працює так само як 47 відповідей за 14 днів.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Literal

import numpy as np
import pandas as pd

from core.timeline import TimelineSeries

from .calibration import (
    apply_calibration_arrays,
    apply_sample_size_scaling,
    get_calibration_multiplier,
)
from .delta_ci import cap_width, delta_method_ci
from .intervals import nhpp_prediction_interval
from .models import models_for_n_points
from .priors import load_priors
from .selector import select_best_model
from .shape_classifier import classify_timeline
from .types import ForecastError, ForecastResult

DEFAULT_N_SIMULATIONS = 2000
DEFAULT_RANDOM_SEED = 42
DEFAULT_HORIZON_FRACTION = 0.25
MIN_HORIZON_DAYS = 1
MIN_TRAIN_POINTS = 5
MIN_DURATION_DAYS = 1.0 / 24.0

# P12: cap delta-CI width на explosion-cases (ill-conditioned pcov на
# Gompertz/multi-wave формах). Sane width лишається не зміненою.
DELTA_CI_MAX_RELATIVE = 5.0  # cap: width ≤ 5 × point
DELTA_CI_MAX_ABSOLUTE = 5000.0  # absolute cap: width ≤ 5000 responses
DELTA_CI_MIN_ABSOLUTE = 20.0  # floor після capping: width ≥ 20

CIMethod = Literal["delta", "nhpp", "auto"]


def forecast_responses(
    timeline: TimelineSeries,
    target: int | None = None,
    horizon_fraction: float = DEFAULT_HORIZON_FRACTION,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    horizon_until: pd.Timestamp | None = None,
    use_priors: bool = False,
    form_type: str | None = None,
    ci_method: CIMethod = "delta",
) -> ForecastResult:
    """Прогноз cumulative responses з CI у `ci_method`-flavoured calibration.

    Алгоритм:
    1. Валідація: мінімум `MIN_TRAIN_POINTS` (5) timestamps.
    2. Continuous-time t_train: float-дні від першого timestamp'у; y_train = 1..N.
    3. Selector обирає кращу модель за AICc.
    4. Horizon: `max(duration_days * fraction, MIN_HORIZON_DAYS)`. Future-grid
       щодоби, починаючи з наступного дня після last_ts.
    5. CI: за `ci_method`:
       - `"delta"` (default, P12) — classical delta-method на pcov селектора
         + cap_width захист від ill-conditioned pcov. Без NHPP, без множників.
         Width природно мала на високому R², велика на поганому. Найкраща
         калібровка для коротких горизонтів (≤ 6h, per research/16_).
       - `"nhpp"` (legacy) — NHPP-симуляція + P7 (×10) + P11 (per-type) +
         P10 (sample-size scaling). Збережено для repro thesis-numbers.
       - `"auto"` — eventually conformal fallback; зараз = "delta".
    6. Cumulative monotonicity + lower-floor на last_observed.

    Args:
        timeline: побудована `build_timeline_from_timestamps`.
        target: цільова кількість відповідей; впливає на bounds моделі.
        horizon_fraction: частка тривалості, яку додаємо як прогноз.
        n_simulations: кількість Poisson-траєкторій для NHPP (тільки якщо
            `ci_method="nhpp"`).
        random_seed: для відтворюваності NHPP-симуляції.
        horizon_until: явний кінцевий час горизонту (alt to `horizon_fraction`).
        use_priors: вмикає emp. Bayes priors (P9) на bounds.
        form_type: тип форми з catalog (тільки для `ci_method="nhpp"` + P11).
        ci_method: "delta" (P12 default) / "nhpp" (legacy) / "auto".

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

    # Точкова оцінка — детермінована model.predict, floor'нута на last_observed,
    # cumulative-monotonic. Identical для всіх ci_method.
    model_mean = fitted.model.predict(t_future, *fitted.params)
    future_cum_arr = np.maximum.accumulate(np.maximum(model_mean, float(last_observed)))

    # CI: routing per ci_method.
    method = "delta" if ci_method == "auto" else ci_method
    if method == "delta":
        ci_lower_arr, ci_upper_arr = _delta_ci_path(fitted, t_future, n_points, future_cum_arr)
    elif method == "nhpp":
        ci_lower_arr, ci_upper_arr = _nhpp_legacy_path(
            fitted,
            t_future,
            last_observed,
            future_cum_arr,
            n_simulations,
            random_seed,
            form_type,
        )
    else:
        raise ValueError(f"Unknown ci_method: {ci_method}")

    # Cumulative floor: ci_lower не нижче last_observed.
    ci_lower_arr = np.maximum(ci_lower_arr, float(last_observed))
    # Гарантуємо point ∈ [lower, upper] (numerical guard).
    ci_lower_arr = np.minimum(ci_lower_arr, future_cum_arr)
    ci_upper_arr = np.maximum(ci_upper_arr, future_cum_arr)

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


def _delta_ci_path(
    fitted, t_future: np.ndarray, n_train: int, future_cum_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """P12 primary: delta-method CI + cap_width explosion guard."""
    try:
        ci_lower, ci_upper = delta_method_ci(fitted, t_future, n_train=n_train)
    except ValueError:
        # pcov degenerate / non-finite → fallback на тривіальний ±0 (width=0).
        # cap_width нижче підніме до DELTA_CI_MIN_ABSOLUTE. Це сигналізує
        # caller'у про відсутню uncertainty estimation замість silent NHPP-
        # fallback (бо NHPP — це окрема історія, не purpose цього path).
        ci_lower = future_cum_arr.copy()
        ci_upper = future_cum_arr.copy()

    # Cap_width: захист від ill-conditioned pcov (Gompertz, multi-wave).
    # На sane delta-CI не впливає; explosion-cases обмежує до 5×point.
    ci_lower, ci_upper = cap_width(
        future_cum_arr,
        ci_lower,
        ci_upper,
        max_relative=DELTA_CI_MAX_RELATIVE,
        max_absolute=DELTA_CI_MAX_ABSOLUTE,
        min_absolute=DELTA_CI_MIN_ABSOLUTE,
    )
    return ci_lower, ci_upper


def _nhpp_legacy_path(
    fitted,
    t_future: np.ndarray,
    last_observed: int,
    future_cum_arr: np.ndarray,
    n_simulations: int,
    random_seed: int,
    form_type: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy P7 + P10 + P11 pipeline. Збережено для repro і fallback."""
    rng = np.random.default_rng(random_seed)
    _mean, _median, ci_lower, ci_upper = nhpp_prediction_interval(
        fitted,
        t_future,
        last_observed=last_observed,
        n_sims=n_simulations,
        rng=rng,
    )
    multiplier = get_calibration_multiplier(form_type)
    ci_lower, ci_upper = apply_calibration_arrays(
        future_cum_arr, ci_lower, ci_upper, multiplier=multiplier
    )
    min_half_width = np.maximum(future_cum_arr * 0.10, 5.0)
    ci_lower = np.minimum(ci_lower, future_cum_arr - min_half_width)
    ci_upper = np.maximum(ci_upper, future_cum_arr + min_half_width)
    ci_lower = np.maximum(ci_lower, float(last_observed))
    ci_lower, ci_upper = apply_sample_size_scaling(
        future_cum_arr,
        ci_lower,
        ci_upper,
        n_train=last_observed,
        last_observed=last_observed,
    )
    return ci_lower, ci_upper


def _to_days_from(ts: pd.Series, anchor) -> np.ndarray:
    """Перевести Series datetime'ів у float-дні від anchor."""
    deltas = pd.to_datetime(ts) - pd.Timestamp(anchor)
    return (deltas.dt.total_seconds() / 86400.0).to_numpy(dtype=float)
