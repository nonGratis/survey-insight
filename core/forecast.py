"""Прогноз кумулятивного потоку відповідей до заданого дедлайну.

Модель — асимптотична експонента `y = a * (1 - exp(-b * t)) + c`:
- `t` — день від першого submit (>= 0)
- `a + c` ≈ асимптота, інтерпретується як "теоретичний максимум відповідей,
  якщо опитування ніколи не закриється"
- `b` — швидкість насичення
- `c` — невеликий зсув (зазвичай ≈ 0)

Чому саме ця модель: процес інтенсивного надходження на старті з
поступовим затуханням добре описує реальний life-cycle опитувань
(хвилі агітування → стабільне поповнення → плато). Поліноміальні фітинги
дають "точні" residuals на історії, але дикі extrapolations.

Довірчий інтервал — bootstrap (1000 ресемплів daily_counts з поверненням),
2.5/97.5 перцентилі фінальної оцінки. Це чесніше за аналітичні CI з
коваріаційної матриці curve_fit при малих вибірках і нелінійній моделі.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from core.logger import get_logger
from core.timeline import TimelineSeries

log = get_logger(__name__)

DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_RANDOM_SEED = 42  # для відтворюваності CI у демо
DEFAULT_HORIZON_FRACTION = 0.25  # 25% від тривалості опитування
MIN_HORIZON_DAYS = 1


class ForecastError(RuntimeError):
    """Прогнозна модель не змогла знайти збіжний фіт."""


@dataclass(frozen=True)
class ForecastResult:
    """Результат прогнозу до дедлайну.

    Attributes:
        model: ідентифікатор моделі ("asymptotic_exp").
        future_dates: pd.DatetimeIndex від наступної доби після останнього
            відомого факту до дедлайну включно.
        future_cum: модельний кумулятив на future_dates.
        ci_lower: 2.5-перцентиль cumulative на кожній даті (bootstrap).
        ci_upper: 97.5-перцентиль.
        final_estimate: цілочислова точкова оцінка cumulative на дедлайн.
        final_ci: (lower, upper) точкових перцентилів на дедлайн.
        rmse: RMSE моделі на тренувальних даних.
        r_squared: R² на тренувальних даних.
    """

    model: str
    future_dates: pd.DatetimeIndex
    future_cum: pd.Series
    ci_lower: pd.Series
    ci_upper: pd.Series
    final_estimate: int
    final_ci: tuple[int, int]
    rmse: float
    r_squared: float


def _asymptotic_exp(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """y = a * (1 - exp(-b * t)) + c."""
    return a * (1.0 - np.exp(-b * t)) + c


def _fit_asymptotic_exp(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Знайти (a, b, c) curve_fit'ом. Кидає ForecastError при non-convergence."""
    # Початкові оцінки: a≈y_max-y_min, b малий додатній, c≈y[0].
    a0 = float(max(y[-1] - y[0], 1.0))
    b0 = 0.05
    c0 = float(y[0])
    # Bounds: a > 0, b > 0; c має бути >= 0 для cumulative.
    try:
        popt, _ = curve_fit(
            _asymptotic_exp,
            t,
            y,
            p0=(a0, b0, c0),
            bounds=([0.0, 1e-6, 0.0], [np.inf, 5.0, np.inf]),
            maxfev=5000,
        )
    except (RuntimeError, ValueError) as exc:
        raise ForecastError(f"Asymptotic exp не зійшовся: {exc}") from exc
    return float(popt[0]), float(popt[1]), float(popt[2])


def _bootstrap_ci(
    daily_counts: pd.Series,
    t_future: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample daily_counts з поверненням, рефіт, beredu percentile."""
    n_days = len(daily_counts)
    daily_array = daily_counts.to_numpy()
    samples: list[np.ndarray] = []
    fails = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_days, size=n_days)
        sampled_daily = daily_array[idx]
        sampled_cum = np.cumsum(sampled_daily)
        t_train = np.arange(n_days, dtype=float)
        try:
            a, b, c = _fit_asymptotic_exp(t_train, sampled_cum.astype(float))
        except ForecastError:
            fails += 1
            continue
        samples.append(_asymptotic_exp(t_future, a, b, c))
    if not samples:
        raise ForecastError("Bootstrap: жоден ресемпл не зійшовся.")
    if fails > 0:
        log.info(
            "forecast_bootstrap_partial",
            extra={"fails": fails, "ok": len(samples), "total": n_bootstrap},
        )
    stacked = np.vstack(samples)
    return (
        np.percentile(stacked, 2.5, axis=0),
        np.percentile(stacked, 97.5, axis=0),
    )


def asymptotic_exp_forecast(
    timeline: TimelineSeries,
    horizon_fraction: float = DEFAULT_HORIZON_FRACTION,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> ForecastResult:
    """Спрогнозувати cumulative на горизонт ~25% від тривалості опитування.

    Дедлайн більше не передається — прогноз йде на фіксовану частку
    від (last - first) timestamp у даних. Це робить прогноз тривимірно
    послідовним: для тиждень-старого опитування — 2 додаткові дні
    наперед; для місячного — 7-8 днів.

    Args:
        timeline: побудована `build_timeline_from_timestamps`.
        horizon_fraction: частка тривалості, яку додаємо як прогноз.
            Default 0.25 (25%). Min горизонт 1 день.
        n_bootstrap: к-сть ресемплів для CI; 1000 — баланс точності/часу.
        random_seed: для відтворюваності.

    Raises:
        ForecastError: якщо timeline порожня, замало точок, або
            curve_fit не зійшовся на історії.
    """
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

    # Точковий фіт
    a, b, c = _fit_asymptotic_exp(t_train, cum_array)
    fitted_cum = _asymptotic_exp(t_train, a, b, c)
    residuals = cum_array - fitted_cum
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((cum_array - cum_array.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Future timeline: last_known_day + 1 ... + horizon_days
    future_dates = pd.date_range(
        start=last_known_day + timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )
    t_future = np.array(
        [(d.to_pydatetime() - first_known).days for d in future_dates],
        dtype=float,
    )
    future_cum_arr = _asymptotic_exp(t_future, a, b, c)

    # Bootstrap CI
    rng = np.random.default_rng(random_seed)
    ci_lower_arr, ci_upper_arr = _bootstrap_ci(timeline.daily_counts, t_future, n_bootstrap, rng)

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
        rmse=rmse,
        r_squared=r_squared,
    )
