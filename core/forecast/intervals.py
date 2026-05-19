"""Побудова довірчого інтервалу для прогнозу.

Поточна реалізація — bootstrap по `daily_counts` з повторним рефітом.
У наступному коміті замінюється на NHPP (Poisson-симуляцію приходів),
що дає монотонний CI ≥ last_observed за побудовою.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logger import get_logger

from .models import asymptotic_exp, fit_asymptotic_exp
from .types import ForecastError

log = get_logger(__name__)


def bootstrap_ci(
    daily_counts: pd.Series,
    t_future: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Ресемпл daily_counts з поверненням, рефіт, percentile."""
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
            a, b, c = fit_asymptotic_exp(t_train, sampled_cum.astype(float))
        except ForecastError:
            fails += 1
            continue
        samples.append(asymptotic_exp(t_future, a, b, c))
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
