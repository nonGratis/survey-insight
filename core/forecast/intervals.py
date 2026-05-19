"""Побудова довірчого інтервалу для прогнозу.

Тут дві реалізації:

1. `bootstrap_ci` — legacy: ресемпл `daily_counts` з поверненням + рефіт
   asymptotic-exp на кожному ресемплі. Дає нестабільні (інколи
   немонотонні, інколи < last_observed) CI на малих N. Лишається тимчасово
   для зворотної сумісності у `service.py` до commit'у 5.

2. `nhpp_prediction_interval` — нова реалізація через NHPP
   (non-homogeneous Poisson process). Денні приходи моделюються як
   Poisson(λ(t)), де λ(t) = predict(t) - predict(t-1) фітованої кривої.
   Cumulative-симуляція = last_observed + cumsum(Poisson(λ)). Звідси:
   - sim_cum[i] >= last_observed завжди (cumsum невід'ємних + база);
   - sim_cum[i] >= sim_cum[i-1] завжди (монотонність cumsum);
   - перцентили зберігають ці властивості.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.logger import get_logger

from .models import asymptotic_exp, fit_asymptotic_exp
from .types import FittedModel, ForecastError

log = get_logger(__name__)


def bootstrap_ci(
    daily_counts: pd.Series,
    t_future: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy: ресемпл daily_counts + рефіт, percentile."""
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


def nhpp_prediction_interval(
    fitted: FittedModel,
    t_future: np.ndarray,
    last_observed: int,
    n_sims: int = 2000,
    rng: np.random.Generator | None = None,
    ci_lower_pct: float = 2.5,
    ci_upper_pct: float = 97.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """95% (за замовчуванням) prediction interval через NHPP-симуляцію.

    Args:
        fitted: фітована модель (з .model.predict і .params).
        t_future: моменти часу для прогнозу (у тих самих одиницях, що й
            при фіті — типово дні від першого спостереження).
        last_observed: останній відомий cumulative-факт; CI не може бути нижче.
        n_sims: кількість симуляційних траєкторій. 2000 — стійкий компроміс.
        rng: генератор (для відтворюваності). None → новий за замовчуванням.
        ci_lower_pct, ci_upper_pct: перцентилі (типово 2.5 і 97.5 = 95% CI).

    Returns:
        (mean_cum, ci_lower, ci_upper) — np.ndarray довжини len(t_future).
        mean_cum — модельне cumulative (детермінований predict).
        ci_lower / ci_upper — монотонно неспадні, ci_lower[0] >= last_observed.

    Raises:
        ValueError: якщо t_future порожній або параметри перцентилів некоректні.
    """
    if len(t_future) == 0:
        raise ValueError("t_future is empty")
    if not 0.0 <= ci_lower_pct < ci_upper_pct <= 100.0:
        raise ValueError(f"Invalid percentiles: {ci_lower_pct}, {ci_upper_pct}")
    if rng is None:
        rng = np.random.default_rng()

    # Анкор: predict у момент t[0]-1, щоб посчитати λ[0] = predict(t[0]) - predict(t[0]-1).
    t_anchor = np.concatenate(([t_future[0] - 1.0], t_future))
    mean_cum_with_anchor = fitted.model.predict(t_anchor, *fitted.params)
    lambdas = np.clip(np.diff(mean_cum_with_anchor), 0.0, None)
    mean_cum = mean_cum_with_anchor[1:]

    # n_sims траєкторій майбутніх денних приходів.
    sim_daily = rng.poisson(lam=lambdas, size=(n_sims, len(t_future)))
    sim_cum = last_observed + np.cumsum(sim_daily, axis=1)

    ci_lower = np.percentile(sim_cum, ci_lower_pct, axis=0)
    ci_upper = np.percentile(sim_cum, ci_upper_pct, axis=0)
    return mean_cum, ci_lower, ci_upper
