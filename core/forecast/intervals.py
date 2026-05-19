"""Побудова довірчого інтервалу для прогнозу через NHPP-симуляцію.

NHPP (non-homogeneous Poisson process): денні приходи моделюються як
Poisson(λ(t)), де λ(t) = predict(t) - predict(t-1) фітованої кривої.
Майбутнє cumulative = last_observed + cumsum(Poisson(λ)).

Властивості, гарантовані за побудовою:
- sim_cum[i] >= last_observed завжди (cumsum невід'ємних + база);
- sim_cum[i] >= sim_cum[i-1] завжди (cumsum монотонний);
- перцентили зберігають ці властивості → CI монотонний і ≥ last_observed.
"""

from __future__ import annotations

import numpy as np

from .types import FittedModel


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
        fitted: фітована модель (.model.predict, .params).
        t_future: моменти часу для прогнозу (у тих самих одиницях, що при фіті —
            типово дні від першого спостереження).
        last_observed: останній відомий cumulative-факт; CI не може бути нижче.
        n_sims: кількість симуляційних траєкторій. 2000 — стійкий компроміс.
        rng: генератор (для відтворюваності). None → новий за замовчуванням.
        ci_lower_pct, ci_upper_pct: перцентилі (типово 2.5/97.5 = 95% CI).

    Returns:
        (mean_cum, ci_lower, ci_upper) — np.ndarray довжини len(t_future).
        mean_cum — детермінований model.predict.
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

    # Анкор у момент t[0]-1, щоб λ[0] = predict(t[0]) - predict(t[0]-1).
    t_anchor = np.concatenate(([t_future[0] - 1.0], t_future))
    mean_cum_with_anchor = fitted.model.predict(t_anchor, *fitted.params)
    lambdas = np.clip(np.diff(mean_cum_with_anchor), 0.0, None)
    mean_cum = mean_cum_with_anchor[1:]

    sim_daily = rng.poisson(lam=lambdas, size=(n_sims, len(t_future)))
    sim_cum = last_observed + np.cumsum(sim_daily, axis=1)

    ci_lower = np.percentile(sim_cum, ci_lower_pct, axis=0)
    ci_upper = np.percentile(sim_cum, ci_upper_pct, axis=0)
    return mean_cum, ci_lower, ci_upper
