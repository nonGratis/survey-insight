"""Побудова довірчого інтервалу для прогнозу через NHPP-симуляцію.

NHPP (non-homogeneous Poisson process) з пробрасуванням parameter uncertainty:

  Для k = 1..n_sims:
    θ_k ~ N(θ̂, pcov)                            # PARAMETER UNCERTAINTY
    λ_k[i] = predict(t_i; θ_k) - predict(t_{i-1}; θ_k)
    daily_k[i] ~ Poisson(λ_k[i])                 # POISSON POINT-PROCESS NOISE
    cum_k[i] = last_observed + Σ daily_k[1..i]

  ci = percentile(cum across k)

Це додає parameter variance поверх Poisson-noise. Без неї (backtest 02)
ми мали coverage 19% при заявлених 95% — модель була переконана у своїх
параметрах і не враховувала їх дисперсію.

Якщо pcov відсутня або не PSD — fallback на point-estimate sampling.

Властивості, гарантовані за побудовою:
- sim_cum[i] >= last_observed завжди (cumsum невід'ємних + база);
- sim_cum[i] >= sim_cum[i-1] завжди (cumsum монотонний);
- перцентили зберігають ці властивості.
"""

from __future__ import annotations

import warnings

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
    """95% (за замовчуванням) prediction interval через NHPP + param-sampling."""
    if len(t_future) == 0:
        raise ValueError("t_future is empty")
    if not 0.0 <= ci_lower_pct < ci_upper_pct <= 100.0:
        raise ValueError(f"Invalid percentiles: {ci_lower_pct}, {ci_upper_pct}")
    if rng is None:
        rng = np.random.default_rng()

    theta_hat = np.asarray(fitted.params, dtype=float)
    t_anchor = np.concatenate(([t_future[0] - 1.0], t_future))
    n_t = len(t_future)

    # Point-estimate mean for return value.
    mean_cum = fitted.model.predict(t_anchor, *theta_hat)[1:]

    # Sample params з апостеріора N(θ̂, pcov).
    if fitted.pcov is not None and _is_psd(fitted.pcov):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            thetas = rng.multivariate_normal(theta_hat, fitted.pcov, size=n_sims)
    else:
        thetas = np.tile(theta_hat, (n_sims, 1))

    sim_cum = np.empty((n_sims, n_t), dtype=np.int64)
    for k in range(n_sims):
        try:
            mean_k = fitted.model.predict(t_anchor, *thetas[k])
            lambdas_k = np.clip(np.diff(mean_k), 0.0, 1e8)
            if not np.all(np.isfinite(lambdas_k)):
                raise ValueError("non-finite lambdas")
            daily_k = rng.poisson(lam=lambdas_k)
        except (FloatingPointError, OverflowError, ValueError):
            # Bad param sample — fallback на θ̂.
            mean_k = fitted.model.predict(t_anchor, *theta_hat)
            lambdas_k = np.clip(np.diff(mean_k), 0.0, 1e8)
            daily_k = rng.poisson(lam=lambdas_k)
        sim_cum[k] = last_observed + np.cumsum(daily_k)

    ci_lower = np.percentile(sim_cum, ci_lower_pct, axis=0)
    ci_upper = np.percentile(sim_cum, ci_upper_pct, axis=0)
    return mean_cum, ci_lower, ci_upper


def _is_psd(matrix: np.ndarray, tol: float = -1e-8) -> bool:
    """Чи позитивно-напівнизнаена матриця (з невеликою числовою tolerance)."""
    if not np.all(np.isfinite(matrix)):
        return False
    try:
        eigvals = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        return False
    return bool(np.min(eigvals) >= tol)
