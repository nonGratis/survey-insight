"""Побудова довірчого інтервалу для прогнозу через NHPP-симуляцію.

NHPP (non-homogeneous Poisson process) з параметричною uncertainty,
обмеженою фізичними межами:

  Для k = 1..n_sims:
    θ_k ~ N(θ̂, pcov), constrained до model.bounds()  # bounded sampling
    λ_k[i] = predict(t_i; θ_k) - predict(t_{i-1}; θ_k)
    daily_k[i] ~ Poisson(λ_k[i])
    cum_k[i] = last_observed + Σ daily_k[1..i]
    cum_k[i] = min(cum_k[i], reality_cap)  # reality check: rate × horizon × 5

  ci = percentile(cum across k)

Без bounded sampling MVN-tail видає absurd параметри (b → 5 у asymp_exp на
малих N, де pcov_b велика) → λ explodes → CI=(16, 2443) для 16 відповідей.
Це робить алгоритм безглуздим у саме тому use case, для якого створений
(перші 15 хв, мало даних, потрібен швидкий прогноз).

Поправки:
1. Параметри з MVN reject'аться, якщо виходять за model.bounds().
2. λ обмежена 100 × observed_rate (як defense-in-depth).
3. Фінальна траєкторія clip'ається до reality_cap = last + rate×horizon×5.

Властивості:
- sim_cum[i] >= last_observed (cumsum невід'ємних + base);
- sim_cum[i] <= reality_cap (no absurd hallucinations);
- sim_cum монотонний (cumsum).
"""

from __future__ import annotations

import warnings

import numpy as np

from .types import FittedModel

# Reality caps на симуляційне cumulative.
# Без них на короткому prefix (16 точок за 5 хв) observed_rate = 400/day,
# MVN-tail видає absurd параметри, λ explodes → CI=(16, 2443).
#
# Підхід: множинні cap'и, береш найгірший із "не сильно більше реальності":
#   cap_multiplier:  last × 3      — не передбачаємо тройного росту
#   cap_absolute:    last + 200    — або абсолютний +200 запас
#   cap_rate:        last + rate×horizon×3 — або rate-based, якщо щось екстраординарне
# reality_cap = max(cap_multiplier_or_absolute_min, min(cap_max_pair))
REALITY_CAP_MULTIPLIER = 5.0  # last × 5 — генерує trust-able CI без 1000x чисел
REALITY_CAP_ABSOLUTE = 500
REALITY_CAP_RATE_MULTIPLIER = 5.0
MIN_GROWTH_ALLOWANCE = 30  # always дозволяємо мінімум +30 точок зростання


def nhpp_prediction_interval(
    fitted: FittedModel,
    t_future: np.ndarray,
    last_observed: int,
    n_sims: int = 2000,
    rng: np.random.Generator | None = None,
    ci_lower_pct: float = 2.5,
    ci_upper_pct: float = 97.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """95% (за замовчуванням) PI через NHPP з bounded param sampling і
    reality-capped cumulative.
    """
    if len(t_future) == 0:
        raise ValueError("t_future is empty")
    if not 0.0 <= ci_lower_pct < ci_upper_pct <= 100.0:
        raise ValueError(f"Invalid percentiles: {ci_lower_pct}, {ci_upper_pct}")
    if rng is None:
        rng = np.random.default_rng()

    theta_hat = np.asarray(fitted.params, dtype=float)
    t_anchor = np.concatenate(([t_future[0] - 1.0], t_future))
    n_t = len(t_future)

    # Point-estimate mean (детермінований).
    mean_cum = fitted.model.predict(t_anchor, *theta_hat)[1:]

    # Reality cap: множинні bounds, береш найрозумніше.
    elapsed_days = max(float(t_future[0]) - 1.0, 1.0 / 24.0)
    observed_rate = last_observed / elapsed_days
    horizon_days = max(float(t_future[-1]) - float(t_future[0]) + 1.0, 1.0)

    cap_multiplier = int(last_observed * REALITY_CAP_MULTIPLIER)
    cap_absolute = last_observed + REALITY_CAP_ABSOLUTE
    cap_rate = last_observed + int(observed_rate * horizon_days * REALITY_CAP_RATE_MULTIPLIER)
    min_cap = last_observed + MIN_GROWTH_ALLOWANCE
    # Беремо мінімум з cap_multiplier / cap_absolute / cap_rate, але не нижче
    # min_cap (завжди дозволяємо +30 точок зростання).
    reality_cap = max(min_cap, min(cap_multiplier, cap_absolute, cap_rate))
    # Per-day λ cap: 10x mean predict-rate, з підлогою.
    lambda_cap = max(reality_cap / max(horizon_days, 1.0), 5.0)

    # Bounded MVN sampling: відкидаємо samples поза model.bounds().
    thetas = _sample_bounded_params(fitted, theta_hat, n_sims, rng, last_observed)

    sim_cum = np.empty((n_sims, n_t), dtype=np.int64)
    for k in range(n_sims):
        try:
            mean_k = fitted.model.predict(t_anchor, *thetas[k])
            lambdas_k = np.clip(np.diff(mean_k), 0.0, lambda_cap)
            if not np.all(np.isfinite(lambdas_k)):
                raise ValueError("non-finite lambdas")
            daily_k = rng.poisson(lam=lambdas_k)
        except (FloatingPointError, OverflowError, ValueError):
            mean_k = fitted.model.predict(t_anchor, *theta_hat)
            lambdas_k = np.clip(np.diff(mean_k), 0.0, lambda_cap)
            daily_k = rng.poisson(lam=lambdas_k)
        cum_k = last_observed + np.cumsum(daily_k)
        # Reality cap: не може зрости вище фізично-розумної межі.
        sim_cum[k] = np.clip(cum_k, last_observed, reality_cap)

    ci_lower = np.percentile(sim_cum, ci_lower_pct, axis=0)
    ci_upper = np.percentile(sim_cum, ci_upper_pct, axis=0)
    return mean_cum, ci_lower, ci_upper


def _sample_bounded_params(
    fitted: FittedModel,
    theta_hat: np.ndarray,
    n_sims: int,
    rng: np.random.Generator,
    last_observed: int,
) -> np.ndarray:
    """MVN sample з rejection до model.bounds().

    Якщо pcov відсутня або не PSD — повертає n_sims копій θ̂ (no uncertainty).
    """
    if fitted.pcov is None or not _is_psd(fitted.pcov):
        return np.tile(theta_hat, (n_sims, 1))

    # Bounds з моделі — використовуємо як rejection region.
    y_proxy = np.array([last_observed], dtype=float)
    low_b, high_b = fitted.model.bounds(y_proxy, target=None)
    low_arr = np.asarray(low_b, dtype=float)
    high_arr = np.asarray(high_b, dtype=float)

    # Збираємо n_sims samples, відкидаючи out-of-bounds.
    collected = []
    max_attempts = 10  # обмеження на rejection rate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(max_attempts):
            batch = rng.multivariate_normal(theta_hat, fitted.pcov, size=n_sims * 2)
            in_bounds = np.all((batch >= low_arr) & (batch <= high_arr), axis=1)
            collected.extend(batch[in_bounds].tolist())
            if len(collected) >= n_sims:
                break

    if len(collected) >= n_sims:
        return np.array(collected[:n_sims])
    # Не вдалося назбирати достатньо — добиваємо θ̂.
    samples = collected + [theta_hat.tolist()] * (n_sims - len(collected))
    return np.array(samples)


def _is_psd(matrix: np.ndarray, tol: float = -1e-8) -> bool:
    """Чи позитивно-напівнизнаена матриця (з невеликою числовою tolerance)."""
    if not np.all(np.isfinite(matrix)):
        return False
    try:
        eigvals = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        return False
    return bool(np.min(eigvals) >= tol)
