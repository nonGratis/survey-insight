"""Empirical Bayes priors з історії 177-форм для стабільності фіту на малих N.

ПРОБЛЕМА:
research/06 показав, що 27% backtest-точок мають CI width=0 — це big-N
форми, де параметрична uncertainty схлопує. На малих N (N=5-15) MVN-sampling
дає absurd MVN-tails → широкі CI без точності.

РІШЕННЯ (Empirical Bayes):
Для кожної (shape, model) пари обчислити **prior** з параметрів зафічених
повних форм історії — median і std. При фіті нової форми звузити bounds
до ±2σ навколо prior_median. Це:
- стабілізує фіт на малих N (модель не може втекти у дику зону)
- зменшує MVN-tail розкид → CI вужча на пристойних формах
- зберігає bias близько 0 бо prior_median = типовий K для shape

USAGE:
    from core.forecast.priors import load_priors
    priors = load_priors()  # читає priors_data.json з repo
    fitted = forecast_responses(timeline, use_priors=True)
    # → selector передає priors у fit_model, який звужує bounds.

Якщо priors_data.json не існує — fallback на default bounds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Кеш у пам'яті — не читаємо JSON багатократно.
_PRIORS_CACHE: dict | None = None
_PRIORS_PATH = Path(__file__).parent / "priors_data.json"

# Bounds-вуження: prior_median ± N_SIGMA × prior_std.
PRIOR_N_SIGMA = 2.0


@dataclass(frozen=True)
class ShapePrior:
    """Розподіл параметрів моделі на історичному датасеті.

    Attributes:
        model_name: "logistic" / "gompertz" / "asymptotic_exp".
        shape: "logarithmic" / "logistic" / "late_burst" / ...
        param_medians: медіани параметрів (порядок як у model.predict).
        param_stds: стандартні відхилення.
        n_samples: кількість форм, з яких рахували prior.
    """

    model_name: str
    shape: str
    param_medians: tuple[float, ...]
    param_stds: tuple[float, ...]
    n_samples: int


def load_priors() -> dict[tuple[str, str], ShapePrior]:
    """Завантажити priors з priors_data.json у repo. Кешує у пам'яті.

    Returns:
        Dict (model_name, shape) → ShapePrior. Порожній dict, якщо файлу
        нема (fallback на default bounds).
    """
    global _PRIORS_CACHE
    if _PRIORS_CACHE is not None:
        return _PRIORS_CACHE
    if not _PRIORS_PATH.exists():
        _PRIORS_CACHE = {}
        return _PRIORS_CACHE
    raw = json.loads(_PRIORS_PATH.read_text(encoding="utf-8"))
    priors = {}
    for entry in raw:
        key = (entry["model_name"], entry["shape"])
        priors[key] = ShapePrior(
            model_name=entry["model_name"],
            shape=entry["shape"],
            param_medians=tuple(entry["param_medians"]),
            param_stds=tuple(entry["param_stds"]),
            n_samples=entry["n_samples"],
        )
    _PRIORS_CACHE = priors
    return priors


def save_priors(priors: dict[tuple[str, str], ShapePrior]) -> None:
    """Записати priors у priors_data.json."""
    data = [
        {
            "model_name": p.model_name,
            "shape": p.shape,
            "param_medians": list(p.param_medians),
            "param_stds": list(p.param_stds),
            "n_samples": p.n_samples,
        }
        for p in priors.values()
    ]
    _PRIORS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def narrow_bounds_with_prior(
    bounds_low: tuple[float, ...],
    bounds_high: tuple[float, ...],
    prior: ShapePrior,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Звузити bounds до перетину з prior_median ± N_SIGMA × prior_std.

    Якщо std=0 (degenerate prior) — bounds не змінюються для цього параметра.
    """
    new_low = []
    new_high = []
    for i, (lo, hi) in enumerate(zip(bounds_low, bounds_high, strict=True)):
        med = prior.param_medians[i]
        std = prior.param_stds[i]
        if std <= 0:
            new_low.append(lo)
            new_high.append(hi)
            continue
        prior_lo = med - PRIOR_N_SIGMA * std
        prior_hi = med + PRIOR_N_SIGMA * std
        # Перетин з default bounds.
        new_low.append(max(lo, prior_lo))
        new_high.append(min(hi, prior_hi))
        # Sanity: якщо перетин порожній — повертаємо default.
        if new_low[-1] >= new_high[-1]:
            new_low[-1] = lo
            new_high[-1] = hi
    return tuple(new_low), tuple(new_high)
