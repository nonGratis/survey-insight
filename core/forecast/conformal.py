"""Conformal calibration: empirical quantile multiplier на delta-CI half-width.

Розв'язує fundamental issue з delta-method CI: width відображає тільки
параметричну uncertainty від pcov, а НЕ "наскільки модель сама-по-собі
помиляється на цьому типі даних". Conformal калібровка додає це через
empirical residual quantile, обчислений на calibration set.

Methodology (split conformal, Vovk 2005; Romano 2019):
1. На calibration set обчислюємо residual r = (truth − point) / half_delta_ci.
2. Per-cell (n_class × horizon_bucket) quantile q_{1−α} = |r|.quantile(0.95).
3. Runtime: new_half = half_delta_ci × q_cell.
4. Hierarchical fallback: exact cell → bucket → global, для rare/missing cells.

Theoretical guarantee: empirical coverage → 1−α якщо calibration set i.i.d.
з test set. На practice (форми мають кореляції): coverage близько до 1−α
але не точно.

Artifact: `core/forecast/data/conformal_quantiles.json` — кальибровано офлайн
скриптом `research/benchmarks/calibrate_conformal.py` на повному backtest
12_prod_points.csv (3710 точок). Регенерується якщо змінюється
delta_ci-arsenal або модельний пул.

Reference:
- Vovk V, Gammerman A, Shafer G (2005). Algorithmic Learning in a Random World.
- Romano Y, Sesia M, Candès E (2019). Conformalized Quantile Regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_QUANTILES_PATH = Path(__file__).parent / "conformal_quantiles.json"
_CACHE: dict | None = None

# Fallback quantile якщо JSON artifact відсутній (свіжа інсталяція без
# калібровки). 1.0 = no adjustment (== raw delta-CI). Безпечно з точки зору
# point estimate (ніколи не псує), але coverage буде delta-CI native (~75%).
_DEFAULT_QUANTILE = 1.0


def _load() -> dict:
    """Lazy-load JSON artifact. Cached."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not _QUANTILES_PATH.exists():
        _CACHE = {"quantiles": {"_|_": _DEFAULT_QUANTILE}, "alpha": 0.05}
        return _CACHE
    _CACHE = json.loads(_QUANTILES_PATH.read_text(encoding="utf-8"))
    return _CACHE


def horizon_bucket_from_days(horizon_days: float) -> str:
    """Map horizon_days → bucket. Має співпадати з calibration script."""
    h_hours = horizon_days * 24.0
    if h_hours <= 6:
        return "short"
    if h_hours <= 72:
        return "mid"
    return "long"


def n_class_from_n(n: int) -> str:
    """Map training-set N → n_class. Same as research/_common.py."""
    if n < 10:
        return "tiny"
    if n < 30:
        return "small"
    if n < 100:
        return "medium"
    if n < 1000:
        return "large"
    return "huge"


def lookup_quantile(n_class: str, horizon_bucket: str) -> float:
    """Hierarchical lookup: exact → bucket → global.

    Returns float multiplier для delta-CI half-width.
    """
    data = _load()
    q = data.get("quantiles", {})
    for key in (f"{n_class}|{horizon_bucket}", f"_|{horizon_bucket}", "_|_"):
        if key in q:
            return float(q[key])
    return _DEFAULT_QUANTILE


def apply_conformal_adjustment(
    point_arr: np.ndarray,
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
    n_train: int,
    horizon_days_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Multiply delta-CI half-width by per-cell empirical quantile.

    Args:
        point_arr: point estimates (НЕ змінюються).
        lower_arr / upper_arr: delta-CI bounds (post cap_width).
        n_train: розмір тренувального вибірки (визначає n_class).
        horizon_days_arr: масив horizon_days per future point (визначає bucket).

    Returns:
        (new_lower, new_upper) — bounds з conformal-scaled half-width.
        Гарантовано: new_lower ≤ point ≤ new_upper.
    """
    nc = n_class_from_n(n_train)
    half = (upper_arr - lower_arr) / 2.0
    new_half = np.zeros_like(half)
    # Один lookup per унікальний bucket для perf.
    cache: dict[str, float] = {}
    for i, h_days in enumerate(horizon_days_arr):
        hb = horizon_bucket_from_days(float(h_days))
        if hb not in cache:
            cache[hb] = lookup_quantile(nc, hb)
        new_half[i] = half[i] * cache[hb]
    return point_arr - new_half, point_arr + new_half


def reset_cache() -> None:
    """Скинути JSON-cache. Для тестів і коли JSON оновився."""
    global _CACHE
    _CACHE = None
