"""Криві моделі надходжень.

Поки що тут лише asymptotic-exp; logistic і Gompertz додаються наступним
комітом. Виділено в окремий модуль, щоб додавання нових кривих не потребувало
правки оркестратора (Open/Closed).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from .types import ForecastError

CURVE_FIT_MAX_NFEV = 500


def asymptotic_exp(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """y = a * (1 - exp(-b * t)) + c."""
    return a * (1.0 - np.exp(-b * t)) + c


def fit_asymptotic_exp(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Знайти (a, b, c) curve_fit'ом. Кидає ForecastError при non-convergence."""
    a0 = float(max(y[-1] - y[0], 1.0))
    b0 = 0.05
    c0 = float(y[0])
    try:
        popt, _ = curve_fit(
            asymptotic_exp,
            t,
            y,
            p0=(a0, b0, c0),
            bounds=([0.0, 1e-6, 0.0], [np.inf, 5.0, np.inf]),
            maxfev=CURVE_FIT_MAX_NFEV,
        )
    except (RuntimeError, ValueError) as exc:
        raise ForecastError(f"Asymptotic exp не зійшовся: {exc}") from exc
    return float(popt[0]), float(popt[1]), float(popt[2])
