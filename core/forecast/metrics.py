"""Чисті функції оцінки якості фіту: RMSE, R², AICc."""

from __future__ import annotations

import math

import numpy as np


def rmse(y_actual: np.ndarray, y_fitted: np.ndarray) -> float:
    residuals = y_actual - y_fitted
    return float(np.sqrt(np.mean(residuals**2)))


def r_squared(y_actual: np.ndarray, y_fitted: np.ndarray) -> float:
    residuals = y_actual - y_fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_actual - y_actual.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def aicc(y_actual: np.ndarray, y_fitted: np.ndarray, n_params: int) -> float:
    """Akaike Information Criterion з поправкою на малий розмір вибірки.

    Формула для нормально-розподілених помилок:
        AIC  = n·ln(SSE/n) + 2k
        AICc = AIC + 2k(k+1)/(n-k-1)

    де k = n_params + 1 (плюс варіанса шуму як вільний параметр).

    Менше — краще. Для n ≤ k+1 повертається +inf (поправка не визначена).
    """
    n = len(y_actual)
    k = n_params + 1
    if n <= k + 1:
        return math.inf
    ss_res = float(np.sum((y_actual - y_fitted) ** 2))
    if ss_res <= 0.0:
        ss_res = 1e-12  # perfect fit → numerical floor, не -inf
    aic = n * math.log(ss_res / n) + 2.0 * k
    correction = 2.0 * k * (k + 1) / (n - k - 1)
    return float(aic + correction)
