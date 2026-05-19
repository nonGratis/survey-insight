"""Чисті функції оцінки якості фіту."""

from __future__ import annotations

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
