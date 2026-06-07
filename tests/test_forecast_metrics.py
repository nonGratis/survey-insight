"""Юніт-тести для функцій метрик."""

from __future__ import annotations

import math

import numpy as np

from core.forecast.metrics import aicc, r_squared, rmse


def test_rmse_zero_on_perfect_fit():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmse(y, y) == 0.0


def test_rmse_basic():
    y_actual = np.array([1.0, 2.0, 3.0])
    y_fitted = np.array([1.5, 2.5, 3.5])
    # residuals 0.5 each → MSE 0.25 → RMSE 0.5
    assert math.isclose(rmse(y_actual, y_fitted), 0.5)


def test_r_squared_one_on_perfect_fit():
    y = np.arange(10, dtype=float)
    assert math.isclose(r_squared(y, y), 1.0)


def test_r_squared_zero_when_constant_fitted_to_mean():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_fitted = np.full_like(y, y.mean())
    assert math.isclose(r_squared(y, y_fitted), 0.0, abs_tol=1e-12)


def test_r_squared_handles_zero_variance():
    y = np.array([5.0, 5.0, 5.0])
    assert r_squared(y, y) == 0.0  # ss_tot=0 → fallback 0


def test_aicc_returns_inf_when_n_too_small():
    y = np.array([1.0, 2.0])
    assert aicc(y, y, n_params=3) == math.inf


def test_aicc_prefers_simpler_when_equal_fit():
    """При однаковому SSE менше параметрів → менший AICc."""
    y = np.linspace(0, 10, 20)
    y_fitted = y + 0.5  # constant offset, однаковий SSE для k=2 vs k=4
    aicc_2 = aicc(y, y_fitted, n_params=2)
    aicc_4 = aicc(y, y_fitted, n_params=4)
    assert aicc_2 < aicc_4


def test_aicc_prefers_better_fit_when_equal_complexity():
    y = np.linspace(0, 10, 20)
    y_good = y + 0.1
    y_bad = y + 2.0
    assert aicc(y, y_good, n_params=3) < aicc(y, y_bad, n_params=3)
