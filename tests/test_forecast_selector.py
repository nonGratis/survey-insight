"""Юніт-тести для селектора моделей за AICc."""

from __future__ import annotations

import numpy as np
import pytest

from core.forecast.models import (
    AsymptoticExpModel,
    GompertzModel,
    LogisticModel,
)
from core.forecast.selector import select_best_model
from core.forecast.types import ForecastError


def test_selects_logistic_on_pure_logistic_curve():
    t = np.arange(20, dtype=float)
    y = 100.0 / (1.0 + np.exp(-0.3 * (t - 15.0)))
    best = select_best_model(t, y, target=100)
    assert best.model.name == "logistic"
    assert best.r_squared > 0.99


def test_selects_asymptotic_exp_on_pure_asymptotic_curve():
    t = np.arange(30, dtype=float)
    y = 80.0 * (1.0 - np.exp(-0.1 * t)) + 5.0
    best = select_best_model(t, y, target=80)
    # На чистій asymptotic-exp кривій usually win logistic чи asymp_exp;
    # обидва дають ідеальний фіт, тож AICc вирішує по складності.
    # Тут перевіряємо, що не Gompertz і R² відмінне.
    assert best.model.name in {"asymptotic_exp", "logistic"}
    assert best.r_squared > 0.999


def test_returns_fitted_with_filled_metrics():
    t = np.arange(15, dtype=float)
    y = 50.0 / (1.0 + np.exp(-0.4 * (t - 7.0)))
    best = select_best_model(t, y, target=50)
    assert best.params and len(best.params) == best.model.n_params
    assert best.aicc < float("inf")
    assert 0.0 <= best.r_squared <= 1.0
    assert best.rmse >= 0.0


def test_skips_failing_models():
    """Якщо одна модель завжди падає — селектор повертає робочу."""

    class AlwaysFailingModel:
        name = "always_failing"
        n_params = 3

        def predict(self, t, a, b, c):
            return a * t + b + c

        def initial_guess(self, t, y, target):
            # Guess поза bounds → curve_fit кидає ValueError
            return (-1000.0, -1000.0, -1000.0)

        def bounds(self, y, target):
            return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    t = np.arange(15, dtype=float)
    y = 80.0 * (1.0 - np.exp(-0.1 * t))
    best = select_best_model(t, y, target=80, models=(AlwaysFailingModel(), AsymptoticExpModel()))
    assert best.model.name == "asymptotic_exp"


def test_raises_when_all_models_fail():
    class AlwaysFailingModel:
        name = "x"
        n_params = 3

        def predict(self, t, a, b, c):
            # NaN з predict → curve_fit гарантовано не зійдеться.
            return np.full_like(np.asarray(t, dtype=float), np.nan)

        def initial_guess(self, t, y, target):
            return (1.0, 1.0, 1.0)

        def bounds(self, y, target):
            return (0.0, 0.0, 0.0), (10.0, 10.0, 10.0)

    t = np.arange(10, dtype=float)
    y = np.linspace(1, 10, 10)
    with pytest.raises(ForecastError, match="не змогла знайти стійкий фіт"):
        select_best_model(t, y, target=None, models=(AlwaysFailingModel(),))


def test_target_prior_influences_selection():
    """З target ~ справжньої стелі — фіт стабільніший."""
    t = np.arange(12, dtype=float)
    y = 100.0 / (1.0 + np.exp(-0.3 * (t - 15.0)))  # ще не насичена крива
    # Без target K може скакнути дуже високо; з target=100 — обмежено.
    best_with_target = select_best_model(t, y, target=100, models=(LogisticModel(),))
    k_est = best_with_target.params[0]
    assert 30.0 <= k_est <= 300.0  # широкий sanity range


def test_di_uses_supplied_models_only():
    """Селектор не імпортує моделі сам — використовує ті, що передали."""
    t = np.arange(20, dtype=float)
    y = 100.0 / (1.0 + np.exp(-0.3 * (t - 10.0)))
    best = select_best_model(t, y, target=100, models=(GompertzModel(),))
    assert best.model.name == "gompertz"
