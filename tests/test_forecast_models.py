"""Юніт-тести для параметричних моделей насиченої кривої."""

from __future__ import annotations

import numpy as np
import pytest

from core.forecast.models import (
    DEFAULT_MODELS,
    AsymptoticExpModel,
    GompertzModel,
    LogisticModel,
    SaturationModel,
    fit_model,
)
from core.forecast.types import ForecastError


@pytest.fixture
def logistic_data():
    """y = 100 / (1 + exp(-0.3·(t-15))), 20 точок з малим шумом."""
    t = np.arange(20, dtype=float)
    y_true = 100.0 / (1.0 + np.exp(-0.3 * (t - 15.0)))
    rng = np.random.default_rng(0)
    y = np.cumsum(np.diff(np.concatenate(([0.0], y_true))) + rng.normal(0, 0.1, size=20))
    y = np.maximum.accumulate(np.clip(y, 0.0, None))
    return t, y


@pytest.fixture
def gompertz_data():
    t = np.arange(25, dtype=float)
    y = 80.0 * np.exp(-np.exp(-0.2 * (t - 10.0)))
    return t, np.maximum.accumulate(y)


@pytest.fixture
def asympt_exp_data():
    t = np.arange(30, dtype=float)
    y = 80.0 * (1.0 - np.exp(-0.1 * t)) + 5.0
    return t, y


@pytest.mark.parametrize("model_cls", [LogisticModel, GompertzModel, AsymptoticExpModel])
def test_implements_protocol(model_cls):
    model = model_cls()
    assert isinstance(model, SaturationModel)
    assert isinstance(model.name, str) and model.name
    assert model.n_params == 3


@pytest.mark.parametrize("model_cls", [LogisticModel, GompertzModel, AsymptoticExpModel])
def test_predict_returns_array_of_correct_shape(model_cls):
    model = model_cls()
    t = np.arange(10, dtype=float)
    params = model.initial_guess(t, np.linspace(1, 50, 10), target=100)
    y = model.predict(t, *params)
    assert y.shape == t.shape
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize("model_cls", [LogisticModel, GompertzModel, AsymptoticExpModel])
def test_initial_guess_within_bounds(model_cls):
    model = model_cls()
    t = np.arange(15, dtype=float)
    y = np.linspace(2.0, 55.0, 15)
    for target in (None, 100):
        p0 = model.initial_guess(t, y, target)
        low, high = model.bounds(y, target)
        for v, lo, hi in zip(p0, low, high, strict=True):
            assert lo <= v <= hi, f"{model.name}: {v} not in [{lo}, {hi}] for target={target}"


def test_logistic_recovers_true_capacity(logistic_data):
    t, y = logistic_data
    model = LogisticModel()
    params, _pcov = fit_model(model, t, y, target=None)
    k_est, _r, _t0 = params
    assert 80.0 <= k_est <= 130.0, f"K={k_est} too far from true 100"


def test_gompertz_recovers_true_capacity(gompertz_data):
    t, y = gompertz_data
    model = GompertzModel()
    params, _pcov = fit_model(model, t, y, target=None)
    k_est, _r, _t0 = params
    # Upper bound розширена бо K_MIN_RELAXATION=1.30 додає conservative
    # bias-correction floor на основі last_observed.
    assert 70.0 <= k_est <= 120.0


def test_asymptotic_exp_recovers_asymptote(asympt_exp_data):
    t, y = asympt_exp_data
    model = AsymptoticExpModel()
    params, _pcov = fit_model(model, t, y, target=None)
    a, _b, c = params
    asymptote = a + c
    assert 75.0 <= asymptote <= 95.0


def test_target_constrains_capacity():
    """Якщо задано target — K не може скакнути на нереально високе."""
    t = np.arange(10, dtype=float)
    y = np.linspace(2.0, 30.0, 10)
    model = LogisticModel()
    (k_no_target, *_), _ = fit_model(model, t, y, target=None)
    (k_with_target, *_), _ = fit_model(model, t, y, target=50)
    assert k_with_target <= 3.0 * 50 + 1e-6
    _ = k_no_target  # not asserted, just confirms fit didn't fail


def test_fit_raises_on_degenerate_input():
    model = LogisticModel()
    t = np.array([0.0, 1.0])
    y = np.array([np.nan, np.inf])
    with pytest.raises(ForecastError):
        fit_model(model, t, y, target=None)


def test_default_models_tuple_includes_three():
    names = {m.name for m in DEFAULT_MODELS}
    assert names == {"logistic", "gompertz", "asymptotic_exp"}


def test_models_for_small_sample_returns_only_asymptotic_exp():
    from core.forecast.models import models_for_n_points

    selected = models_for_n_points(7)
    assert len(selected) == 1
    assert selected[0].name == "asymptotic_exp"


def test_models_for_large_sample_returns_all_three():
    from core.forecast.models import models_for_n_points

    selected = models_for_n_points(50)
    assert {m.name for m in selected} == {"logistic", "gompertz", "asymptotic_exp"}


def test_models_for_threshold_boundary():
    from core.forecast.models import SMALL_SAMPLE_THRESHOLD, models_for_n_points

    # На самому порозі — повний набір
    assert len(models_for_n_points(SMALL_SAMPLE_THRESHOLD)) == 3
    # На 1 нижче — лише AsymptoticExp
    assert len(models_for_n_points(SMALL_SAMPLE_THRESHOLD - 1)) == 1
