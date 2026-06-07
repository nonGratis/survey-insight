"""Tests for core.forecast.delta_ci (P12)."""

from __future__ import annotations

import numpy as np
import pytest

from core.forecast.delta_ci import cap_width, delta_method_ci
from core.forecast.metrics import aicc, r_squared, rmse
from core.forecast.models import AsymptoticExpModel, fit_model
from core.forecast.types import FittedModel


def _make_fitted(n: int = 20, rng_seed: int = 42) -> FittedModel:
    """Helper: fit AsympExp to clean synthetic data."""
    rng = np.random.default_rng(rng_seed)
    t = np.arange(n, dtype=float)
    y_true = 50.0 * (1.0 - np.exp(-0.15 * t)) + 5.0
    y = y_true + rng.normal(0, 0.5, size=n)
    y = np.maximum.accumulate(y)  # ensure monotonic for cumulative

    model = AsymptoticExpModel()
    params, pcov = fit_model(model, t, y, target=None)
    y_fit = model.predict(t, *params)
    return FittedModel(
        model=model,
        params=params,
        aicc=aicc(y, y_fit, model.n_params),
        rmse=rmse(y, y_fit),
        r_squared=r_squared(y, y_fit),
        pcov=pcov,
    )


class TestDeltaMethodCI:
    def test_returns_finite_arrays(self):
        fitted = _make_fitted()
        t_future = np.array([25.0, 30.0, 35.0])
        lower, upper = delta_method_ci(fitted, t_future, n_train=20)

        assert lower.shape == t_future.shape
        assert upper.shape == t_future.shape
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))

    def test_upper_geq_lower(self):
        fitted = _make_fitted()
        t_future = np.linspace(20, 50, 10)
        lower, upper = delta_method_ci(fitted, t_future, n_train=20)
        assert np.all(upper >= lower)

    def test_point_inside_ci(self):
        """y_pred повинен бути всередині [lower, upper] (CI симетричний навколо predict)."""
        fitted = _make_fitted()
        t_future = np.array([25.0])
        lower, upper = delta_method_ci(fitted, t_future, n_train=20)
        y_pred = fitted.model.predict(t_future, *fitted.params)[0]
        assert lower[0] <= y_pred <= upper[0]

    def test_higher_confidence_wider_ci(self):
        """95% CI ширше за 50% CI."""
        fitted = _make_fitted()
        t_future = np.array([25.0])
        lo_95, hi_95 = delta_method_ci(fitted, t_future, n_train=20, confidence=0.95)
        lo_50, hi_50 = delta_method_ci(fitted, t_future, n_train=20, confidence=0.50)
        assert (hi_95[0] - lo_95[0]) > (hi_50[0] - lo_50[0])

    def test_wider_extrapolation_wider_ci(self):
        """CI на дальшому горизонті ширша (більше extrapolation uncertainty)."""
        fitted = _make_fitted()
        lo_near, hi_near = delta_method_ci(fitted, np.array([21.0]), n_train=20)
        lo_far, hi_far = delta_method_ci(fitted, np.array([100.0]), n_train=20)
        assert (hi_far[0] - lo_far[0]) > (hi_near[0] - lo_near[0])

    def test_none_pcov_raises(self):
        fitted = _make_fitted()
        bad = FittedModel(
            model=fitted.model,
            params=fitted.params,
            aicc=fitted.aicc,
            rmse=fitted.rmse,
            r_squared=fitted.r_squared,
            pcov=None,
        )
        with pytest.raises(ValueError, match="pcov is None"):
            delta_method_ci(bad, np.array([25.0]), n_train=20)

    def test_non_finite_pcov_raises(self):
        fitted = _make_fitted()
        bad_pcov = np.full_like(fitted.pcov, np.nan)
        bad = FittedModel(
            model=fitted.model,
            params=fitted.params,
            aicc=fitted.aicc,
            rmse=fitted.rmse,
            r_squared=fitted.r_squared,
            pcov=bad_pcov,
        )
        with pytest.raises(ValueError, match="NaN/Inf"):
            delta_method_ci(bad, np.array([25.0]), n_train=20)


class TestCapWidth:
    def test_no_clamp_when_sane(self):
        """sane CI лишається не зміненою."""
        point = np.array([100.0])
        lo = np.array([95.0])
        hi = np.array([105.0])
        clo, chi = cap_width(point, lo, hi, max_relative=5.0)
        # width = 10, cap = min(100*5, 5000)/2 = 250. half=5 ≤ 250. No clamp.
        assert chi[0] - clo[0] == pytest.approx(10.0)

    def test_clamps_excessive_width(self):
        """Width > cap → clamped to cap."""
        point = np.array([50.0])
        lo = np.array([10.0])
        hi = np.array([1000.0])
        clo, chi = cap_width(point, lo, hi, max_relative=5.0, max_absolute=5000.0)
        # cap = min(50*5, 5000) = 250. width capped to 250.
        assert chi[0] - clo[0] == pytest.approx(250.0)

    def test_floor_at_min_absolute(self):
        """Tiny point → floor floor."""
        point = np.array([2.0])
        lo = np.array([1.0])
        hi = np.array([100.0])
        clo, chi = cap_width(point, lo, hi, max_relative=5.0, min_absolute=20.0)
        # 2*5=10 < min_absolute=20 → cap=20. width capped at 20.
        assert chi[0] - clo[0] == pytest.approx(20.0)

    def test_symmetric_output(self):
        """cap_width виводить симетричний CI навколо point."""
        point = np.array([100.0])
        lo = np.array([0.0])
        hi = np.array([1000.0])
        clo, chi = cap_width(point, lo, hi, max_relative=2.0)
        # cap=200, half=100. CI = [100-100, 100+100] = [0, 200].
        assert chi[0] - point[0] == pytest.approx(point[0] - clo[0])

    def test_array_shape_preserved(self):
        point = np.array([10.0, 20.0, 30.0])
        lo = np.array([8.0, 15.0, 25.0])
        hi = np.array([12.0, 25.0, 35.0])
        clo, chi = cap_width(point, lo, hi)
        assert clo.shape == (3,)
        assert chi.shape == (3,)
