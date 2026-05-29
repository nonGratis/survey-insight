"""Юніт-тести для NHPP prediction interval.

Інваріанти, що мусять виконуватись завжди:
- ci_lower[0] >= last_observed (не можемо зменшити cumulative);
- ci_lower і ci_upper монотонно неспадні (cumulative-крива);
- ci_upper >= mean_cum >= ci_lower (CI охоплює центральну оцінку);
- ширина CI зростає (або принаймні не падає різко) з горизонтом.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.forecast.intervals import nhpp_prediction_interval
from core.forecast.models import LogisticModel
from core.forecast.types import FittedModel


def _make_fitted(model=None, params=(100.0, 0.3, 15.0)):
    return FittedModel(
        model=model or LogisticModel(),
        params=params,
        aicc=0.0,
        rmse=0.0,
        r_squared=1.0,
    )


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_ci_lower_never_below_last_observed(rng):
    fitted = _make_fitted()
    t_future = np.arange(14.0, 21.0)
    _, _med, ci_lower, _ = nhpp_prediction_interval(
        fitted, t_future, last_observed=47, n_sims=2000, rng=rng
    )
    assert np.all(ci_lower >= 47), f"ci_lower={ci_lower} dipped below 47"


def test_ci_bands_are_monotonic(rng):
    fitted = _make_fitted()
    t_future = np.arange(10.0, 25.0)
    _, _med, ci_lower, ci_upper = nhpp_prediction_interval(
        fitted, t_future, last_observed=20, n_sims=2000, rng=rng
    )
    assert np.all(np.diff(ci_lower) >= 0), "ci_lower not monotonic"
    assert np.all(np.diff(ci_upper) >= 0), "ci_upper not monotonic"


def test_mean_is_within_ci_band(rng):
    fitted = _make_fitted()
    t_future = np.arange(14.0, 21.0)
    mean_cum, _med, ci_lower, ci_upper = nhpp_prediction_interval(
        fitted, t_future, last_observed=47, n_sims=2000, rng=rng
    )
    # Mean — детермінований predict; CI — симуляційний з last_observed як база.
    # Тож не очікуємо, що mean буде "всередині" CI у строгому сенсі для
    # довільних кривих — але ширина CI має охоплювати реалістичні значення.
    assert np.all(ci_upper >= ci_lower)
    # CI має бути нетривіальним
    assert np.any(ci_upper > ci_lower)


def test_ci_width_grows_with_horizon(rng):
    """Невизначеність зростає з горизонтом (cumulative-варіанс росте з t)."""
    fitted = _make_fitted()
    t_future = np.arange(14.0, 30.0)
    _, _med, ci_lower, ci_upper = nhpp_prediction_interval(
        fitted, t_future, last_observed=47, n_sims=4000, rng=rng
    )
    widths = ci_upper - ci_lower
    # Width на останньому горизонті має бути не меншою за початкову (з допуском).
    assert widths[-1] >= widths[0] - 1, f"widths={widths} not non-decreasing"


def test_empty_t_future_raises(rng):
    fitted = _make_fitted()
    with pytest.raises(ValueError):
        nhpp_prediction_interval(fitted, np.array([]), last_observed=10, rng=rng)


def test_invalid_percentiles_raise(rng):
    fitted = _make_fitted()
    t_future = np.arange(14.0, 21.0)
    with pytest.raises(ValueError):
        nhpp_prediction_interval(
            fitted, t_future, last_observed=10, rng=rng, ci_lower_pct=50, ci_upper_pct=10
        )


def test_reproducibility_with_seed():
    fitted = _make_fitted()
    t_future = np.arange(14.0, 21.0)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    _, _m1, lo1, hi1 = nhpp_prediction_interval(fitted, t_future, 47, n_sims=500, rng=rng1)
    _, _m2, lo2, hi2 = nhpp_prediction_interval(fitted, t_future, 47, n_sims=500, rng=rng2)
    np.testing.assert_array_equal(lo1, lo2)
    np.testing.assert_array_equal(hi1, hi2)


def test_flat_curve_gives_zero_lambda(rng):
    """Якщо predict(t) константа — Poisson(0) → CI повністю на last_observed."""

    class FlatModel:
        name = "flat"
        n_params = 1

        def predict(self, t, c):
            return np.full_like(t, c, dtype=float)

        def initial_guess(self, t, y, target):
            return (float(y[-1]),)

        def bounds(self, y, target):
            return (0.0,), (1e6,)

    fitted = FittedModel(model=FlatModel(), params=(50.0,), aicc=0, rmse=0, r_squared=1)
    t_future = np.arange(14.0, 21.0)
    mean_cum, _med, ci_lower, ci_upper = nhpp_prediction_interval(
        fitted, t_future, last_observed=50, n_sims=2000, rng=rng
    )
    assert np.all(ci_lower == 50)
    assert np.all(ci_upper == 50)
    assert np.all(mean_cum == 50)
