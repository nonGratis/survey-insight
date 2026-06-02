"""Wave-conditional response estimator.

Two modes:

A) CURRENT WAVE (horizon ≤ 3h):
   Fit asymptotic_exp on responses since wave start.
   Predict at horizon_h with delta-CI.
   Principle: 3h window is well-conditioned for saturating models;
   first hour carries ~50% so we have meaningful shape signal after ~5 resp.

B) FINAL FORM ESTIMATE:
   current_cumulative
   + current_wave_remaining (mode A)
   + expected_future_waves * expected_wave_size_per_type
   CI from historical wave-size spread (wave_priors.json).

Implemented as standalone functions, not coupled to `forecast_responses`.
Use directly when you have wave-level context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .delta_ci import cap_width, delta_method_ci
from .selector import select_best_model
from .types import ForecastError

_PRIORS_PATH = Path(__file__).parent / "wave_priors.json"
_PRIORS_CACHE: dict | None = None


def _priors() -> dict:
    global _PRIORS_CACHE
    if _PRIORS_CACHE is None and _PRIORS_PATH.exists():
        _PRIORS_CACHE = json.loads(_PRIORS_PATH.read_text(encoding="utf-8"))
    return _PRIORS_CACHE or {}


def _type_prior(form_type: str | None) -> dict:
    p = _priors()
    if form_type and "per_type" in p and form_type in p["per_type"]:
        return p["per_type"][form_type]
    # Fallback: global medians
    return {
        "frac_1h_med": 0.50,
        "wave_size_med": 7.0,
        "wave_size_p25": 3.0,
        "wave_size_p75": 20.0,
        "n_waves_med": 4.0,
        "n_waves_p75": 6.0,
    }


@dataclass
class WaveForecast:
    """Result of within-wave estimation."""

    # Point estimate at horizon
    point: int
    ci_lower: int
    ci_upper: int
    # Diagnostics
    model_name: str
    r_squared: float
    n_wave_responses: int  # responses used for fit
    horizon_h: float
    # Extrapolation flag
    is_extrapolating: bool  # True if horizon >> train_span (warning)


@dataclass
class FinalFormEstimate:
    """Full form final estimate (current + future waves)."""

    point: int  # best estimate of total final responses
    ci_lower: int  # conservative (fewer future waves)
    ci_upper: int  # optimistic (more future waves)
    current_cum: int  # responses received so far
    wave_remaining: int  # current wave expected remaining
    future_waves_expected: float  # E[future wave count]
    future_size_per_wave: float  # E[responses per future wave]


def estimate_wave(
    wave_timestamps: list | pd.Series,
    horizon_h: float = 3.0,
    form_type: str | None = None,
) -> WaveForecast:
    """Fit saturating model on current wave and predict at horizon_h.

    Args:
        wave_timestamps: response timestamps FROM wave start (not full form).
            Should contain only current wave's responses.
        horizon_h: prediction horizon in hours from wave start.
        form_type: for CI width fallback via wave_priors.

    Returns:
        WaveForecast with point ± CI at horizon_h.

    Raises:
        ForecastError: if < 5 responses (can't fit reliably).
    """
    ts = pd.Series(pd.to_datetime(wave_timestamps)).sort_values().reset_index(drop=True)
    n = len(ts)
    if n < 5:
        raise ForecastError(f"estimate_wave: need ≥5 responses, got {n}")

    t0 = ts.iloc[0].to_pydatetime()
    t_train = np.array([(t.to_pydatetime() - t0).total_seconds() / 3600.0 for t in ts])
    y_train = np.arange(1, n + 1, dtype=float)
    train_span_h = float(t_train[-1])

    is_extrapolating = horizon_h > 3.0 * max(train_span_h, 0.01)

    # Future grid: hourly from now to horizon
    t_future = np.linspace(
        max(t_train[-1] + 0.1, 0.1),
        max(horizon_h, t_train[-1] + 0.1),
        max(10, int(horizon_h * 4)),
    )

    # Model selection — all models if n ≥ 10, else just asymp_exp
    from .models import models_for_n_points as _mfn

    models = _mfn(n)
    fitted = select_best_model(t_train, y_train, target=None, models=models)

    point_arr = fitted.model.predict(t_future, *fitted.params)
    # Monotonic + floor
    point_arr = np.maximum.accumulate(np.maximum(point_arr, float(n)))
    point_at_h = float(point_arr[-1])

    # Delta-CI
    try:
        lo_arr, hi_arr = delta_method_ci(fitted, t_future, n_train=n)
        lo_arr, hi_arr = cap_width(point_arr, lo_arr, hi_arr, max_relative=2.0, min_absolute=10.0)
    except ValueError:
        lo_arr = point_arr.copy()
        hi_arr = point_arr.copy()

    lo = max(float(lo_arr[-1]), float(n))
    hi = max(float(hi_arr[-1]), float(n), point_at_h)

    return WaveForecast(
        point=int(round(point_at_h)),
        ci_lower=int(round(lo)),
        ci_upper=int(round(hi)),
        model_name=fitted.model.name,
        r_squared=float(fitted.r_squared),
        n_wave_responses=n,
        horizon_h=horizon_h,
        is_extrapolating=is_extrapolating,
    )


def estimate_final(
    all_timestamps: list | pd.Series,
    wave_timestamps: list | pd.Series,
    wave_forecast: WaveForecast,
    form_type: str | None = None,
    n_waves_seen: int = 1,
) -> FinalFormEstimate:
    """Estimate final total responses for the entire form.

    Logic:
      total_point = current_cum + wave_remaining + future_waves * per_wave_size
      CI from wave_size_p25/p75 * future wave count range

    Args:
        all_timestamps: all responses so far (full form).
        wave_timestamps: current wave responses (subset of above).
        wave_forecast: result from estimate_wave() on current wave.
        form_type: for per-type priors.
        n_waves_seen: how many waves detected so far (including current).

    Returns:
        FinalFormEstimate.
    """
    current_cum = len(pd.Series(all_timestamps))
    prior = _type_prior(form_type)

    # Wave remaining in current wave
    wave_remaining = max(0, wave_forecast.point - len(pd.Series(wave_timestamps)))

    # Future waves: E[total waves] - waves_already_seen
    n_waves_expected_total = float(prior.get("n_waves_med", 4.0))
    n_waves_p75 = float(prior.get("n_waves_p75", 6.0))
    future_waves_med = max(0.0, n_waves_expected_total - n_waves_seen)
    future_waves_p75 = max(0.0, n_waves_p75 - n_waves_seen)

    # Per-future-wave size
    wave_size_med = float(prior.get("wave_size_med", 7.0))
    wave_size_p25 = float(prior.get("wave_size_p25", 3.0))
    wave_size_p75 = float(prior.get("wave_size_p75", 20.0))

    point = int(round(current_cum + wave_remaining + future_waves_med * wave_size_med))
    ci_lo = int(round(current_cum + wave_remaining + 0 * wave_size_p25))  # no more waves
    ci_hi = int(round(current_cum + wave_remaining + future_waves_p75 * wave_size_p75))

    return FinalFormEstimate(
        point=point,
        ci_lower=max(ci_lo, current_cum),
        ci_upper=ci_hi,
        current_cum=current_cum,
        wave_remaining=wave_remaining,
        future_waves_expected=future_waves_med,
        future_size_per_wave=wave_size_med,
    )
