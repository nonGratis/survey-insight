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
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .delta_ci import cap_width, delta_method_ci
from .models import models_for_n_points
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


def _apply_conformal_ci(
    point_arr: np.ndarray,
    lo_arr: np.ndarray,
    hi_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mondrian-conformal CI calibration (winner of benchmark 19_).

    Scales the raw delta-CI half-width by a per-pred-bucket factor calibrated
    on annotated data, then applies the hard cap_width sanity policy. Bucketing
    is by the within-wave point at horizon (point_arr[-1]); pre_count shifts
    bounds and truth equally, so coverage is preserved when pre_count is added
    downstream. Holdout coverage ~86% (see wave_priors.json:ci_calibration).

    Falls back to a plain cap when no calibration is present.
    """
    calib = _priors().get("ci_calibration")
    if not calib:
        return cap_width(point_arr, lo_arr, hi_arr, max_relative=2.0, min_absolute=10.0)

    edges = calib["bucket_edges"]
    q_by_bucket = calib["bucket_q"]
    point_at_h = float(point_arr[-1])
    bucket = 0
    for e in edges:
        if point_at_h >= e:
            bucket += 1
    q = float(q_by_bucket[min(bucket, len(q_by_bucket) - 1)])

    half = (hi_arr - lo_arr) / 2.0
    scaled_lo = point_arr - q * half
    scaled_hi = point_arr + q * half
    return cap_width(
        point_arr,
        scaled_lo,
        scaled_hi,
        max_relative=calib["cap_max_relative"],
        min_absolute=calib["cap_min_absolute"],
    )


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


@dataclass
class WaveCurve:
    """Full within-wave forecast curve over a future-hours grid.

    Spine used by both estimate_wave (single-horizon point) and
    wave_service.forecast_current_wave (full curve → ForecastResult).
    """

    t0: datetime  # wave start
    t_future_h: np.ndarray  # hours from wave start
    cum: np.ndarray  # point estimate (monotone, >= n)
    lower: np.ndarray  # conformal CI lower (>= n)
    upper: np.ndarray  # conformal CI upper (>= cum)
    model_name: str
    aicc: float
    rmse: float
    r_squared: float
    n: int  # wave response count
    train_span_h: float


def wave_forecast_curve(
    wave_timestamps: list | pd.Series,
    t_future_h: np.ndarray,
    form_type: str | None = None,
) -> WaveCurve:
    """Fit saturating model on the current wave, predict over t_future_h.

    Single fit path (DRY) for the within-wave forecast: model selection +
    monotone point curve + delta-CI + Mondrian-conformal calibration + hard cap.

    Args:
        wave_timestamps: response timestamps FROM the current wave start only.
        t_future_h: future grid in HOURS from wave start (predict points).
        form_type: reserved for future per-type fitting (CI is calibrated
            globally by predicted magnitude; detector handles per-type).

    Returns:
        WaveCurve with cum / lower / upper over t_future_h + fitted diagnostics.

    Raises:
        ForecastError: if < 5 responses (can't fit reliably).
    """
    ts = pd.Series(pd.to_datetime(wave_timestamps)).sort_values().reset_index(drop=True)
    n = len(ts)
    if n < 5:
        raise ForecastError(f"wave_forecast_curve: need >=5 responses, got {n}")

    t0 = ts.iloc[0].to_pydatetime()
    t_train = np.array([(t.to_pydatetime() - t0).total_seconds() / 3600.0 for t in ts])
    y_train = np.arange(1, n + 1, dtype=float)
    t_future = np.asarray(t_future_h, dtype=float)

    fitted = select_best_model(t_train, y_train, target=None, models=models_for_n_points(n))

    cum = np.asarray(fitted.model.predict(t_future, *fitted.params), dtype=float)
    cum = np.maximum.accumulate(np.maximum(cum, float(n)))  # monotone + floor

    # Delta-CI + Mondrian-conformal calibration + hard cap (benchmark 19_ winner).
    try:
        lower, upper = delta_method_ci(fitted, t_future, n_train=n)
        lower, upper = _apply_conformal_ci(cum, lower, upper)
    except ValueError:
        lower = cum.copy()
        upper = cum.copy()

    lower = np.maximum(lower, float(n))  # cumulative floor
    upper = np.maximum(np.maximum(upper, cum), float(n))

    return WaveCurve(
        t0=t0,
        t_future_h=t_future,
        cum=cum,
        lower=lower,
        upper=upper,
        model_name=fitted.model.name,
        aicc=float(fitted.aicc),
        rmse=float(fitted.rmse),
        r_squared=float(fitted.r_squared),
        n=n,
        train_span_h=float(t_train[-1]),
    )


def estimate_wave(
    wave_timestamps: list | pd.Series,
    horizon_h: float = 3.0,
    form_type: str | None = None,
) -> WaveForecast:
    """Fit saturating model on current wave and predict at horizon_h.

    Thin wrapper over `wave_forecast_curve` returning the single horizon point.

    Args:
        wave_timestamps: response timestamps FROM wave start (current wave only).
        horizon_h: prediction horizon in hours from wave start.
        form_type: passed through to wave_forecast_curve (reserved).

    Returns:
        WaveForecast with point ± CI at horizon_h.

    Raises:
        ForecastError: if < 5 responses (can't fit reliably).
    """
    ts = pd.Series(pd.to_datetime(wave_timestamps)).sort_values().reset_index(drop=True)
    if len(ts) < 5:
        raise ForecastError(f"estimate_wave: need >=5 responses, got {len(ts)}")
    last_h = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 3600.0
    is_extrapolating = horizon_h > 3.0 * max(last_h, 0.01)

    # Future grid: from last observed point to horizon (>=2 points for delta-CI).
    t_future = np.linspace(
        max(last_h + 0.1, 0.1),
        max(horizon_h, last_h + 0.1),
        max(10, int(horizon_h * 4)),
    )
    curve = wave_forecast_curve(ts, t_future, form_type=form_type)

    return WaveForecast(
        point=int(round(float(curve.cum[-1]))),
        ci_lower=int(round(float(curve.lower[-1]))),
        ci_upper=int(round(float(curve.upper[-1]))),
        model_name=curve.model_name,
        r_squared=curve.r_squared,
        n_wave_responses=curve.n,
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
