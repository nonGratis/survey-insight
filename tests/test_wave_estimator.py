"""Tests for core.forecast.wave_estimator (within-wave + final estimate)."""

from __future__ import annotations

import pandas as pd
import pytest

from core.forecast.types import ForecastError
from core.forecast.wave_estimator import (
    FinalFormEstimate,
    WaveForecast,
    estimate_final,
    estimate_wave,
)

BASE = pd.Timestamp("2025-01-01 09:00:00")

# A decaying-rate wave: inter-arrivals grow → saturating cumulative.
_DECAY_MINUTES = [0, 2, 4, 7, 10, 14, 19, 25, 32, 40, 50, 62, 75, 90]
DECAY_WAVE = [BASE + pd.Timedelta(minutes=m) for m in _DECAY_MINUTES]


def test_estimate_wave_basic_contract():
    wf = estimate_wave(DECAY_WAVE, horizon_h=2.0, form_type="survey")
    assert isinstance(wf, WaveForecast)
    n = len(DECAY_WAVE)
    # Point must be >= observed count (cumulative monotone floor).
    assert wf.point >= n
    # CI brackets the point.
    assert wf.ci_lower <= wf.point <= wf.ci_upper
    # Lower bound never below observed count.
    assert wf.ci_lower >= n
    assert wf.model_name in {"logistic", "gompertz", "asymptotic_exp", "linear", "bass"}
    assert 0.0 <= wf.r_squared <= 1.0
    assert wf.n_wave_responses == n


def test_estimate_wave_raises_below_five():
    with pytest.raises(ForecastError):
        estimate_wave(DECAY_WAVE[:4], horizon_h=2.0)


def test_estimate_wave_exactly_five_ok():
    wf = estimate_wave(DECAY_WAVE[:5], horizon_h=1.0)
    assert wf.point >= 5


def test_estimate_wave_longer_horizon_not_smaller():
    # Saturating: cumulative at 6h >= cumulative at 1h.
    wf_short = estimate_wave(DECAY_WAVE, horizon_h=1.0, form_type="survey")
    wf_long = estimate_wave(DECAY_WAVE, horizon_h=6.0, form_type="survey")
    assert wf_long.point >= wf_short.point


def test_estimate_wave_extrapolation_flag():
    # Train span ~1.5h; horizon 24h → flagged as extrapolating.
    wf = estimate_wave(DECAY_WAVE, horizon_h=24.0)
    assert wf.is_extrapolating is True
    # Horizon within train span → not extrapolating.
    wf2 = estimate_wave(DECAY_WAVE, horizon_h=1.0)
    assert wf2.is_extrapolating is False


def test_estimate_final_at_least_current_cum():
    all_ts = DECAY_WAVE
    wave_ts = DECAY_WAVE
    wf = estimate_wave(wave_ts, horizon_h=3.0, form_type="survey")
    fe = estimate_final(all_ts, wave_ts, wf, form_type="survey", n_waves_seen=1)
    assert isinstance(fe, FinalFormEstimate)
    assert fe.point >= fe.current_cum
    assert fe.ci_lower >= fe.current_cum
    assert fe.ci_upper >= fe.point
    assert fe.current_cum == len(all_ts)


def test_estimate_final_future_waves_add_volume():
    # If many waves expected but few seen, point > current_cum.
    all_ts = DECAY_WAVE
    wf = estimate_wave(DECAY_WAVE, horizon_h=3.0, form_type="event_registration")
    fe = estimate_final(all_ts, DECAY_WAVE, wf, form_type="event_registration", n_waves_seen=1)
    # event_registration expects ~5 waves → future volume added.
    assert fe.future_waves_expected >= 0
    assert fe.point >= fe.current_cum


def test_estimate_final_all_waves_seen_no_future():
    all_ts = DECAY_WAVE
    wf = estimate_wave(DECAY_WAVE, horizon_h=3.0, form_type="survey")
    # Seen more waves than the type median → no future waves added.
    fe = estimate_final(all_ts, DECAY_WAVE, wf, form_type="survey", n_waves_seen=99)
    assert fe.future_waves_expected == 0.0


def test_estimate_wave_accepts_pandas_series():
    wf = estimate_wave(pd.Series(DECAY_WAVE), horizon_h=2.0)
    assert wf.point >= len(DECAY_WAVE)
