"""Tests for core.forecast.wave_service.forecast_current_wave (prod entry)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.forecast import ForecastError, ForecastResult, forecast_current_wave
from core.timeline import build_timeline_from_timestamps

BASE = pd.Timestamp("2025-01-01 09:00:00")


def _burst(start: pd.Timestamp, n: int, span_min: float) -> list:
    step_s = round(span_min * 60 / max(n - 1, 1))
    return [(start + pd.Timedelta(seconds=step_s * i)).to_pydatetime() for i in range(n)]


def _two_wave_timeline():
    # Wave 1: 8 resp / 40 min. Gap 15h. Wave 2: 8 resp / 40 min.
    w1 = _burst(BASE, 8, 40)
    w2 = _burst(BASE + pd.Timedelta(hours=15), 8, 40)
    return build_timeline_from_timestamps(w1 + w2)


def test_returns_forecast_result_and_changepoints():
    tl = _two_wave_timeline()
    fc, cps = forecast_current_wave(tl, form_type="event_registration")
    assert isinstance(fc, ForecastResult)
    assert isinstance(cps, list)
    # Second wave detected → at least one agitation changepoint (wave #0 excluded).
    assert len(cps) >= 1


def test_ci_brackets_point_and_caps():
    tl = _two_wave_timeline()
    fc, _ = forecast_current_wave(tl, form_type="event_registration")
    lo, hi = fc.final_ci
    assert lo <= fc.final_estimate <= hi
    # cap policy: half-width <= 2x point (never absurd).
    half = (hi - lo) / 2.0
    assert half <= 2.0 * max(fc.final_estimate, 1) + 1


def test_future_cum_monotone_nondecreasing():
    tl = _two_wave_timeline()
    fc, _ = forecast_current_wave(tl, form_type="event_registration")
    vals = fc.future_cum.to_numpy()
    assert np.all(np.diff(vals) >= -1e-9)


def test_final_estimate_at_least_current():
    tl = _two_wave_timeline()
    current = int(tl.cumulative.iloc[-1])
    fc, _ = forecast_current_wave(tl, form_type="event_registration")
    assert fc.final_estimate >= current
    assert fc.final_ci[0] >= 0


def test_fallback_short_current_wave():
    # Wave 1 (8 resp), gap 15h, wave 2 only 3 resp → current wave < 5 → fallback
    # to forecast_responses (P17 capped). Must still return a sane ForecastResult.
    w1 = _burst(BASE, 8, 40)
    w2 = _burst(BASE + pd.Timedelta(hours=15), 3, 20)
    tl = build_timeline_from_timestamps(w1 + w2)
    fc, cps = forecast_current_wave(tl, form_type="event_registration")
    assert isinstance(fc, ForecastResult)
    # Fallback output also respects cap (no absurd interval).
    half = (fc.final_ci[1] - fc.final_ci[0]) / 2.0
    assert half <= 2.0 * max(fc.final_estimate, 1) + 1


def test_empty_timeline_raises_forecast_error():
    # Empty input → ForecastError (same contract as forecast_responses; the UI
    # catches it in _cached_forecast and shows "—").
    tl = build_timeline_from_timestamps([])
    with pytest.raises(ForecastError):
        forecast_current_wave(tl)


def test_unknown_form_type_uses_default():
    tl = _two_wave_timeline()
    fc, _ = forecast_current_wave(tl, form_type="nonexistent_xyz")
    assert isinstance(fc, ForecastResult)
    assert fc.final_estimate >= 0
