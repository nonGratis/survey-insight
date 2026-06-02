"""Tests for core.forecast.wave_detector (CUSUM agitation detector)."""

from __future__ import annotations

import pandas as pd
import pytest

from core.forecast.wave_detector import (
    WaveStart,
    current_wave_timestamps,
    detect_wave_starts,
)

BASE = pd.Timestamp("2025-01-01 09:00:00")


def _burst(start: pd.Timestamp, n: int, span_min: float) -> list[pd.Timestamp]:
    """n timestamps evenly spread over span_min minutes from start."""
    if n == 1:
        return [start]
    step = span_min / (n - 1)
    return [start + pd.Timedelta(minutes=step * i) for i in range(n)]


def test_empty_input_returns_empty():
    assert detect_wave_starts([]) == []


def test_single_burst_is_one_wave():
    ts = _burst(BASE, 8, span_min=40)
    waves = detect_wave_starts(ts, form_type="event_registration")
    assert len(waves) == 1
    assert waves[0].wave_index == 0
    assert waves[0].timestamp == BASE


def test_first_response_always_wave_zero():
    ts = _burst(BASE, 6, span_min=30)
    waves = detect_wave_starts(ts)
    assert waves[0].timestamp == ts[0]
    assert isinstance(waves[0], WaveStart)


def test_two_waves_separated_by_silence():
    # Wave 1: 6 resp in 40 min. Gap 15h. Wave 2: 6 resp in 40 min.
    w1 = _burst(BASE, 6, span_min=40)
    w2_start = BASE + pd.Timedelta(hours=15)
    w2 = _burst(w2_start, 6, span_min=40)
    waves = detect_wave_starts(w1 + w2, form_type="event_registration")
    assert len(waves) == 2
    # Second wave start should be near w2_start (within the burst)
    assert waves[1].timestamp >= w2_start
    assert waves[1].silence_before_h > 6.0


def test_two_waves_with_default_params():
    w1 = _burst(BASE, 6, span_min=40)
    w2 = _burst(BASE + pd.Timedelta(hours=15), 6, span_min=40)
    waves = detect_wave_starts(w1 + w2, form_type=None)
    assert len(waves) == 2


def test_test_skip_strips_initial_responses():
    # 2 isolated test responses, then a real wave 30h later.
    tests = [BASE, BASE + pd.Timedelta(minutes=5)]
    real = _burst(BASE + pd.Timedelta(hours=30), 6, span_min=40)
    waves = detect_wave_starts(tests + real, form_type="event_registration", test_skip=2)
    # After skipping 2, first real response is wave 0.
    assert waves[0].timestamp == real[0]


def test_unsorted_input_is_sorted():
    ts = _burst(BASE, 6, span_min=30)
    shuffled = [ts[3], ts[0], ts[5], ts[1], ts[4], ts[2]]
    waves = detect_wave_starts(shuffled)
    assert waves[0].timestamp == ts[0]


def test_short_gap_does_not_split_wave():
    # survey min_silence=12h. Two bursts 50 min apart never reach 12h
    # silence → must stay a single wave.
    w1 = _burst(BASE, 4, span_min=20)
    w2 = _burst(BASE + pd.Timedelta(minutes=50), 4, span_min=20)
    waves = detect_wave_starts(w1 + w2, form_type="survey")
    assert len(waves) == 1


def test_current_wave_returns_last_wave_only():
    w1 = _burst(BASE, 6, span_min=40)
    w2_start = BASE + pd.Timedelta(hours=15)
    w2_end = w2_start + pd.Timedelta(minutes=40)
    w2 = _burst(w2_start, 6, span_min=40)
    start, wave_ts = current_wave_timestamps(w1 + w2, form_type="event_registration")
    # Detector triggers ~2-3 responses into the burst, so the detected
    # start lags the true wave start but stays inside the w2 window.
    assert w2_start <= start <= w2_end
    assert (wave_ts >= start).all()
    # At most the full w2 burst; at least the tail after the trigger point.
    assert 3 <= len(wave_ts) <= 6


def test_current_wave_single_wave_returns_all():
    ts = _burst(BASE, 8, span_min=40)
    start, wave_ts = current_wave_timestamps(ts, form_type="event_registration")
    assert start == BASE
    assert len(wave_ts) == 8


def test_unknown_form_type_falls_back_to_default():
    ts = _burst(BASE, 6, span_min=30)
    # A made-up type not in priors → default params, no crash.
    waves = detect_wave_starts(ts, form_type="nonexistent_type_xyz")
    assert len(waves) >= 1
