"""Tests for core.forecast.test_responses.detect_test_responses."""

from __future__ import annotations

import pandas as pd

from core.forecast.test_responses import detect_test_responses

BASE = pd.Timestamp("2025-01-01 09:00:00")


def _burst(start, n, step_min=8):
    return [start + pd.Timedelta(minutes=step_min * i) for i in range(n)]


def test_short_series_returns_zero():
    assert detect_test_responses(_burst(BASE, 6)) == 0


def test_pure_burst_no_test_skip():
    # Dense first wave (no isolated leaders) → 0 (must NOT chop a real wave).
    ts = _burst(BASE, 20, step_min=5)
    assert detect_test_responses(ts) == 0


def test_single_isolated_test_response():
    # 1 isolated creator test, then 30h gap, then a real burst.
    ts = [BASE] + _burst(BASE + pd.Timedelta(hours=30), 19, step_min=5)
    assert detect_test_responses(ts) == 1


def test_two_isolated_test_responses():
    # 2 sparse leaders (each separated by a big gap), then real burst.
    leaders = [BASE, BASE + pd.Timedelta(hours=5)]
    real = _burst(BASE + pd.Timedelta(hours=40), 18, step_min=5)
    assert detect_test_responses(leaders + real) == 2


def test_skip_capped_by_max_skip():
    ts = [BASE] + _burst(BASE + pd.Timedelta(hours=30), 19, step_min=5)
    assert detect_test_responses(ts, max_skip=0) == 0


def test_accepts_pandas_series():
    ts = [BASE] + _burst(BASE + pd.Timedelta(hours=30), 19, step_min=5)
    assert detect_test_responses(pd.Series(ts)) == 1
