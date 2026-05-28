"""Unit tests для ``core.detection`` (CP detection + ETL)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.detection import (
    Changepoint,
    InsufficientDataError,
    detect_changepoints,
    median_smooth,
    to_cumulative,
    to_rate_series,
)


def _datetime_index(n: int, freq: str = "1h", start: str = "2025-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq=freq)


# ---------- to_rate_series --------------------------------------------------


def test_to_rate_series_groups_and_preserves_gaps() -> None:
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2025-01-01 00:30",
                "2025-01-01 00:45",
                "2025-01-01 01:10",
                "2025-01-01 03:00",  # дві порожні години в проміжку
            ]
        )
    )
    rate = to_rate_series(timestamps, freq="1h")
    assert rate.index[0] == pd.Timestamp("2025-01-01 00:00")
    assert rate.index[-1] == pd.Timestamp("2025-01-01 03:00")
    assert rate.loc["2025-01-01 00:00"] == 2
    assert rate.loc["2025-01-01 01:00"] == 1
    assert rate.loc["2025-01-01 02:00"] == 0  # gap зберігається


def test_to_rate_series_empty_raises() -> None:
    with pytest.raises(InsufficientDataError):
        to_rate_series(pd.Series([], dtype="datetime64[ns]"))


def test_to_rate_series_unparseable_raises() -> None:
    with pytest.raises(InsufficientDataError):
        to_rate_series(pd.Series(["abc", "def"], dtype="object"))


def test_to_rate_series_single_event() -> None:
    rate = to_rate_series(pd.Series(pd.to_datetime(["2025-01-01 12:00"])))
    assert len(rate) == 1
    assert int(rate.iloc[0]) == 1


# ---------- median_smooth ---------------------------------------------------


def test_median_smooth_neutralises_spike() -> None:
    series = pd.Series([5, 5, 5, 500, 5, 5, 5], dtype="int64")
    smoothed = median_smooth(series, window=3)
    # Медіана [5, 500, 5] == 5; спайк нейтралізується.
    assert int(smoothed.iloc[3]) == 5
    assert int(smoothed.iloc[2]) == 5
    assert int(smoothed.iloc[4]) == 5


def test_median_smooth_invalid_window_raises() -> None:
    with pytest.raises(ValueError):
        median_smooth(pd.Series([1, 2, 3]), window=0)


def test_median_smooth_preserves_dtype_and_length() -> None:
    series = pd.Series(np.arange(10), dtype="int64")
    smoothed = median_smooth(series, window=3)
    assert smoothed.dtype == series.dtype
    assert len(smoothed) == len(series)


# ---------- to_cumulative ---------------------------------------------------


def test_to_cumulative_basic() -> None:
    idx = _datetime_index(5)
    rate = pd.Series([1, 2, 3, 4, 5], index=idx, dtype="int64")
    cumulative = to_cumulative(rate)
    assert cumulative.tolist() == [1, 3, 6, 10, 15]
    assert cumulative.index.equals(idx)


# ---------- detect_changepoints ---------------------------------------------


def test_detect_changepoints_finds_step_changes() -> None:
    # Дві хвилі: тиха фаза (mean=1) → активна (mean=20) → тиха (mean=1).
    rng = np.random.default_rng(0)
    segments = [
        np.full(30, 1.0),
        np.full(30, 20.0),
        np.full(30, 1.0),
    ]
    values = np.concatenate(segments) + rng.normal(0, 0.5, size=90)
    values = np.maximum(values, 0).round().astype("int64")
    rate = pd.Series(values, index=_datetime_index(90), dtype="int64")

    cps = detect_changepoints(rate, penalty=5.0, min_segment=5)
    detected = [cp.index for cp in cps]

    # Очікуємо принаймні дві точки розриву поблизу i=30 та i=60.
    assert any(abs(i - 30) <= 3 for i in detected), f"Не знайдено CP біля 30, отримано: {detected}"
    assert any(abs(i - 60) <= 3 for i in detected), f"Не знайдено CP біля 60, отримано: {detected}"


def test_detect_changepoints_returns_empty_for_short_series() -> None:
    idx = _datetime_index(5)
    rate = pd.Series([1, 2, 1, 2, 1], index=idx, dtype="int64")
    assert detect_changepoints(rate, min_segment=5) == []


def test_detect_changepoints_no_changes_in_stationary_process() -> None:
    rng = np.random.default_rng(1)
    idx = _datetime_index(60)
    rate = pd.Series(rng.poisson(5, size=60).astype("int64"), index=idx)
    # Стаціонарний процес — при достатньому penalty очікуємо 0 changepoints.
    assert detect_changepoints(rate, penalty=50.0) == []


def test_detect_changepoints_returns_changepoint_dataclass() -> None:
    rng = np.random.default_rng(2)
    values = np.concatenate([np.full(20, 1.0), np.full(20, 15.0)]) + rng.normal(0, 0.3, size=40)
    rate = pd.Series(
        np.maximum(values, 0).round().astype("int64"),
        index=_datetime_index(40),
    )
    cps = detect_changepoints(rate, penalty=3.0, min_segment=5)
    assert len(cps) >= 1
    for cp in cps:
        assert isinstance(cp, Changepoint)
        assert isinstance(cp.index, int)
        assert isinstance(cp.timestamp, pd.Timestamp)
        assert cp.timestamp == rate.index[cp.index]
