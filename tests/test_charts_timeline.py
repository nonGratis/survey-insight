from __future__ import annotations

import pandas as pd

from core.charts_timeline import forecast_window_axis_ranges
from core.forecast import ForecastResult


def _forecast() -> ForecastResult:
    future_dates = pd.date_range("2025-01-01 06:00", periods=3, freq="h")
    return ForecastResult(
        model="gompertz",
        aicc=10.0,
        future_dates=future_dates,
        future_cum=pd.Series([7.0, 9.0, 11.0]),
        ci_lower=pd.Series([6.0, 7.0, 8.0]),
        ci_upper=pd.Series([8.0, 12.0, 16.0]),
        final_estimate=11,
        final_ci=(8, 16),
        rmse=1.0,
        r_squared=0.9,
    )


def test_forecast_window_axis_ranges_focuses_selected_window_without_forecast() -> None:
    timestamps = pd.date_range("2025-01-01", periods=10, freq="h")

    axis_ranges = forecast_window_axis_ranges(timestamps, start_idx=4, end_idx=6, forecast=None)

    assert axis_ranges is not None
    assert axis_ranges.x[0] < timestamps[3]
    assert axis_ranges.x[1] > timestamps[5]
    assert axis_ranges.x[1] < timestamps[-1]
    assert axis_ranges.y[0] < 4
    assert axis_ranges.y[1] > 6


def test_forecast_window_axis_ranges_includes_forecast_horizon_and_ci() -> None:
    timestamps = pd.date_range("2025-01-01", periods=6, freq="h")

    axis_ranges = forecast_window_axis_ranges(
        timestamps,
        start_idx=4,
        end_idx=6,
        forecast=_forecast(),
    )

    assert axis_ranges is not None
    assert axis_ranges.x[0] < timestamps[3]
    assert axis_ranges.x[1] > pd.Timestamp("2025-01-01 08:00")
    assert axis_ranges.y[0] < 4
    assert axis_ranges.y[1] > 16


def test_forecast_window_axis_ranges_clamps_invalid_indices() -> None:
    timestamps = pd.date_range("2025-01-01", periods=3, freq="h")

    axis_ranges = forecast_window_axis_ranges(timestamps, start_idx=-10, end_idx=99, forecast=None)

    assert axis_ranges is not None
    assert axis_ranges.x[0] < timestamps[0]
    assert axis_ranges.x[1] > timestamps[-1]
    assert axis_ranges.y[0] == 0.0
    assert axis_ranges.y[1] > 3


def test_forecast_window_axis_ranges_returns_none_for_empty_timestamps() -> None:
    assert forecast_window_axis_ranges([], start_idx=1, end_idx=1, forecast=None) is None
