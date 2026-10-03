from __future__ import annotations

import numpy as np
import pandas as pd

from core.charts_timeline import (
    BAND_LABEL,
    EXCLUDED_LABEL,
    FACT_LABEL,
    FORECAST_LABEL,
    forecast_window_axis_ranges,
    plot_timeline_with_forecast,
)
from core.detection import Changepoint
from core.forecast import ForecastResult
from core.timeline import TimelineSeries, build_timeline_from_timestamps


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


def _timeline(n: int = 6) -> TimelineSeries:
    stamps = list(pd.date_range("2025-01-01", periods=n, freq="h").to_pydatetime())
    return build_timeline_from_timestamps(stamps)


def _spec(**kwargs: object) -> dict:
    """Vega-Lite spec of the chart; ``to_dict`` also validates it against the schema."""
    return plot_timeline_with_forecast(**kwargs).to_dict()  # type: ignore[arg-type]


def _rows(spec: dict, layer: dict) -> list[dict]:
    # with a single layer Altair moves its data up to the chart itself
    data = layer.get("data", spec.get("data"))
    return spec["datasets"][data["name"]]


def _legend(spec: dict) -> dict[str, str]:
    scale = spec["layer"][0]["encoding"]["color"]["scale"]
    return dict(zip(scale["domain"], scale["range"], strict=True))


def test_chart_without_forecast_draws_every_response_as_utc_instants() -> None:
    spec = _spec(timeline=_timeline(), forecast=None)

    assert list(_legend(spec)) == [FACT_LABEL]
    [fact] = spec["layer"]
    rows = _rows(spec, fact)
    assert [row["Відповідь №"] for row in rows] == [1, 2, 3, 4, 5, 6]
    # naive Forms API timestamps are UTC; the browser shows them in local time
    assert rows[0]["Дата"].startswith("2025-01-01T00:00:00")
    assert rows[0]["Дата"].endswith("+00:00")
    assert fact["mark"]["interpolate"] == "step-after"


def test_forecast_continues_from_the_last_fact_point_with_its_band() -> None:
    spec = _spec(timeline=_timeline(), forecast=_forecast())

    assert list(_legend(spec)) == [FACT_LABEL, BAND_LABEL, FORECAST_LABEL]
    band, fact, line = spec["layer"]
    assert band["mark"]["type"] == "area"
    assert line["mark"]["strokeDash"] == [6, 4]
    first = _rows(spec, line)[0]
    assert first["Прогноз"] == first["Нижня межа"] == first["Верхня межа"] == 6.0
    assert first["Дата"].startswith("2025-01-01T05:00:00")
    assert [row["Модель"] for row in _rows(spec, line)] == ["gompertz"] * 4
    assert {"field": "Модель", "type": "nominal"} in line["encoding"]["tooltip"]


def test_excluded_points_are_a_grey_series_under_the_fact_curve() -> None:
    mask = np.array([True, True, False, False, False, False])

    spec = _spec(timeline=_timeline(), forecast=_forecast(), excluded_mask=mask)

    legend = _legend(spec)
    assert list(legend) == [EXCLUDED_LABEL, FACT_LABEL, BAND_LABEL, FORECAST_LABEL]
    assert legend[EXCLUDED_LABEL].startswith("rgba(150, 150, 150")
    _band, excluded, fact, _line = spec["layer"]
    # global numbering: the grey and blue curves keep their heights
    assert [row["Відповідь №"] for row in _rows(spec, excluded)] == [1, 2]
    assert [row["Відповідь №"] for row in _rows(spec, fact)] == [3, 4, 5, 6]


def test_changepoints_are_dashed_rules_with_one_label_outside_the_legend() -> None:
    timeline = _timeline()
    changepoints = [
        Changepoint(timeline.timestamps.iloc[2], 2),
        Changepoint(timeline.timestamps.iloc[4], 4),
    ]

    spec = _spec(timeline=timeline, forecast=None, changepoints=changepoints)

    _fact, rules, label = spec["layer"]
    assert rules["mark"]["type"] == "rule"
    assert len(_rows(spec, rules)) == 2
    assert [row["Підпис"] for row in _rows(spec, label)] == ["🔶 хвиль виявлено: 2"]
    assert "color" not in rules["encoding"]
    assert list(_legend(spec)) == [FACT_LABEL]


def test_axis_ranges_fix_both_domains_and_clip_the_marks() -> None:
    timeline = _timeline()
    ranges = forecast_window_axis_ranges(timeline.timestamps, 4, 6, _forecast())
    assert ranges is not None

    spec = _spec(timeline=timeline, forecast=_forecast(), axis_ranges=ranges)

    for layer in spec["layer"]:
        assert layer["mark"]["clip"] is True
    fact = spec["layer"][1]
    x_start, x_end = fact["encoding"]["x"]["scale"]["domain"]
    assert x_start["utc"] is True and x_end["utc"] is True
    assert (x_start["date"], x_start["hours"]) == (ranges.x[0].day, ranges.x[0].hour)
    assert (x_end["date"], x_end["hours"]) == (ranges.x[1].day, ranges.x[1].hour)
    assert fact["encoding"]["y"]["scale"]["domain"] == list(ranges.y)


def test_empty_timeline_still_builds_a_valid_chart() -> None:
    spec = _spec(timeline=build_timeline_from_timestamps([]), forecast=None)

    [fact] = spec["layer"]
    assert _rows(spec, fact) == []
    assert list(_legend(spec)) == [FACT_LABEL]
