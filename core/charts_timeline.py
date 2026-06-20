"""Plotly-чарт для timeline + forecast.

Окремий модуль (а не доповнення до `core/charts.py`), бо це composite
figure з кількома trace'ами і вертикальними/горизонтальними markerами,
з власною семантикою — не вписується у "fabricate one chart per question
type" сцена `charts.py`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.detection import Changepoint
from core.forecast import ForecastResult
from core.timeline import TimelineSeries

_INCLUDED_COLOR = "#1f77b4"
_EXCLUDED_COLOR = "rgba(150, 150, 150, 0.55)"
_CHANGEPOINT_COLOR = "#ff7f0e"  # помаранчевий — хвилі агітації


@dataclass(frozen=True)
class ChartAxisRanges:
    """Рекомендовані межі осей для сфокусованого перегляду forecast-графіка."""

    x: tuple[pd.Timestamp, pd.Timestamp]
    y: tuple[float, float]


def forecast_window_axis_ranges(
    timestamps: Iterable,
    start_idx: int,
    end_idx: int,
    forecast: ForecastResult | None,
) -> ChartAxisRanges | None:
    """Обчислити межі осей для вибраного вікна навчання і його прогнозу.

    Індекси у UI 1-based, бо слайсер показує відповіді як 1..N. Межі включають
    вибраний фактичний інтервал, прогнозний горизонт і CI, а також невеликий
    запас, щоб лінії та підписи не притискались до рамки графіка.
    """
    parsed = _to_datetime_series(timestamps)
    if parsed.empty:
        return None

    n = len(parsed)
    start = max(1, min(int(start_idx), n))
    end = max(start, min(int(end_idx), n))
    selected = parsed.iloc[start - 1 : end]
    if selected.empty:
        return None

    x_values: list[pd.Timestamp] = [selected.iloc[0], selected.iloc[-1]]
    y_values: list[float] = [float(start), float(end)]

    if forecast is not None:
        future_dates = _to_datetime_series(forecast.future_dates)
        if not future_dates.empty:
            x_values.extend([future_dates.iloc[0], future_dates.iloc[-1]])
        y_values.extend(_finite_numeric_values(forecast.future_cum))
        y_values.extend(_finite_numeric_values(forecast.ci_lower))
        y_values.extend(_finite_numeric_values(forecast.ci_upper))

    x_min = min(x_values)
    x_max = max(x_values)
    x_span = x_max - x_min
    x_pad = max(x_span * 0.05, pd.Timedelta(minutes=30))

    y_min = min(y_values)
    y_max = max(y_values)
    y_span = y_max - y_min
    y_pad = max(y_span * 0.08, 1.0)

    return ChartAxisRanges(
        x=(x_min - x_pad, x_max + x_pad),
        y=(max(0.0, y_min - y_pad), y_max + y_pad),
    )


def plot_timeline_with_forecast(
    timeline: TimelineSeries,
    forecast: ForecastResult | None,
    excluded_mask: np.ndarray | None = None,
    changepoints: list[Changepoint] | None = None,
) -> go.Figure:
    """Скомпонувати чарт кумулятиву + прогнозу + хвиль агітації.

    Лейаут:
    - Step-крива з маркерами: кожна відповідь — окрема точка.
    - Якщо `excluded_mask` — точки з True сірі (виключені з фіту).
    - Пунктирна синя лінія: прогнозний future_cum (якщо forecast).
    - Затемнена зона: 95% prediction interval.
    - Помаранчеві вертикальні dashed-лінії: виявлені CP (хвилі агітації).

    Args:
        timeline: повний timeline з усіма timestamps.
        forecast: результат прогнозу або None.
        excluded_mask: bool-масив довжини N; True → виключено з фіту.
        changepoints: список виявлених CP для візуалізації. None або
            пустий → не малюємо маркери.
    """
    fig = go.Figure()

    _add_fact_traces(fig, timeline, excluded_mask)
    boundary = _compute_forecast_boundary(timeline, excluded_mask)
    _add_forecast_traces(fig, forecast, boundary)
    _add_changepoint_markers(fig, changepoints)

    fig.update_layout(
        title="Динаміка надходження відповідей",
        xaxis_title="Дата",
        yaxis_title="К-сть відповідей (кумулятив)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def _to_datetime_series(values: Iterable) -> pd.Series:
    parsed = pd.to_datetime(list(values), errors="coerce")
    return pd.Series(parsed).dropna().reset_index(drop=True)


def _finite_numeric_values(values: Iterable) -> list[float]:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return [float(value) for value in numeric]


def _add_fact_traces(
    fig: go.Figure,
    timeline: TimelineSeries,
    excluded_mask: np.ndarray | None,
) -> None:
    """Додати трасу(и) фактичної кумулятивної кривої.

    Якщо mask=None або всі False — один трейс синім. Інакше два трейси:
    включені (сині) і виключені (сірі); y-значення зберігають глобальну
    нумерацію 1..N, щоб точки лишалися на своїх "висотах" у cumulative.
    """
    if timeline.timestamps.empty:
        return

    n = len(timeline.timestamps)
    ts_array = np.asarray(timeline.timestamps)
    y_global = np.arange(1, n + 1)

    has_exclusions = excluded_mask is not None and bool(np.asarray(excluded_mask).any())

    if not has_exclusions:
        fig.add_trace(
            go.Scatter(
                x=ts_array,
                y=y_global,
                mode="lines+markers",
                name="Фактично",
                line=dict(color=_INCLUDED_COLOR, width=2, shape="hv"),
                marker=dict(size=5),
            )
        )
        return

    mask = np.asarray(excluded_mask, dtype=bool)

    # Виключені: малюємо ПЕРШИМ, щоб синя крива була зверху.
    if mask.any():
        fig.add_trace(
            go.Scatter(
                x=ts_array[mask],
                y=y_global[mask],
                mode="lines+markers",
                name="Виключено з фіту",
                line=dict(color=_EXCLUDED_COLOR, width=2, shape="hv"),
                marker=dict(size=5, color=_EXCLUDED_COLOR),
            )
        )
    if (~mask).any():
        fig.add_trace(
            go.Scatter(
                x=ts_array[~mask],
                y=y_global[~mask],
                mode="lines+markers",
                name="Фактично",
                line=dict(color=_INCLUDED_COLOR, width=2, shape="hv"),
                marker=dict(size=5),
            )
        )


def _compute_forecast_boundary(
    timeline: TimelineSeries, excluded_mask: np.ndarray | None
) -> tuple | None:
    """Останній факт-point, від якого має «стартувати» forecast-крива.

    Без mask — останній timestamp і його глобальна y-координата (= N).
    З mask — останній *включений* timestamp і його глобальний індекс.
    Потрібно, щоб прогноз візуально продовжував факт, а не висів окремо.
    """
    if timeline.timestamps.empty:
        return None
    if excluded_mask is None or not bool(np.asarray(excluded_mask).any()):
        return timeline.timestamps.iloc[-1], len(timeline.timestamps)
    mask = np.asarray(excluded_mask, dtype=bool)
    included_idx = np.where(~mask)[0]
    if len(included_idx) == 0:
        return None
    last_inc = int(included_idx[-1])
    return timeline.timestamps.iloc[last_inc], last_inc + 1


def _add_forecast_traces(
    fig: go.Figure,
    forecast: ForecastResult | None,
    boundary: tuple | None,
) -> None:
    if forecast is None or forecast.future_cum.empty:
        return

    # Приплюсовуємо boundary-point (останній факт): дає візуальну неперервність
    # між фактом і прогнозом, і гарантує ≥ 2 точки навіть для horizon=1
    # (інакше mode="lines" нічого не малює, CI-polygon degenerate).
    future_dates = list(forecast.future_dates)
    future_cum = list(forecast.future_cum.values)
    ci_lower = list(forecast.ci_lower.values)
    ci_upper = list(forecast.ci_upper.values)
    if boundary is not None:
        b_ts, b_y = boundary
        future_dates = [b_ts] + future_dates
        future_cum = [float(b_y)] + future_cum
        ci_lower = [float(b_y)] + ci_lower
        ci_upper = [float(b_y)] + ci_upper

    # CI band — додаємо першим, щоб лінія прогнозу була зверху.
    fig.add_trace(
        go.Scatter(
            x=future_dates + future_dates[::-1],
            y=ci_upper + ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="95% CI",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=future_cum,
            mode="lines+markers",  # markers — щоб single-point horizon було видно
            name=f"Прогноз ({forecast.model})",
            line=dict(color=_INCLUDED_COLOR, width=2, dash="dash"),
            marker=dict(size=6, symbol="diamond-open"),
        )
    )


def _add_changepoint_markers(fig: go.Figure, changepoints: list[Changepoint] | None) -> None:
    """Вертикальні dashed-лінії на timestamp'ах виявлених хвиль агітації.

    Малюємо як shapes (на paper-y-axis), не як scatter — щоб не з'являлись
    у legend і не "ламались" hovermode="x unified".
    """
    if not changepoints:
        return
    for cp in changepoints:
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=cp.timestamp,
            x1=cp.timestamp,
            y0=0,
            y1=1,
            line=dict(color=_CHANGEPOINT_COLOR, width=1, dash="dash"),
        )
    # Один annotation на ВСІ маркери — щоб не дублювати legend-noise.
    fig.add_annotation(
        x=changepoints[-1].timestamp,
        y=1.02,
        xref="x",
        yref="paper",
        text=f"🔶 хвиль виявлено: {len(changepoints)}",
        showarrow=False,
        font=dict(color=_CHANGEPOINT_COLOR, size=10),
        xanchor="right",
    )
