"""Plotly-чарт для timeline + forecast.

Окремий модуль (а не доповнення до `core/charts.py`), бо це composite
figure з кількома trace'ами і вертикальними/горизонтальними markerами,
з власною семантикою — не вписується у "fabricate one chart per question
type" сцена `charts.py`.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from core.forecast import ForecastResult
from core.timeline import TimelineSeries

_INCLUDED_COLOR = "#1f77b4"
_EXCLUDED_COLOR = "rgba(150, 150, 150, 0.55)"


def plot_timeline_with_forecast(
    timeline: TimelineSeries,
    forecast: ForecastResult | None,
    target: int | None,
    excluded_mask: np.ndarray | None = None,
) -> go.Figure:
    """Скомпонувати чарт кумулятиву + прогнозу + цільового маркера.

    Лейаут:
    - Step-крива з маркерами: кожна відповідь — окрема точка, y стрибає +1
      у момент її timestamp'у (shape="hv" — горизонталь, потім вертикаль).
    - Якщо передано `excluded_mask` — точки з True малюються сірими
      напівпрозорими (виключені з фіту прогнозу), інші — синіми.
    - Пунктирна синя лінія: прогнозний future_cum (якщо forecast).
    - Затемнена зона: 95% prediction interval (ci_lower..ci_upper).
    - Зелена горизонталь з підписом: target N (якщо задано).

    Дедлайн прибрано — модель сама визначає горизонт як 25% від
    тривалості опитування (див. core.forecast.forecast_responses).

    Графік малюється з `timeline.timestamps` (per-response), тоді як прогноз
    усе ще працює на denoised daily — це дві незалежні концерни.

    Args:
        timeline: побудована TimelineSeries з повним списком timestamps.
        forecast: результат прогнозу або None.
        target: цільова кількість відповідей або None.
        excluded_mask: bool-масив довжини len(timeline.timestamps); True
            означає "виключено з вікна прогнозу". None → усе включено.
    """
    fig = go.Figure()

    _add_fact_traces(fig, timeline, excluded_mask)
    _add_forecast_traces(fig, forecast)
    _add_target_marker(fig, target)

    fig.update_layout(
        title="Динаміка надходження відповідей",
        xaxis_title="Дата",
        yaxis_title="К-сть відповідей (кумулятив)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


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


def _add_forecast_traces(fig: go.Figure, forecast: ForecastResult | None) -> None:
    if forecast is None or forecast.future_cum.empty:
        return
    # CI band — додаємо першим, щоб лінія прогнозу була зверху.
    fig.add_trace(
        go.Scatter(
            x=list(forecast.future_dates) + list(forecast.future_dates[::-1]),
            y=list(forecast.ci_upper.values) + list(forecast.ci_lower.values[::-1]),
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
            x=forecast.future_dates,
            y=forecast.future_cum.values,
            mode="lines",
            name=f"Прогноз ({forecast.model})",
            line=dict(color=_INCLUDED_COLOR, width=2, dash="dash"),
        )
    )


def _add_target_marker(fig: go.Figure, target: int | None) -> None:
    if target is None or target <= 0:
        return
    # Той самий патерн для горизонталі — заради консистентності і захисту
    # від майбутніх Plotly-регресій (hline+annotation на числовій осі
    # зараз працює, але краще не покладатись).
    fig.add_shape(
        type="line",
        xref="paper",
        yref="y",
        x0=0,
        x1=1,
        y0=target,
        y1=target,
        line=dict(color="#2ca02c", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=1.0,
        xref="paper",
        y=target,
        yref="y",
        text=f"Мета: {target}",
        showarrow=False,
        xanchor="right",
        yshift=10,
        font=dict(color="#2ca02c"),
    )
