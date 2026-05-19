"""Plotly-чарт для timeline + forecast.

Окремий модуль (а не доповнення до `core/charts.py`), бо це composite
figure з кількома trace'ами і вертикальними/горизонтальними markerами,
з власною семантикою — не вписується у "fabricate one chart per question
type" сцена `charts.py`.
"""

from __future__ import annotations

import plotly.graph_objects as go

from core.forecast import ForecastResult
from core.timeline import TimelineSeries


def plot_timeline_with_forecast(
    timeline: TimelineSeries,
    forecast: ForecastResult | None,
    target: int | None,
) -> go.Figure:
    """Скомпонувати чарт кумулятиву + прогнозу + цільового маркера.

    Лейаут:
    - Суцільна синя лінія: фактичний cumulative (timeline).
    - Пунктирна синя лінія: прогнозний future_cum (якщо forecast).
    - Затемнена зона: 95% prediction interval (ci_lower..ci_upper).
    - Зелена горизонталь з підписом: target N (якщо задано).

    Дедлайн прибрано — модель сама визначає горизонт як 25% від
    тривалості опитування (див. core.forecast.forecast_responses).
    """
    fig = go.Figure()

    if not timeline.cumulative.empty:
        fig.add_trace(
            go.Scatter(
                x=timeline.cumulative.index,
                y=timeline.cumulative.values,
                mode="lines+markers",
                name="Фактично",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=5),
            )
        )

    if forecast is not None and not forecast.future_cum.empty:
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
                line=dict(color="#1f77b4", width=2, dash="dash"),
            )
        )

    if target is not None and target > 0:
        # Той самий patern для горизонталі — заради консистентності і захисту
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

    fig.update_layout(
        title="Динаміка надходження відповідей",
        xaxis_title="Дата",
        yaxis_title="К-сть відповідей (кумулятив)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
