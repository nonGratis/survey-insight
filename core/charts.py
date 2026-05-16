"""Plotly-фабрики графіків під типи питань форми.

Чисті функції, що приймають pandas DataFrame і назву колонки,
повертають готовий Figure або DataFrame (для частотних таблиць).
Streamlit-агностично — щоб можна було перевикористати у PDF-експорті,
ноутбуках, тестах.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def bar_categorical(df: pd.DataFrame, col: str, title: str | None = None) -> Figure:
    """Горизонтальний bar-chart частот для MULTIPLE_CHOICE / CHECKBOX."""
    series = df[col].fillna("(порожньо)").replace("", "(порожньо)")
    counts = series.value_counts()
    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        title=title or col,
        labels={"x": "К-сть відповідей", "y": ""},
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def hist_ordinal(df: pd.DataFrame, col: str, title: str | None = None) -> Figure:
    """Гістограма для LINEAR_SCALE (числові шкали 1-5, 1-10 тощо)."""
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        return Figure().update_layout(
            title=title or col,
            annotations=[
                dict(
                    text="Немає числових значень",
                    showarrow=False,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                )
            ],
        )
    nbins = max(int(vals.max() - vals.min() + 1), 5)
    fig = px.histogram(
        vals,
        nbins=nbins,
        title=title or col,
        labels={"value": col, "count": "Частота"},
    )
    fig.update_layout(showlegend=False, xaxis_title=col, yaxis_title="Частота")
    return fig


def freq_table(df: pd.DataFrame, col: str, top_n: int = 20) -> pd.DataFrame:
    """Топ-N частот для SHORT_ANSWER — за відсутністю кращої агрегації."""
    series = df[col].fillna("").astype(str)
    counts = series[series != ""].value_counts().head(top_n)
    return counts.reset_index().rename(columns={"index": col, "count": "К-сть"})


def response_count(df: pd.DataFrame, col: str) -> int:
    """К-сть непорожніх відповідей у колонці (для метрик)."""
    return int(df[col].fillna("").astype(str).str.strip().astype(bool).sum())
