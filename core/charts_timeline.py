"""Altair-чарт для timeline + forecast.

Шари: факт сходинками, прогноз із довірчим інтервалом і позначки хвиль агітації.
Модуль повертає специфікацію Vega-Lite (`alt.LayerChart`): її малює Streamlit, а
`chart.to_dict()` — будь-який інший фронтенд.

Часові мітки відповідей — naive UTC (Forms API `createTime`). На графік вони йдуть
як UTC-моменти, тож браузер показує їх у місцевому часі глядача.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd

from core.detection import Changepoint
from core.forecast import ForecastResult
from core.timeline import TimelineSeries

_INCLUDED_COLOR = "#1f77b4"
_EXCLUDED_COLOR = "rgba(150, 150, 150, 0.55)"
_BAND_COLOR = "rgba(31, 119, 180, 0.15)"
_CHANGEPOINT_COLOR = "#ff7f0e"  # помаранчевий — хвилі агітації

FACT_LABEL = "Фактично"
EXCLUDED_LABEL = "Виключено з фіту"
BAND_LABEL = "95% CI"
FORECAST_LABEL = "Прогноз"

_DATE = "Дата"
_SERIES = "Ряд"
_RESPONSE = "Відповідь №"
_FORECAST = "Прогноз"
_LOWER = "Нижня межа"
_UPPER = "Верхня межа"
_MODEL = "Модель"
_Y_TITLE = "К-сть відповідей (кумулятив)"
_TOOLTIP_TIME = "%d.%m.%Y %H:%M"
# Опівнічні поділки підписуємо датою, решту — часом: вбудовані формати Vega
# пишуть місяці англійською. Поділок небагато, щоб Vega не ховав підписи від
# тісноти (тоді зникали саме дати); крок він обирає сам, і при масштабуванні теж.
_X_LABEL_EXPR = (
    "hours(datum.value) == 0 && minutes(datum.value) == 0"
    " ? timeFormat(datum.value, '%d.%m') : timeFormat(datum.value, '%H:%M')"
)
_X_TICK_COUNT = 8
_X_PADDING_PX = 10


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
    axis_ranges: ChartAxisRanges | None = None,
) -> alt.LayerChart:
    """Скомпонувати чарт кумулятиву + прогнозу + хвиль агітації.

    Лейаут:
    - Сходинкова крива з маркерами: кожна відповідь — окрема точка.
    - Якщо `excluded_mask` — точки з True сірі (виключені з фіту).
    - Пунктирна синя лінія: прогнозний future_cum (якщо forecast).
    - Затемнена зона: 95% prediction interval.
    - Помаранчеві вертикальні пунктири: виявлені CP (хвилі агітації).

    Args:
        timeline: повний timeline з усіма timestamps.
        forecast: результат прогнозу або None.
        excluded_mask: bool-масив довжини N; True → виключено з фіту.
        changepoints: список виявлених CP для візуалізації. None або
            пустий → не малюємо маркери.
        axis_ranges: межі осей для фокусу на вікні прогнозу; None → увесь ряд.
            Те, що виходить за межі, обрізається.
    """
    x = _x_encoding(axis_ranges)
    y_scale = alt.Scale(domain=list(axis_ranges.y)) if axis_ranges is not None else alt.Scale()
    legend: dict[str, str] = {}  # підпис ряду → колір, у порядку легенди

    fact = _fact_frame(timeline, excluded_mask)
    fact_layers: list[alt.Chart] = []
    # Виключені малюємо першими, щоб синя крива була зверху.
    for label, series_color in ((EXCLUDED_LABEL, _EXCLUDED_COLOR), (FACT_LABEL, _INCLUDED_COLOR)):
        rows = fact[fact[_SERIES] == label]
        if not rows.empty:
            legend[label] = series_color
            fact_layers.append(_fact_layer(rows, x, y_scale))
    if not fact_layers:  # відповідей ще немає: лишаємо порожні осі
        legend[FACT_LABEL] = _INCLUDED_COLOR
        fact_layers.append(_fact_layer(fact, x, y_scale))

    layers: list[alt.Chart] = []
    boundary = _compute_forecast_boundary(timeline, excluded_mask)
    projected = _forecast_frame(forecast, boundary)
    if forecast is not None and projected is not None:
        # Назва моделі — у підказці: у легенді довгий підпис не вміщується на телефоні.
        projected[_MODEL] = forecast.model
        legend[BAND_LABEL] = _BAND_COLOR
        legend[FORECAST_LABEL] = _INCLUDED_COLOR
        # CI band — першим, щоб лінія прогнозу була зверху.
        layers.append(
            alt.Chart(projected.assign(**{_SERIES: BAND_LABEL}))
            .mark_area(clip=True)
            .encode(
                x=x,
                y=alt.Y(field=_LOWER, type="quantitative", title=_Y_TITLE, scale=y_scale),
                y2=alt.Y2(field=_UPPER),
            )
        )
        layers.extend(fact_layers)
        layers.append(
            alt.Chart(projected.assign(**{_SERIES: FORECAST_LABEL}))
            .mark_line(
                strokeDash=[6, 4],
                clip=True,
                # маркери — щоб single-point horizon було видно
                point=alt.OverlayMarkDef(shape="diamond", filled=False, size=50, clip=True),
            )
            .encode(
                x=x,
                y=alt.Y(field=_FORECAST, type="quantitative", title=_Y_TITLE, scale=y_scale),
                tooltip=[
                    alt.Tooltip(field=_DATE, type="temporal", format=_TOOLTIP_TIME),
                    alt.Tooltip(field=_FORECAST, type="quantitative", format=".0f"),
                    alt.Tooltip(field=_LOWER, type="quantitative", format=".0f"),
                    alt.Tooltip(field=_UPPER, type="quantitative", format=".0f"),
                    alt.Tooltip(field=_MODEL, type="nominal"),
                ],
            )
        )
    else:
        layers.extend(fact_layers)

    color = alt.Color(
        field=_SERIES,
        type="nominal",
        scale=alt.Scale(domain=list(legend), range=list(legend.values())),
        # Два стовпчики: в один рядок легенда не вміщується на телефоні.
        legend=alt.Legend(orient="top", title=None, columns=2),
    )
    layers = [layer.encode(color=color) for layer in layers]
    layers.extend(_changepoint_layers(changepoints, x))

    # Масштаб і зсув мишею, як у .interactive().
    return (
        alt.LayerChart(layer=layers)
        .properties(title="Динаміка надходження відповідей", height=420)
        .add_params(alt.selection_interval(bind="scales"))
    )


def _to_datetime_series(values: Iterable) -> pd.Series:
    parsed = pd.to_datetime(list(values), errors="coerce")
    return pd.Series(parsed).dropna().reset_index(drop=True)


def _finite_numeric_values(values: Iterable) -> list[float]:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return [float(value) for value in numeric]


def _as_utc(values: Iterable) -> pd.Series:
    """Naive-мітки вважаємо UTC; aware переводимо в UTC."""
    parsed = pd.Series(pd.to_datetime(list(values)))
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize("UTC")
    return parsed.dt.tz_convert("UTC")


def _utc_datetime(value: pd.Timestamp) -> alt.DateTime:
    """Межа осі як UTC-момент — так само, як дані на графіку."""
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return alt.DateTime(
        year=ts.year,
        month=ts.month,
        date=ts.day,
        hours=ts.hour,
        minutes=ts.minute,
        seconds=ts.second,
        utc=True,
    )


def _x_encoding(axis_ranges: ChartAxisRanges | None) -> alt.X:
    # Без фокусу — запас у кілька пікселів, щоб крайні маркери не різались навпіл;
    # межі фокусу вже містять свій запас.
    scale = (
        alt.Scale(domain=[_utc_datetime(axis_ranges.x[0]), _utc_datetime(axis_ranges.x[1])])
        if axis_ranges is not None
        else alt.Scale(padding=_X_PADDING_PX)
    )
    return alt.X(
        field=_DATE,
        type="temporal",
        title="Дата",
        scale=scale,
        axis=alt.Axis(labelExpr=_X_LABEL_EXPR, tickCount=_X_TICK_COUNT),
    )


def _fact_frame(timeline: TimelineSeries, excluded_mask: np.ndarray | None) -> pd.DataFrame:
    """Фактичні відповіді з глобальною нумерацією 1..N.

    Виключені й включені точки лишаються на своїх «висотах» кумулятиву, тож
    сіра і синя криві не зсуваються одна відносно одної.
    """
    n = len(timeline.timestamps)
    excluded = np.zeros(n, dtype=bool) if excluded_mask is None else np.asarray(excluded_mask, bool)
    return pd.DataFrame(
        {
            _DATE: _as_utc(timeline.timestamps),
            _RESPONSE: np.arange(1, n + 1),
            _SERIES: np.where(excluded, EXCLUDED_LABEL, FACT_LABEL),
        }
    )


def _fact_layer(rows: pd.DataFrame, x: alt.X, y_scale: alt.Scale) -> alt.Chart:
    """Сходинкова крива з маркером на кожній відповіді."""
    return (
        alt.Chart(rows)
        .mark_line(
            interpolate="step-after",
            clip=True,
            point=alt.OverlayMarkDef(size=25, clip=True),
        )
        .encode(
            x=x,
            y=alt.Y(field=_RESPONSE, type="quantitative", title=_Y_TITLE, scale=y_scale),
            tooltip=[
                alt.Tooltip(field=_DATE, type="temporal", format=_TOOLTIP_TIME),
                alt.Tooltip(field=_RESPONSE, type="quantitative"),
                alt.Tooltip(field=_SERIES, type="nominal"),
            ],
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


def _forecast_frame(forecast: ForecastResult | None, boundary: tuple | None) -> pd.DataFrame | None:
    """Точки прогнозу і CI, починаючи з останнього факту.

    Boundary-point дає візуальну неперервність між фактом і прогнозом і
    гарантує ≥ 2 точки навіть для horizon=1 (інакше лінія й смуга вироджені).
    """
    if forecast is None or forecast.future_cum.empty:
        return None

    future_dates = list(forecast.future_dates)
    future_cum = [float(v) for v in forecast.future_cum.values]
    ci_lower = [float(v) for v in forecast.ci_lower.values]
    ci_upper = [float(v) for v in forecast.ci_upper.values]
    if boundary is not None:
        b_ts, b_y = boundary
        future_dates = [b_ts] + future_dates
        future_cum = [float(b_y)] + future_cum
        ci_lower = [float(b_y)] + ci_lower
        ci_upper = [float(b_y)] + ci_upper

    return pd.DataFrame(
        {
            _DATE: _as_utc(future_dates),
            _FORECAST: future_cum,
            _LOWER: ci_lower,
            _UPPER: ci_upper,
        }
    )


def _changepoint_layers(changepoints: list[Changepoint] | None, x: alt.X) -> list[alt.Chart]:
    """Вертикальні пунктири на хвилях агітації і один підпис на всі.

    Поза легендою: це позначки подій, а не ряд даних.
    """
    if not changepoints:
        return []
    frame = pd.DataFrame({_DATE: _as_utc(cp.timestamp for cp in changepoints)})
    rules = (
        alt.Chart(frame)
        .mark_rule(color=_CHANGEPOINT_COLOR, strokeDash=[4, 4], strokeWidth=1, clip=True)
        .encode(
            x=x,
            tooltip=[alt.Tooltip(field=_DATE, type="temporal", format=_TOOLTIP_TIME)],
        )
    )
    label = (
        alt.Chart(frame.tail(1).assign(Підпис=f"🔶 хвиль виявлено: {len(changepoints)}"))
        .mark_text(
            align="right",
            baseline="top",
            dx=-4,
            dy=4,
            color=_CHANGEPOINT_COLOR,
            fontSize=10,
            clip=True,
        )
        .encode(x=x, y=alt.value(0), text=alt.Text(field="Підпис", type="nominal"))
    )
    return [rules, label]
