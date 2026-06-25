"""Сторінка «Динаміка» — кумулятив відповідей і прогноз поточної хвилі.

Сторінка відповідає за один сценарій: зрозуміти темп надходження відповідей,
побачити старт нових хвиль агітації та оцінити посадку поточної хвилі без
змішування з аналізом питань або зважуванням.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import streamlit as st

from core.charts_timeline import forecast_window_axis_ranges, plot_timeline_with_forecast
from core.detection import Changepoint
from core.forecast import (
    ForecastError,
    ForecastResult,
    classify_form_type,
    detect_test_responses,
    forecast_current_wave,
)
from core.forms_api import get_linked_sheet_id
from core.logger import get_logger
from core.timeline import build_timeline_from_timestamps
from ui.components.action_bar import ActionBarStatus, render_action_bar, render_action_status
from ui.components.auth_widget import ensure_api_access
from ui.components.form_picker import clear_forms_cache
from ui.components.metric_bar import MetricItem, render_metric_bar
from ui.components.page_shell import (
    render_empty_state,
    render_error_state,
    render_page_header,
    render_state,
)
from ui.google_data import (
    cache_token,
    clear_form_cache,
    get_form_structure,
    list_response_timestamps,
)

log = get_logger(__name__)

render_page_header("Динаміка")

if not ensure_api_access():
    st.stop()

action = render_action_bar(
    refresh_scope="dynamics",
    show_status=False,
)
if not action.selected_form:
    st.stop()
form_id = action.selected_form["id"]


@st.cache_data(ttl=120, show_spinner="Завантажую структуру форми…")
def _cached_structure(form_id_: str, _cache_token: str) -> dict:
    return get_form_structure(form_id_)


try:
    structure = _cached_structure(form_id, cache_token())
except Exception as exc:  # noqa: BLE001
    log.exception("ui_dynamics_get_structure_failed", extra={"form_id": form_id})
    render_error_state("Не вдалося завантажити форму.", details=str(exc))
    st.stop()

form_title = structure.get("info", {}).get("title", "—")
sheet_id = get_linked_sheet_id(structure)

if not sheet_id:
    # Огляд tab працює напряму з Forms API і Sheet не потребує.
    # Решта tabs (По питаннях / Крос-таби / Якість) у майбутніх PR'ах
    # покажуть власне попередження про потребу Sheet — там, де вони
    # реально читатимуть answer values.
    render_state(
        "Ця форма не має прив'язаного Google Sheet. "
        "Огляд працює через Forms API безпосередньо; інші вкладки "
        "потребуватимуть Sheet (Responses → Link to Sheets у формі)."
    )


@st.cache_data(ttl=60, show_spinner="Завантажую відповіді…")
def _cached_timestamps(form_id_: str, _cache_token: str) -> list[datetime]:
    """Forms API timestamps, кешовано на 60s за access_token+form_id."""
    return list_response_timestamps(form_id_)


@st.cache_data(ttl=300, show_spinner="Обчислюю прогноз…")
def _cached_forecast(
    _form_id: str,
    n_responses: int,
    first_ts: datetime,
    last_ts: datetime,
    start_idx: int,
    end_idx: int,
    horizon_until: datetime | None,
    form_type: str,
    _timestamps: list[datetime],
) -> tuple[ForecastResult | None, list[Changepoint], str | None]:
    """Кешований current-wave прогноз для subset'у `_timestamps[start_idx-1:end_idx]`.

    Cache key: (form_id, n_responses, first_ts, last_ts, start_idx, end_idx,
                horizon_until, form_type).

    Прогноз посадки ПОТОЧНОЇ хвилі (`forecast_current_wave`): CUSUM-детектор
    хвиль агітації + within-wave saturation fit + Mondrian-conformal CI.
    Повертає (forecast, changepoints, error_msg); CP = старти хвиль агітації
    (окрім першої) для маркерів на графіку.
    """
    timeline = build_timeline_from_timestamps(_timestamps)
    try:
        fc, cps = forecast_current_wave(timeline, form_type=form_type, horizon_until=horizon_until)
        return fc, cps, None
    except ForecastError as exc:
        return None, [], str(exc)


if action.refresh_clicked:
    clear_forms_cache()
    clear_form_cache(form_id)
    _cached_structure.clear()
    _cached_timestamps.clear()
    _cached_forecast.clear()
    st.rerun()


def _shift_forecast(forecast: ForecastResult, offset: int) -> ForecastResult:
    """Зсунути cumulative-значення прогнозу на `offset` (для subset → global).

    Прогноз рахується на subset'і timestamps, тож його cumulative починається
    з 1 для першої точки subset'у. Графік показує global-нумерацію (1..N
    усіх timestamps); щоб forecast curve візуально продовжувала факт,
    додаємо `offset = start_idx - 1` (= кількість виключених на початку).
    """
    if offset == 0:
        return forecast
    return ForecastResult(
        model=forecast.model,
        aicc=forecast.aicc,
        future_dates=forecast.future_dates,
        future_cum=forecast.future_cum + offset,
        ci_lower=forecast.ci_lower + offset,
        ci_upper=forecast.ci_upper + offset,
        final_estimate=forecast.final_estimate + offset,
        final_ci=(forecast.final_ci[0] + offset, forecast.final_ci[1] + offset),
        rmse=forecast.rmse,
        r_squared=forecast.r_squared,
    )


def _render_forecast_details(
    forecast: ForecastResult,
    *,
    horizon_label: str,
    horizon_end: object,
    changepoints_count: int,
    n_used: int,
    n_total: int,
) -> None:
    """Render compact model details without turning metadata into a caption wall."""
    chips = [
        f"`Модель: {forecast.model}`",
        f"`Горизонт: {horizon_label} · до {horizon_end:%d.%m.%Y}`",
        f"`95% CI: {forecast.final_ci[0]}–{forecast.final_ci[1]}`",
        f"`R²={forecast.r_squared:.3f}`",
        f"`Використано: {n_used}/{n_total}`",
    ]
    if changepoints_count:
        chips.append(f"`Хвиль: {changepoints_count}`")

    st.markdown("**Деталі прогнозу**")
    st.markdown(" ".join(chips))


try:
    timestamps = _cached_timestamps(form_id, cache_token())
except Exception as exc:  # noqa: BLE001
    log.exception(
        "ui_dynamics_list_timestamps_failed",
        extra={"form_id": form_id, "error_code": type(exc).__name__},
    )
    render_error_state("Не вдалося отримати timestamps відповідей.", details=str(exc))
    st.stop()

if timestamps:
    _last_response = timestamps[-1]
    _ago_seconds = max(
        (datetime.now(tz=_last_response.tzinfo) - _last_response).total_seconds(),
        0,
    )
    if _ago_seconds < 3600:
        _fresh_label = f"{int(_ago_seconds // 60)} хв тому"
    elif _ago_seconds < 86400:
        _fresh_label = f"{int(_ago_seconds // 3600)} год тому"
    else:
        _fresh_label = f"{int(_ago_seconds // 86400)} дн тому"
    render_action_status(
        ActionBarStatus(
            responses=len(timestamps),
            updated=_fresh_label,
            note="оперативний моніторинг",
        )
    )
else:
    render_action_status(ActionBarStatus(responses=0, note="оперативний моніторинг"))

# Ця сторінка — лише ДИНАМІКА (прогноз надходження). Якість питань і аналіз
# відповідей живуть на сторінці «Питання». Дизайн-аналіз форми — там само.
with st.container():
    timeline = build_timeline_from_timestamps(timestamps)

    if timeline.cumulative.empty:
        render_empty_state("Поки немає валідних timestamps у відповідях.")
    else:
        # Вікно прогнозу — у cache key. session_state може зберігати stale
        # значення (форма оновилась), тому clamp'имо у [1, n_ts].
        _window_key = f"analysis_window_{form_id}"
        n_ts = len(timestamps)
        # Авто-пропуск провідних тестових відповідей (слайдер стартує після
        # них; користувач може скоригувати вручну).
        test_skip = detect_test_responses(timestamps)
        default_window = (test_skip + 1, max(n_ts, 1))
        start_idx, end_idx = st.session_state.get(_window_key, default_window)
        start_idx = max(1, min(int(start_idx), n_ts))
        end_idx = max(start_idx, min(int(end_idx), n_ts))

        subset_timestamps = timestamps[start_idx - 1 : end_idx]
        excluded_mask = np.zeros(n_ts, dtype=bool)
        excluded_mask[: start_idx - 1] = True
        excluded_mask[end_idx:] = True

        # Горизонт прогнозу: завжди від ОСТАННЬОГО timestamp'у повного
        # timeline + 25% span'у. Це гарантує, що навіть trim'нутий префікс
        # проектує далеко вперед — користувач бачить, чи модель з 16 точок
        # коректно "вгадує" реальні 47 і трошки далі.
        from datetime import timedelta as _td

        full_span_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400.0
        extra_days = max(int(full_span_days * 0.25), 1)
        horizon_until = timestamps[-1] + _td(days=extra_days)

        # Тип форми визначається автоматично за назвою (per-type пороги
        # детектора хвиль). Ручний вибір прибрано — алгоритм обирає краще.
        form_type_choice = classify_form_type(form_title)

        if subset_timestamps:
            forecast, changepoints, forecast_error = _cached_forecast(
                _form_id=form_id,
                n_responses=len(timestamps),
                first_ts=timestamps[0],
                last_ts=timestamps[-1],
                start_idx=start_idx,
                end_idx=end_idx,
                horizon_until=horizon_until,
                form_type=form_type_choice,
                _timestamps=subset_timestamps,
            )
            # Subset → global coordinate shift.
            if forecast is not None:
                forecast = _shift_forecast(forecast, offset=start_idx - 1)
        else:
            forecast, changepoints, forecast_error = None, [], None

        # Рядок 1: BANs — "Зараз", "Прогноз (ця хвиля)", "Стан хвилі" (порадник).
        current = int(timeline.cumulative.iloc[-1])
        metric_items = [MetricItem("Зараз", current)]
        if forecast is not None:
            ci_half = (forecast.final_ci[1] - forecast.final_ci[0]) // 2
            metric_items.append(
                MetricItem(
                    "Прогноз (ця хвиля)",
                    forecast.final_estimate,
                    delta=f"±{ci_half}",
                    delta_color="off",
                    help=(
                        "Скільки набере ПОТОЧНА хвиля, якщо без нової агітації. "
                        "Нова хвиля (нагадування/пост) додасть ще — це твоє рішення."
                    ),
                ),
            )
            # Порадник: % посадки, досягнутий на КІНЕЦЬ ВІКНА прогнозу
            # (не глобальний current — інакше при звуженні вікна завжди 100%).
            wave_now = end_idx  # к-сть відповідей на кінець вікна (глобальна нумерація)
            landing = max(forecast.final_estimate, wave_now, 1)
            pct = min(int(round(wave_now / landing * 100)), 100)
            status = "на плато" if pct >= 100 else ("майже" if pct >= 80 else "набирає")
            metric_items.append(
                MetricItem(
                    "Стан хвилі",
                    f"{pct}%",
                    delta=status,
                    delta_color="off",
                    help=(
                        "Скільки прогнозованої посадки хвилі вже зібрано (на кінець "
                        "вікна). 100% «на плато» → модель вважає цю хвилю сталою; для "
                        "більшого потрібна нова агітація (майбутні хвилі тут не враховані)."
                    ),
                ),
            )
        else:
            metric_items.extend(
                [
                    MetricItem("Прогноз (ця хвиля)", "-"),
                    MetricItem("Стан хвилі", "-"),
                ]
            )
        render_metric_bar(metric_items, columns=3)

        auto_scale = st.checkbox(
            "Автомасштабування вікна прогнозу",
            value=True,
            key=f"dynamics_forecast_auto_scale_{form_id}",
            help=(
                "Наближує графік до вибраного слайсером інтервалу та прогнозного "
                "горизонту. Вимкніть, щоб бачити весь ряд відповідей."
            ),
        )

        # Свіжість даних — анотацією ПОВЕРХ графіка (правий нижній кут).
        # Рядок 2: графік (з CP-маркерами, якщо знайдені хвилі агітації).
        fig = plot_timeline_with_forecast(
            timeline=timeline,
            forecast=forecast,
            excluded_mask=excluded_mask,
            changepoints=changepoints,
        )
        if auto_scale:
            axis_ranges = forecast_window_axis_ranges(timestamps, start_idx, end_idx, forecast)
            if axis_ranges is not None:
                fig.update_xaxes(range=list(axis_ranges.x))
                fig.update_yaxes(range=list(axis_ranges.y))
        fig.add_annotation(
            text=f"Оновлено {_fresh_label}",
            xref="paper",
            yref="paper",
            x=1.0,
            y=0.0,
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            font=dict(color="#888", size=10),
        )
        st.plotly_chart(fig, width="stretch")

        # Рядок 3: маленька іконка-завантаження прогнозу у CSV.
        if forecast is not None:
            _rows = ["date,forecast,ci_lower,ci_upper"]
            for _d, _c, _lo, _hi in zip(
                forecast.future_dates,
                forecast.future_cum,
                forecast.ci_lower,
                forecast.ci_upper,
                strict=False,
            ):
                _rows.append(f"{_d.isoformat()},{int(_c)},{int(_lo)},{int(_hi)}")
            st.download_button(
                ":material/download:",
                data="\n".join(_rows),
                file_name=f"forecast_{form_id}.csv",
                mime="text/csv",
                help="Завантажити прогнозну криву (дата, прогноз, нижня/верхня межа CI) у CSV",
                width="content",
            )

        # Рядок 4: range-slider — обирає вікно для фіту прогнозу. Full-width
        # під графіком; рендериться завжди (а не лише при n_ts ≥ 2), щоб
        # компонент займав те саме місце і не "стрибав" UI.
        if n_ts >= 2:
            st.slider(
                "Вікно навчання прогнозу",
                min_value=1,
                max_value=n_ts,
                value=(start_idx, end_idx),
                step=1,
                key=_window_key,
                help=(
                    "Перетягни ручки, щоб обмежити, які відповіді "
                    "використовуються для фіту прогнозу. Сірі точки на графіку — "
                    "виключені."
                ),
            )

        # Рядок 5: compact details — модель/якість фіту/горизонт + хвилі.
        n_used = end_idx - start_idx + 1
        if forecast is not None:
            horizon_end = forecast.future_dates[-1].date()
            _last_obs = timestamps[end_idx - 1]
            _ahead_h = (
                forecast.future_dates[-1].to_pydatetime() - _last_obs
            ).total_seconds() / 3600.0
            _proj = (
                f"~{int(round(_ahead_h))} год вперед"
                if _ahead_h < 48
                else f"~{int(round(_ahead_h / 24))} дн вперед"
            )
            _render_forecast_details(
                forecast,
                horizon_label=_proj,
                horizon_end=horizon_end,
                changepoints_count=len(changepoints),
                n_used=n_used,
                n_total=n_ts,
            )
        elif forecast_error:
            # Якщо помилка — "замало точок" І користувач звузив вікно — пораду
            # "розширити вікно" виносимо явно, бо це найшвидший фікс.
            hint = ""
            if "Замало точок" in forecast_error and n_used < n_ts:
                hint = f" Розширте вікно — повний набір має {n_ts} відповідей."
            window_suffix = f" Використано {n_used} з {n_ts} відповідей." if n_used < n_ts else ""
            st.warning(f"Прогноз недоступний: {forecast_error}{hint}{window_suffix}")
