"""Сторінка «Аналіз» — обробка відповідей обраної Google Form.

5 вкладок:
- 📈 Огляд          — часовий кумулятив + прогноз + дедлайн (PR1)
- 📊 По питаннях    — дескриптивна статистика per question (PR4)
- 🔀 Крос-таби      — таблиці спряженості питання × питання (PR5)
- 🔍 Якість         — fill-duration, drop-out, аномалії (PR6)
- 🎯 Репрезентативність — постстратифікація + DEFF Кіша (PR7)

PR1 заповнює тільки вкладку Огляд; решта показують 'Скоро у PR…' info.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from core.auth import credentials_from_dict
from core.charts_timeline import plot_timeline_with_forecast
from core.forecast import ForecastError, ForecastResult, asymptotic_exp_forecast
from core.forms_api import (
    FormsApiError,
    get_form_structure,
    get_linked_sheet_id,
    list_response_timestamps,
    list_user_forms,
)
from core.logger import get_logger
from core.timeline import build_timeline_from_timestamps
from ui.components.auth_widget import ensure_api_access

log = get_logger(__name__)

st.title("Аналіз")

if not ensure_api_access():
    st.stop()

creds = credentials_from_dict(st.session_state["credentials"])

# Selectbox-driven form picker. URL `?form_id=` (з Каталог-кнопки 📊) і
# `preselected_form_id` (legacy) використовуємо ЛИШЕ як default-індекс —
# завжди показуємо selectbox зі списку всіх форм, бо cross-tab auth поки
# не зберігається і LinkColumn у Каталозі відкриває новий tab з порожнім
# session_state.


@st.cache_data(ttl=120, show_spinner="Завантажую список форм…")
def _cached_forms(_creds_token: str) -> list[dict]:
    """Список форм користувача, кешований на 2 хв за access_token."""
    return list_user_forms(creds)


try:
    forms = _cached_forms(creds.token or "")
except FormsApiError as exc:
    log.exception("ui_analysis_list_forms_failed", extra={"status": exc.status})
    st.error(f"Не вдалося отримати список форм: {exc}")
    st.stop()

if not forms:
    st.info(
        "На цьому акаунті не знайдено жодної Google Form. "
        "Створи форму в google.com/forms і повернись сюди."
    )
    st.stop()

# Preselect із URL (з Каталог-кнопки) або session_state (legacy fallback).
preselected_id = st.query_params.get("form_id") or st.session_state.get("preselected_form_id")
default_idx = 0
if preselected_id:
    matched = next((i for i, f in enumerate(forms) if f["id"] == preselected_id), None)
    if matched is not None:
        default_idx = matched
    # Прибираємо обидва preselect-джерела, щоб юзер міг вільно міняти selectbox
    # без "залипання" URL/session_state при наступному rerun.
    if "form_id" in st.query_params:
        del st.query_params["form_id"]
    st.session_state.pop("preselected_form_id", None)

choice = st.selectbox(
    "Форма для аналізу",
    options=forms,
    format_func=lambda f: f["name"],
    index=default_idx,
)
if not choice:
    st.stop()

form_id = choice["id"]


@st.cache_data(ttl=120, show_spinner="Завантажую структуру форми…")
def _cached_structure(form_id_: str, _creds_token: str) -> dict:
    return get_form_structure(creds, form_id_)


try:
    structure = _cached_structure(form_id, creds.token or "")
except FormsApiError as exc:
    log.exception("ui_analysis_get_structure_failed", extra={"form_id": form_id})
    st.error(f"Не вдалося завантажити форму: {exc}")
    st.stop()

form_title = structure.get("info", {}).get("title", "—")
sheet_id = get_linked_sheet_id(structure)

st.caption(f"Форма: **{form_title}**")
if not sheet_id:
    # Огляд tab працює напряму з Forms API і Sheet не потребує.
    # Решта tabs (По питаннях / Крос-таби / Якість) у майбутніх PR'ах
    # покажуть власне попередження про потребу Sheet — там, де вони
    # реально читатимуть answer values.
    st.info(
        "Ця форма не має прив'язаного Google Sheet. "
        "Огляд працює через Forms API безпосередньо; інші вкладки "
        "потребуватимуть Sheet (Responses → Link to Sheets у формі)."
    )


@st.cache_data(ttl=60, show_spinner="Завантажую відповіді…")
def _cached_timestamps(form_id_: str, _creds_token: str) -> list[datetime]:
    """Forms API timestamps, кешовано на 60s за access_token+form_id."""
    return list_response_timestamps(creds, form_id_)


@st.cache_data(ttl=300, show_spinner="Обчислюю прогноз…")
def _cached_forecast(
    _form_id: str,
    n_responses: int,
    first_ts: datetime,
    last_ts: datetime,
    _timestamps: list[datetime],
) -> tuple[ForecastResult | None, str | None]:
    """Кешований прогноз.

    Cache key: (form_id, к-сть відповідей, перший+останній timestamp).
    Якщо нові відповіді не прийшли — instant cache hit (forecast НЕ
    залежить від target, тож зміна input'у його не invalidate'ить).
    `_timestamps` з підкреслення — Streamlit пропускає його в hashing,
    передаємо як payload.

    Returns (result, error_msg). На фейлі фіту result=None, error=повідомлення.
    """
    timeline = build_timeline_from_timestamps(_timestamps)
    try:
        return asymptotic_exp_forecast(timeline), None
    except ForecastError as exc:
        return None, str(exc)


try:
    timestamps = _cached_timestamps(form_id, creds.token or "")
except FormsApiError as exc:
    log.exception(
        "ui_analysis_list_timestamps_failed",
        extra={"form_id": form_id, "status": exc.status},
    )
    st.error(f"Не вдалося отримати timestamps відповідей: {exc}")
    st.stop()

tab_overview, tab_per_q, tab_crosstabs, tab_quality, tab_repr = st.tabs(
    [
        "📈 Огляд",
        "📊 По питаннях",
        "🔀 Крос-таби",
        "🔍 Якість",
        "🎯 Репрезентативність",
    ]
)

with tab_overview:
    timeline = build_timeline_from_timestamps(timestamps)

    if timeline.cumulative.empty:
        st.info("Поки немає валідних timestamps у відповідях.")
    else:
        # Прогноз — на 25% вперед від тривалості опитування (last - first).
        # Кешований: при зміні цільової кількості переобчислення не відбувається.
        if timestamps:
            forecast, forecast_error = _cached_forecast(
                _form_id=form_id,
                n_responses=len(timestamps),
                first_ts=timestamps[0],
                last_ts=timestamps[-1],
                _timestamps=timestamps,
            )
        else:
            forecast, forecast_error = None, None

        # Цільова кількість живе у session_state per form_id.
        _target_key = f"analysis_target_{form_id}"
        target = int(st.session_state.get(_target_key, 100))

        # Рядок 1: BANs (current / forecast / target) — над усім.
        ban_cols = st.columns(3)
        current = int(timeline.cumulative.iloc[-1])
        ban_cols[0].metric("Зараз", current)
        if forecast is not None:
            ci_half = (forecast.final_ci[1] - forecast.final_ci[0]) // 2
            ban_cols[1].metric(
                "Прогноз",
                forecast.final_estimate,
                delta=f"±{ci_half}",
                delta_color="off",
            )
        else:
            ban_cols[1].metric("Прогноз", "—")
        ban_cols[2].metric("Мета", target)

        # Рядок 2: controls (target + refresh) — компактно над графіком.
        ctrl_target, ctrl_refresh = st.columns([4, 1])
        with ctrl_target:
            target = int(
                st.number_input(
                    "Цільова кількість відповідей",
                    min_value=1,
                    value=target,
                    step=10,
                    key=_target_key,
                    label_visibility="collapsed",
                    placeholder="Цільова кількість",
                )
            )
        with ctrl_refresh:
            if st.button(
                "Оновити",
                key=f"refresh_{form_id}",
                width="stretch",
                help="Скинути кеш і перечитати свіжі timestamps з Forms API",
            ):
                st.cache_data.clear()
                st.rerun()

        # Рядок 3: графік (без deadline-вертикалі; target — горизонталь).
        fig = plot_timeline_with_forecast(
            timeline=timeline,
            forecast=forecast,
            target=target,
        )
        st.plotly_chart(fig, width="stretch")

        # Рядок 4: caption — модель/якість фіту/горизонт.
        if forecast is not None:
            horizon_end = forecast.future_dates[-1].date()
            st.caption(
                f"Прогноз: asymptotic exp · горизонт до {horizon_end:%d.%m.%Y} "
                f"(25% тривалості) · 95% CI: {forecast.final_ci[0]}–{forecast.final_ci[1]} · "
                f"RMSE={forecast.rmse:.2f} · R²={forecast.r_squared:.3f}"
            )
        elif forecast_error:
            st.caption(f"Прогноз недоступний: {forecast_error}")

with tab_per_q:
    st.info("Скоро у PR4: дескриптивна статистика по кожному питанню.")

with tab_crosstabs:
    st.info("Скоро у PR5: крос-табуляції питання × питання + χ²-тест.")

with tab_quality:
    st.info("Скоро у PR6: час заповнення, drop-out rate, виявлення підозрілих патернів відповідей.")

with tab_repr:
    st.info(
        "Скоро у PR7: постстратифікаційне зважування, DEFF Кіша, метрика "
        "«ще треба X відповідей з страти Y»."
    )
