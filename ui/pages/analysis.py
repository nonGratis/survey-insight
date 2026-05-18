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

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from core.auth import credentials_from_dict
from core.charts_timeline import plot_timeline_with_forecast
from core.forecast import ForecastError, asymptotic_exp_forecast
from core.forms_api import (
    FormsApiError,
    get_form_structure,
    get_linked_sheet_id,
    list_user_forms,
)
from core.logger import get_logger
from core.sheets_api import SheetsApiError, fetch_responses
from core.timeline import build_timeline
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
    st.warning(
        "Ця форма не має привʼязаного Google Sheet. "
        "Прив'яжи Sheet у формі (Responses → Link to Sheets), щоб увімкнути аналіз."
    )
    st.stop()

# Sidebar: deadline + target (персистяться в session_state per form_id).
_config_key = f"analysis_config_{form_id}"
_default_deadline = date.today() + timedelta(days=14)
_config = st.session_state.get(_config_key, {"deadline": _default_deadline, "target": 100})

with st.sidebar:
    st.subheader("Параметри аналізу")
    _config["deadline"] = st.date_input(
        "Дедлайн опитування",
        value=_config["deadline"],
        key=f"deadline_{form_id}",
    )
    _config["target"] = st.number_input(
        "Цільова к-сть відповідей",
        min_value=1,
        value=int(_config["target"]),
        step=10,
        key=f"target_{form_id}",
    )
    if st.button("Оновити дані", key=f"refresh_{form_id}"):
        st.cache_data.clear()
        st.rerun()
st.session_state[_config_key] = _config


@st.cache_data(ttl=60, show_spinner="Завантажую відповіді…")
def _cached_responses(sheet_id_: str, _creds_token: str) -> pd.DataFrame:
    return fetch_responses(creds, sheet_id_)


try:
    df = _cached_responses(sheet_id, creds.token or "")
except SheetsApiError as exc:
    log.exception(
        "ui_analysis_fetch_responses_failed",
        extra={"sheet_id": sheet_id, "status": exc.status},
    )
    st.error(f"Не вдалося завантажити відповіді: {exc}")
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
    timeline = build_timeline(df)

    if timeline.cumulative.empty:
        st.info("Поки немає валідних timestamps у відповідях.")
    else:
        forecast = None
        forecast_error: str | None = None
        try:
            forecast = asymptotic_exp_forecast(timeline, deadline=_config["deadline"])
        except ForecastError as exc:
            forecast_error = str(exc)

        fig = plot_timeline_with_forecast(
            timeline=timeline,
            forecast=forecast,
            target=int(_config["target"]),
            deadline=_config["deadline"],
        )
        st.plotly_chart(fig, width="stretch")

        cols = st.columns(4)
        current = int(timeline.cumulative.iloc[-1])
        days_left = (_config["deadline"] - date.today()).days
        cols[0].metric("Зараз", current)
        if forecast is not None:
            ci_half = (forecast.final_ci[1] - forecast.final_ci[0]) // 2
            cols[1].metric(
                "Прогноз на дедлайн",
                forecast.final_estimate,
                delta=f"±{ci_half}",
            )
        else:
            cols[1].metric("Прогноз на дедлайн", "—")
        cols[2].metric("До дедлайну", f"{days_left} днів")
        cols[3].metric("Мета", int(_config["target"]))

        if forecast is not None:
            st.caption(
                f"Asymptotic exp · RMSE={forecast.rmse:.2f} · "
                f"R²={forecast.r_squared:.3f} · "
                f"95% CI на дедлайн: {forecast.final_ci[0]}–{forecast.final_ci[1]}"
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
