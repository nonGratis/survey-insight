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

import streamlit as st

from core.auth import credentials_from_dict
from core.forms_api import FormsApiError, get_form_structure, get_linked_sheet_id
from core.logger import get_logger
from ui.components.auth_widget import ensure_api_access

log = get_logger(__name__)

st.title("Аналіз")

if not ensure_api_access():
    st.stop()

creds = credentials_from_dict(st.session_state["credentials"])

# Resolve form_id: пріоритет — query param (з Каталог-кнопки), fallback — session_state.
form_id = st.query_params.get("form_id") or st.session_state.get("preselected_form_id")
if not form_id:
    st.info(
        "Перейди до **Каталогу** і клікни «📊 Аналіз» біля форми, "
        "щоб почати аналіз відповідей."
    )
    st.stop()


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
    st.info(
        "Скоро у наступному коміті цього PR: часовий кумулятив відповідей з "
        "прогнозом до дедлайну і довірчим інтервалом."
    )

with tab_per_q:
    st.info("Скоро у PR4: дескриптивна статистика по кожному питанню.")

with tab_crosstabs:
    st.info("Скоро у PR5: крос-табуляції питання × питання + χ²-тест.")

with tab_quality:
    st.info(
        "Скоро у PR6: час заповнення, drop-out rate, виявлення підозрілих "
        "патернів відповідей."
    )

with tab_repr:
    st.info(
        "Скоро у PR7: постстратифікаційне зважування, DEFF Кіша, метрика "
        "«ще треба X відповідей з страти Y»."
    )
