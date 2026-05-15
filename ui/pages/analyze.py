"""Сторінка аналізу: вибір Google Form, структура, реальні відповіді."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.auth import credentials_from_dict
from core.forms_api import (
    get_form_structure,
    get_linked_sheet_id,
    list_user_forms,
    parse_question_types,
)
from core.sheets_api import fetch_responses
from ui.components.auth_widget import ensure_api_access

st.title("Аналіз")

if not ensure_api_access():
    st.stop()

creds = credentials_from_dict(st.session_state["credentials"])


@st.cache_data(ttl=120, show_spinner="Завантажую список форм…")
def _cached_forms(_creds_token: str) -> list[dict]:
    """Кешуємо список форм на 2 хв, ключ — access_token (для multi-user)."""
    return list_user_forms(creds)


forms = _cached_forms(creds.token or "")

if not forms:
    st.info(
        "На цьому акаунті не знайдено жодної Google Form. "
        "Створи форму в google.com/forms і повернись сюди."
    )
    st.stop()

choice = st.selectbox(
    "Виберіть форму",
    options=forms,
    format_func=lambda f: f["name"],
)

if not choice:
    st.stop()


@st.cache_data(ttl=120, show_spinner="Завантажую структуру форми…")
def _cached_structure(form_id: str, _creds_token: str) -> dict:
    return get_form_structure(creds, form_id)


structure = _cached_structure(choice["id"], creds.token or "")
questions = parse_question_types(structure)
sheet_id = get_linked_sheet_id(structure)


@st.cache_data(ttl=60, show_spinner="Завантажую відповіді…")
def _cached_responses(sheet_id_: str, _creds_token: str) -> pd.DataFrame:
    return fetch_responses(creds, sheet_id_)


df: pd.DataFrame
if sheet_id:
    df = _cached_responses(sheet_id, creds.token or "")
else:
    df = pd.DataFrame()

col1, col2, col3 = st.columns(3)
col1.metric("Назва форми", structure.get("info", {}).get("title", "—"))
col2.metric("Питань", len(questions))
col3.metric("Відповідей", len(df) if sheet_id else "—")

if not sheet_id:
    st.warning(
        "Ця форма ще не має привʼязаного Google Sheet з відповідями. "
        "Відкрий форму → вкладка **Responses** → **Link to Sheets**, "
        "і повертайся."
    )
elif df.empty:
    st.info("Sheet привʼязаний, але відповідей ще немає.")

st.divider()
st.subheader("Питання та типи")
for idx, q in enumerate(questions, start=1):
    with st.container(border=True):
        st.markdown(f"**{idx}. {q.title}**")
        st.caption(f"Тип: `{q.type}`")
        if q.options:
            st.write("Варіанти: " + ", ".join(q.options))

if not df.empty:
    with st.expander("Перші 5 рядків відповідей (raw)", expanded=False):
        st.dataframe(df.head(), use_container_width=True, hide_index=True)
