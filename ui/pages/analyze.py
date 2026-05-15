"""Сторінка аналізу: вибір Google Form і перегляд її структури."""
from __future__ import annotations

import streamlit as st

from core.auth import credentials_from_dict
from core.forms_api import get_form_structure, list_user_forms, parse_question_types
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

col1, col2 = st.columns(2)
col1.metric("Назва форми", structure.get("info", {}).get("title", "—"))
col2.metric("Питань", len(questions))

st.divider()
st.subheader("Питання та типи")
for idx, q in enumerate(questions, start=1):
    with st.container(border=True):
        st.markdown(f"**{idx}. {q.title}**")
        st.caption(f"Тип: `{q.type}`")
        if q.options:
            st.write("Варіанти: " + ", ".join(q.options))
