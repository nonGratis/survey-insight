"""Сторінка аналізу: вибір Google Form, структура, реальні відповіді."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.auth import credentials_from_dict
from core.charts import bar_categorical, freq_table, hist_ordinal, response_count
from core.forms_api import (
    FormsApiError,
    Question,
    get_form_structure,
    get_linked_sheet_id,
    list_user_forms,
    parse_question_types,
)
from core.logger import get_logger
from core.sheets_api import SheetsApiError, fetch_responses
from ui.components.auth_widget import ensure_api_access

log = get_logger(__name__)

st.title("Аналіз")

if not ensure_api_access():
    st.stop()

creds = credentials_from_dict(st.session_state["credentials"])


@st.cache_data(ttl=120, show_spinner="Завантажую список форм…")
def _cached_forms(_creds_token: str) -> list[dict]:
    """Кешуємо список форм на 2 хв, ключ — access_token (для multi-user)."""
    return list_user_forms(creds)


try:
    forms = _cached_forms(creds.token or "")
except FormsApiError as exc:
    log.exception("ui_analyze_list_forms_failed", extra={"status": exc.status})
    st.error(f"Не вдалося отримати список форм: {exc}")
    st.stop()

if not forms:
    st.info(
        "На цьому акаунті не знайдено жодної Google Form. "
        "Створи форму в google.com/forms і повернись сюди."
    )
    st.stop()

# Якщо користувач прийшов сюди з Каталог через LinkColumn-кнопку "Аналіз",
# URL міститиме ?form_id=ABC; підбираємо її як default у selectbox.
preselected_id = st.query_params.get("form_id")
default_idx = 0
if preselected_id:
    matched = next(
        (i for i, f in enumerate(forms) if f["id"] == preselected_id), None
    )
    if matched is not None:
        default_idx = matched
    # Прибираємо query-param, щоб користувач міг вільно міняти selectbox,
    # а URL не "тягнув" стару форму при наступному rerun.
    if "form_id" in st.query_params:
        del st.query_params["form_id"]

choice = st.selectbox(
    "Виберіть форму",
    options=forms,
    format_func=lambda f: f["name"],
    index=default_idx,
)

if not choice:
    st.stop()


@st.cache_data(ttl=120, show_spinner="Завантажую структуру форми…")
def _cached_structure(form_id: str, _creds_token: str) -> dict:
    return get_form_structure(creds, form_id)


try:
    structure = _cached_structure(choice["id"], creds.token or "")
except FormsApiError as exc:
    log.exception(
        "ui_analyze_get_structure_failed",
        extra={"form_id": choice["id"], "status": exc.status},
    )
    st.error(f"Не вдалося завантажити форму: {exc}")
    st.stop()

questions = parse_question_types(structure)
sheet_id = get_linked_sheet_id(structure)


@st.cache_data(ttl=60, show_spinner="Завантажую відповіді…")
def _cached_responses(sheet_id_: str, _creds_token: str) -> pd.DataFrame:
    return fetch_responses(creds, sheet_id_)


df = pd.DataFrame()
if sheet_id:
    try:
        df = _cached_responses(sheet_id, creds.token or "")
    except SheetsApiError as exc:
        log.exception(
            "ui_analyze_fetch_responses_failed",
            extra={"sheet_id": sheet_id, "status": exc.status},
        )
        st.error(f"Не вдалося завантажити відповіді: {exc}")

col1, col2, col3 = st.columns(3)
col1.metric("Назва форми", structure.get("info", {}).get("title", "—"))
col2.metric("Питань", len(questions))
col3.metric("Відповідей", str(len(df)) if sheet_id else "—")

if not sheet_id:
    st.warning(
        "Ця форма ще не має привʼязаного Google Sheet з відповідями. "
        "Відкрий форму → вкладка **Responses** → **Link to Sheets**, "
        "і повертайся."
    )
elif df.empty:
    st.info("Sheet привʼязаний, але відповідей ще немає.")

if not df.empty:
    with st.expander("Перші 5 рядків відповідей (raw)", expanded=False):
        st.dataframe(df.head(), use_container_width=True, hide_index=True)


st.divider()
st.subheader("Аналіз по питаннях")


def _render_question(idx: int, q: Question, df: pd.DataFrame) -> None:
    """Відмалювати один expander під питання: метрика + графік/таблиця."""
    label = f"{idx}. {q.title}  ·  `{q.type}`"
    with st.expander(label, expanded=(idx == 1)):
        if q.title not in df.columns:
            st.caption("У Sheet немає колонки з такою назвою — пропускаю.")
            return

        st.metric("Відповідей на це питання", response_count(df, q.title))

        if q.type in ("MULTIPLE_CHOICE", "CHECKBOX"):
            st.plotly_chart(bar_categorical(df, q.title), use_container_width=True)
        elif q.type == "LINEAR_SCALE":
            st.plotly_chart(hist_ordinal(df, q.title), use_container_width=True)
        elif q.type == "SHORT_ANSWER":
            table = freq_table(df, q.title)
            if table.empty:
                st.caption("Немає непорожніх відповідей.")
            else:
                st.dataframe(table, use_container_width=True, hide_index=True)
        else:
            st.caption(f"Графік для `{q.type}` буде у наступних блоках.")


if df.empty:
    st.info("Відповіді ще не зʼявились — графіки покажуться, коли надійдуть.")
else:
    for idx, q in enumerate(questions, start=1):
        _render_question(idx, q, df)
