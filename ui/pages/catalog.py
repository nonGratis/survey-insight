"""Каталог — табличний огляд усіх Google Forms, до яких є доступ.

Tier 1 (цей файл): один Drive API виклик → миттєва таблиця з
базовими метаданими (id, name, owner, дати). Tier 2/3 (форма
structure + response stats) додаються в наступних комітах через
@st.fragment-ticker, щоб таблиця не блокувалась на повний enrichment.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.auth import credentials_from_dict
from core.forms_api import FormsApiError
from core.forms_catalog import FormDriveMeta, list_forms_with_drive_meta
from ui.components.auth_widget import ensure_api_access

st.title("Каталог")

if not ensure_api_access():
    st.stop()

creds = credentials_from_dict(st.session_state["credentials"])


@st.cache_data(ttl=900, show_spinner="Завантажую каталог форм…")
def _cached_drive_list(_creds_token: str) -> list[FormDriveMeta]:
    """Drive list форм, кешується на 15 хв за access_token."""
    return list_forms_with_drive_meta(creds)


col_metric, col_refresh = st.columns([4, 1])
try:
    forms_meta = _cached_drive_list(creds.token or "")
except FormsApiError as exc:
    st.error(f"Не вдалося завантажити каталог: {exc}")
    st.stop()

col_metric.metric("Форм у каталозі", len(forms_meta))
if col_refresh.button("Оновити", help="Скинути кеш і перечитати з Drive"):
    _cached_drive_list.clear()
    st.rerun()

if not forms_meta:
    st.info(
        "Жодної Google Form не знайдено на цьому акаунті. "
        "Створи форму на forms.google.com і повернись."
    )
    st.stop()


def _build_tier1_dataframe(forms: list[FormDriveMeta]) -> pd.DataFrame:
    """Зібрати DataFrame з Tier 1 даних. Tier 2/3 додамо у наступних комітах."""
    return pd.DataFrame(
        {
            "Name": [f"{f.edit_url}|{f.name}" for f in forms],
            "Owner": [f.owner_email for f in forms],
            "Modified": [f.modified_time for f in forms],
            "Created": [f.created_time for f in forms],
            "FullID": [f.id for f in forms],
        }
    )


df = _build_tier1_dataframe(forms_meta)

st.dataframe(
    df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Name": st.column_config.LinkColumn(
            "Назва",
            help="Клік відкриває форму в редакторі Google Forms (новий таб)",
            display_text=r"^.*\|(.*)$",
        ),
        "Owner": st.column_config.TextColumn("Власник"),
        "Modified": st.column_config.DatetimeColumn(
            "Змінено", format="DD.MM.YYYY HH:mm"
        ),
        "Created": st.column_config.DatetimeColumn(
            "Створено", format="DD.MM.YYYY HH:mm"
        ),
        "FullID": st.column_config.TextColumn("Form ID", width="small"),
    },
)

st.caption(
    "Tier 1: базові метадані з Drive. Структура форм і кількість "
    "відповідей будуть додані у наступних оновленнях сторінки."
)
