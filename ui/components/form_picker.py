"""Глобальний вибір форми — один раз, спільний для всіх сторінок.

Раніше кожна сторінка мала власний selectbox форми → користувач обирав
форму заново на кожній вкладці. Тут — спільний sidebar-пікер зі стабільним
ключем `global_form_id`: вибір зберігається у session_state і переноситься
між сторінками автоматично.

Підхоплює `?form_id=` (кнопка «Динаміка» у Каталозі) як одноразовий preselect.
"""

from __future__ import annotations

import streamlit as st
from google.oauth2.credentials import Credentials

from core.forms_api import FormsApiError, list_user_forms

_FORM_KEY = "global_form_id"


@st.cache_data(ttl=120, show_spinner="Завантажую список форм…")
def _fetch_forms(_creds: Credentials, _token: str) -> list[dict]:
    """Список форм користувача (кеш за access_token; _creds не хешується)."""
    return list_user_forms(_creds)


def render_form_picker(creds: Credentials) -> dict | None:
    """Відмалювати спільний sidebar-selectbox форми; повернути обрану форму.

    Вибір персистентний (session_state[`global_form_id`]) — один раз на всі
    сторінки. Повертає dict форми (`id`, `name`) або None, якщо форм немає.
    """
    try:
        forms = _fetch_forms(creds, creds.token or "")
    except FormsApiError as exc:
        st.sidebar.error(f"Не вдалося отримати форми: {exc}")
        return None
    if not forms:
        st.sidebar.info("Немає Google Forms на акаунті.")
        return None

    by_id = {f["id"]: f for f in forms}
    ids = list(by_id)

    # Одноразовий preselect із ?form_id (перехід із Каталогу).
    pre = st.query_params.get("form_id")
    if pre in by_id:
        st.session_state[_FORM_KEY] = pre
        del st.query_params["form_id"]
    if st.session_state.get(_FORM_KEY) not in by_id:
        st.session_state[_FORM_KEY] = ids[0]

    chosen_id = st.sidebar.selectbox(
        "📋 Форма",
        options=ids,
        format_func=lambda i: by_id[i]["name"],
        key=_FORM_KEY,
    )
    return by_id.get(chosen_id)
