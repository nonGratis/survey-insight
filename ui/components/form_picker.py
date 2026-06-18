"""Global form selection state shared by all pages."""

from __future__ import annotations

import streamlit as st
from google.oauth2.credentials import Credentials

from core.forms_api import FormsApiError, list_user_forms

FORM_KEY = "global_form_id"
FORM_WIDGET_PREFIX = "global_form_select"


@st.cache_data(ttl=120, show_spinner="Завантажую список форм...")
def _fetch_forms(_creds: Credentials, _token: str) -> list[dict]:
    """Return user's forms; cache key includes the access token."""
    return list_user_forms(_creds)


def fetch_forms(creds: Credentials) -> list[dict]:
    """Return cached forms for the current access token."""
    return _fetch_forms(creds, creds.token or "")


def clear_forms_cache() -> None:
    """Clear cached Drive forms list."""
    _fetch_forms.clear()


def form_widget_key(scope: str) -> str:
    """Return a page-scoped widget key for the shared form selector."""
    return f"{FORM_WIDGET_PREFIX}_{scope}"


def sync_form_widget(widget_key: str) -> None:
    """Copy a page-scoped selector value into persistent global form state."""
    selected = st.session_state.get(widget_key)
    if selected:
        st.session_state[FORM_KEY] = selected


def prepare_form_widget(scope: str, by_id: dict[str, dict]) -> tuple[str, str]:
    """Prepare page-local widget state from the persistent global selection.

    Streamlit widget keys are page-local in practice: using the persistent
    global key directly as the selectbox key can be reset when navigating
    between pages. We keep `FORM_KEY` as durable app state and use a
    page-scoped widget key only for the current selectbox.
    """
    ids = list(by_id)
    if st.session_state.get(FORM_KEY) not in by_id:
        st.session_state[FORM_KEY] = ids[0]

    widget_key = form_widget_key(scope)
    if st.session_state.get(widget_key) != st.session_state[FORM_KEY]:
        st.session_state[widget_key] = st.session_state[FORM_KEY]
    return widget_key, st.session_state[FORM_KEY]


def select_form_from_query(by_id: dict[str, dict]) -> None:
    """Apply one-shot ?form_id= preselect to the global form state."""
    pre = st.query_params.get("form_id")
    if pre in by_id:
        st.session_state[FORM_KEY] = pre
        del st.query_params["form_id"]


def ensure_selected_form(forms: list[dict]) -> dict | None:
    """Resolve the globally selected form from a list of form dictionaries."""
    if not forms:
        return None
    by_id = {form["id"]: form for form in forms}
    select_form_from_query(by_id)
    if st.session_state.get(FORM_KEY) not in by_id:
        st.session_state[FORM_KEY] = next(iter(by_id))
    return by_id.get(st.session_state[FORM_KEY])


def render_form_picker(creds: Credentials) -> dict | None:
    """Backward-compatible sidebar picker used by pages not yet migrated."""
    try:
        forms = fetch_forms(creds)
    except FormsApiError as exc:
        st.sidebar.error(f"Не вдалося отримати форми: {exc}")
        return None
    if not forms:
        st.sidebar.info("Немає Google Forms на акаунті.")
        return None

    by_id = {form["id"]: form for form in forms}
    ids = list(by_id)
    select_form_from_query(by_id)
    widget_key, _selected_id = prepare_form_widget("sidebar", by_id)

    chosen_id = st.sidebar.selectbox(
        "Форма",
        options=ids,
        format_func=lambda form_id: by_id[form_id]["name"],
        key=widget_key,
        on_change=sync_form_widget,
        args=(widget_key,),
    )
    return by_id.get(chosen_id)
