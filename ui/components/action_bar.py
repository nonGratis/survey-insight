"""Top action bar for form-scoped pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import streamlit as st
from google.oauth2.credentials import Credentials

from core.forms_api import FormsApiError
from ui.components.form_picker import (
    fetch_forms,
    prepare_form_widget,
    select_form_from_query,
    sync_form_widget,
)
from ui.components.page_shell import render_empty_state, render_error_state

_ACTION_BAR_CSS = """
<style>
.si-action-label {
  display: inline-flex;
  align-items: center;
  height: 2.5rem;
  white-space: nowrap;
  line-height: 1;
}
[class*="st-key-action_bar_refresh_"] button,
[class*="st-key-action_bar_open_"] a {
  width: 2.5rem !important;
  min-width: 2.5rem !important;
  height: 2.5rem !important;
  min-height: 2.5rem !important;
  padding: 0 !important;
}
[class*="st-key-action_bar_refresh_"] button > div,
[class*="st-key-action_bar_open_"] a > div {
  justify-content: center;
}
[class*="st-key-action_bar_open_"] {
  margin-left: 0.25rem;
}
</style>
"""


class _ActionContainer(Protocol):
    def columns(
        self,
        spec: list[float],
        gap: str = "small",
        vertical_alignment: str = "top",
    ) -> list[object]: ...

    def caption(self, body: str) -> object: ...


@dataclass(frozen=True)
class ActionBarStatus:
    """Compact status shown in the action bar."""

    responses: int | None = None
    updated: str | None = None
    note: str | None = None

    def render(self) -> str:
        parts: list[str] = []
        if self.responses is not None:
            parts.append(f"{self.responses} відповідей")
        if self.updated:
            parts.append(f"оновлено {self.updated}")
        if self.note:
            parts.append(self.note)
        return " · ".join(parts)


@dataclass(frozen=True)
class ActionBarState:
    """Current form selection and top-bar action flags."""

    selected_form: dict | None
    refresh_clicked: bool
    forms_count: int


def _form_edit_url(form_id: str) -> str:
    return f"https://docs.google.com/forms/d/{form_id}/edit"


def render_action_status(
    status: ActionBarStatus | None,
    *,
    container: _ActionContainer = st,
) -> None:
    """Render a compact status line directly below the action bar."""
    if status is None:
        return
    rendered = status.render()
    if rendered:
        container.caption(rendered)


def render_action_bar(
    creds: Credentials,
    *,
    show_status: bool = True,
    refresh_scope: str,
    status: ActionBarStatus | None = None,
    container: _ActionContainer = st,
) -> ActionBarState:
    """Render top form picker, refresh, open-form action and compact status."""
    st.markdown(_ACTION_BAR_CSS, unsafe_allow_html=True)
    try:
        forms = fetch_forms(creds)
    except FormsApiError as exc:
        render_error_state("Не вдалося отримати форми.", details=str(exc))
        return ActionBarState(selected_form=None, refresh_clicked=False, forms_count=0)

    if not forms:
        render_empty_state("Немає Google Forms на акаунті.")
        return ActionBarState(selected_form=None, refresh_clicked=False, forms_count=0)

    by_id = {form["id"]: form for form in forms}
    ids = list(by_id)
    select_form_from_query(by_id)
    widget_key, selected_id = prepare_form_widget(refresh_scope, by_id)

    label_col, select_col, refresh_col, open_col = container.columns(
        [0.8, 8.25, 0.55, 0.55],
        gap="small",
        vertical_alignment="center",
    )
    with label_col:
        st.markdown(
            '<span class="si-action-label"><strong>Форма:</strong></span>', unsafe_allow_html=True
        )
    with select_col:
        chosen_id = st.selectbox(
            "Форма",
            options=ids,
            format_func=lambda form_id: by_id[form_id]["name"],
            key=widget_key,
            on_change=sync_form_widget,
            args=(widget_key,),
            label_visibility="collapsed",
        )
    with refresh_col, st.container(key=f"action_bar_refresh_{refresh_scope}"):
        refresh_clicked = st.button(
            "",
            icon=":material/refresh:",
            help="Оновити дані",
            key=f"refresh_{refresh_scope}",
            width="stretch",
        )
    with open_col, st.container(key=f"action_bar_open_{refresh_scope}"):
        st.link_button(
            "",
            _form_edit_url(chosen_id or selected_id),
            icon=":material/open_in_new:",
            help="Відкрити Google Forms",
            width="stretch",
        )

    if show_status:
        render_action_status(status, container=container)

    return ActionBarState(
        selected_form=by_id.get(chosen_id or selected_id),
        refresh_clicked=refresh_clicked,
        forms_count=len(forms),
    )
