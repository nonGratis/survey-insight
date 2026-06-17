"""Small shared UI shell helpers for Streamlit pages.

The goal is boring consistency: page title, selected form caption and common
state messages should look and read the same on every page.
"""

from __future__ import annotations

from typing import Literal, Protocol

import streamlit as st

StateKind = Literal["info", "success", "warning", "error"]


class _MessageContainer(Protocol):
    def title(self, body: str) -> object: ...

    def caption(self, body: str) -> object: ...

    def info(self, body: str, *, icon: str | None = None) -> object: ...

    def success(self, body: str, *, icon: str | None = None) -> object: ...

    def warning(self, body: str, *, icon: str | None = None) -> object: ...

    def error(self, body: str, *, icon: str | None = None) -> object: ...


_STATE_ICONS: dict[StateKind, str] = {
    "info": ":material/info:",
    "success": ":material/check_circle:",
    "warning": ":material/warning:",
    "error": ":material/error:",
}


def format_form_caption(form_title: str | None, label: str = "Форма") -> str:
    """Return the standard selected-form caption used under page headers."""
    title = str(form_title or "—").strip() or "—"
    return f"{label}: **{title}**"


def render_page_header(
    title: str,
    caption: str | None = None,
    *,
    container: _MessageContainer = st,
) -> None:
    """Render a consistent page title and optional caption."""
    container.title(title)
    if caption:
        container.caption(caption)


def render_form_caption(
    form_title: str | None,
    *,
    label: str = "Форма",
    container: _MessageContainer = st,
) -> None:
    """Render the selected form caption in the same format everywhere."""
    container.caption(format_form_caption(form_title, label=label))


def render_state(
    message: str,
    *,
    kind: StateKind = "info",
    details: str | None = None,
    icon: str | None = None,
    container: _MessageContainer = st,
) -> None:
    """Render a common page state message.

    `details` is intentionally appended as a separate sentence-like fragment so
    call sites can keep error context without each page inventing formatting.
    """
    body = message if not details else f"{message}\n\n{details}"
    renderer = getattr(container, kind)
    renderer(body, icon=icon or _STATE_ICONS[kind])


def render_empty_state(
    message: str,
    *,
    details: str | None = None,
    container: _MessageContainer = st,
) -> None:
    """Render a neutral empty state."""
    render_state(message, kind="info", details=details, container=container)


def render_error_state(
    message: str,
    *,
    details: str | None = None,
    container: _MessageContainer = st,
) -> None:
    """Render an error state with consistent icon and formatting."""
    render_state(message, kind="error", details=details, container=container)


def loading_message(message: str):
    """Shared spinner wrapper for ad-hoc page loading work."""
    return st.spinner(message)
