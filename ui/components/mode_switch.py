"""Shared page-mode switch for lazy Streamlit sections."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st


def render_mode_switch(label: str, options: Sequence[str], *, key: str) -> str:
    """Render one visual style for lazy page sections and return active mode."""
    if not options:
        raise ValueError("options must contain at least one mode.")
    selected = st.segmented_control(
        label,
        list(options),
        default=options[0],
        key=key,
        label_visibility="collapsed",
        width="stretch",
    )
    return str(selected or options[0])
