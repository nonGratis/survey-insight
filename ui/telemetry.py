"""Analytics events about page runs: one structured log line per Streamlit script run.

``ui_page_run`` fields: ``page`` (url path), ``run_kind`` (``full`` page or ``fragment``),
``outcome`` (``completed``, ``stopped`` by st.stop, ``rerun``, ``error``) and
``duration_ms``. The logging filter adds ``session_ref`` and ``user_id``, so runs group by
browser session and by user, and line up with ``ui_saas_api_request`` in time.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.scriptrunner_utils.exceptions import RerunException, StopException

from core.logger import bind_session_user, get_logger

log = get_logger(__name__)


def bind_user() -> None:
    """Tell the log filter who is signed in to this browser session (call once per run)."""
    ctx = get_script_run_ctx(suppress_warning=True)
    if ctx is None:
        return
    user = st.session_state.get("user") or {}
    user_id = user.get("id")
    bind_session_user(ctx.session_id, user_id if isinstance(user_id, str) else None)


@contextmanager
def page_run(page: str, *, started: float | None = None, fragment: bool = False) -> Iterator[None]:
    """Log one ``ui_page_run`` when the wrapped part of the run ends, however it ends.

    ``started`` is a ``time.perf_counter()`` taken at the top of the run. With
    ``fragment=True`` the event is written only when Streamlit reruns the fragment alone:
    as part of a full run the fragment is already counted there.
    """
    ctx = get_script_run_ctx(suppress_warning=True)
    fragment_run = bool(ctx and ctx.fragment_ids_this_run)
    if fragment and not fragment_run:
        yield
        return
    begin = time.perf_counter() if started is None else started
    outcome = "error"
    try:
        yield
        outcome = "completed"
    except StopException:
        outcome = "stopped"
        raise
    except RerunException:
        outcome = "rerun"
        raise
    finally:
        log.info(
            "ui_page_run",
            extra={
                "page": page,
                "run_kind": "fragment" if fragment_run else "full",
                "outcome": outcome,
                "duration_ms": round((time.perf_counter() - begin) * 1000, 1),
            },
        )
