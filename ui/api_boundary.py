"""Central UI error boundary for SaaS API failures.

st.rerun() and st.stop() raise BaseException subclasses, so they pass through
the pages' own ``except Exception`` blocks. That is what lets one decorator on
the data facade replace a traceback with a proper state on every page.

Scope errors (MissingGoogleScopesError) are intentionally not handled here:
pages treat Sheets access as optional and degrade in place.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from core.logger import get_logger
from ui.data_access_cache import session_cache_key
from ui.saas_api import (
    ApiServerError,
    GoogleTokenRevokedError,
    GoogleUnavailableError,
    SessionExpiredError,
)

log = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

GOOGLE_UNAVAILABLE_MESSAGE = "Тимчасова проблема з Google API. Спробуй оновити сторінку."


def handle_api_errors(func: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except (SessionExpiredError, GoogleTokenRevokedError) as exc:
            if not _in_script_thread():
                raise
            log.warning("ui_saas_session_invalidated", extra={"error_code": type(exc).__name__})
            _sign_out()
            st.rerun()
            raise
        except GoogleUnavailableError:
            if not _in_script_thread():
                raise
            st.warning(GOOGLE_UNAVAILABLE_MESSAGE)
            st.stop()
            raise
        except ApiServerError as exc:
            if not _in_script_thread():
                raise
            reference = _support_reference()
            log.error(
                "ui_saas_api_server_error",
                extra={"status": exc.status_code, "error_code": exc.error_code, "ref": reference},
            )
            st.error(f"Сервіс тимчасово недоступний. Код для підтримки: {reference}")
            st.stop()
            raise

    return wrapper


def _in_script_thread() -> bool:
    return get_script_run_ctx() is not None


def _support_reference() -> str:
    session_id = st.session_state.get("saas_session_id")
    return session_cache_key(session_id)[:8] if isinstance(session_id, str) else "n/a"


def _sign_out() -> None:
    # Imported lazily: auth_widget imports google_data, which uses this module.
    from ui.components.auth_widget import _logout_saas_session

    _logout_saas_session()
