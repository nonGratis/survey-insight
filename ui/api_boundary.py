"""Central UI error boundary for SaaS API failures.

st.rerun() and st.stop() raise BaseException subclasses, so they pass through
the pages' own ``except Exception`` blocks. That is what lets one decorator on
the data facade replace a traceback with a proper state on every page.

Scope errors are handled only for Forms, the data every page needs: the user is
offered the reconnect button instead of a raw 403. Sheets access is optional, so
pages degrade in place and its scope errors pass through untouched.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from core.logger import get_logger
from ui.data_access_cache import session_cache_key
from ui.saas_api import (
    ApiServerError,
    GoogleTokenRevokedError,
    GoogleUnavailableError,
    MissingGoogleScopesError,
    SessionExpiredError,
)

log = get_logger(__name__)


GOOGLE_UNAVAILABLE_MESSAGE = "Тимчасова проблема з Google API. Спробуй оновити сторінку."


def handle_api_errors[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    """Guard a data call: session, Google, server and missing-Forms-scope failures."""
    return _guarded(func, scope_errors=True)


def handle_access_check_errors[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    """Guard the access decision call, where missing scopes are the expected answer."""
    return _guarded(func, scope_errors=False)


def _guarded[**P, T](func: Callable[P, T], *, scope_errors: bool) -> Callable[P, T]:
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
        except MissingGoogleScopesError as exc:
            if not (scope_errors and exc.purpose == "forms" and _in_script_thread()):
                raise
            _offer_forms_reconnect(exc)
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


def _offer_forms_reconnect(exc: MissingGoogleScopesError) -> None:
    st.warning(
        "Цій сторінці потрібен доступ до Google Forms. "
        "Натисни кнопку нижче, щоб надати дозвіл через захищений SaaS API."
    )
    if exc.connect_url:
        st.link_button("Підключити Google Forms", exc.connect_url, type="primary")


def _in_script_thread() -> bool:
    return get_script_run_ctx() is not None


def _support_reference() -> str:
    session_id = st.session_state.get("saas_session_id")
    return session_cache_key(session_id)[:8] if isinstance(session_id, str) else "n/a"


def _sign_out() -> None:
    # Imported lazily: auth_widget imports google_data, which uses this module.
    from ui.components.auth_widget import _logout_saas_session

    _logout_saas_session()
