"""Google auth widget for local demo OAuth and production SaaS sessions."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime, timedelta

import extra_streamlit_components as stx
import httpx
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from core.auth import (
    API_SCOPES,
    IDENTITY_SCOPES,
    build_flow,
    clear_verifier,
    credentials_from_dict,
    credentials_to_dict,
    exchange_code,
    get_auth_url,
    get_user_info,
    has_api_scopes,
    load_verifier,
    refresh_if_needed,
    save_verifier,
)
from core.logger import get_logger, hash_email
from core.saas.settings import load_web_settings
from ui.api_boundary import handle_access_check_errors
from ui.data_access_cache import ACCESS_TTL_SECONDS, CacheKey, get_or_load, session_cache_key
from ui.google_data import clear_google_data_cache
from ui.saas_api import MissingGoogleScopesError, SaaSApiClient, SaaSSession

log = get_logger(__name__)

_SAAS_SESSION_COOKIE = "survey_insight_session_id"
_SAAS_SESSION_DAYS = 30
_SAAS_VALIDATE_TTL_SECONDS = 30
_SAAS_COOKIE_PROBE_RUNS = "saas_cookie_probe_runs"
_SAAS_AUTH_RESTORE_PENDING = "saas_auth_restore_pending"
_SAAS_SESSION_RETRIES = "saas_session_retries"
# Cookie writes and deletes waiting for the browser to confirm them (see _SessionCookie).
_SAAS_COOKIE_WRITE = "saas_cookie_write"
_SAAS_COOKIE_DELETE = "saas_cookie_delete"
# Sessions this tab signed out from: the cookie component may still report the old value
# until the page reloads, and it must not sign the tab in again.
_SAAS_SIGNED_OUT = "saas_signed_out_sessions"
_SAAS_COOKIE_ABSENT_LOGGED = "saas_cookie_absent_logged"
SESSION_RESTORE_MAX_RETRIES = 3
SESSION_RESTORE_RETRY_DELAY_SECONDS = 2.0


def _saas_auth_enabled() -> bool:
    return load_web_settings().signs_in_through_api


def _app_base_url() -> str:
    return load_web_settings().app_base_url


def _api_base_url() -> str:
    return load_web_settings().api_base_url


@st.cache_resource
def _saas_client(base_url: str) -> SaaSApiClient:
    return SaaSApiClient(base_url)


def _cookie_manager() -> stx.CookieManager:
    return stx.CookieManager(key="saas_auth_cookies")


class _SessionCookie:
    """The browser cookie that keeps the sign-in between visits.

    Its components are rendered only in the auth slot at the top of the page (``app.py``):
    they load in the browser and rerun the page when they answer, so anywhere else they
    shift the page, and inside a cached function they break it. A write or a delete waits
    in session state and is rendered on every run until the browser confirms it: a single
    render can be removed by the next run before the browser has carried it out.
    """

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled
        self._manager: stx.CookieManager | None = None

    def _get_manager(self) -> stx.CookieManager | None:
        # One manager per run: its component key may appear only once.
        if self._enabled and self._manager is None:
            self._manager = _cookie_manager()
        return self._manager

    def read(self) -> str | None:
        manager = self._get_manager()
        value = manager.get(_SAAS_SESSION_COOKIE) if manager else None
        if not isinstance(value, str) or not value:
            return None
        return None if value in st.session_state.get(_SAAS_SIGNED_OUT, ()) else value

    def answered(self) -> bool:
        """Whether the browser reported its cookies (it always has Streamlit's own)."""
        manager = self._get_manager()
        return bool(manager and manager.cookies)

    def sync(self) -> None:
        write = st.session_state.get(_SAAS_COOKIE_WRITE)
        if write:
            self._write(write)
        delete_key = st.session_state.get(_SAAS_COOKIE_DELETE)
        if delete_key:
            self._delete(delete_key)

    def _write(self, write: dict) -> None:
        if st.session_state.get(write["key"]) is True:
            st.session_state.pop(_SAAS_COOKIE_WRITE, None)
            log.info("ui_session_cookie_saved")
            return
        manager = self._get_manager()
        if manager is None:
            return
        manager.set(
            _SAAS_SESSION_COOKIE,
            write["value"],
            key=write["key"],
            path="/",
            # Without expires_at the component also sends Expires = now + 1 day; with both
            # set browsers follow Max-Age, but the two should not disagree.
            expires_at=write["expires_at"],
            max_age=float(_SAAS_SESSION_DAYS * 24 * 60 * 60),
            secure=_app_base_url().startswith("https://"),
            same_site="lax",
        )

    def _delete(self, delete_key: str) -> None:
        if st.session_state.get(delete_key) is True:
            st.session_state.pop(_SAAS_COOKIE_DELETE, None)
            log.info("ui_session_cookie_deleted")
            return
        manager = self._get_manager()
        if manager is None:
            return
        # Straight to the component: CookieManager.delete() also drops the name from the
        # browser's last report and fails while there is none (right after a sign-out).
        # Deleting a cookie that is not there is a no-op in the browser.
        manager.cookie_manager(
            method="delete", cookie=_SAAS_SESSION_COOKIE, key=delete_key, default=False
        )


def _queue_cookie_write(session_id: str) -> None:
    st.session_state.pop(_SAAS_COOKIE_DELETE, None)
    st.session_state[_SAAS_COOKIE_WRITE] = {
        "value": session_id,
        # The API session expires 30 days after login whatever the activity: the cookie
        # gets the same lifetime once, there is nothing to refresh later.
        "expires_at": datetime.now(UTC) + timedelta(days=_SAAS_SESSION_DAYS),
        "key": f"set_saas_session_{uuid.uuid4().hex[:8]}",
    }


def _queue_cookie_delete() -> None:
    st.session_state.pop(_SAAS_COOKIE_WRITE, None)
    st.session_state[_SAAS_COOKIE_DELETE] = f"delete_saas_session_{uuid.uuid4().hex[:8]}"


def _query_param(name: str) -> str | None:
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _get_container(location: str) -> DeltaGenerator:
    if location == "sidebar":
        return st.sidebar
    return st.container()


def _handle_local_oauth_callback() -> None:
    code = _query_param("code")
    if not code:
        return

    verifier = load_verifier()
    if not verifier:
        st.error("Сесія входу зламалась. Натисни «Увійти через Google» ще раз.")
        st.query_params.clear()
        return

    scope_param = _query_param("scope") or ""
    scopes = scope_param.split() if scope_param else IDENTITY_SCOPES

    try:
        flow = build_flow(scopes, code_verifier=verifier)
        creds = exchange_code(flow, code)
        user = get_user_info(creds)
    except Exception as exc:  # noqa: BLE001
        log.exception("auth_callback_failed")
        st.error(f"Помилка входу: {exc}")
        st.query_params.clear()
        clear_verifier()
        return

    clear_verifier()
    st.session_state["credentials"] = credentials_to_dict(creds)
    st.session_state["user"] = user
    email = user.get("email", "")
    log.info(
        "auth_login_ok",
        extra={
            "user_hash": hash_email(email) if email else "",
            "scopes_count": len(scopes),
        },
    )
    st.query_params.clear()
    st.rerun()


def _handle_saas_login_ticket() -> bool:
    ticket = _query_param("login_ticket")
    if not ticket:
        return False

    try:
        session = _saas_client(_api_base_url()).exchange_login_ticket(ticket)
    except httpx.HTTPError as exc:
        log.exception("saas_login_ticket_exchange_failed")
        st.error(f"Не вдалося завершити вхід: {exc}")
        st.query_params.clear()
        return False

    if not session.authenticated or not session.session_id:
        st.error("API не підтвердив сесію. Спробуй увійти ще раз.")
        st.query_params.clear()
        return False

    _remember_saas_session(session)
    _queue_cookie_write(session.session_id)
    st.query_params.clear()
    return True


def _restore_saas_session(*, manage_cookie: bool) -> bool:
    st.session_state[_SAAS_AUTH_RESTORE_PENDING] = False

    auth_error = _query_param("auth_error")
    if auth_error:
        st.error("Не вдалося завершити Google-вхід. Спробуй увійти ще раз.")
        st.query_params.clear()
        return False

    cookie = _SessionCookie(enabled=manage_cookie)
    if _handle_saas_login_ticket():
        cookie.sync()
        return True
    cookie.sync()

    if _has_fresh_saas_session():
        return True

    # A session this tab holds is re-checked through the API alone; the cookie is read only
    # to find a session the tab does not hold yet (page load, or a lost connection).
    session_id = st.session_state.get("saas_session_id")
    from_cookie = not isinstance(session_id, str) or not session_id
    if from_cookie:
        if st.session_state.get(_SAAS_COOKIE_DELETE):
            return False  # signing out: the cookie still names the old session
        session_id = cookie.read()
    if not isinstance(session_id, str) or not session_id:
        if not cookie.answered() and _should_wait_for_cookie_probe():
            st.session_state[_SAAS_AUTH_RESTORE_PENDING] = True
        elif manage_cookie and not st.session_state.get(_SAAS_COOKIE_ABSENT_LOGGED):
            st.session_state[_SAAS_COOKIE_ABSENT_LOGGED] = True
            log.info("ui_session_cookie_absent")
        return False

    try:
        session = _saas_client(_api_base_url()).read_session(session_id)
    except httpx.HTTPError:
        log.exception("saas_session_restore_failed")
        _clear_saas_session(session_id)
        cookie.sync()
        return False

    if session is None:
        return _retry_session_check_or_give_up()
    st.session_state.pop(_SAAS_SESSION_RETRIES, None)

    if not session.authenticated:
        _clear_saas_session(session_id)
        cookie.sync()
        return False

    _remember_saas_session(session)
    if from_cookie:
        log.info("ui_session_restored_from_cookie")
    return True


def _retry_session_check_or_give_up() -> bool:
    """Handle an API that did not answer the session check in time (cold start).

    The stored session is kept: the API being slow says nothing about whether
    the session is valid. After a few attempts fall back to the login UI.
    """
    retries = int(st.session_state.get(_SAAS_SESSION_RETRIES, 0))
    if retries >= SESSION_RESTORE_MAX_RETRIES:
        st.session_state.pop(_SAAS_SESSION_RETRIES, None)
        return False
    st.session_state[_SAAS_SESSION_RETRIES] = retries + 1
    st.info("Перевірка з'єднання…")
    time.sleep(SESSION_RESTORE_RETRY_DELAY_SECONDS)
    st.rerun()
    return False


def _should_wait_for_cookie_probe() -> bool:
    runs = int(st.session_state.get(_SAAS_COOKIE_PROBE_RUNS, 0))
    st.session_state[_SAAS_COOKIE_PROBE_RUNS] = runs + 1
    return runs == 0


def _has_fresh_saas_session() -> bool:
    checked_at = st.session_state.get("saas_session_checked_at")
    return (
        isinstance(st.session_state.get("saas_session_id"), str)
        and bool(st.session_state.get("user"))
        and isinstance(checked_at, datetime)
        and datetime.now(UTC) - checked_at < timedelta(seconds=_SAAS_VALIDATE_TTL_SECONDS)
    )


def _remember_saas_session(session: SaaSSession) -> None:
    """Keep the checked session in this tab (the cookie is written once, at login)."""
    if not session.session_id:
        return

    st.session_state["saas_session_id"] = session.session_id
    st.session_state["saas_session_checked_at"] = datetime.now(UTC)
    st.session_state[_SAAS_AUTH_RESTORE_PENDING] = False
    st.session_state[_SAAS_COOKIE_PROBE_RUNS] = 0
    st.session_state["user"] = {
        "id": session.user_id,
        "email": session.email,
        "name": session.name,
        "plan": session.plan,
    }


def _clear_saas_session(session_id: str | None = None) -> None:
    """Forget the session in this tab and queue the cookie delete for the auth slot.

    Renders nothing: it also runs deep inside pages, even inside cached functions.
    ``session_id`` is the one read from the cookie when the tab did not hold it yet.
    """
    forgotten = session_id or st.session_state.get("saas_session_id")
    if isinstance(forgotten, str) and forgotten:
        signed_out = list(st.session_state.get(_SAAS_SIGNED_OUT, []))
        st.session_state[_SAAS_SIGNED_OUT] = [*signed_out, forgotten][-5:]
    for key in (
        "saas_session_id",
        "saas_session_checked_at",
        "user",
        _SAAS_AUTH_RESTORE_PENDING,
        _SAAS_COOKIE_PROBE_RUNS,
    ):
        st.session_state.pop(key, None)
    _queue_cookie_delete()


def _logout_saas_session() -> None:
    """Revoke the API session, then drop every trace of it from this tab.

    The API call goes first so the server-side session is gone even if local
    cleanup fails. Its failure (timeout, API down) must never keep the user
    signed in locally.
    """
    session_id = st.session_state.get("saas_session_id")
    try:
        _saas_client(_api_base_url()).logout(session_id if isinstance(session_id, str) else None)
    except httpx.HTTPError as exc:
        log.warning("saas_logout_failed", extra={"error_code": type(exc).__name__})
    if isinstance(session_id, str):
        clear_google_data_cache(session_id=session_id)
    _clear_saas_session()
    # Kept through the clear: the next runs delete the cookie and ignore its old value,
    # or it would sign the tab in again.
    kept = {key: st.session_state.get(key) for key in (_SAAS_COOKIE_DELETE, _SAAS_SIGNED_OUT)}
    st.session_state.clear()
    for key, value in kept.items():
        if value:
            st.session_state[key] = value


def _render_local_login_button(location: str = "sidebar") -> None:
    flow = build_flow(IDENTITY_SCOPES)
    auth_url, verifier = get_auth_url(flow)
    save_verifier(verifier)
    container = _get_container(location)
    container.link_button(
        "Увійти через Google",
        auth_url,
        use_container_width=True,
    )
    container.caption("Demo в Testing-режимі: працює лише для test users.")


def _render_saas_login_button(location: str = "sidebar") -> None:
    auth_url = _saas_client(_api_base_url()).google_auth_start_url(f"{_app_base_url()}/")
    container = _get_container(location)
    container.link_button(
        "Увійти через Google",
        auth_url,
        use_container_width=True,
    )
    container.caption("Production OAuth через захищений API.")


def _render_logged_in(location: str = "sidebar") -> None:
    user = st.session_state.get("user", {})
    email = user.get("email", "—")
    name = user.get("name", "")
    picture = user.get("picture")

    container = _get_container(location)
    container.subheader("Профіль")
    if picture:
        container.image(picture, width=64)
    if name:
        container.text(name)
    container.caption(email)
    if container.button("Вийти", use_container_width=True):
        if _saas_auth_enabled():
            _logout_saas_session()
        else:
            st.session_state.pop("credentials", None)
            st.session_state.pop("user", None)
        st.rerun()


def ensure_login_state(*, manage_cookie: bool = False) -> bool:
    """Refresh auth state and return True if the user is logged in.

    ``manage_cookie`` only from the auth slot in ``app.py``: the one call that may render
    the session cookie components. Pages call it after that one, with the session just
    checked.
    """
    if _saas_auth_enabled():
        return _restore_saas_session(manage_cookie=manage_cookie)

    _handle_local_oauth_callback()
    if "credentials" in st.session_state:
        creds = credentials_from_dict(st.session_state["credentials"])
        creds = refresh_if_needed(creds)
        st.session_state["credentials"] = credentials_to_dict(creds)
        return True

    return False


def is_auth_restore_pending() -> bool:
    """Return True while the Streamlit cookie component is restoring a session."""
    return bool(st.session_state.get(_SAAS_AUTH_RESTORE_PENDING))


def render_login_button(location: str = "sidebar") -> None:
    """Render the login button in the requested area."""
    if _saas_auth_enabled():
        _render_saas_login_button(location)
        return
    _render_local_login_button(location)


def render_profile(location: str = "sidebar") -> None:
    """Render profile and logout button in the requested area."""
    _render_logged_in(location)


def render_login(
    location: str = "sidebar",
    profile_location: str | None = "sidebar",
) -> bool:
    """Render the auth widget and return True when the user is logged in."""
    logged_in = ensure_login_state()

    if logged_in:
        if profile_location:
            _render_logged_in(profile_location)
        return True

    render_login_button(location)
    return False


def ensure_api_access(purpose: str = "forms") -> bool:
    """Gate pages that need Google API data."""
    if _saas_auth_enabled():
        if not ensure_login_state():
            return False
        session_id = st.session_state.get("saas_session_id")
        if not isinstance(session_id, str) or not session_id:
            return False
        try:
            get_or_load(
                CacheKey(
                    session_key=session_cache_key(session_id),
                    data_kind="access_check",
                    purpose=purpose,
                ),
                ttl_seconds=ACCESS_TTL_SECONDS,
                loader=handle_access_check_errors(
                    lambda: _saas_client(_api_base_url()).check_google_access(
                        session_id,
                        purpose=purpose,
                        next_url=f"{_app_base_url()}/",
                    )
                ),
            )
        except MissingGoogleScopesError as exc:
            st.warning(
                "Цій сторінці потрібен доступ до Google Forms. "
                "Натисни кнопку нижче, щоб додати дозвіл через захищений SaaS API."
            )
            st.link_button("Підключити Google Forms", exc.connect_url, type="primary")
            return False
        except httpx.HTTPError as exc:
            log.exception("saas_google_access_check_failed")
            st.error(f"Не вдалося перевірити доступ до Google API: {exc}")
            return False
        return True

    creds_dict = st.session_state.get("credentials")
    if has_api_scopes(creds_dict):
        return True

    flow = build_flow(API_SCOPES)
    auth_url, verifier = get_auth_url(flow)
    save_verifier(verifier)

    st.warning(
        "Цій сторінці потрібен доступ до твоїх Google Forms і Sheets. "
        "Натисни кнопку нижче, щоб додати дозвіл."
    )
    st.link_button(
        "Підключити Google Forms / Sheets",
        auth_url,
        type="primary",
    )
    return False
