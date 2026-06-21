"""Google auth widget for local demo OAuth and production SaaS sessions."""

from __future__ import annotations

import os
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
from ui.saas_api import SaaSApiClient, SaaSSession

log = get_logger(__name__)

_SAAS_SESSION_COOKIE = "survey_insight_session_id"
_SAAS_SESSION_DAYS = 30
_SAAS_VALIDATE_TTL_SECONDS = 30


def _saas_auth_enabled() -> bool:
    return os.environ.get("APP_ENV") == "production" and bool(os.environ.get("API_BASE_URL"))


def _app_base_url() -> str:
    return os.environ.get("APP_BASE_URL", "http://localhost:8501").rstrip("/")


def _api_base_url() -> str:
    return os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")


@st.cache_resource
def _saas_client(base_url: str) -> SaaSApiClient:
    return SaaSApiClient(base_url)


def _cookie_manager() -> stx.CookieManager:
    return stx.CookieManager(key="saas_auth_cookies")


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
    st.query_params.clear()
    st.rerun()
    return True


def _restore_saas_session() -> bool:
    auth_error = _query_param("auth_error")
    if auth_error:
        st.error("Не вдалося завершити Google-вхід. Спробуй увійти ще раз.")
        st.query_params.clear()
        return False

    if _handle_saas_login_ticket():
        return True

    if _has_fresh_saas_session():
        return True

    cookie_manager = _cookie_manager()
    session_id = st.session_state.get("saas_session_id") or cookie_manager.get(_SAAS_SESSION_COOKIE)
    if not isinstance(session_id, str) or not session_id:
        return False

    try:
        session = _saas_client(_api_base_url()).read_session(session_id)
    except httpx.HTTPError:
        log.exception("saas_session_restore_failed")
        _clear_saas_session(cookie_manager)
        return False

    if not session.authenticated:
        _clear_saas_session(cookie_manager)
        return False

    _remember_saas_session(session, cookie_manager)
    return True


def _has_fresh_saas_session() -> bool:
    checked_at = st.session_state.get("saas_session_checked_at")
    return (
        isinstance(st.session_state.get("saas_session_id"), str)
        and bool(st.session_state.get("user"))
        and isinstance(checked_at, datetime)
        and datetime.now(UTC) - checked_at < timedelta(seconds=_SAAS_VALIDATE_TTL_SECONDS)
    )


def _remember_saas_session(
    session: SaaSSession,
    cookie_manager: stx.CookieManager | None = None,
) -> None:
    if not session.session_id:
        return

    st.session_state["saas_session_id"] = session.session_id
    st.session_state["saas_session_checked_at"] = datetime.now(UTC)
    st.session_state["user"] = {
        "id": session.user_id,
        "email": session.email,
        "name": session.name,
        "plan": session.plan,
    }
    manager = cookie_manager or _cookie_manager()
    manager.set(
        _SAAS_SESSION_COOKIE,
        session.session_id,
        key="set_saas_session",
        path="/",
        max_age=float(_SAAS_SESSION_DAYS * 24 * 60 * 60),
        secure=_app_base_url().startswith("https://"),
        same_site="lax",
    )


def _clear_saas_session(cookie_manager: stx.CookieManager | None = None) -> None:
    for key in ("saas_session_id", "saas_session_checked_at", "user"):
        st.session_state.pop(key, None)
    manager = cookie_manager or _cookie_manager()
    if manager.get(_SAAS_SESSION_COOKIE) is not None:
        manager.delete(_SAAS_SESSION_COOKIE, key="delete_saas_session")


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
            session_id = st.session_state.get("saas_session_id")
            try:
                _saas_client(_api_base_url()).logout(
                    session_id if isinstance(session_id, str) else None
                )
            except httpx.HTTPError:
                log.exception("saas_logout_failed")
            _clear_saas_session()
        else:
            st.session_state.pop("credentials", None)
            st.session_state.pop("user", None)
        st.rerun()


def ensure_login_state() -> bool:
    """Refresh auth state and return True if the user is logged in."""
    if _saas_auth_enabled():
        return _restore_saas_session()

    _handle_local_oauth_callback()
    if "credentials" in st.session_state:
        creds = credentials_from_dict(st.session_state["credentials"])
        creds = refresh_if_needed(creds)
        st.session_state["credentials"] = credentials_to_dict(creds)
        return True

    return False


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


def ensure_api_access() -> bool:
    """Gate pages that still use direct Google API credentials."""
    if _saas_auth_enabled():
        if not ensure_login_state():
            return False
        st.info(
            "Production-вхід працює. Ця сторінка ще очікує міграцію читання "
            "Google Forms у SaaS API, щоб Streamlit не тримав OAuth-токени."
        )
        return False

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
