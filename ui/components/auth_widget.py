"""Sidebar-віджет автентифікації Google.

render_login() показує одне з двох:
- кнопку "Увійти через Google" (якщо не залогінений);
- профіль + кнопку "Вийти" (якщо залогінений).

OAuth-callback (повернення з accounts.google.com у вигляді ?code=...)
обробляється через st.query_params. PKCE verifier тягнемо з temp-файла
(див. core.auth), бо Streamlit session_state не виживає редиректу.
Credentials серіалізуємо в session_state — достатньо для single-user демо.
"""
from __future__ import annotations

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


def _handle_oauth_callback() -> None:
    """Обробити OAuth callback (?code=...): обміняти code на token.

    Працює і для identity-логіну, і для incremental upgrade — Google
    повертає реальні granted scopes у `?scope=...`, ми використовуємо їх
    при перебудові Flow (з RELAX_TOKEN_SCOPE=1 це робить fetch_token
    толерантним до будь-яких розбіжностей).
    """
    code = st.query_params.get("code")
    if not code:
        return

    verifier = load_verifier()
    if not verifier:
        st.error("Сесія входу зламалась. Натисни «Увійти через Google» ще раз.")
        st.query_params.clear()
        return

    scope_param = st.query_params.get("scope") or ""
    scopes = scope_param.split() if scope_param else IDENTITY_SCOPES

    try:
        flow = build_flow(scopes, code_verifier=verifier)
        creds = exchange_code(flow, code)
        user = get_user_info(creds)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Помилка входу: {exc}")
        st.query_params.clear()
        clear_verifier()
        return

    clear_verifier()
    st.session_state["credentials"] = credentials_to_dict(creds)
    st.session_state["user"] = user
    st.query_params.clear()
    st.rerun()


def _get_container(location: str) -> DeltaGenerator:
    if location == "sidebar":
        return st.sidebar
    return st.container()


def _render_login_button(location: str = "sidebar") -> None:
    flow = build_flow(IDENTITY_SCOPES)
    auth_url, verifier = get_auth_url(flow)
    save_verifier(verifier)
    container = _get_container(location)
    container.link_button("Увійти через Google", auth_url, use_container_width=True)
    container.caption("Demo в Testing-режимі: працює лише для test users.")


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
        st.session_state.pop("credentials", None)
        st.session_state.pop("user", None)
        st.rerun()


def ensure_login_state() -> bool:
    """Оновити стан входу та повернути True, якщо є валідні credentials."""
    _handle_oauth_callback()

    if "credentials" in st.session_state:
        creds = credentials_from_dict(st.session_state["credentials"])
        creds = refresh_if_needed(creds)
        st.session_state["credentials"] = credentials_to_dict(creds)
        return True

    return False


def render_login_button(location: str = "sidebar") -> None:
    """Відмалювати кнопку входу в заданій області."""
    _render_login_button(location)


def render_profile(location: str = "sidebar") -> None:
    """Відмалювати профіль та кнопку виходу в заданій області."""
    _render_logged_in(location)


def render_login(
    location: str = "sidebar",
    profile_location: str | None = "sidebar",
) -> bool:
    """Відмалювати auth-віджет. Повертає True, якщо залогінений."""
    logged_in = ensure_login_state()

    if logged_in:
        if profile_location:
            _render_logged_in(profile_location)
        return True

    _render_login_button(location)
    return False


def ensure_api_access() -> bool:
    """Гейт для сторінок, які працюють з Google API.

    Якщо identity-вхід є, але API-scopes ще не надані — рендеримо кнопку
    "Підключити Google Forms/Sheets", яка запускає incremental authorization.
    Повертає True, якщо доступ є; False, якщо потрібен upgrade (і сторінка
    має викликати st.stop()).
    """
    creds_dict = st.session_state.get("credentials")
    if has_api_scopes(creds_dict):
        return True

    flow = build_flow(API_SCOPES)
    auth_url, verifier = get_auth_url(flow)
    save_verifier(verifier)

    st.warning(
        "Цій сторінці потрібен доступ до твоїх Google Forms і Sheets. "
        "Натисни кнопку нижче, щоб додати дозволи."
    )
    st.link_button(
        "Підключити Google Forms / Sheets",
        auth_url,
        type="primary",
    )
    return False
