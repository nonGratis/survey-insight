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


def _render_login_button() -> None:
    flow = build_flow(IDENTITY_SCOPES)
    auth_url, verifier = get_auth_url(flow)
    save_verifier(verifier)
    st.sidebar.markdown("### Доступ")
    st.sidebar.link_button("Увійти через Google", auth_url, use_container_width=True)
    st.sidebar.caption("Demo в Testing-режимі: працює лише для test users.")


def _render_logged_in() -> None:
    user = st.session_state.get("user", {})
    email = user.get("email", "—")
    name = user.get("name", "")
    picture = user.get("picture")

    st.sidebar.markdown("### Профіль")
    if picture:
        st.sidebar.image(picture, width=64)
    st.sidebar.markdown(f"**{name}**")
    st.sidebar.caption(email)
    if st.sidebar.button("Вийти", use_container_width=True):
        st.session_state.pop("credentials", None)
        st.session_state.pop("user", None)
        st.rerun()


def render_login() -> bool:
    """Відмалювати auth-віджет у sidebar. Повертає True, якщо залогінений."""
    _handle_oauth_callback()

    if "credentials" in st.session_state:
        creds = credentials_from_dict(st.session_state["credentials"])
        creds = refresh_if_needed(creds)
        st.session_state["credentials"] = credentials_to_dict(creds)
        _render_logged_in()
        return True

    _render_login_button()
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
