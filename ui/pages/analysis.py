"""Сторінка «Аналіз» — обробка відповідей обраної Google Form.

5 вкладок:
- 📈 Огляд          — часовий кумулятив + прогноз + дедлайн (PR1)
- 📊 По питаннях    — дескриптивна статистика per question (PR4)
- 🔀 Крос-таби      — таблиці спряженості питання × питання (PR5)
- 🔍 Якість         — fill-duration, drop-out, аномалії (PR6)
- 🎯 Репрезентативність — постстратифікація + DEFF Кіша (PR7)

PR1 заповнює тільки вкладку Огляд; решта показують 'Скоро у PR…' info.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import streamlit as st

from core.auth import credentials_from_dict
from core.charts_timeline import plot_timeline_with_forecast
from core.forecast import ForecastError, ForecastResult, forecast_responses
from core.forms_api import (
    FormsApiError,
    get_form_structure,
    get_linked_sheet_id,
    list_response_timestamps,
    list_user_forms,
)
from core.logger import get_logger
from core.timeline import build_timeline_from_timestamps
from ui.components.auth_widget import ensure_api_access

log = get_logger(__name__)

st.title("Аналіз")

if not ensure_api_access():
    st.stop()

creds = credentials_from_dict(st.session_state["credentials"])

# Selectbox-driven form picker. URL `?form_id=` (з Каталог-кнопки 📊) і
# `preselected_form_id` (legacy) використовуємо ЛИШЕ як default-індекс —
# завжди показуємо selectbox зі списку всіх форм, бо cross-tab auth поки
# не зберігається і LinkColumn у Каталозі відкриває новий tab з порожнім
# session_state.


@st.cache_data(ttl=120, show_spinner="Завантажую список форм…")
def _cached_forms(_creds_token: str) -> list[dict]:
    """Список форм користувача, кешований на 2 хв за access_token."""
    return list_user_forms(creds)


try:
    forms = _cached_forms(creds.token or "")
except FormsApiError as exc:
    log.exception("ui_analysis_list_forms_failed", extra={"status": exc.status})
    st.error(f"Не вдалося отримати список форм: {exc}")
    st.stop()

if not forms:
    st.info(
        "На цьому акаунті не знайдено жодної Google Form. "
        "Створи форму в google.com/forms і повернись сюди."
    )
    st.stop()

# Preselect із URL (з Каталог-кнопки) або session_state (legacy fallback).
preselected_id = st.query_params.get("form_id") or st.session_state.get("preselected_form_id")
default_idx = 0
if preselected_id:
    matched = next((i for i, f in enumerate(forms) if f["id"] == preselected_id), None)
    if matched is not None:
        default_idx = matched
    # Прибираємо обидва preselect-джерела, щоб юзер міг вільно міняти selectbox
    # без "залипання" URL/session_state при наступному rerun.
    if "form_id" in st.query_params:
        del st.query_params["form_id"]
    st.session_state.pop("preselected_form_id", None)

choice = st.selectbox(
    "Форма для аналізу",
    options=forms,
    format_func=lambda f: f["name"],
    index=default_idx,
)
if not choice:
    st.stop()

form_id = choice["id"]


@st.cache_data(ttl=120, show_spinner="Завантажую структуру форми…")
def _cached_structure(form_id_: str, _creds_token: str) -> dict:
    return get_form_structure(creds, form_id_)


try:
    structure = _cached_structure(form_id, creds.token or "")
except FormsApiError as exc:
    log.exception("ui_analysis_get_structure_failed", extra={"form_id": form_id})
    st.error(f"Не вдалося завантажити форму: {exc}")
    st.stop()

form_title = structure.get("info", {}).get("title", "—")
sheet_id = get_linked_sheet_id(structure)

st.caption(f"Форма: **{form_title}**")
if not sheet_id:
    # Огляд tab працює напряму з Forms API і Sheet не потребує.
    # Решта tabs (По питаннях / Крос-таби / Якість) у майбутніх PR'ах
    # покажуть власне попередження про потребу Sheet — там, де вони
    # реально читатимуть answer values.
    st.info(
        "Ця форма не має прив'язаного Google Sheet. "
        "Огляд працює через Forms API безпосередньо; інші вкладки "
        "потребуватимуть Sheet (Responses → Link to Sheets у формі)."
    )


@st.cache_data(ttl=60, show_spinner="Завантажую відповіді…")
def _cached_timestamps(form_id_: str, _creds_token: str) -> list[datetime]:
    """Forms API timestamps, кешовано на 60s за access_token+form_id."""
    return list_response_timestamps(creds, form_id_)


@st.cache_data(ttl=300, show_spinner="Обчислюю прогноз…")
def _cached_forecast(
    _form_id: str,
    n_responses: int,
    first_ts: datetime,
    last_ts: datetime,
    target: int,
    start_idx: int,
    end_idx: int,
    _timestamps: list[datetime],
) -> tuple[ForecastResult | None, str | None]:
    """Кешований прогноз для subset'у `_timestamps[start_idx-1:end_idx]`.

    Cache key: (form_id, n_responses, first_ts, last_ts, target, start_idx, end_idx).
    `_timestamps` з підкреслення — Streamlit пропускає його в hashing,
    передаємо як payload (вже відрізаний субсет).

    Результат повертається у subset-локальних координатах (cumulative
    1..len(subset)). UI зсуває значення на `start_idx - 1`, щоб згрупувати
    з global-нумерацією графіка.

    Returns (result, error_msg). На фейлі фіту result=None, error=повідомлення.
    """
    timeline = build_timeline_from_timestamps(_timestamps)
    try:
        return forecast_responses(timeline, target=target), None
    except ForecastError as exc:
        return None, str(exc)


def _shift_forecast(forecast: ForecastResult, offset: int) -> ForecastResult:
    """Зсунути cumulative-значення прогнозу на `offset` (для subset → global).

    Прогноз рахується на subset'і timestamps, тож його cumulative починається
    з 1 для першої точки subset'у. Графік показує global-нумерацію (1..N
    усіх timestamps); щоб forecast curve візуально продовжувала факт,
    додаємо `offset = start_idx - 1` (= кількість виключених на початку).
    """
    if offset == 0:
        return forecast
    return ForecastResult(
        model=forecast.model,
        aicc=forecast.aicc,
        future_dates=forecast.future_dates,
        future_cum=forecast.future_cum + offset,
        ci_lower=forecast.ci_lower + offset,
        ci_upper=forecast.ci_upper + offset,
        final_estimate=forecast.final_estimate + offset,
        final_ci=(forecast.final_ci[0] + offset, forecast.final_ci[1] + offset),
        rmse=forecast.rmse,
        r_squared=forecast.r_squared,
    )


try:
    timestamps = _cached_timestamps(form_id, creds.token or "")
except FormsApiError as exc:
    log.exception(
        "ui_analysis_list_timestamps_failed",
        extra={"form_id": form_id, "status": exc.status},
    )
    st.error(f"Не вдалося отримати timestamps відповідей: {exc}")
    st.stop()

tab_overview, tab_per_q, tab_crosstabs, tab_quality, tab_repr = st.tabs(
    [
        "📈 Огляд",
        "📊 По питаннях",
        "🔀 Крос-таби",
        "🔍 Якість",
        "🎯 Репрезентативність",
    ]
)

with tab_overview:
    timeline = build_timeline_from_timestamps(timestamps)

    if timeline.cumulative.empty:
        st.info("Поки немає валідних timestamps у відповідях.")
    else:
        # Target і вікно прогнозу — обидва впливають на forecast, тож читаємо
        # ДО виклику, щоб увійшли в cache key. session_state може зберігати
        # stale значення (форма оновилась), тому clamp'имо у [1, n_ts].
        _target_key = f"analysis_target_{form_id}"
        target = int(st.session_state.get(_target_key, 100))

        _window_key = f"analysis_window_{form_id}"
        n_ts = len(timestamps)
        default_window = (1, max(n_ts, 1))
        start_idx, end_idx = st.session_state.get(_window_key, default_window)
        start_idx = max(1, min(int(start_idx), n_ts))
        end_idx = max(start_idx, min(int(end_idx), n_ts))

        subset_timestamps = timestamps[start_idx - 1 : end_idx]
        excluded_mask = np.zeros(n_ts, dtype=bool)
        excluded_mask[: start_idx - 1] = True
        excluded_mask[end_idx:] = True

        # Прогноз — на 25% вперед від тривалості subset'у.
        # Кеш інвалідується при зміні target або вікна.
        if subset_timestamps:
            forecast, forecast_error = _cached_forecast(
                _form_id=form_id,
                n_responses=len(timestamps),
                first_ts=timestamps[0],
                last_ts=timestamps[-1],
                target=target,
                start_idx=start_idx,
                end_idx=end_idx,
                _timestamps=subset_timestamps,
            )
            # Subset → global coordinate shift.
            if forecast is not None:
                forecast = _shift_forecast(forecast, offset=start_idx - 1)
        else:
            forecast, forecast_error = None, None

        # Рядок 1: BANs (current / forecast / target) — над усім.
        ban_cols = st.columns(3)
        current = int(timeline.cumulative.iloc[-1])
        ban_cols[0].metric("Зараз", current)
        if forecast is not None:
            ci_half = (forecast.final_ci[1] - forecast.final_ci[0]) // 2
            ban_cols[1].metric(
                "Прогноз",
                forecast.final_estimate,
                delta=f"±{ci_half}",
                delta_color="off",
            )
        else:
            ban_cols[1].metric("Прогноз", "—")
        ban_cols[2].metric("Мета", target)

        # Рядок 2: controls (target + refresh) — компактно над графіком.
        ctrl_target, ctrl_refresh = st.columns([4, 1])
        with ctrl_target:
            target = int(
                st.number_input(
                    "Цільова кількість відповідей",
                    min_value=1,
                    value=target,
                    step=10,
                    key=_target_key,
                    label_visibility="collapsed",
                    placeholder="Цільова кількість",
                )
            )
        with ctrl_refresh:
            if st.button(
                "Оновити",
                key=f"refresh_{form_id}",
                width="stretch",
                help="Скинути кеш і перечитати свіжі timestamps з Forms API",
            ):
                st.cache_data.clear()
                st.rerun()

        # Рядок 3: графік (без deadline-вертикалі; target — горизонталь).
        fig = plot_timeline_with_forecast(
            timeline=timeline,
            forecast=forecast,
            target=target,
            excluded_mask=excluded_mask,
        )
        st.plotly_chart(fig, width="stretch")

        # Рядок 4: range-slider — обирає вікно для фіту прогнозу. Full-width
        # під графіком; рендериться завжди (а не лише при n_ts ≥ 2), щоб
        # компонент займав те саме місце і не "стрибав" UI.
        if n_ts >= 2:
            st.slider(
                "Вікно для прогнозу",
                min_value=1,
                max_value=n_ts,
                value=(start_idx, end_idx),
                step=1,
                key=_window_key,
                help=(
                    "Перетягни ручки, щоб обмежити, які відповіді "
                    "використовуються для фіту прогнозу. Сірі точки на графіку — "
                    "виключені."
                ),
            )

        # Рядок 5: caption — модель/якість фіту/горизонт.
        n_used = end_idx - start_idx + 1
        window_suffix = f" · використано {n_used} з {n_ts} відповідей" if n_used < n_ts else ""
        if forecast is not None:
            horizon_end = forecast.future_dates[-1].date()
            st.caption(
                f"Прогноз: {forecast.model} (AICc={forecast.aicc:.1f}) · "
                f"горизонт до {horizon_end:%d.%m.%Y} (25% тривалості) · "
                f"95% CI: {forecast.final_ci[0]}–{forecast.final_ci[1]} · "
                f"RMSE={forecast.rmse:.2f} · R²={forecast.r_squared:.3f}"
                f"{window_suffix}"
            )
        elif forecast_error:
            # Якщо помилка — "замало точок" І користувач звузив вікно — пораду
            # "розширити вікно" виносимо явно, бо це найшвидший фікс.
            hint = ""
            if "Замало точок" in forecast_error and n_used < n_ts:
                hint = f" Розширте вікно — повний набір має {n_ts} відповідей."
            st.caption(f"Прогноз недоступний: {forecast_error}{hint}{window_suffix}")

with tab_per_q:
    st.info("Скоро у PR4: дескриптивна статистика по кожному питанню.")

with tab_crosstabs:
    st.info("Скоро у PR5: крос-табуляції питання × питання + χ²-тест.")

with tab_quality:
    st.info("Скоро у PR6: час заповнення, drop-out rate, виявлення підозрілих патернів відповідей.")

with tab_repr:
    st.info(
        "Скоро у PR7: постстратифікаційне зважування, DEFF Кіша, метрика "
        "«ще треба X відповідей з страти Y»."
    )
