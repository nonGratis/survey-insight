"""Каталог — табличний огляд усіх Google Forms, до яких є доступ.

Tier 1: один Drive API виклик → миттєва таблиця з базовими метаданими.
Tier 2: forms.get() для кожної форми → title, опис, секції, питання,
        linkedSheetId, accepting.
Tier 3: Sheets API на колонку A привʼязаного Sheet → total / last
        response timestamps.

Tier 2/3 виконуються у фоні через @st.fragment(run_every) ticker, що
кожні N секунд бере чанк з 5 форм у ThreadPoolExecutor і поповнює
session_state. Користувач не чекає на повний enrichment — таблиця
відображається одразу з Tier 1 і дозаповнюється рядок-за-рядком.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import streamlit as st

from core.auth import credentials_from_dict
from core.forms_api import FormsApiError
from core.forms_catalog import (
    FormDriveMeta,
    FormEnrichment,
    ResponseStats,
    enrich_form,
    fetch_response_stats,
    list_forms_with_drive_meta,
)
from core.google_throttle import DEFAULT_MAX_WORKERS, parallel_map
from core.logger import get_logger
from ui.components.auth_widget import ensure_api_access

log = get_logger(__name__)

ENRICHMENT_TICK_SECONDS = 2

# Усі колонки таблиці у канонічному порядку. UI-користувач у settings
# panel вибирає підмножину і її ж порядок — задавання default тут.
ALL_COLUMNS = [
    "📊",
    "↗",
    "FormName",
    "Title",
    "Owner",
    "Questions",
    "Sections",
    "Accepting",
    "Total",
    "LastResponse",
    "Modified",
    "Created",
    "SheetID",
    "Description",
]
DEFAULT_VISIBLE_COLUMNS = [
    "📊",
    "↗",
    "FormName",
    "Owner",
    "Questions",
    "Accepting",
    "Total",
    "LastResponse",
    "Modified",
]

if not ensure_api_access():
    st.stop()

creds = credentials_from_dict(st.session_state["credentials"])


@st.cache_data(ttl=900, show_spinner="Завантажую каталог форм…")
def _cached_drive_list(_creds_token: str) -> list[FormDriveMeta]:
    """Drive list форм, кешується на 15 хв за access_token."""
    return list_forms_with_drive_meta(creds)


try:
    forms_meta = _cached_drive_list(creds.token or "")
except FormsApiError as exc:
    log.exception("ui_catalog_drive_list_failed", extra={"status": exc.status})
    st.error(f"Не вдалося завантажити каталог: {exc}")
    st.stop()

title_col, metric_col, col_refresh = st.columns([6, 1.5, 0.6])
with title_col:
    st.markdown("## Каталог")
with metric_col:
    st.metric("Форм у каталозі", len(forms_meta))
with col_refresh:
    if col_refresh.button("", icon=":material/refresh:", help="Скинути кеш і перечитати з Drive"):
        _cached_drive_list.clear()
        st.session_state["form_enrichments"] = {}
        st.session_state["form_response_stats"] = {}
        st.rerun()

if not forms_meta:
    st.info(
        "Жодної Google Form не знайдено на цьому акаунті. "
        "Створи форму на forms.google.com і повернись."
    )
    st.stop()


# Sentinel-маркер: None означає "пробували enrich-ити, але отримали HTTP-помилку".
# Це різнить "ще не пробували" (ключ відсутній) від "пробували — failed" (None).
st.session_state.setdefault("form_enrichments", {})
st.session_state.setdefault("form_response_stats", {})


def _render_table_filters(forms: list[FormDriveMeta]) -> dict:
    """Намалювати фільтри над таблицею, повернути значення."""
    top_left, top_mid, top_right = st.columns([2, 1, 1])
    with top_left:
        search = st.text_input("Пошук за назвою", key="catalog_search")
    with top_mid:
        owner_options = sorted({f.owner_email for f in forms if f.owner_email != "—"})
        owners = st.multiselect("Власник", options=owner_options, key="catalog_owners")
    with top_right:
        accepting = st.selectbox(
            "Стан",
            options=["Усі", "Приймає відповіді", "Не приймає"],
            key="catalog_accepting",
        )

    bottom_left, bottom_right = st.columns([2, 1])
    with bottom_left:
        date_range = st.date_input(
            "Змінено в діапазоні",
            value=[],  # порожній список = немає дефолтних меж; користувач задає обидві
            key="catalog_date_range",
        )
    with bottom_right:
        sheet = st.selectbox(
            "Sheet",
            options=["Усі", "З привʼязаним Sheet", "Без Sheet"],
            key="catalog_sheet",
        )

    return {
        "search": search.strip(),
        "owners": owners,
        "date_range": date_range,
        "accepting": accepting,
        "sheet": sheet,
    }


def _apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    """Послідовно застосувати фільтри. Pending-рядки (без enrichment)
    проходять тільки коли селектор стоїть на 'Усі'."""
    out = df

    if f["search"]:
        out = out[out["FormName"].astype(str).str.contains(f["search"], case=False, na=False)]

    if f["owners"]:
        out = out[out["Owner"].isin(f["owners"])]

    if isinstance(f["date_range"], (tuple, list)) and len(f["date_range"]) == 2:
        start, end = f["date_range"]
        if isinstance(start, date) and isinstance(end, date):
            start_ts = pd.Timestamp(datetime.combine(start, datetime.min.time()), tz=UTC)
            end_ts = pd.Timestamp(datetime.combine(end, datetime.max.time()), tz=UTC)
            out = out[(out["Modified"] >= start_ts) & (out["Modified"] <= end_ts)]

    if f["accepting"] == "Приймає відповіді":
        out = out[out["Accepting"] == True]  # noqa: E712 — pandas truth
    elif f["accepting"] == "Не приймає":
        out = out[out["Accepting"] == False]  # noqa: E712

    if f["sheet"] != "Усі":
        has_sheet = out["SheetID"].astype(str).str.len() > 0
        if f["sheet"] == "З привʼязаним Sheet":  # noqa: SIM108 — if/else тут читабельніший за ternary
            out = out[has_sheet]
        else:  # "Без Sheet"
            out = out[~has_sheet]

    return out


def _build_dataframe(
    forms: list[FormDriveMeta],
    enrichments: dict[str, FormEnrichment | None],
    stats: dict[str, ResponseStats],
) -> pd.DataFrame:
    """Зібрати DataFrame, підставляючи placeholders для ще-не-enriched рядків."""
    rows = []
    for f in forms:
        enr = enrichments.get(f.id)
        stat = stats.get(f.id)
        row = {
            "📊": f"/dynamics?form_id={f.id}",
            "↗": f.edit_url,
            "FormName": f.name,
            "Title": enr.title if enr else "",
            "Owner": f.owner_email,
            "Questions": enr.questions_count if enr else None,
            "Sections": enr.sections_count if enr else None,
            "Accepting": enr.accepting_responses if enr else None,
            "Total": stat.total if stat else None,
            "LastResponse": stat.last_response if stat else "",
            "Modified": f.modified_time,
            "Created": f.created_time,
            "SheetID": (enr.linked_sheet_id or "") if enr else "",
            "Description": (enr.description or "") if enr else "",
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    # ISO 8601 з Drive — pandas парсить охайно. Sheet timestamps локалізовані
    # (DD.MM.YYYY HH:MM:SS), залишаємо як рядок щоб не наламати дров.
    df["Modified"] = pd.to_datetime(df["Modified"], errors="coerce", utc=True)
    df["Created"] = pd.to_datetime(df["Created"], errors="coerce", utc=True)
    return df


filter_values = _render_table_filters(forms_meta)


@st.fragment(run_every=ENRICHMENT_TICK_SECONDS)
def _table_with_enrichment() -> None:
    """Один chunk enrichment'у + рендер таблиці. Тікає кожні 2с."""
    enrichments = st.session_state["form_enrichments"]
    stats = st.session_state["form_response_stats"]

    pending = [f for f in forms_meta if f.id not in enrichments]
    if pending:
        chunk = pending[:DEFAULT_MAX_WORKERS]
        enrich_results = parallel_map(lambda f: enrich_form(creds, f.id), chunk)
        sheet_targets: list[tuple[str, str]] = []
        for form, result in enrich_results:
            if isinstance(result, Exception):
                enrichments[form.id] = None  # failed — не пробуємо знову
                st.toast(f"⚠️ {form.name}: {result}", icon="⚠️")
                continue
            enrichments[form.id] = result
            if result.linked_sheet_id:
                sheet_targets.append((form.id, result.linked_sheet_id))

        if sheet_targets:
            stat_results = parallel_map(
                lambda pair: fetch_response_stats(creds, pair[1]),
                sheet_targets,
            )
            for (form_id, _sheet_id), stat_result in stat_results:
                if isinstance(stat_result, Exception):
                    st.toast(f"⚠️ stats fetch: {stat_result}", icon="⚠️")
                    continue
                stats[form_id] = stat_result

    loaded = sum(1 for f in forms_meta if f.id in enrichments)
    total = len(forms_meta)
    if loaded < total:
        st.progress(
            loaded / total,
            text=f"Підвантажую деталі: {loaded}/{total}",
        )

    df = _build_dataframe(forms_meta, enrichments, stats)
    filtered = _apply_filters(df, filter_values)
    table_columns = list(ALL_COLUMNS)
    display = filtered[[c for c in table_columns if c in filtered.columns]].copy()
    if "Accepting" in display.columns:
        display["Accepting"] = display["Accepting"].map({True: "✓", False: "✗"}).fillna("")

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "📊": st.column_config.LinkColumn(
                help="Перейти на сторінку Динаміка",
                display_text="📊",
                width="small",
            ),
            "↗": st.column_config.LinkColumn(
                help="Відкрити форму в редакторі Google Forms",
                display_text="↗",
                width="small",
            ),
            "FormName": st.column_config.TextColumn("Назва"),
            "Title": st.column_config.TextColumn("Внутрішня назва"),
            "Owner": st.column_config.TextColumn("Власник"),
            "Questions": st.column_config.NumberColumn("Питань", format="%d"),
            "Sections": st.column_config.NumberColumn("Секцій", format="%d"),
            "Accepting": st.column_config.TextColumn("Приймає"),
            "Total": st.column_config.NumberColumn("Відповідей", format="%d"),
            "LastResponse": st.column_config.TextColumn("Остання відповідь"),
            "Modified": st.column_config.DatetimeColumn("Змінено", format="DD.MM.YYYY HH:mm"),
            "Created": st.column_config.DatetimeColumn("Створено", format="DD.MM.YYYY HH:mm"),
            "SheetID": st.column_config.TextColumn("Sheet ID", width="small"),
            "Description": st.column_config.TextColumn("Опис"),
        },
    )


_table_with_enrichment()
