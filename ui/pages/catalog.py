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

from core.forms_catalog import (
    FormDriveMeta,
    FormEnrichment,
    ResponseStats,
)
from core.google_throttle import DEFAULT_MAX_WORKERS, parallel_map
from core.logger import get_logger
from ui.components.action_bar import render_action_bar
from ui.components.auth_widget import ensure_api_access
from ui.components.form_picker import FORM_KEY, clear_forms_cache
from ui.components.metric_bar import MetricItem, render_metric_bar
from ui.components.page_shell import render_empty_state, render_error_state, render_page_header
from ui.google_data import (
    cache_token,
    clear_catalog_cache,
    google_data_client,
    list_catalog_snapshot,
)

log = get_logger(__name__)

ENRICHMENT_TICK_SECONDS = 2
TABLE_HEADER_HEIGHT_PX = 38
TABLE_ROW_HEIGHT_PX = 35
TABLE_MIN_HEIGHT_PX = 360
TABLE_MAX_HEIGHT_PX = 680
STATUS_ALL = "Усі"
STATUS_OPEN = "Відкриті"
STATUS_CLOSED = "Закриті"
STATUS_UNPUBLISHED = "Неопубліковані"
STATUS_UNKNOWN = "Невідомо"
STATUS_OPTIONS = [STATUS_ALL, STATUS_OPEN, STATUS_CLOSED, STATUS_UNPUBLISHED, STATUS_UNKNOWN]

# Усі колонки таблиці у канонічному порядку. UI-користувач у settings
# panel вибирає підмножину і її ж порядок — задавання default тут.
ALL_COLUMNS = [
    "FormName",
    "PublicationStatus",
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
    "FormName",
    "PublicationStatus",
    "Owner",
    "Questions",
    "Accepting",
    "Total",
    "LastResponse",
    "Modified",
]

if not ensure_api_access():
    st.stop()


@st.cache_data(ttl=900, show_spinner="Завантажую каталог форм…")
def _cached_catalog_snapshot(
    _cache_token: str,
) -> tuple[list[FormDriveMeta], dict[str, FormEnrichment | None], dict[str, ResponseStats]]:
    """Catalog snapshot; SaaS uses aggregate API, local mode keeps Drive list fallback."""
    return list_catalog_snapshot()


try:
    forms_meta, initial_enrichments, initial_stats = _cached_catalog_snapshot(cache_token())
except Exception as exc:  # noqa: BLE001
    log.exception("ui_catalog_drive_list_failed", extra={"error_code": type(exc).__name__})
    render_error_state("Не вдалося завантажити каталог.", details=str(exc))
    st.stop()

render_page_header("Каталог")
action = render_action_bar(
    refresh_scope="catalog",
)
if action.refresh_clicked:
    clear_forms_cache()
    clear_catalog_cache()
    _cached_catalog_snapshot.clear()
    st.session_state["form_enrichments"] = {}
    st.session_state["form_response_stats"] = {}
    st.rerun()

if not forms_meta:
    render_empty_state(
        "Жодної Google Form не знайдено на цьому акаунті. "
        "Створи форму на forms.google.com і повернись."
    )
    st.stop()


# Sentinel-маркер: None означає "пробували enrich-ити, але отримали HTTP-помилку".
# Це різнить "ще не пробували" (ключ відсутній) від "пробували — failed" (None).
st.session_state.setdefault("form_enrichments", {})
st.session_state.setdefault("form_response_stats", {})
st.session_state["form_enrichments"].update(initial_enrichments)
st.session_state["form_response_stats"].update(initial_stats)


def _render_table_filters(forms: list[FormDriveMeta]) -> dict:
    """Намалювати фільтри над таблицею, повернути значення."""
    top_left, top_mid, top_right = st.columns([2, 1, 1])
    with top_left:
        search = st.text_input("Пошук за назвою", key="catalog_search")
    with top_mid:
        owner_options = sorted({f.owner_email for f in forms if f.owner_email != "—"})
        owners = st.multiselect("Власник", options=owner_options, key="catalog_owners")
    with top_right:
        publication_status = st.selectbox(
            "Статус",
            options=STATUS_OPTIONS,
            key="catalog_publication_status",
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
        "publication_status": publication_status,
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

    if f["publication_status"] != STATUS_ALL:
        out = out[out["PublicationStatus"] == f["publication_status"]]

    if f["sheet"] != "Усі":
        has_sheet = out["SheetID"].astype(str).str.len() > 0
        if f["sheet"] == "З привʼязаним Sheet":  # noqa: SIM108 — if/else тут читабельніший за ternary
            out = out[has_sheet]
        else:  # "Без Sheet"
            out = out[~has_sheet]

    return out


def _publication_status(enr: FormEnrichment | None) -> str:
    if enr is None:
        return STATUS_UNKNOWN
    if enr.is_published is True and enr.accepting_responses is True:
        return STATUS_OPEN
    if enr.is_published is True and enr.accepting_responses is False:
        return STATUS_CLOSED
    if enr.is_published is False:
        return STATUS_UNPUBLISHED
    return STATUS_UNKNOWN


def _render_catalog_metrics(df: pd.DataFrame) -> None:
    counts = df["PublicationStatus"].value_counts()
    render_metric_bar(
        [
            MetricItem("Форм у каталозі", len(df)),
            MetricItem("Відкритих", int(counts.get(STATUS_OPEN, 0))),
            MetricItem("Закритих", int(counts.get(STATUS_CLOSED, 0))),
            MetricItem("Неопублікованих", int(counts.get(STATUS_UNPUBLISHED, 0))),
            MetricItem("Невідомо", int(counts.get(STATUS_UNKNOWN, 0))),
        ],
        columns=5,
    )


def _table_height(row_count: int) -> int:
    content_height = TABLE_HEADER_HEIGHT_PX + max(row_count, 1) * TABLE_ROW_HEIGHT_PX
    return max(TABLE_MIN_HEIGHT_PX, min(TABLE_MAX_HEIGHT_PX, content_height))


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
            "FormID": f.id,
            "FormName": f.name,
            "PublicationStatus": _publication_status(enr),
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


def _render_table_with_enrichment() -> None:
    """One enrichment chunk plus table render."""
    enrichments = st.session_state["form_enrichments"]
    stats = st.session_state["form_response_stats"]
    data = google_data_client()

    pending = [f for f in forms_meta if f.id not in enrichments]
    if pending:
        chunk = pending[:DEFAULT_MAX_WORKERS]
        enrich_results = parallel_map(
            lambda f: data.get_form_summary(f.id),
            chunk,
        )
        stat_targets: list[str] = []
        for form, result in enrich_results:
            if isinstance(result, Exception):
                enrichments[form.id] = None  # failed — не пробуємо знову
                st.toast(f"⚠️ {form.name}: {result}", icon="⚠️")
                continue
            enrichments[form.id] = result
            stat_targets.append(form.id)

        if stat_targets:
            stat_results = parallel_map(
                data.get_response_stats,
                stat_targets,
            )
            for form_id, stat_result in stat_results:
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
    _render_catalog_metrics(df)
    filtered = _apply_filters(df, filter_values)
    selection_source = filtered.reset_index(drop=True)
    table_columns = list(ALL_COLUMNS)
    display = selection_source[[c for c in table_columns if c in selection_source.columns]].copy()
    if "Accepting" in display.columns:
        display["Accepting"] = display["Accepting"].map({True: "✓", False: "✗"}).fillna("")

    selection = st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=_table_height(len(display)),
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "FormName": st.column_config.TextColumn("Назва"),
            "PublicationStatus": st.column_config.TextColumn("Статус"),
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
    selected_rows = getattr(getattr(selection, "selection", None), "rows", [])
    if selected_rows:
        selected_form_id = selection_source.iloc[selected_rows[0]]["FormID"]
        if selected_form_id and st.session_state.get(FORM_KEY) != selected_form_id:
            st.session_state[FORM_KEY] = selected_form_id
            st.rerun()


@st.fragment(run_every=ENRICHMENT_TICK_SECONDS)
def _table_with_enrichment_fragment() -> None:
    _render_table_with_enrichment()


if any(f.id not in st.session_state["form_enrichments"] for f in forms_meta):
    _table_with_enrichment_fragment()
else:
    _render_table_with_enrichment()
