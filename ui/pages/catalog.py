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
from ui.components.auth_widget import ensure_api_access

ENRICHMENT_TICK_SECONDS = 2

st.title("Каталог")

if not ensure_api_access():
    st.stop()

creds = credentials_from_dict(st.session_state["credentials"])


@st.cache_data(ttl=900, show_spinner="Завантажую каталог форм…")
def _cached_drive_list(_creds_token: str) -> list[FormDriveMeta]:
    """Drive list форм, кешується на 15 хв за access_token."""
    return list_forms_with_drive_meta(creds)


col_metric, col_refresh = st.columns([4, 1])
try:
    forms_meta = _cached_drive_list(creds.token or "")
except FormsApiError as exc:
    st.error(f"Не вдалося завантажити каталог: {exc}")
    st.stop()

col_metric.metric("Форм у каталозі", len(forms_meta))
if col_refresh.button("Оновити", help="Скинути кеш і перечитати з Drive"):
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
            "Name": f"{f.edit_url}|{f.name}",
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
            "FullID": f.id,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    # ISO 8601 з Drive — pandas парсить охайно. Sheet timestamps локалізовані
    # (DD.MM.YYYY HH:MM:SS), залишаємо як рядок щоб не наламати дров.
    df["Modified"] = pd.to_datetime(df["Modified"], errors="coerce", utc=True)
    df["Created"] = pd.to_datetime(df["Created"], errors="coerce", utc=True)
    return df


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

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Name": st.column_config.LinkColumn(
                "Назва",
                help="Клік відкриває форму в редакторі Google Forms",
                display_text=r"^.*\|(.*)$",
            ),
            "Title": st.column_config.TextColumn("Внутрішня назва"),
            "Owner": st.column_config.TextColumn("Власник"),
            "Questions": st.column_config.NumberColumn("Питань", format="%d"),
            "Sections": st.column_config.NumberColumn("Секцій", format="%d"),
            "Accepting": st.column_config.CheckboxColumn("Приймає"),
            "Total": st.column_config.NumberColumn("Відповідей", format="%d"),
            "LastResponse": st.column_config.TextColumn("Остання відповідь"),
            "Modified": st.column_config.DatetimeColumn(
                "Змінено", format="DD.MM.YYYY HH:mm"
            ),
            "Created": st.column_config.DatetimeColumn(
                "Створено", format="DD.MM.YYYY HH:mm"
            ),
            "SheetID": st.column_config.TextColumn("Sheet ID", width="small"),
            "Description": st.column_config.TextColumn("Опис"),
            "FullID": st.column_config.TextColumn("Form ID", width="small"),
        },
    )


_table_with_enrichment()
