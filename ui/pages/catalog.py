"""Каталог — табличний огляд усіх Google Forms, до яких є доступ.

Tier 1: один Drive API виклик → миттєва таблиця з базовими метаданими.
Tier 2: forms.get() для кожної форми → title, опис, секції, питання,
        linkedSheetId, accepting.
Tier 3: Forms API responses.list → кількість відповідей, перша й остання.

Tier 2/3 виконуються у фоні через @st.fragment(run_every) ticker: кожен крок
надсилає до CATALOG_ENRICH_PARALLEL_REQUESTS запитів по CATALOG_ENRICH_CHUNK_SIZE
форм паралельно і поповнює session_state. Рядки, які притримав ліміт Google
(його стереже API), повторюються самі через паузу — core/catalog_loading.py.
Користувач не чекає на повний enrichment — таблиця відображається одразу з
Tier 1 і дозаповнюється.
"""

from __future__ import annotations

import functools
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime

import pandas as pd
import streamlit as st

from core.catalog_loading import Retry, is_loading, next_form_batches, schedule_retry
from core.forms_catalog import (
    FormDriveMeta,
    FormEnrichment,
    ResponseStats,
)
from core.logger import get_logger
from ui.components.action_bar import render_action_bar
from ui.components.auth_widget import ensure_api_access
from ui.components.form_picker import FORM_KEY, clear_forms_cache
from ui.components.metric_bar import MetricItem, render_metric_bar
from ui.components.page_shell import render_empty_state, render_error_state, render_page_header
from ui.google_data import (
    CatalogEnrichmentResult,
    cache_token,
    clear_catalog_cache,
    google_data_client,
    list_catalog_snapshot,
)

log = get_logger(__name__)

ENRICHMENT_TICK_SECONDS = 2
# Найбільша порція, яку приймає API (SI_CATALOG_ENRICH_MAX_IDS).
CATALOG_ENRICH_CHUNK_SIZE = 20
CATALOG_ENRICH_PARALLEL_REQUESTS = 3
ACTIVE_RECENT_DAYS = 7
RETRYABLE_DATA_STATUSES = {"timeout", "api_error", "rate_limited"}
TABLE_HEADER_HEIGHT_PX = 38
TABLE_ROW_HEIGHT_PX = 35
TABLE_MIN_HEIGHT_PX = 360
TABLE_MAX_HEIGHT_PX = 680
TABLE_KEY = "catalog_table"
# FormID рядків у порядку, в якому таблицю показано востаннє: вибір приходить номером рядка.
TABLE_FORM_IDS_KEY = "catalog_table_form_ids"
TABLE_PICKED_KEY = "catalog_table_picked_form"
# Форма, з якою сторінку намальовано востаннє, і прапорець «цей запуск без порції».
TABLE_DRAWN_FORM_KEY = "catalog_table_drawn_form"
SKIP_LOADING_STEP_KEY = "catalog_skip_loading_step"
LOADING_STATUS_HEIGHT_PX = 72
STATUS_ALL = "Усі"
STATUS_OPEN = "Відкриті"
STATUS_CLOSED = "Закриті"
STATUS_UNPUBLISHED = "Неопубліковані"
STATUS_UNKNOWN = "Невідомо"
# Деталі форми ще не прийшли. Окремо від «Невідомо», щоб черга довантаження не виглядала
# як форми, про які Google нічого не каже.
STATUS_LOADING = "Завантажується"
STATUS_OPTIONS = [STATUS_ALL, STATUS_OPEN, STATUS_CLOSED, STATUS_UNPUBLISHED, STATUS_UNKNOWN]

# Колонки таблиці в порядку показу. Сховані (HIDDEN_COLUMNS) користувач вмикає кнопкою
# «Показати/сховати колонки» над таблицею.
TABLE_COLUMNS = [
    "FormName",
    "PublicationStatus",
    "DataStatus",
    "Total",
    "LastResponse",
    "Activity",
    "Questions",
    "Owner",
    "Modified",
    "Created",
    "Sections",
    "Title",
    "Description",
    "UpdatedAgo",
]
HIDDEN_COLUMNS = {"Sections", "Title", "Description", "UpdatedAgo"}
# Стан даних показуємо, лише коли якийсь рядок не довантажився; «Ок» і черга — не новина.
DATA_STATUS_QUIET = {"Ок", "Завантажується"}
# У рядку — про одну форму; категорії фільтра й лічильники лишаються в множині.
STATUS_ROW_LABELS = {
    STATUS_OPEN: "Відкрита",
    STATUS_CLOSED: "Закрита",
    STATUS_UNPUBLISHED: "Не опублікована",
}
DATETIME_FORMAT = "DD.MM.YYYY HH:mm"

if not ensure_api_access():
    st.stop()


@st.cache_data(ttl=900, show_spinner="Завантажую каталог форм…")
def _cached_catalog_snapshot(
    session_token: str,
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
    st.session_state["form_data_status"] = {}
    st.session_state["form_data_fetched_at"] = {}
    st.session_state["form_retries"] = {}
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
st.session_state.setdefault("form_data_status", {})
st.session_state.setdefault("form_data_fetched_at", {})
# Рядки, що чекають автоматичного повтору: form_id → Retry.
st.session_state.setdefault("form_retries", {})
st.session_state["form_enrichments"].update(initial_enrichments)
st.session_state["form_response_stats"].update(initial_stats)


OWNERSHIP_ALL = "Усі"
OWNERSHIP_MINE = "Мої"
OWNERSHIP_OTHERS = "Чужі"
OWNERSHIP_OPTIONS = [OWNERSHIP_ALL, OWNERSHIP_MINE, OWNERSHIP_OTHERS]


def _signed_in_email() -> str:
    user = st.session_state.get("user")
    email = user.get("email") if isinstance(user, dict) else None
    return email if isinstance(email, str) else ""


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

    bottom_left, bottom_mid, bottom_right = st.columns([2, 1, 1])
    with bottom_left:
        date_range = st.date_input(
            "Змінено в діапазоні",
            value=[],  # порожній список = немає дефолтних меж; користувач задає обидві
            key="catalog_date_range",
        )
    with bottom_mid:
        sheet = st.selectbox(
            "Sheet",
            options=["Усі", "З привʼязаним Sheet", "Без Sheet"],
            key="catalog_sheet",
        )
    user_email = _signed_in_email()
    with bottom_right:
        ownership = st.segmented_control(
            "Чиї форми",
            options=OWNERSHIP_OPTIONS,
            default=OWNERSHIP_ALL,
            key="catalog_ownership",
            disabled=not user_email,
            help="«Мої» — форми, власник яких ви; «Чужі» — ті, якими з вами поділились.",
        )

    return {
        "search": search.strip(),
        "owners": owners,
        "date_range": date_range,
        "publication_status": publication_status,
        "sheet": sheet,
        # Повторний клік знімає вибір: тоді показуємо всі форми.
        "ownership": ownership or OWNERSHIP_ALL,
        "user_email": user_email,
    }


def _apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    """Послідовно застосувати фільтри. Pending-рядки (без enrichment)
    проходять тільки коли селектор стоїть на 'Усі'."""
    out = df

    if f["search"]:
        out = out[out["FormName"].astype(str).str.contains(f["search"], case=False, na=False)]

    if f["owners"]:
        out = out[out["Owner"].isin(f["owners"])]

    if f["ownership"] != OWNERSHIP_ALL and f["user_email"]:
        # Drive і вхід Google можуть писати ту саму адресу різним регістром.
        mine = out["Owner"].astype(str).str.casefold() == f["user_email"].casefold()
        out = out[mine] if f["ownership"] == OWNERSHIP_MINE else out[~mine]

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


def _publication_status(enr: FormEnrichment | None, *, loaded: bool) -> str:
    if not loaded:
        return STATUS_LOADING
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


def _response_activity(stat: ResponseStats | None) -> tuple[str, int | None]:
    if stat is None:
        return "", None
    if stat.total <= 0 or not stat.last_response:
        return "Без відповідей", None

    last_response = pd.to_datetime(stat.last_response, errors="coerce", utc=True)
    if pd.isna(last_response):
        return STATUS_UNKNOWN, None

    delta = pd.Timestamp.now(tz=UTC) - last_response
    days = max(int(delta.total_seconds() // 86400), 0)
    if days <= ACTIVE_RECENT_DAYS:
        return "Активна", days
    return "Затухла", days


def _data_status_label(
    form_id: str,
    enrichments: dict[str, FormEnrichment | None],
    statuses: dict[str, str],
    retries: dict[str, Retry],
) -> str:
    # Рядок, що чекає автоматичного повтору, ще вантажиться: помилку показуємо лише тоді,
    # коли повтори скінчились.
    if form_id not in enrichments or form_id in retries:
        return "Завантажується"
    status = statuses.get(form_id)
    if status == "ok":
        return "Ок"
    if status == "timeout":
        return "Таймаут"
    if status == "rate_limited":
        return "Ліміт Google"
    if status == "no_access":
        return "Немає доступу"
    if status == "deleted":
        return "Видалена"
    if status == "unsupported":
        return "Не підтримується"
    if status == "api_error":
        return "Помилка API"
    return "Помилка" if enrichments.get(form_id) is None else "Ок"


def _retryable_enrichment_ids(
    forms: list[FormDriveMeta],
    statuses: dict[str, str],
    retries: dict[str, Retry],
) -> list[str]:
    """Рядки з тимчасовою помилкою, які сторінка вже не повторює сама."""
    return [
        form.id
        for form in forms
        if statuses.get(form.id) in RETRYABLE_DATA_STATUSES and form.id not in retries
    ]


def _clear_enrichment_state_for(form_ids: list[str]) -> None:
    for form_id in form_ids:
        st.session_state["form_enrichments"].pop(form_id, None)
        st.session_state["form_response_stats"].pop(form_id, None)
        st.session_state["form_data_status"].pop(form_id, None)
        st.session_state["form_data_fetched_at"].pop(form_id, None)
        st.session_state["form_retries"].pop(form_id, None)


def _updated_ago_label(value: str | None) -> str:
    if not value:
        return ""
    fetched_at = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(fetched_at):
        return ""
    delta_seconds = max((pd.Timestamp.now(tz=UTC) - fetched_at).total_seconds(), 0)
    if delta_seconds < 60:
        return "щойно"
    if delta_seconds < 3600:
        return f"{int(delta_seconds // 60)} хв тому"
    if delta_seconds < 86400:
        return f"{int(delta_seconds // 3600)} год тому"
    return f"{int(delta_seconds // 86400)} дн тому"


def _build_dataframe(
    forms: list[FormDriveMeta],
    enrichments: dict[str, FormEnrichment | None],
    stats: dict[str, ResponseStats],
    statuses: dict[str, str],
    fetched_at: dict[str, str],
    retries: dict[str, Retry],
) -> pd.DataFrame:
    """Зібрати DataFrame, підставляючи placeholders для ще-не-enriched рядків."""
    rows = []
    for f in forms:
        enr = enrichments.get(f.id)
        stat = stats.get(f.id)
        activity, _ = _response_activity(stat)
        # Статус відомий, щойно прийшли деталі форми, навіть якщо відповіді ще в черзі.
        status_loaded = enr is not None or (f.id in enrichments and f.id not in retries)
        row = {
            "FormID": f.id,
            "FormName": f.name,
            "PublicationStatus": _publication_status(enr, loaded=status_loaded),
            "DataStatus": _data_status_label(f.id, enrichments, statuses, retries),
            "Title": enr.title if enr else "",
            "Owner": f.owner_email,
            "Questions": enr.questions_count if enr else None,
            "Sections": enr.sections_count if enr else None,
            "Total": stat.total if stat else None,
            "LastResponse": stat.last_response if stat else None,
            "Activity": activity,
            "UpdatedAgo": _updated_ago_label(fetched_at.get(f.id)),
            "Modified": f.modified_time,
            "Created": f.created_time,
            "SheetID": (enr.linked_sheet_id or "") if enr else "",
            "Description": (enr.description or "") if enr else "",
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    # Drive дає ISO 8601, час відповідей — теж ISO (UTC): дати сортуються як дати.
    for column in ("Modified", "Created", "LastResponse"):
        df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)
    return df


def _table_display(rows: pd.DataFrame) -> pd.DataFrame:
    """Колонки, які бачить користувач, у порядку показу; статус — про одну форму."""
    display = rows[[column for column in TABLE_COLUMNS if column in rows.columns]].copy()
    display["PublicationStatus"] = display["PublicationStatus"].replace(STATUS_ROW_LABELS)
    return display


def _table_column_config(*, hide_owner: bool, show_data_status: bool) -> dict:
    """Підписи й вигляд колонок. Час — у часовому поясі браузера, як його бачить людина."""
    timezone = st.context.timezone

    def when(label: str):  # returns a Streamlit column config
        return st.column_config.DatetimeColumn(label, format=DATETIME_FORMAT, timezone=timezone)

    config = {
        "FormName": st.column_config.TextColumn("Назва"),
        "PublicationStatus": st.column_config.TextColumn("Статус"),
        "DataStatus": st.column_config.TextColumn("Стан даних"),
        "Total": st.column_config.NumberColumn("Відповідей", format="%d"),
        "LastResponse": when("Остання відповідь"),
        "Activity": st.column_config.TextColumn(
            "Активність",
            help=f"Активна — остання відповідь не давніше {ACTIVE_RECENT_DAYS} днів.",
        ),
        "Questions": st.column_config.NumberColumn("Запитань", format="%d"),
        "Owner": st.column_config.TextColumn("Власник"),
        "Modified": when("Змінено"),
        "Created": when("Створено"),
        "Sections": st.column_config.NumberColumn("Секцій", format="%d"),
        "Title": st.column_config.TextColumn(
            "Заголовок для респондентів",
            help="Заголовок, який бачать респонденти. «Назва» — ім'я файлу в Google Drive.",
        ),
        "Description": st.column_config.TextColumn("Опис"),
        "UpdatedAgo": st.column_config.TextColumn(
            "Дані отримано",
            help=(
                "Коли сервіс востаннє отримав дані цієї форми з Google. "
                "Коли змінювали саму форму, показує стовпець «Змінено»."
            ),
        ),
    }
    hidden = set(HIDDEN_COLUMNS)
    if not show_data_status:
        hidden.add("DataStatus")
    if hide_owner:
        # Коли обрано «Мої», власник у всіх рядках той самий — колонка нічого не каже.
        hidden.add("Owner")
    # Сховану колонку користувач може показати кнопкою над таблицею.
    for column in hidden:
        config[column]["hidden"] = True
    return config


filter_values = _render_table_filters(forms_meta)


def _table_key(filters: dict) -> str:
    """Ключ таблиці: сталий, поки довантажуються деталі, новий — коли змінились фільтри.

    Streamlit пам'ятає позначений рядок за номером, а після зміни фільтрів на цьому номері
    вже інша форма. Тому з новими фільтрами таблиця починається наново й позначає поточну.
    """
    signature = hashlib.sha256(repr(sorted(filters.items())).encode("utf-8")).hexdigest()
    return f"{TABLE_KEY}_{signature[:12]}"


def _pick_form_from_table(table_key: str) -> None:
    """Зробити поточною форму клікнутого рядка.

    Streamlit викликає це лише тоді, коли користувач змінив вибір, і передає номер рядка.
    Номер читаємо за рядками, які користувач бачив, тож рядки, що зсунулись під час
    довантаження чи фільтрування, самі форму не перемикають.
    """
    rows = st.session_state[table_key]["selection"]["rows"]
    form_ids = st.session_state.get(TABLE_FORM_IDS_KEY, [])
    if not rows or rows[0] >= len(form_ids):
        return
    selected_form_id = form_ids[rows[0]]
    if st.session_state.get(FORM_KEY) != selected_form_id:
        st.session_state[FORM_KEY] = selected_form_id
        st.session_state[TABLE_PICKED_KEY] = True


def _mark_current_form(table_key: str, form_ids: list[str], current_form_id: str | None) -> None:
    """Позначка в таблиці йде за поточною формою, звідки б її не змінили.

    Нову таблицю позначає selection_default. Наявна пам'ятає свій вибір, тож форму,
    обрану зверху, позначаємо через Session State: таблиця та сама, прокрутка лишається.
    """
    state = st.session_state.get(table_key)
    if state is None:
        return
    wanted = [form_ids.index(current_form_id)] if current_form_id in form_ids else []
    if list(state["selection"]["rows"]) != wanted:
        st.session_state[table_key] = {"selection": {"rows": wanted, "columns": [], "cells": []}}


def _render_loading_status(
    details_loaded: int, finished: int, total: int, retryable_ids: list[str]
) -> None:
    """Прогрес довантаження й кнопка повтору в рядку сталої висоти.

    Поки дані вантажаться, рядок не змінює висоти, тож лічильники й таблиця під ним не
    стрибають. Коли все довантажено і повторювати нічого, рядок зникає: скільки форм і
    які в них статуси, кажуть лічильники над таблицею.
    """
    if finished >= total and not retryable_ids:
        # st.empty() тримає місце рядка в дереві сторінки, тож таблиця під ним
        # не будується наново і прокрутка лишається.
        st.empty()
        return
    with st.container(height=LOADING_STATUS_HEIGHT_PX, border=False):
        progress_col, retry_col = st.columns([3, 2], vertical_alignment="center")
        if details_loaded < total:
            progress_col.progress(
                details_loaded / total, text=f"Підвантажую деталі: {details_loaded}/{total}"
            )
        elif finished < total:
            # Статуси вже є; кількість відповідей Google віддає обмежено (180 запитів
            # за хвилину), тож решта рядків чекає повтору.
            progress_col.progress(
                finished / total,
                text=f"Відповіді: {finished}/{total} — решта за хвилину-дві (ліміт Google)",
            )
        if retryable_ids and retry_col.button(
            f"Повторити проблемні рядки ({len(retryable_ids)})",
            key="catalog_retry_failed_rows",
            help="Повторно завантажити рядки зі статусами timeout, api_error або rate_limited.",
        ):
            _clear_enrichment_state_for(retryable_ids)
            st.rerun()


def _remember_result(result: CatalogEnrichmentResult, now: float) -> None:
    """Записати результат рядка й, якщо помилка тимчасова, запланувати повтор."""
    form_id = result.form_id
    enrichments = st.session_state["form_enrichments"]
    retries = st.session_state["form_retries"]
    st.session_state["form_data_status"][form_id] = result.status
    if result.fetched_at:
        st.session_state["form_data_fetched_at"][form_id] = result.fetched_at
    # Деталі, отримані раніше, лишаються, якщо невдалим був лише повтор.
    if result.summary is not None or form_id not in enrichments:
        enrichments[form_id] = result.summary
    if result.response_stats is not None:
        st.session_state["form_response_stats"][form_id] = result.response_stats
    retry = schedule_retry(result.status, retries.get(form_id), now)
    if retry is None:
        retries.pop(form_id, None)
    else:
        retries[form_id] = retry


def _enrich_next_batches() -> None:
    """Один крок довантаження: до CATALOG_ENRICH_PARALLEL_REQUESTS запитів паралельно."""
    enrichments = st.session_state["form_enrichments"]
    batches = next_form_batches(
        [f.id for f in forms_meta],
        done=enrichments.keys(),
        retries=st.session_state["form_retries"],
        have_summary={form_id for form_id, enr in enrichments.items() if enr is not None},
        now=time.monotonic(),
        batch_size=CATALOG_ENRICH_CHUNK_SIZE,
        max_batches=CATALOG_ENRICH_PARALLEL_REQUESTS,
    )
    if not batches:
        return
    data = google_data_client()  # створюється тут: потоки не читають session_state
    with ThreadPoolExecutor(max_workers=len(batches)) as pool:
        futures = [
            pool.submit(
                data.enrich_catalog_forms,
                batch.form_ids,
                include_summary=batch.include_summary,
                include_stats=True,
            )
            for batch in batches
        ]
    now = time.monotonic()
    failure: Exception | None = None
    for batch, future in zip(batches, futures, strict=True):
        try:
            by_id = {result.form_id: result for result in future.result()}
        except Exception as exc:  # noqa: BLE001
            failure, by_id = exc, {}
        # Без тосту на кожен рядок: проблемні рядки видно у «Стан даних» і на кнопці повтору.
        for form_id in batch.form_ids:
            _remember_result(
                by_id.get(form_id) or CatalogEnrichmentResult(form_id=form_id, status="api_error"),
                now,
            )
    if failure is not None:
        st.toast(f"⚠️ Не вдалося дозавантажити каталог: {failure}", icon="⚠️")


def _render_table_with_enrichment(*, in_fragment: bool) -> None:
    """One enrichment step plus table render."""
    enrichments = st.session_state["form_enrichments"]
    stats = st.session_state["form_response_stats"]
    statuses = st.session_state["form_data_status"]
    fetched_at = st.session_state["form_data_fetched_at"]
    retries = st.session_state["form_retries"]

    # Запуск, що змінив поточну форму (клік у таблиці чи вибір зверху), лише перемальовує
    # сторінку: порцію з Google візьме наступний тік, а вибір не чекає на неї 2-5 с.
    current_form = st.session_state.get(FORM_KEY)
    form_changed = st.session_state.get(TABLE_DRAWN_FORM_KEY, current_form) != current_form
    st.session_state[TABLE_DRAWN_FORM_KEY] = current_form
    if not (form_changed or st.session_state.pop(SKIP_LOADING_STEP_KEY, False)):
        _enrich_next_batches()

    finished = sum(1 for f in forms_meta if f.id in enrichments and f.id not in retries)
    details_loaded = sum(
        1
        for f in forms_meta
        if enrichments.get(f.id) is not None or (f.id in enrichments and f.id not in retries)
    )
    _render_loading_status(
        details_loaded,
        finished,
        len(forms_meta),
        _retryable_enrichment_ids(forms_meta, statuses, retries),
    )

    df = _build_dataframe(forms_meta, enrichments, stats, statuses, fetched_at, retries)
    _render_catalog_metrics(df)
    filtered = _apply_filters(df, filter_values)
    selection_source = filtered.reset_index(drop=True)
    display = _table_display(selection_source)

    form_ids = list(selection_source["FormID"])
    st.session_state[TABLE_FORM_IDS_KEY] = form_ids
    current_form_id = st.session_state.get(FORM_KEY)
    table_key = _table_key(filter_values)
    _mark_current_form(table_key, form_ids, current_form_id)
    st.dataframe(
        display,
        # Сталий key: інакше Streamlit виводить ідентичність таблиці з даних і на кожному
        # кроці довантаження створює її наново, скидаючи прокрутку, сортування й вибір.
        key=table_key,
        hide_index=True,
        width="stretch",
        height=_table_height(len(display)),
        on_select=functools.partial(_pick_form_from_table, table_key),
        selection_mode="single-row",
        # Нова таблиця (перша або з іншими фільтрами) позначає поточну форму.
        selection_default=(
            {"selection": {"rows": [form_ids.index(current_form_id)]}}
            if current_form_id in form_ids
            else None
        ),
        column_config=_table_column_config(
            hide_owner=filter_values["ownership"] == OWNERSHIP_MINE,
            show_data_status=not set(display["DataStatus"]).issubset(DATA_STATUS_QUIET),
        ),
    )
    # Клік у фрагменті перезапускає лише фрагмент, а вибрану форму показує панель над ним.
    if st.session_state.pop(TABLE_PICKED_KEY, False) and in_fragment:
        st.session_state[SKIP_LOADING_STEP_KEY] = True
        st.rerun()


def _has_pending_forms() -> bool:
    return is_loading(
        [f.id for f in forms_meta],
        done=st.session_state["form_enrichments"].keys(),
        retries=st.session_state["form_retries"],
    )


@st.fragment(run_every=ENRICHMENT_TICK_SECONDS)
def _table_with_enrichment_fragment() -> None:
    _render_table_with_enrichment(in_fragment=True)
    if not _has_pending_forms():
        # Усе довантажено: повний перезапуск малює сторінку вже без таймера фрагмента,
        # який інакше й далі перемальовував би таблицю кожні ENRICHMENT_TICK_SECONDS.
        st.rerun()


if _has_pending_forms():
    _table_with_enrichment_fragment()
else:
    # Фрагмент малює свій вміст у власному контейнері. Той самий контейнер тут тримає
    # таблицю на тому ж місці сторінки, коли довантаження завершується, — інакше браузер
    # будує її наново й скидає прокрутку.
    with st.container():
        _render_table_with_enrichment(in_fragment=False)
