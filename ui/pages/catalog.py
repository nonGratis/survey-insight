"""Каталог — табличний огляд усіх Google Forms, до яких є доступ.

Tier 1: один Drive API виклик → миттєва таблиця з базовими метаданими.
Tier 2: forms.get() для кожної форми → title, опис, секції, питання,
        linkedSheetId, accepting.
Tier 3: Forms API responses.list → кількість відповідей, перша й остання.

Tier 2/3 вантажить один фоновий потік: один запит до API, що віддає кожну форму, щойно
вона готова, і сам чекає на ліміт Google (core.catalog_stream, ui.catalog_load). Сторінка
лише малює те, що вже прийшло: таблиця з'являється одразу з Tier 1, а фрагмент раз на
LOAD_TICK_SECONDS перемальовує її, не чекаючи на Google, доки завантаження не скінчиться.
"""

from __future__ import annotations

import functools
import hashlib
import math
from datetime import UTC, date, datetime

import pandas as pd
import streamlit as st

from core.forms_catalog import (
    FormDriveMeta,
    FormEnrichment,
    ResponseStats,
)
from core.logger import get_logger
from ui.api_boundary import handle_api_errors
from ui.catalog_load import CatalogLoad, CatalogSnapshot
from ui.components.action_bar import render_action_bar
from ui.components.auth_widget import ensure_api_access
from ui.components.form_picker import FORM_KEY, clear_forms_cache
from ui.components.metric_bar import MetricItem, render_metric_bar
from ui.components.page_shell import render_empty_state, render_error_state, render_page_header
from ui.google_data import (
    clear_catalog_cache,
    google_data_client,
    list_catalog_forms,
)
from ui.saas_api import SaaSApiError
from ui.telemetry import page_run

log = get_logger(__name__)

# Як часто сторінка перемальовує таблицю, поки фоновий потік вантажить каталог.
LOAD_TICK_SECONDS = 2
ACTIVE_RECENT_DAYS = 7
TABLE_HEADER_HEIGHT_PX = 38
TABLE_ROW_HEIGHT_PX = 35
TABLE_MIN_HEIGHT_PX = 360
TABLE_MAX_HEIGHT_PX = 680
TABLE_KEY = "catalog_table"
# FormID рядків у порядку, в якому таблицю показано востаннє: вибір приходить номером рядка.
TABLE_FORM_IDS_KEY = "catalog_table_form_ids"
TABLE_PICKED_KEY = "catalog_table_picked_form"
# Завантаження каталогу цієї сесії (ui.catalog_load.CatalogLoad).
CATALOG_LOAD_KEY = "catalog_load"
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
DATA_STATUS_LABELS = {
    "ok": "Ок",
    "timeout": "Таймаут",
    "rate_limited": "Ліміт Google",
    "no_access": "Немає доступу",
    "deleted": "Видалена",
    "unsupported": "Не підтримується",
    "api_error": "Помилка API",
}

if not ensure_api_access():
    st.stop()


try:
    forms_meta = list_catalog_forms()
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
    if isinstance(previous := st.session_state.pop(CATALOG_LOAD_KEY, None), CatalogLoad):
        previous.stop()
    st.rerun()

if not forms_meta:
    render_empty_state(
        "Жодної Google Form не знайдено на цьому акаунті. "
        "Створи форму на forms.google.com і повернись."
    )
    st.stop()


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


def _data_status_label(form_id: str, snapshot: CatalogSnapshot) -> str:
    if form_id not in snapshot.finished:
        # Завантаження обірвалось (видно в рядку стану), а цю форму не встигло.
        return "Помилка" if snapshot.done else "Завантажується"
    return DATA_STATUS_LABELS.get(snapshot.statuses.get(form_id, "ok"), "Помилка")


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


def _build_dataframe(forms: list[FormDriveMeta], snapshot: CatalogSnapshot) -> pd.DataFrame:
    """Зібрати DataFrame, підставляючи placeholders для ще не завантажених рядків."""
    rows = []
    for f in forms:
        enr = snapshot.summaries.get(f.id)
        stat = snapshot.stats.get(f.id)
        activity, _ = _response_activity(stat)
        row = {
            "FormID": f.id,
            "FormName": f.name,
            # Статус відомий, щойно прийшли деталі форми, навіть якщо відповіді ще в черзі.
            "PublicationStatus": _publication_status(enr, loaded=f.id in snapshot.summaries),
            "DataStatus": _data_status_label(f.id, snapshot),
            "Title": enr.title if enr else "",
            "Owner": f.owner_email,
            "Questions": enr.questions_count if enr else None,
            "Sections": enr.sections_count if enr else None,
            "Total": stat.total if stat else None,
            "LastResponse": stat.last_response if stat else None,
            "Activity": activity,
            "UpdatedAgo": _updated_ago_label(snapshot.fetched_at.get(f.id)),
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


def _render_loading_status(snapshot: CatalogSnapshot, total: int) -> None:
    """Прогрес завантаження в рядку сталої висоти.

    Поки дані вантажаться, рядок не змінює висоти, тож лічильники й таблиця під ним не
    стрибають. Коли все завантажено, рядок зникає: скільки форм і які в них статуси, кажуть
    лічильники над таблицею. Якщо завантаження обірвалось, рядок про це каже.
    """
    if snapshot.done and snapshot.error is None:
        # st.empty() тримає місце рядка в дереві сторінки, тож таблиця під ним
        # не будується наново і прокрутка лишається.
        st.empty()
        return
    with st.container(height=LOADING_STATUS_HEIGHT_PX, border=False):
        details_loaded = len(snapshot.summaries)
        finished = len(snapshot.finished)
        if snapshot.error is not None:
            st.warning(
                "Не вдалося довантажити каталог. Натисни «Оновити» над таблицею.",
                icon=":material/sync_problem:",
            )
        elif details_loaded < total:
            st.progress(
                details_loaded / total, text=f"Підвантажую деталі: {details_loaded}/{total}"
            )
        else:
            st.progress(
                finished / total,
                text=_responses_progress_text(finished, total, snapshot.wait_seconds),
            )


def _responses_progress_text(finished: int, total: int, quota_wait: float | None) -> str:
    """Скільки форм уже мають кількість відповідей і коли чекати решту.

    Статуси вже є; кількість відповідей Google віддає обмежено (180 запитів за хвилину),
    тож решта рядків чекає свого місця в ліміті. ``quota_wait`` — секунди до запуску
    останнього з них (його назвав API), плюс тік таймера, на якому сторінка його покаже.
    """
    text = f"Відповіді: {finished}/{total}"
    if quota_wait is None:
        return text
    seconds = math.ceil(quota_wait + LOAD_TICK_SECONDS)
    wait = f"{seconds} с" if seconds < 90 else f"{math.ceil(seconds / 60)} хв"
    return f"{text} — решта приблизно за {wait} (ліміт Google)"


@handle_api_errors
def _surface_load_error(error: Exception) -> None:
    """Hand a sign-in or Google error from the background load to the API boundary.

    The load runs in its own thread, where the boundary does not act; here, in the page
    run, it signs out, offers to reconnect or stops as for any page call.
    """
    if isinstance(error, SaaSApiError):
        raise error


def _catalog_load() -> CatalogLoad:
    """This session's load of the catalog's details: started once, kept between runs.

    A load left alone while the user was on another page has stopped; a new one
    takes its place (the API's cache answers for what the old one got).
    """
    form_ids = [f.id for f in forms_meta]
    load = st.session_state.get(CATALOG_LOAD_KEY)
    if not isinstance(load, CatalogLoad) or load.form_ids != form_ids or load.stopped:
        # The client is made here: the load's thread does not read session state.
        load = CatalogLoad(form_ids, google_data_client().stream_catalog).start()
        st.session_state[CATALOG_LOAD_KEY] = load
    return load


def _render_catalog(snapshot: CatalogSnapshot, *, in_fragment: bool) -> None:
    """Loading status, counters and the table, from what the load has so far."""
    if snapshot.error is not None:
        _surface_load_error(snapshot.error)
    _render_loading_status(snapshot, len(forms_meta))

    df = _build_dataframe(forms_meta, snapshot)
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
        st.rerun()


catalog_load = _catalog_load()


@st.fragment(run_every=LOAD_TICK_SECONDS)
def _catalog_while_loading() -> None:
    with page_run("catalog", fragment=True):
        snapshot = catalog_load.snapshot()
        _render_catalog(snapshot, in_fragment=True)
        if snapshot.done:
            # Усе завантажено: повний перезапуск малює сторінку вже без таймера фрагмента,
            # який інакше й далі перемальовував би таблицю кожні LOAD_TICK_SECONDS.
            st.rerun()


if not catalog_load.snapshot().done:
    _catalog_while_loading()
else:
    # Фрагмент малює свій вміст у власному контейнері. Той самий контейнер тут тримає
    # таблицю на тому ж місці сторінки, коли завантаження завершується, — інакше браузер
    # будує її наново й скидає прокрутку.
    with st.container():
        _render_catalog(catalog_load.snapshot(), in_fragment=False)
