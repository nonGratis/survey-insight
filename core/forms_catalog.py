"""Каталог Google Forms — композитний слой над Drive, Forms, Sheets API.

Tier 1 (`FormDriveMeta`) — миттєвий список форм користувача з Drive API.
Tier 2 (`FormEnrichment`) — структурні деталі форми з Forms API.
Tier 3 (`ResponseStats`) — статистика відповідей із привʼязаного Sheet.

UI шар рендерить Tier 1 одразу і прогресивно дозаповнює Tier 2/3
через background fragment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from core.forms_api import FORM_MIME_TYPE, FormsApiError
from core.logger import get_logger, log_call
from core.sheets_api import SheetsApiError, find_response_sheet_name

log = get_logger(__name__)


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return min(max(value, minimum), maximum)


DRIVE_FIELDS = (
    "nextPageToken,"
    "files(id,name,createdTime,modifiedTime,owners(emailAddress,displayName),"
    "webViewLink)"
)
DRIVE_PAGE_SIZE = _env_int("SI_DRIVE_FORMS_PAGE_SIZE", 100, minimum=1, maximum=1000)
DRIVE_MAX_FORMS = _env_int("SI_DRIVE_FORMS_MAX", 300, minimum=1, maximum=5000)
FORM_EDIT_URL_TEMPLATE = "https://docs.google.com/forms/d/{form_id}/edit"
CATALOG_SUMMARY_FIELDS = (
    "info(title,description),"
    "items(pageBreakItem,questionItem/question/questionId),"
    "linkedSheetId,"
    "publishSettings/publishState(isPublished,isAcceptingResponses)"
)


@dataclass(frozen=True)
class FormDriveMeta:
    """Tier 1 — все що дає Drive API за один list call."""

    id: str
    name: str
    owner_email: str
    owner_name: str
    created_time: str
    modified_time: str
    edit_url: str


@dataclass(frozen=True)
class FormEnrichment:
    """Tier 2 — Forms API forms.get(): структурні дані форми."""

    title: str
    description: str
    sections_count: int
    questions_count: int
    linked_sheet_id: str | None
    is_published: bool | None  # publishSettings може бути відсутнім у legacy-форм
    accepting_responses: bool | None  # publishSettings може бути відсутнім


@dataclass(frozen=True)
class ResponseStats:
    """Tier 3 — стат-метрики, обчислені з колонки A privʼязаного Sheet."""

    total: int
    first_response: str | None
    second_response: str | None
    last_response: str | None


def list_forms_with_drive_meta(creds: Credentials) -> list[FormDriveMeta]:
    """Список усіх Google Forms користувача (own + shared) з Drive.

    Drive за замовчуванням повертає файли в My Drive + ті, що shared з
    користувачем — саме що нам потрібно. Без owner-фільтра.

    Raises:
        FormsApiError: на HttpError від Drive API.
    """
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    items: list[FormDriveMeta] = []
    page_token: str | None = None
    page_idx = 0
    try:
        while len(items) < DRIVE_MAX_FORMS:
            page_idx += 1
            page_size = min(DRIVE_PAGE_SIZE, DRIVE_MAX_FORMS - len(items))
            with log_call(
                "api_call_ok",
                target="drive.files.list",
                scope="catalog",
                page=page_idx,
                logger=log,
            ):
                resp = (
                    service.files()
                    .list(
                        q=f"mimeType='{FORM_MIME_TYPE}' and trashed=false",
                        fields=DRIVE_FIELDS,
                        pageSize=page_size,
                        pageToken=page_token,
                        orderBy="modifiedTime desc",
                    )
                    .execute()
                )
            for raw in resp.get("files", []):
                items.append(_parse_drive_file(raw))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        if page_token:
            log.warning(
                "forms_catalog_drive_limit_reached",
                extra={"max_forms": DRIVE_MAX_FORMS, "page_size": DRIVE_PAGE_SIZE},
            )
    except HttpError as exc:
        raise FormsApiError(
            f"Не вдалося отримати каталог форм з Drive: {exc.reason or exc}",
            status=exc.resp.status,
        ) from exc
    return items


def _parse_drive_file(raw: dict[str, Any]) -> FormDriveMeta:
    """Drive file dict → FormDriveMeta. Захищено від відсутніх полів."""
    form_id = raw["id"]
    owners = raw.get("owners") or []
    first_owner = owners[0] if owners else {}
    return FormDriveMeta(
        id=form_id,
        name=raw.get("name", "—"),
        owner_email=first_owner.get("emailAddress", "—"),
        owner_name=first_owner.get("displayName", "—"),
        created_time=raw.get("createdTime", ""),
        modified_time=raw.get("modifiedTime", ""),
        edit_url=FORM_EDIT_URL_TEMPLATE.format(form_id=form_id),
    )


def enrich_form(creds: Credentials, form_id: str) -> FormEnrichment:
    """Forms API forms.get() → структурні деталі форми.

    Раises:
        FormsApiError: на 403 (нема forms.body.readonly), 404 (видалено).
    """
    service = build("forms", "v1", credentials=creds, cache_discovery=False)
    try:
        with log_call(
            "api_call_ok",
            target="forms.forms.get",
            scope="enrich",
            form_id=form_id,
            logger=log,
        ):
            form = service.forms().get(formId=form_id, fields=CATALOG_SUMMARY_FIELDS).execute()
    except HttpError as exc:
        raise FormsApiError(
            f"Не вдалося завантажити форму {form_id}: {exc.reason or exc}",
            status=exc.resp.status,
        ) from exc
    return _parse_form(form)


def _parse_form(form: dict[str, Any]) -> FormEnrichment:
    """Розкласти Forms API response у FormEnrichment."""
    info = form.get("info", {})
    items = form.get("items", [])
    sections = sum(1 for item in items if "pageBreakItem" in item)
    questions = sum(1 for item in items if "questionItem" in item)
    is_published, accepting_responses = _publish_state(form)
    return FormEnrichment(
        title=info.get("title", "—"),
        description=info.get("description", ""),
        sections_count=sections,
        questions_count=questions,
        linked_sheet_id=form.get("linkedSheetId"),
        is_published=is_published,
        accepting_responses=accepting_responses,
    )


def _publish_state(form: dict[str, Any]) -> tuple[bool | None, bool | None]:
    """(опублікована, приймає відповіді) з `publishSettings` форми.

    Legacy-форми не мають `publishSettings` зовсім, і Google не повідомляє їхній стан.
    В інших Google пропускає булеві поля зі значенням false, тож відсутнє поле — це false:
    закрита форма приходить лише з `isPublished`, неопублікована — з порожнім `publishState`.
    """
    settings = form.get("publishSettings")
    if settings is None:
        return None, None
    state = settings.get("publishState") or {}
    return bool(state.get("isPublished", False)), bool(state.get("isAcceptingResponses", False))


def fetch_response_stats(creds: Credentials, sheet_id: str) -> ResponseStats:
    """Статистика відповідей з колонки A привʼязаного Sheet.

    Перша колонка форм-Sheet'у — це Timestamp. Читаємо її повністю,
    обчислюємо total / first / second / last.

    Raises:
        SheetsApiError: на 403/404/інші HTTP-помилки Sheets API.
    """
    service = build("sheets", "v4", credentials=creds, cache_discovery=False)
    try:
        sheet_name = find_response_sheet_name(service, sheet_id)
        # Беремо тільки колонку A — мінімум payload, але достатньо для timestamps.
        range_name = f"'{sheet_name}'!A:A"
        with log_call(
            "api_call_ok",
            target="sheets.values.get",
            scope="response_stats_column_a",
            sheet_id=sheet_id,
            logger=log,
        ):
            resp = (
                service.spreadsheets()
                .values()
                .get(spreadsheetId=sheet_id, range=range_name)
                .execute()
            )
    except HttpError as exc:
        raise SheetsApiError(
            f"Не вдалося отримати timestamps з Sheet {sheet_id}: {exc.reason or exc}",
            status=exc.resp.status,
        ) from exc

    values = resp.get("values", [])
    # Перший рядок — заголовок ("Timestamp" або локалізована назва).
    timestamps = [row[0] for row in values[1:] if row]
    total = len(timestamps)
    first = timestamps[0] if total >= 1 else None
    second = timestamps[1] if total >= 2 else None
    last = timestamps[-1] if total >= 1 else None
    return ResponseStats(
        total=total,
        first_response=first,
        second_response=second,
        last_response=last,
    )
