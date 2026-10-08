"""Робота з Google Forms і Drive API.

Drive API використовуємо лише для одного — переліку Forms користувача
(Forms API не має методу list, треба фільтрувати у Drive за mimeType).
Forms API дає структуру форми: питання, типи, варіанти.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from google.oauth2.credentials import Credentials
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import build_http

from core.google_errors import google_error_reason
from core.logger import get_logger, log_call

log = get_logger(__name__)

FORM_MIME_TYPE = "application/vnd.google-apps.form"
RESPONSE_TIMESTAMPS_FIELDS = "responses(createTime),nextPageToken"
# How long a catalog call (a form's details or one page of its response times) may go
# without an answer. Google answers these in ~0.4 s (p99 ~2 s); a call it holds for tens
# of seconds is dropped and asked again (core.catalog_stream) instead of holding the
# whole catalog. The heavier calls (whole form, full responses) keep the client default.
CATALOG_CALL_TIMEOUT_SECONDS = float(os.getenv("SI_GOOGLE_CALL_TIMEOUT_SECONDS", "10"))
DEFAULT_FORMS_PAGE_SIZE = 50
# This thread's Forms client and the credentials it was built for (forms_service).
_clients = threading.local()

QuestionType = Literal[
    "MULTIPLE_CHOICE",
    "CHECKBOX",
    "SHORT_ANSWER",
    "LINEAR_SCALE",
    "DATE",
    "TIME",
    "UNKNOWN",
]


class FormsApiError(RuntimeError):
    """Доменна помилка під будь-який збій Forms/Drive API.

    Перехоплює googleapiclient.errors.HttpError і дає UI-шару змістовне
    повідомлення замість сирого traceback. Зберігає HTTP-статус, щоб
    caller міг розрізняти "очікувані" коди
    (403 shared form без access, 404 видалена форма) від справжніх збоїв.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        reason: str | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.reason = reason


@dataclass(frozen=True)
class Question:
    """Нормалізований опис одного питання форми."""

    id: str
    title: str
    type: QuestionType
    options: list[str]  # для CHOICE/CHECKBOX — список варіантів, інакше []


def list_user_forms(
    creds: Credentials, page_size: int = DEFAULT_FORMS_PAGE_SIZE
) -> list[dict[str, Any]]:
    """Повернути список Google Forms користувача через Drive API.

    Args:
        creds: OAuth credentials з drive.metadata.readonly scope.
        page_size: ліміт на одну сторінку (Google API max 1000).

    Returns:
        Список dict: [{id, name, modifiedTime}, ...] відсортований за
        modifiedTime descending.

    Raises:
        FormsApiError: при будь-якій HTTP-помилці від Drive API
            (403 — нема scope, 401 — токен невалідний, тощо).
    """
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    try:
        with log_call(
            "api_call_ok",
            target="drive.files.list",
            scope="forms_only",
            page_size=page_size,
            logger=log,
        ):
            resp = (
                service.files()
                .list(
                    q=f"mimeType='{FORM_MIME_TYPE}' and trashed=false",
                    fields="files(id,name,modifiedTime)",
                    pageSize=page_size,
                    orderBy="modifiedTime desc",
                )
                .execute()
            )
    except HttpError as exc:
        raise FormsApiError(
            f"Не вдалося отримати список форм з Drive: {exc.reason or exc}",
            status=exc.resp.status,
            reason=google_error_reason(exc),
        ) from exc
    return resp.get("files", [])


def forms_service(creds: Credentials, *, timeout: float | None = None) -> Any:
    """This thread's Forms API client for these credentials (and socket ``timeout``).

    A client keeps its HTTPS connection open between calls, so the catalog's workers reuse
    one each instead of opening a connection per form. Measured locally, a fresh client per
    call took about 1 s a call with 10 calls in parallel and 2 s with 30, against 0.6 s
    reused, with twice the memory. httplib2 connections are not thread-safe, hence one
    client per thread; and a client serves only the very credentials object it was built
    for (one user's request).
    """
    cached = getattr(_clients, "forms", None)
    if cached is None or cached[0] is not creds or cached[1] != timeout:
        # What build(credentials=...) does, with the timeout set: build_http() keeps the
        # library's default (60 s) when none is given.
        http = build_http()
        if timeout is not None:
            http.timeout = timeout
        service = build("forms", "v1", http=AuthorizedHttp(creds, http=http), cache_discovery=False)
        cached = (creds, timeout, service)
        _clients.forms = cached
    return cached[2]


def get_form_structure(creds: Credentials, form_id: str) -> dict[str, Any]:
    """Завантажити повну структуру форми через Forms API.

    Raises:
        FormsApiError: 403 (нема forms.body.readonly), 404 (форма видалена
            або недоступна), інші HTTP-помилки.
    """
    service = forms_service(creds)
    try:
        with log_call("api_call_ok", target="forms.forms.get", form_id=form_id, logger=log):
            return service.forms().get(formId=form_id).execute()
    except HttpError as exc:
        raise FormsApiError(
            f"Не вдалося завантажити форму {form_id}: {exc.reason or exc}",
            status=exc.resp.status,
            reason=google_error_reason(exc),
        ) from exc


def list_response_timestamps(creds: Credentials, form_id: str) -> list[datetime]:
    """Завантажити всі timestamps відповідей форми через Forms API.

    Використовує `forms.responses.list` з pagination через `nextPageToken`.
    Повертає список naive UTC datetime, відсортований за зростанням.
    Працює навіть для форм без linked Sheet — канонічне джерело
    timestamps це поле `createTime`, яке Google виставляє при submit.

    Scope: достатньо `https://www.googleapis.com/auth/forms.responses.readonly`
    (read-only для responses). Цей scope уже в `core.auth.API_SCOPES`.

    Raises:
        FormsApiError: 403 (нема scope), 404 (форма видалена), інші
            HTTP-помилки Forms API.
    """
    service = forms_service(creds, timeout=CATALOG_CALL_TIMEOUT_SECONDS)
    timestamps: list[datetime] = []
    page_token: str | None = None
    try:
        while True:
            with log_call(
                "api_call_ok",
                target="forms.forms.responses.list",
                form_id=form_id,
                logger=log,
            ):
                resp = (
                    service.forms()
                    .responses()
                    .list(
                        formId=form_id,
                        pageToken=page_token,
                        fields=RESPONSE_TIMESTAMPS_FIELDS,
                    )
                    .execute()
                )
            for r in resp.get("responses", []):
                ct = r.get("createTime")
                if not ct:
                    continue
                # RFC 3339 ("2026-05-15T14:30:45.123Z") → naive UTC datetime
                # (повсюди у проєкті використовуємо naive UTC: core.store,
                # core.timeline). `astimezone(UTC).replace(tzinfo=None)` нормалізує
                # незалежно від суфіксу (Z, +00:00, +03:00 тощо).
                timestamps.append(
                    datetime.fromisoformat(ct.replace("Z", "+00:00"))
                    .astimezone(UTC)
                    .replace(tzinfo=None)
                )
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    except HttpError as exc:
        raise FormsApiError(
            f"Не вдалося отримати відповіді форми {form_id}: {exc.reason or exc}",
            status=exc.resp.status,
            reason=google_error_reason(exc),
        ) from exc
    timestamps.sort()
    return timestamps


def list_form_responses(creds: Credentials, form_id: str) -> list[dict[str, Any]]:
    """Завантажити повні відповіді форми (createTime + answers) через Forms API.

    Той самий `forms.responses.list`, що й timestamps, але повертає сирі
    об'єкти відповідей із полем `answers` (значення по кожному questionId).
    Потрібно для per-question аналізу. Sheet НЕ потрібен.

    Raises:
        FormsApiError: 403 (нема scope), 404 (форма видалена), інші HTTP-помилки.
    """
    service = forms_service(creds)
    responses: list[dict[str, Any]] = []
    page_token: str | None = None
    try:
        while True:
            with log_call(
                "api_call_ok",
                target="forms.forms.responses.list",
                form_id=form_id,
                logger=log,
            ):
                resp = (
                    service.forms().responses().list(formId=form_id, pageToken=page_token).execute()
                )
            responses.extend(resp.get("responses", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    except HttpError as exc:
        raise FormsApiError(
            f"Не вдалося отримати відповіді форми {form_id}: {exc.reason or exc}",
            status=exc.resp.status,
            reason=google_error_reason(exc),
        ) from exc
    return responses


def get_linked_sheet_id(form: dict[str, Any]) -> str | None:
    """Повернути id привʼязаного Google Sheet або None.

    Forms API заповнює top-level поле `linkedSheetId` лише коли власник
    форми створив link до Sheet через Responses → Link to Sheets.
    """
    return form.get("linkedSheetId")


def parse_question_types(form: dict[str, Any]) -> list[Question]:
    """Витягти питання та класифікувати типи.

    Forms API повертає `items[].questionItem.question.<typeQuestion>`,
    де ключ під questionItem — це і є тип. Маpимо у наші константи.
    """
    questions: list[Question] = []
    for item in form.get("items", []):
        questions.extend(_extract_questions(item))
    return questions


def _extract_questions(item: dict[str, Any]) -> list[Question]:
    """Розпарсити item у 0..N питань, включно з matrix/grid question groups."""
    single = _extract_question(item)
    if single is not None:
        return [single]

    group = item.get("questionGroupItem")
    if not group:
        return []

    columns = group.get("grid", {}).get("columns", {})
    if not columns:
        return []

    qtype = columns.get("type", "RADIO")
    normalized: QuestionType = "CHECKBOX" if qtype == "CHECKBOX" else "MULTIPLE_CHOICE"
    options = [opt.get("value", "") for opt in columns.get("options", [])]
    title = item.get("title", "")
    out: list[Question] = []
    for sub in group.get("questions", []):
        qid = sub.get("questionId", "")
        if not qid:
            continue
        row_title = sub.get("rowQuestion", {}).get("title", "")
        label = f"{title} — {row_title}".strip(" —")
        out.append(Question(id=qid, title=label, type=normalized, options=options))
    return out


def _extract_question(item: dict[str, Any]) -> Question | None:
    """Розпарсити одну item-структуру у Question, або None якщо не питання."""
    question_item = item.get("questionItem")
    if not question_item:
        return None  # секція / зображення / відео — не питання

    question = question_item.get("question", {})
    qid = question.get("questionId", "")
    title = item.get("title", "")

    if "choiceQuestion" in question:
        choice = question["choiceQuestion"]
        qtype = choice.get("type", "RADIO")
        normalized: QuestionType = "CHECKBOX" if qtype == "CHECKBOX" else "MULTIPLE_CHOICE"
        options = [opt.get("value", "") for opt in choice.get("options", [])]
        return Question(id=qid, title=title, type=normalized, options=options)

    if "textQuestion" in question:
        return Question(id=qid, title=title, type="SHORT_ANSWER", options=[])

    if "scaleQuestion" in question:
        return Question(id=qid, title=title, type="LINEAR_SCALE", options=[])

    if "dateQuestion" in question:
        return Question(id=qid, title=title, type="DATE", options=[])

    if "timeQuestion" in question:
        return Question(id=qid, title=title, type="TIME", options=[])

    return Question(id=qid, title=title, type="UNKNOWN", options=[])
