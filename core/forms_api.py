"""Робота з Google Forms і Drive API.

Drive API використовуємо лише для одного — переліку Forms користувача
(Forms API не має методу list, треба фільтрувати у Drive за mimeType).
Forms API дає структуру форми: питання, типи, варіанти.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

FORM_MIME_TYPE = "application/vnd.google-apps.form"
DEFAULT_FORMS_PAGE_SIZE = 50

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
    повідомлення замість сирого traceback.
    """


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
        resp = service.files().list(
            q=f"mimeType='{FORM_MIME_TYPE}' and trashed=false",
            fields="files(id,name,modifiedTime)",
            pageSize=page_size,
            orderBy="modifiedTime desc",
        ).execute()
    except HttpError as exc:
        raise FormsApiError(
            f"Не вдалося отримати список форм з Drive: {exc.reason or exc}"
        ) from exc
    return resp.get("files", [])


def get_form_structure(creds: Credentials, form_id: str) -> dict[str, Any]:
    """Завантажити повну структуру форми через Forms API.

    Raises:
        FormsApiError: 403 (нема forms.body.readonly), 404 (форма видалена
            або недоступна), інші HTTP-помилки.
    """
    service = build("forms", "v1", credentials=creds, cache_discovery=False)
    try:
        return service.forms().get(formId=form_id).execute()
    except HttpError as exc:
        raise FormsApiError(
            f"Не вдалося завантажити форму {form_id}: {exc.reason or exc}"
        ) from exc


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
        q = _extract_question(item)
        if q is not None:
            questions.append(q)
    return questions


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
        normalized = "CHECKBOX" if qtype == "CHECKBOX" else "MULTIPLE_CHOICE"
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
