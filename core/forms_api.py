"""Робота з Google Forms і Drive API.

Drive API використовуємо лише для одного — переліку Forms користувача
(Forms API не має методу list, треба фільтрувати у Drive за mimeType).
Forms API дає структуру форми: питання, типи, варіанти.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

FORM_MIME_TYPE = "application/vnd.google-apps.form"


@dataclass(frozen=True)
class Question:
    """Нормалізований опис одного питання форми."""

    id: str
    title: str
    type: str  # MULTIPLE_CHOICE | CHECKBOX | SHORT_ANSWER | LINEAR_SCALE | DATE | TIME | UNKNOWN
    options: list[str]  # для CHOICE/CHECKBOX — список варіантів, інакше []


def list_user_forms(creds: Credentials, page_size: int = 50) -> list[dict]:
    """Повернути список Google Forms користувача через Drive API.

    Args:
        creds: OAuth credentials з drive.metadata.readonly scope.
        page_size: ліміт на одну сторінку (Google API max 1000).

    Returns:
        Список dict: [{id, name, modifiedTime}, ...] відсортований за
        modifiedTime descending.
    """
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    resp = service.files().list(
        q=f"mimeType='{FORM_MIME_TYPE}' and trashed=false",
        fields="files(id,name,modifiedTime)",
        pageSize=page_size,
        orderBy="modifiedTime desc",
    ).execute()
    return resp.get("files", [])


def get_form_structure(creds: Credentials, form_id: str) -> dict[str, Any]:
    """Завантажити повну структуру форми через Forms API."""
    service = build("forms", "v1", credentials=creds, cache_discovery=False)
    return service.forms().get(formId=form_id).execute()


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
