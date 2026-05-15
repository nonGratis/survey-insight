"""Google Sheets API: завантаження відповідей у pandas DataFrame.

Sheets, привʼязаний до Google Form, містить аркуш типу GRID із одним
header-рядком (питання форми + Timestamp) і одним рядком на кожну
відповідь. Імʼя аркуша Google локалізує під мову акаунта
("Form Responses 1" / "Відповіді форми 1"), тому ми завжди спочатку
читаємо metadata.
"""
from __future__ import annotations

import pandas as pd
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Sheets API "A:ZZ" покриває до 702 колонок — більш ніж достатньо для
# будь-якої реальної форми (максимум у Google Forms ~300 питань).
DEFAULT_COLUMN_RANGE = "A:ZZ"


class SheetsApiError(RuntimeError):
    """Доменна помилка Sheets API — для змістовного UI-повідомлення."""


def _find_response_sheet_name(service, sheet_id: str) -> str:
    """Повернути імʼя першого GRID-аркуша у привʼязаному Spreadsheet.

    Forms-linked spreadsheets завжди мають мінімум один GRID-аркуш із
    відповідями. Беремо перший такий — це робастно для українських,
    англійських та будь-яких інших локалізацій, а також для випадку,
    коли користувач додав другий tab вручну.
    """
    meta = service.spreadsheets().get(
        spreadsheetId=sheet_id,
        fields="sheets(properties(title,sheetType))",
    ).execute()
    for sheet in meta.get("sheets", []):
        props = sheet.get("properties", {})
        if props.get("sheetType", "GRID") == "GRID":
            return props["title"]
    raise RuntimeError(f"No GRID sheet found in spreadsheet {sheet_id}.")


def fetch_responses(creds: Credentials, sheet_id: str) -> pd.DataFrame:
    """Завантажити всі відповіді з привʼязаного Sheet у DataFrame.

    Args:
        creds: OAuth credentials зі scope spreadsheets.readonly.
        sheet_id: id Spreadsheet (form.linkedSheetId).

    Returns:
        DataFrame, де колонки — заголовки з першого рядка аркуша
        (зазвичай "Timestamp" + назви питань форми). Порожній DataFrame,
        якщо відповідей ще немає.

    Raises:
        SheetsApiError: на 403/404 та інші HTTP-помилки Sheets API.
    """
    service = build("sheets", "v4", credentials=creds, cache_discovery=False)
    try:
        sheet_name = _find_response_sheet_name(service, sheet_id)
        range_name = f"'{sheet_name}'!{DEFAULT_COLUMN_RANGE}"
        resp = service.spreadsheets().values().get(
            spreadsheetId=sheet_id, range=range_name
        ).execute()
    except HttpError as exc:
        raise SheetsApiError(
            f"Не вдалося прочитати Sheet {sheet_id}: {exc.reason or exc}"
        ) from exc

    values = resp.get("values", [])
    if not values:
        return pd.DataFrame()

    headers, *rows = values
    # Sheets API обрізає trailing порожні cells. Нормалізуємо кожен рядок
    # точно до довжини headers: padding порожніми або truncate, якщо
    # користувач випадково додав колонки поза header-рядком.
    width = len(headers)
    rows = [(row + [""] * max(0, width - len(row)))[:width] for row in rows]
    return pd.DataFrame(rows, columns=headers)
