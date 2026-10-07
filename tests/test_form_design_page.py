"""Сторінка «Дизайн форми»: справжні web і API, фейковий лише Google."""

from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest
from google.oauth2.credentials import Credentials
from streamlit import dataframe_util

from api.google_data_cache import clear_api_cache
from tests.test_e2e_web_api import _app_with, _signed_in_web, _web_talking_to
from tests.test_saas_api import (
    _FakeGoogleFormsClient,
    _seed_google_grant,
    _seed_user_session,
    _test_container,
)

PAGE = "ui/pages/form_design.py"


class _ChoiceAndOpenQuestions(_FakeGoogleFormsClient):
    def get_form_structure(self, creds: Credentials, form_id: str) -> dict:
        def question(qid: str, title: str, body: dict) -> dict:
            return {"title": title, "questionItem": {"question": {"questionId": qid, **body}}}

        return {
            "formId": form_id,
            "info": {"title": "Admissions poll"},
            "items": [
                question(
                    "q1",
                    "Стать?",
                    {
                        "choiceQuestion": {
                            "type": "RADIO",
                            "options": [{"value": "Ч"}, {"value": "Ж"}],
                        }
                    },
                ),
                question("q2", "Коментар", {"textQuestion": {"paragraph": True}}),
            ],
        }


@pytest.fixture(autouse=True)
def _production_web(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("APP_BASE_URL", "https://app.example.com")
    clear_api_cache()


def test_questions_table_reaches_the_browser_without_arrow_fixes() -> None:
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    _, api_app = _app_with(container, _ChoiceAndOpenQuestions())
    # Streamlit falls back to this when a column mixes types, and logs a traceback each time.
    arrow_fixes = mock.patch.object(
        dataframe_util,
        "fix_arrow_incompatible_column_types",
        wraps=dataframe_util.fix_arrow_incompatible_column_types,
    )

    with _web_talking_to(api_app), arrow_fixes as fixed:
        at = _signed_in_web(session_id)
        at.run()
        at.switch_page(PAGE).run()

    assert not at.exception, [e.value for e in at.exception]
    table = at.dataframe[0].value
    assert list(table["Запитання"]) == ["Стать?", "Коментар"]
    assert fixed.call_count == 0
    # A number column sorts as numbers; an open question has no options to count.
    assert pd.api.types.is_numeric_dtype(table["Опцій"])
    assert table["Опцій"].iloc[0] == 2
    assert pd.isna(table["Опцій"].iloc[1])
