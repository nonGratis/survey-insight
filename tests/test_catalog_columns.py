"""Стовпці таблиці «Каталогу»: справжні web і API, фейковий лише Google."""

from __future__ import annotations

import json

import pytest

from api.google_data_cache import clear_api_cache
from tests.test_e2e_web_api import _app_with, _signed_in_web, _web_talking_to
from tests.test_saas_api import (
    _FakeGoogleFormsClient,
    _seed_google_grant,
    _seed_user_session,
    _test_container,
)


@pytest.fixture(autouse=True)
def _production_web(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("APP_BASE_URL", "https://app.example.com")
    clear_api_cache()


def _catalog_column_config() -> dict:
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    _, api_app = _app_with(container, _FakeGoogleFormsClient())

    with _web_talking_to(api_app):
        at = _signed_in_web(session_id)
        at.run()

    assert not at.exception, [e.value for e in at.exception]
    return json.loads(at.dataframe[0].proto.columns)


def test_the_data_age_column_says_when_the_data_was_fetched() -> None:
    column = _catalog_column_config()["UpdatedAgo"]

    # «Оновлено» read as "the form was edited"; that is what «Змінено» shows.
    assert column["label"] == "Дані отримано"
    assert "Змінено" in column["help"]
