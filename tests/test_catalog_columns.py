"""Стовпці таблиці «Каталогу»: справжні web і API, фейковий лише Google."""

from __future__ import annotations

import json

import pytest
from google.oauth2.credentials import Credentials

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


class _TwoForms(_FakeGoogleFormsClient):
    def list_forms(self, creds: Credentials) -> list[dict]:
        base = super().list_forms(creds)[0]
        return [
            {**base, "id": "form_a", "name": "Poll A"},
            {**base, "id": "form_b", "name": "Poll B"},
        ]

    def get_form_summary(self, creds: Credentials, form_id: str) -> dict:
        return super().get_form_summary(creds, "form_1")

    def get_response_stats(self, creds: Credentials, form_id: str) -> dict:
        return super().get_response_stats(creds, "form_1")


def _catalog_table(*, ownership: str | None = None):  # type: ignore[no-untyped-def]
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    _, api_app = _app_with(container, _TwoForms())

    with _web_talking_to(api_app):
        at = _signed_in_web(session_id)
        at.run()
        if ownership:
            at.segmented_control(key="catalog_ownership").set_value(ownership).run()

    assert not at.exception, [e.value for e in at.exception]
    table = at.dataframe[0]
    return table.value, json.loads(table.proto.columns)


def _shown(values, config: dict) -> list[str]:  # type: ignore[no-untyped-def]
    return [column for column in values.columns if not config.get(column, {}).get("hidden")]


def test_the_table_shows_the_agreed_columns_in_order() -> None:
    values, config = _catalog_table()

    assert _shown(values, config) == [
        "FormName",
        "PublicationStatus",
        "Total",
        "LastResponse",
        "Activity",
        "Questions",
        "Owner",
        "Modified",
        "Created",
    ]
    assert config["Questions"]["label"] == "Запитань"
    # Duplicates of other columns are gone: «Приймає» is the status, «Днів без відповіді»
    # is the last response; the raw Sheet id says nothing to a person.
    assert not {"Accepting", "DaysNoResponse", "SheetID"} & set(values.columns)
    # Kept but hidden, one click away in the table toolbar; data status while all is fine.
    for column in ("Sections", "Title", "Description", "UpdatedAgo", "DataStatus"):
        assert config[column]["hidden"] is True, column
    # The responder-facing title is not the internal one: the Drive file name is.
    assert config["Title"]["label"] == "Заголовок для респондентів"


def test_a_row_status_speaks_of_one_form() -> None:
    values, _ = _catalog_table()

    assert set(values["PublicationStatus"]) == {"Відкрита"}


def test_the_owner_column_steps_aside_for_my_forms_only() -> None:
    _, everyone = _catalog_table(ownership="Усі")
    _, mine = _catalog_table(ownership="Мої")

    assert not everyone["Owner"].get("hidden")
    assert mine["Owner"]["hidden"] is True


def test_a_field_the_api_adds_later_does_not_break_the_catalog() -> None:
    # During a deploy the new API answers the old web for a few seconds, and back.
    from ui.google_data import _drive_meta_from_payload

    meta = _drive_meta_from_payload(
        {
            "id": "form_1",
            "name": "Poll",
            "owner_email": "owner@example.com",
            "owner_name": "Owner",
            "created_time": "2026-06-01T10:00:00Z",
            "modified_time": "2026-06-02T10:00:00Z",
            "edit_url": "https://docs.google.com/forms/d/form_1/edit",
            "field_from_a_newer_api": "ignored",
        }
    )

    assert meta.id == "form_1"
