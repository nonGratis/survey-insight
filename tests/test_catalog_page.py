"""Сторінка «Каталог»: справжні web і API, фейковий лише Google."""

from __future__ import annotations

import json

import pytest
from google.oauth2.credentials import Credentials

from api.google_data_cache import clear_api_cache
from api.google_quota import FormsQuotaGuards, RollingWindowGuard
from tests.test_e2e_web_api import _app_with, _signed_in_web, _web_talking_to
from tests.test_saas_api import (
    _FakeGoogleFormsClient,
    _seed_google_grant,
    _seed_user_session,
    _test_container,
)

# Більше, ніж сторінка довантажує за один прохід: частина рядків лишається в черзі.
FORM_COUNT = 70


class _ManyOpenForms(_FakeGoogleFormsClient):
    def list_forms(self, creds: Credentials) -> list[dict]:
        return [
            {
                "id": f"form_{index:02d}",
                "name": f"Poll {index:02d}",
                "owner_email": "owner@example.com",
                "owner_name": "Owner",
                "created_time": "2026-06-01T10:00:00Z",
                "modified_time": "2026-06-02T10:00:00Z",
                "edit_url": f"https://docs.google.com/forms/d/form_{index:02d}/edit",
            }
            for index in range(FORM_COUNT)
        ]

    def get_form_summary(self, creds: Credentials, form_id: str) -> dict:
        return {
            "title": form_id,
            "description": "",
            "sections_count": 1,
            "questions_count": 3,
            "linked_sheet_id": None,
            "is_published": True,
            "accepting_responses": True,
        }

    def get_response_stats(self, creds: Credentials, form_id: str) -> dict:
        return {"total": 0, "first_response": None, "second_response": None, "last_response": None}


@pytest.fixture(autouse=True)
def _production_web(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("APP_BASE_URL", "https://app.example.com")
    clear_api_cache()


def test_rows_still_loading_are_not_counted_as_unknown() -> None:
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    _, api_app = _app_with(container, _ManyOpenForms())

    with _web_talking_to(api_app):
        at = _signed_in_web(session_id)
        at.run()

    assert not at.exception, [e.value for e in at.exception]
    table = at.dataframe[0].value
    loading = table["DataStatus"] == "Завантажується"
    assert 0 < loading.sum() < FORM_COUNT
    assert set(table.loc[loading, "PublicationStatus"]) == {"Завантажується"}
    assert set(table.loc[~loading, "PublicationStatus"]) == {"Відкрита"}
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Невідомо"] == "0"


def test_table_keeps_its_identity_while_details_load() -> None:
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    _, api_app = _app_with(container, _ManyOpenForms())

    with _web_talking_to(api_app):
        at = _signed_in_web(session_id)
        at.run()
        before = at.dataframe[0]
        loaded_before = int((before.value["DataStatus"] != "Завантажується").sum())
        at.run()  # the next loading step brings more rows
        after = at.dataframe[0]

    assert not at.exception, [e.value for e in at.exception]
    assert int((after.value["DataStatus"] != "Завантажується").sum()) > loaded_before
    # Streamlit derives an unkeyed table's identity from its data, and a new identity
    # remounts the table in the browser: scroll, sorting and selection are lost.
    assert after.proto.id == before.proto.id
    assert before.key.startswith("catalog_table")


def test_changing_a_filter_starts_a_fresh_table_that_marks_the_current_form() -> None:
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    _, api_app = _app_with(container, _ManyOpenForms())

    with _web_talking_to(api_app):
        at = _signed_in_web(session_id)
        at.session_state["global_form_id"] = "form_05"
        at.run()
        before = at.dataframe[0]
        at.text_input(key="catalog_search").set_value("5").run()
        after = at.dataframe[0]

    assert not at.exception, [e.value for e in at.exception]
    # A row is marked by its position, and after filtering another form sits there.
    assert after.proto.id != before.proto.id
    assert json.loads(before.proto.selection_default)["selection"]["rows"] == [5]
    assert after.value["FormName"].iloc[0] == "Poll 05"
    assert json.loads(after.proto.selection_default)["selection"]["rows"] == [0]


def _position_of_table(node, path=()):
    """Index path of the table in the page tree, the way the browser places it."""
    for index, child in getattr(node, "children", {}).items():
        if getattr(child, "type", None) == "dataframe":
            return (*path, index)
        found = _position_of_table(child, (*path, index))
        if found:
            return found
    return None


def test_the_table_stays_in_place_when_loading_ends() -> None:
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    _, api_app = _app_with(container, _ManyOpenForms())

    with _web_talking_to(api_app):
        at = _signed_in_web(session_id)
        at.run()
        while_loading = (_position_of_table(at._tree), at.dataframe[0].proto.id)
        for _ in range(20):
            if "Завантажується" not in set(at.dataframe[0].value["DataStatus"]):
                break
            at.run()
        at.run()  # the page without the loading timer
        loaded = (_position_of_table(at._tree), at.dataframe[0].proto.id)

    assert not at.exception, [e.value for e in at.exception]
    assert "Завантажується" not in set(at.dataframe[0].value["DataStatus"])
    # Once everything is loaded the status row is gone: the counters above say it already.
    assert not at.get("progress")
    assert not any("Деталі" in caption.value for caption in at.caption)
    # A fragment draws inside its own container; drawn anywhere else when loading ends, the
    # table would be built anew in the browser and lose its scroll position.
    assert loaded == while_loading


def test_one_loading_step_sends_several_batches() -> None:
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    _, api_app = _app_with(container, _ManyOpenForms())

    with _web_talking_to(api_app):
        at = _signed_in_web(session_id)
        at.run()

    assert not at.exception, [e.value for e in at.exception]
    table = at.dataframe[0].value
    # Three parallel requests of 20 forms, the most the API takes in one request.
    assert int((table["DataStatus"] != "Завантажується").sum()) == 60


def test_rows_held_back_by_the_quota_show_their_status_and_wait_for_a_retry() -> None:
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    _, api_app = _app_with(container, _ManyOpenForms())
    api_app.state.forms_quota = FormsQuotaGuards(
        reads=RollingWindowGuard(1000), response_lists=RollingWindowGuard(5)
    )

    with _web_talking_to(api_app):
        at = _signed_in_web(session_id)
        at.run()

    assert not at.exception, [e.value for e in at.exception]
    table = at.dataframe[0].value
    counted = table["Total"].notna()
    assert int(counted.sum()) == 5
    # Every loaded form has its status; the held-back ones only wait for the response count,
    # quietly, without an error label or the manual retry button.
    loaded = table["PublicationStatus"] != "Завантажується"
    assert int(loaded.sum()) == 60
    assert set(table.loc[loaded, "PublicationStatus"]) == {"Відкрита"}
    assert set(table.loc[loaded & ~counted, "DataStatus"]) == {"Завантажується"}
    assert not any("Повторити" in button.label for button in at.button)


def test_choosing_a_form_above_moves_the_mark_in_the_same_table() -> None:
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    _, api_app = _app_with(container, _ManyOpenForms())

    with _web_talking_to(api_app):
        at = _signed_in_web(session_id)
        at.session_state["global_form_id"] = "form_05"
        at.run()
        before = at.dataframe[0]
        # The browser keeps the table's selection between runs; AppTest has no browser, so set
        # the row the table marks, or every run would start again from selection_default.
        at.session_state[before.key] = {"selection": {"rows": [5], "columns": [], "cells": []}}
        at.selectbox(key="global_form_select_catalog").set_value("form_10").run()
        after = at.dataframe[0]

    assert not at.exception, [e.value for e in at.exception]
    assert json.loads(before.proto.selection_default)["selection"]["rows"] == [5]
    # The same table, so the browser keeps its scroll; only the mark moves to the new form.
    assert after.proto.id == before.proto.id
    assert list(at.session_state[after.key]["selection"]["rows"]) == [10]


class _CountingDriveLists(_ManyOpenForms):
    def __init__(self) -> None:
        self.drive_lists = 0

    def list_forms(self, creds: Credentials) -> list[dict]:
        self.drive_lists += 1
        return super().list_forms(creds)


def test_the_form_picker_reuses_the_catalog_list_from_drive() -> None:
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    google = _CountingDriveLists()
    _, api_app = _app_with(container, google)

    with _web_talking_to(api_app):
        at = _signed_in_web(session_id)
        at.run()

    assert not at.exception, [e.value for e in at.exception]
    # One Drive list a page load: the picker above used to fetch the same list again,
    # 2.5-3.5 s each time its shorter cache ran out.
    assert google.drive_lists == 1
    assert len(at.selectbox(key="global_form_select_catalog").options) == FORM_COUNT
