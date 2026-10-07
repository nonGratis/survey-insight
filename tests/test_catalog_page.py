"""Сторінка «Каталог»: справжні web і API, фейковий лише Google."""

from __future__ import annotations

import json
import logging
import queue
import time
from collections.abc import Callable, Iterator
from unittest import mock

import pytest
from google.oauth2.credentials import Credentials
from streamlit.testing.v1 import AppTest

from api.google_data_cache import clear_api_cache
from tests.test_e2e_web_api import _app_with, _signed_in_web, _web_talking_to
from tests.test_saas_api import (
    _FakeGoogleFormsClient,
    _seed_google_grant,
    _seed_user_session,
    _test_container,
)
from ui.catalog_load import CatalogLoad, CatalogSnapshot
from ui.google_data import GoogleDataClient

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


def _catalog_api():  # type: ignore[no-untyped-def]
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    _, api_app = _app_with(container, _ManyOpenForms())
    return session_id, api_app


def _load(at: AppTest) -> CatalogLoad:
    return at.session_state["catalog_load"]


def _run_loaded(at: AppTest) -> None:
    """Run the page, let its background load finish, and run it again to draw the result."""
    at.run()
    _load(at).join(15)
    at.run()


def _settle(at: AppTest, ready: Callable[[CatalogSnapshot], bool]) -> None:
    """Wait until the background load has taken in what the test handed it."""
    deadline = time.monotonic() + 5
    while not ready(_load(at).snapshot()):
        assert time.monotonic() < deadline, "the load did not take the events in"
        time.sleep(0.01)


class _Feed:
    """Catalog events handed to the page's load by the test; the load waits for more."""

    def __init__(self) -> None:
        self._events: queue.Queue[dict | None] = queue.Queue()

    def push(self, *events: dict) -> None:
        for event in events:
            self._events.put(event)

    def end(self) -> None:
        self._events.put(None)

    def install(self):  # type: ignore[no-untyped-def]
        def stream(client, form_ids: list[str]) -> Iterator[dict]:  # type: ignore[no-untyped-def]
            while (event := self._events.get(timeout=10)) is not None:
                yield event

        return mock.patch.object(GoogleDataClient, "stream_catalog", stream)


def _summary(form_id: str) -> dict:
    data = {
        "title": form_id,
        "description": "",
        "sections_count": 1,
        "questions_count": 3,
        "linked_sheet_id": None,
        "is_published": True,
        "accepting_responses": True,
    }
    return {"event": "summary", "form_id": form_id, "status": "ok", "data": data}


def _count(form_id: str) -> dict:
    data = {"total": 0, "first_response": None, "second_response": None, "last_response": None}
    return {"event": "stats", "form_id": form_id, "status": "ok", "data": data}


FORM_IDS = [f"form_{index:02d}" for index in range(FORM_COUNT)]


def test_the_catalog_loads_through_one_streamed_request(caplog: pytest.LogCaptureFixture) -> None:
    session_id, api_app = _catalog_api()
    caplog.set_level(logging.INFO)

    with _web_talking_to(api_app):
        at = _signed_in_web(session_id)
        _run_loaded(at)

    assert not at.exception, [e.value for e in at.exception]
    table = at.dataframe[0].value
    assert set(table["PublicationStatus"]) == {"Відкрита"}
    assert table["Total"].notna().all()
    paths = [r.path for r in caplog.records if r.getMessage() == "ui_saas_api_request"]
    # The Drive list, then the whole catalog in one request instead of a batch per tick.
    assert paths.count("/v1/forms/catalog/stream") == 1
    assert "/v1/forms/catalog/enrich" not in paths


def test_rows_still_loading_are_not_counted_as_unknown() -> None:
    session_id, api_app = _catalog_api()
    feed = _Feed()

    with _web_talking_to(api_app), feed.install():
        at = _signed_in_web(session_id)
        at.run()
        feed.push(*(_summary(form_id) for form_id in FORM_IDS[:10]))
        _settle(at, lambda snapshot: len(snapshot.summaries) == 10)
        at.run()
        feed.end()

    assert not at.exception, [e.value for e in at.exception]
    table = at.dataframe[0].value
    loading = table["PublicationStatus"] == "Завантажується"
    assert int(loading.sum()) == FORM_COUNT - 10
    assert set(table.loc[~loading, "PublicationStatus"]) == {"Відкрита"}
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Невідомо"] == "0"


def test_table_keeps_its_identity_while_details_load() -> None:
    session_id, api_app = _catalog_api()
    feed = _Feed()

    with _web_talking_to(api_app), feed.install():
        at = _signed_in_web(session_id)
        at.run()
        before = at.dataframe[0]
        feed.push(*(_summary(form_id) for form_id in FORM_IDS[:10]))
        _settle(at, lambda snapshot: len(snapshot.summaries) == 10)
        at.run()
        after = at.dataframe[0]
        feed.end()

    assert not at.exception, [e.value for e in at.exception]
    assert int((after.value["PublicationStatus"] != "Завантажується").sum()) == 10
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
    session_id, api_app = _catalog_api()
    feed = _Feed()

    with _web_talking_to(api_app), feed.install():
        at = _signed_in_web(session_id)
        at.run()
        while_loading = (_position_of_table(at._tree), at.dataframe[0].proto.id)
        feed.push(
            *(event for form_id in FORM_IDS for event in (_summary(form_id), _count(form_id)))
        )
        feed.push({"event": "done"})
        _settle(at, lambda snapshot: snapshot.done)
        feed.end()
        at.run()  # the fragment sees the end and reruns the page without its timer
        at.run()
        loaded = (_position_of_table(at._tree), at.dataframe[0].proto.id)

    assert not at.exception, [e.value for e in at.exception]
    assert "Завантажується" not in set(at.dataframe[0].value["DataStatus"])
    # Once everything is loaded the status row is gone: the counters above say it already.
    assert not at.get("progress")
    assert not any("Деталі" in caption.value for caption in at.caption)
    # A fragment draws inside its own container; drawn anywhere else when loading ends, the
    # table would be built anew in the browser and lose its scroll position.
    assert loaded == while_loading


def test_the_status_row_tells_how_long_the_held_back_counts_wait() -> None:
    session_id, api_app = _catalog_api()
    feed = _Feed()

    with _web_talking_to(api_app), feed.install():
        at = _signed_in_web(session_id)
        at.run()
        feed.push(*(_summary(form_id) for form_id in FORM_IDS))
        feed.push(*(_count(form_id) for form_id in FORM_IDS[:5]))
        # The API says how long until the last count held by the Google quota runs.
        feed.push({"event": "waiting", "seconds": 40.0, "forms": FORM_COUNT - 5})
        _settle(at, lambda snapshot: snapshot.wait_seconds is not None)
        at.run()
        feed.end()

    assert not at.exception, [e.value for e in at.exception]
    [text] = [bar.proto.text for bar in at.get("progress")]
    prefix = f"Відповіді: 5/{FORM_COUNT} — решта приблизно за "
    assert text.startswith(prefix) and text.endswith(" с (ліміт Google)"), text
    assert 40 <= int(text.removeprefix(prefix).split()[0]) <= 42
    table = at.dataframe[0].value
    # Every form has its status; the held-back ones only wait for the count, quietly.
    assert set(table["PublicationStatus"]) == {"Відкрита"}
    assert int((table["DataStatus"] == "Завантажується").sum()) == FORM_COUNT - 5
    assert not any("Повторити" in button.label for button in at.button)


def test_a_grant_revoked_during_the_load_signs_out() -> None:
    session_id, api_app = _catalog_api()
    feed = _Feed()

    with _web_talking_to(api_app), feed.install():
        at = _signed_in_web(session_id)
        at.run()
        feed.push({"event": "error", "error_code": "GoogleTokenRevoked"})
        feed.end()
        _load(at).join(5)
        at.run()

    # The load's thread cannot sign out; the page run hands the error to the API boundary.
    assert not at.exception, [e.value for e in at.exception]
    assert "Вхід" in [s.value for s in at.subheader]


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
