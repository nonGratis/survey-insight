"""Перемикач «Усі / Мої / Чужі» у «Каталозі»: справжні web і API, фейковий лише Google."""

from __future__ import annotations

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

# _signed_in_web signs in as owner@example.com; Drive may spell the address differently.
OWNERS = {
    "form_mine": "owner@example.com",
    "form_mine_capitalized": "Owner@Example.com",
    "form_shared": "colleague@example.com",
}


class _FormsOfSeveralOwners(_FakeGoogleFormsClient):
    def list_forms(self, creds: Credentials) -> list[dict]:
        return [
            {
                "id": form_id,
                "name": form_id,
                "owner_email": owner,
                "owner_name": owner,
                "created_time": "2026-06-01T10:00:00Z",
                "modified_time": "2026-06-02T10:00:00Z",
                "edit_url": f"https://docs.google.com/forms/d/{form_id}/edit",
            }
            for form_id, owner in OWNERS.items()
        ]

    def get_form_summary(self, creds: Credentials, form_id: str) -> dict:
        return super().get_form_summary(creds, "form_1")

    def get_response_stats(self, creds: Credentials, form_id: str) -> dict:
        return super().get_response_stats(creds, "form_1")


@pytest.fixture(autouse=True)
def _production_web(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("APP_BASE_URL", "https://app.example.com")
    clear_api_cache()


@pytest.mark.parametrize(
    ("ownership", "expected"),
    [
        ("Усі", {"form_mine", "form_mine_capitalized", "form_shared"}),
        ("Мої", {"form_mine", "form_mine_capitalized"}),
        ("Чужі", {"form_shared"}),
    ],
)
def test_the_ownership_switch_keeps_my_forms_or_forms_shared_with_me(
    ownership: str, expected: set[str]
) -> None:
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    _, api_app = _app_with(container, _FormsOfSeveralOwners())

    with _web_talking_to(api_app):
        at = _signed_in_web(session_id)
        at.run()
        at.segmented_control(key="catalog_ownership").set_value(ownership).run()

    assert not at.exception, [e.value for e in at.exception]
    assert set(at.dataframe[0].value["FormName"]) == expected
