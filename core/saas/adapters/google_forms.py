"""Google Forms/Drive adapter used by the SaaS API."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from google.oauth2.credentials import Credentials

from core.forms_api import (
    get_form_structure,
    list_form_responses,
    list_response_timestamps,
)
from core.forms_catalog import enrich_form, list_forms_with_drive_meta


class GoogleFormsApiClient:
    """Thin server-side adapter over the existing Google API helpers."""

    def list_forms(self, creds: Credentials) -> list[dict[str, Any]]:
        return [asdict(item) for item in list_forms_with_drive_meta(creds)]

    def get_form_summary(self, creds: Credentials, form_id: str) -> dict[str, Any]:
        return asdict(enrich_form(creds, form_id))

    def get_response_stats(self, creds: Credentials, form_id: str) -> dict[str, Any]:
        timestamps = list_response_timestamps(creds, form_id)
        return {
            "total": len(timestamps),
            "first_response": _format_timestamp(timestamps[0]) if timestamps else None,
            "second_response": _format_timestamp(timestamps[1]) if len(timestamps) >= 2 else None,
            "last_response": _format_timestamp(timestamps[-1]) if timestamps else None,
        }

    def list_response_timestamps(self, creds: Credentials, form_id: str) -> list[str]:
        return [_format_timestamp(item) for item in list_response_timestamps(creds, form_id)]

    def get_form_structure(self, creds: Credentials, form_id: str) -> dict[str, Any]:
        return get_form_structure(creds, form_id)

    def list_responses(self, creds: Credentials, form_id: str) -> list[dict[str, Any]]:
        return list_form_responses(creds, form_id)


def _format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")
