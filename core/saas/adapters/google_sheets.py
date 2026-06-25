"""Google Sheets adapter used by the SaaS API.

The SaaS API does not return raw spreadsheet grids to Streamlit. It scans
server-side and exposes only detected population tables used for weighting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, cast

from google.oauth2.credentials import Credentials

from core.context_tables import scan_sheets_for_tables
from core.sheets_api import fetch_all_grids


class GoogleSheetsApiClient:
    """Thin server-side adapter for population-table discovery."""

    def scan_population_tables(self, creds: Credentials, sheet_id: str) -> list[dict[str, Any]]:
        grids = cast(dict[str, Sequence[Sequence[str]]], fetch_all_grids(creds, sheet_id))
        tables = scan_sheets_for_tables(grids)
        return [asdict(table) for table in tables]
