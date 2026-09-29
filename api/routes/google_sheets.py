"""Google Sheets routes owned by the SaaS API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from api.dependencies import require_session
from api.google_errors import google_http_exception as map_google_error
from api.routes.google_forms import require_google_credentials
from core.saas.models import Session
from core.saas.ports import GoogleSheetsClient
from core.sheets_api import SheetsApiError

router = APIRouter(prefix="/v1", tags=["google-sheets"])


class PopulationTableResponse(BaseModel):
    source: str
    label_header: str
    count_header: str
    population: dict[str, int]


@router.get("/sheets/{sheet_id}/population-tables", response_model=list[PopulationTableResponse])
def list_population_tables(
    sheet_id: str,
    request: Request,
    session: Annotated[Session, Depends(require_session)],
    next_url: Annotated[str, Query(max_length=2048)] = "/",
) -> list[PopulationTableResponse]:
    creds = require_google_credentials(request, session, purpose="sheets", next_url=next_url)
    try:
        return [
            PopulationTableResponse.model_validate(item)
            for item in _sheets_client(request).scan_population_tables(creds, sheet_id)
        ]
    except SheetsApiError as exc:
        raise sheets_http_exception(exc) from exc


def sheets_http_exception(exc: SheetsApiError) -> HTTPException:
    return map_google_error(exc, purpose="sheets", error_code="google_sheets_error")


def _sheets_client(request: Request) -> GoogleSheetsClient:
    return request.app.state.google_sheets_client
