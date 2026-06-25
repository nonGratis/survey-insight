"""Google Sheets routes owned by the SaaS API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

from api.dependencies import require_session
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
    if exc.status in {401, 403}:
        code = status.HTTP_403_FORBIDDEN
    elif exc.status == 404:
        code = status.HTTP_404_NOT_FOUND
    elif exc.status in {429, 500, 502, 503, 504}:
        code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        code = status.HTTP_502_BAD_GATEWAY
    return HTTPException(
        status_code=code,
        detail={"code": "google_sheets_error", "message": str(exc)},
    )


def _sheets_client(request: Request) -> GoogleSheetsClient:
    return request.app.state.google_sheets_client
