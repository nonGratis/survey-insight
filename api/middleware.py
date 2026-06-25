"""FastAPI middleware for privacy-safe request timing."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

from core.logger import get_logger

log = get_logger(__name__)


async def request_timing_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Log route-level timing without query values or payload."""
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        log.info(
            "api_request_timing",
            extra={
                "method": request.method,
                "path": str(route_path),
                "status": status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
