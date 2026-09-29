from __future__ import annotations

import json

import httplib2
from googleapiclient.errors import HttpError

from core.google_errors import google_error_reason


def _http_error(status: int, payload: object) -> HttpError:
    content = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return HttpError(httplib2.Response({"status": status}), content)


def test_reason_is_read_from_first_error_entry() -> None:
    exc = _http_error(
        403, {"error": {"code": 403, "errors": [{"reason": "insufficientPermissions"}]}}
    )
    assert google_error_reason(exc) == "insufficientPermissions"


def test_reason_is_read_from_error_details_when_errors_missing() -> None:
    exc = _http_error(
        403,
        {"error": {"code": 403, "details": [{"reason": "accessNotConfigured"}]}},
    )
    assert google_error_reason(exc) == "accessNotConfigured"


def test_reason_is_none_for_unparseable_body() -> None:
    assert google_error_reason(_http_error(500, b"<html>oops</html>")) is None
