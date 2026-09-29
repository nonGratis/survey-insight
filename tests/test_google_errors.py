from __future__ import annotations

import json

import httplib2
from googleapiclient.errors import HttpError

from core.google_errors import google_error_reason, is_scope_error


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


# Shape returned by Drive/Forms for a token that lacks the required scope. It is the
# gRPC-style error: the machine reason sits in details[], there is no errors[] array.
_GRPC_SCOPE_ERROR = {
    "error": {
        "code": 403,
        "message": "Request had insufficient authentication scopes.",
        "status": "PERMISSION_DENIED",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "ACCESS_TOKEN_SCOPE_INSUFFICIENT",
                "domain": "googleapis.com",
                "metadata": {
                    "method": "google.apps.drive.v3.DriveFiles.List",
                    "service": "drive.googleapis.com",
                },
            }
        ],
    }
}


def test_grpc_style_scope_error_reason_is_extracted() -> None:
    assert (
        google_error_reason(_http_error(403, _GRPC_SCOPE_ERROR))
        == "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
    )


def test_a_scope_reason_wins_over_an_earlier_generic_one() -> None:
    payload = {
        "error": {
            "code": 403,
            "errors": [{"reason": "forbidden"}],
            "details": [{"reason": "ACCESS_TOKEN_SCOPE_INSUFFICIENT"}],
        }
    }

    assert google_error_reason(_http_error(403, payload)) == "ACCESS_TOKEN_SCOPE_INSUFFICIENT"


def test_details_without_a_reason_are_skipped() -> None:
    payload = {
        "error": {
            "code": 403,
            "details": [
                {"@type": "type.googleapis.com/google.rpc.Help", "links": []},
                {"reason": "rateLimitExceeded"},
            ],
        }
    }

    assert google_error_reason(_http_error(403, payload)) == "rateLimitExceeded"


def test_is_scope_error_uses_reason_or_google_message() -> None:
    assert is_scope_error(403, "ACCESS_TOKEN_SCOPE_INSUFFICIENT", "")
    assert is_scope_error(403, "insufficientPermissions", "")
    assert is_scope_error(403, None, "Failed: Request had insufficient authentication scopes.")
    assert not is_scope_error(403, "forbidden", "The caller does not have permission")
    assert not is_scope_error(401, "ACCESS_TOKEN_SCOPE_INSUFFICIENT", "")
    assert not is_scope_error(500, None, "insufficient authentication scopes")
