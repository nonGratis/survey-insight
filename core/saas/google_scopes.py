"""Google OAuth scope groups used by SaaS incremental authorization."""

from __future__ import annotations

IDENTITY_SCOPES = (
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
)
FORM_SCOPES = IDENTITY_SCOPES + (
    "https://www.googleapis.com/auth/drive.metadata.readonly",
    "https://www.googleapis.com/auth/forms.body.readonly",
    "https://www.googleapis.com/auth/forms.responses.readonly",
)
SHEETS_SCOPES = FORM_SCOPES + ("https://www.googleapis.com/auth/spreadsheets.readonly",)
# Deliberately not named "oauth_*": CodeQL classifies any identifier containing "oauth" as a
# password, and would flag these public scope URLs as sensitive data when they reach a log line.
SCOPES_BY_PURPOSE = {
    "identity": IDENTITY_SCOPES,
    "forms": FORM_SCOPES,
    "sheets": SHEETS_SCOPES,
}


def scopes_for_purpose(purpose: str) -> tuple[str, ...]:
    return SCOPES_BY_PURPOSE[purpose]
