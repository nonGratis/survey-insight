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
OAUTH_PURPOSE_SCOPES = {
    "identity": IDENTITY_SCOPES,
    "forms": FORM_SCOPES,
    "sheets": SHEETS_SCOPES,
}


def scopes_for_purpose(purpose: str) -> tuple[str, ...]:
    return OAUTH_PURPOSE_SCOPES[purpose]
