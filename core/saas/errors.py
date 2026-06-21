"""Explicit SaaS error taxonomy for auth, quota, Google access, and reports."""

from __future__ import annotations

# ruff: noqa: N818 - domain error names intentionally match the SaaS plan.


class SaaSError(Exception):
    """Base class for SaaS-domain errors."""


class AuthError(SaaSError):
    """Authentication or authorization failed closed."""


class InvalidOAuthState(AuthError):
    """OAuth callback state is absent, malformed, or unknown."""


class ExpiredOAuthState(AuthError):
    """OAuth callback state exists but is expired."""


class ReplayedOAuthState(AuthError):
    """OAuth callback state was already consumed."""


class InvalidLoginTicket(AuthError):
    """Login ticket is absent, malformed, or unknown."""


class ExpiredLoginTicket(AuthError):
    """Login ticket exists but is expired."""


class ReplayedLoginTicket(AuthError):
    """Login ticket was already consumed."""


class InvalidSession(AuthError):
    """Session cookie is missing, expired, revoked, or tampered with."""


class MissingRequiredScopes(AuthError):
    """Stored Google grant does not contain required scopes."""


class QuotaExceeded(SaaSError):
    """User plan quota does not allow the requested action."""


class JobConflict(SaaSError):
    """Requested job transition is not valid for the current state."""


class GoogleApiTemporaryError(SaaSError):
    """Temporary Google API problem; safe to retry."""


class GoogleApiPermissionError(SaaSError):
    """Google API permission or scope problem; user action is required."""


class ReportGenerationError(SaaSError):
    """Report generation failed permanently."""
