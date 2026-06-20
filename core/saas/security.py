"""Security primitives for opaque sessions, tickets, and state hashes."""

from __future__ import annotations

import hmac
import secrets
from datetime import UTC, datetime
from hashlib import sha256


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


def generate_opaque_token(nbytes: int = 32) -> str:
    """Generate a URL-safe opaque secret with at least 256-bit entropy by default."""
    if nbytes < 32:
        raise ValueError("Opaque SaaS tokens must use at least 32 random bytes.")
    return secrets.token_urlsafe(nbytes)


def hash_secret(secret: str, pepper: str) -> str:
    """Return HMAC-SHA256(secret) using a server-side pepper.

    Firestore stores only this digest for sessions, OAuth states, and login tickets.
    """
    if not secret:
        raise ValueError("secret must not be empty")
    if not pepper:
        raise ValueError("pepper must not be empty")
    digest = hmac.new(pepper.encode("utf-8"), secret.encode("utf-8"), sha256).hexdigest()
    return f"hmac-sha256:{digest}"


def is_expired(expires_at: datetime, *, now: datetime | None = None) -> bool:
    current = now or utcnow()
    return expires_at <= current
