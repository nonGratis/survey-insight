"""URL checks for redirect targets supplied by API callers."""

from __future__ import annotations


def safe_next_url(next_url: str, app_base_url: str) -> str:
    if next_url.startswith("/"):
        return next_url
    if app_base_url and next_url.startswith(app_base_url.rstrip("/") + "/"):
        return next_url
    return "/"
