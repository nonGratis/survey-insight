"""URL checks for redirect targets supplied by API callers."""

from __future__ import annotations


def safe_next_url(next_url: str, app_base_url: str) -> str:
    """Return ``next_url`` when it leads back into our own app, else ``"/"``.

    The OAuth callback redirects the browser to this value with a one-time login
    ticket appended, so a target that lands on another host hands that ticket to
    whoever picked the host. Two shapes are trusted:

    * an absolute URL inside ``app_base_url`` (or equal to it);
    * a path on the current host: a single leading ``/``, never ``//``.
    """
    trusted = not _has_ambiguous_characters(next_url) and (
        _is_local_path(next_url) or _is_inside_app(next_url, app_base_url)
    )
    return next_url if trusted else "/"


def _has_ambiguous_characters(url: str) -> bool:
    """True if a browser may read the URL differently from a plain string check.

    Browsers drop tab/CR/LF anywhere in a URL and treat "\\" as "/", so
    "/<TAB>/host" and "/\\host" both mean "//host" although they look like local
    paths. Other whitespace and invisible characters never occur in a link we issue.
    """
    return any(ch == "\\" or ch.isspace() or not ch.isprintable() for ch in url)


def _is_local_path(url: str) -> bool:
    # "//host/path" is scheme-relative: the browser sends it to host.
    return url.startswith("/") and not url.startswith("//")


def _is_inside_app(url: str, app_base_url: str) -> bool:
    base = app_base_url.rstrip("/")
    # An empty base would become the prefix "/", which every local path starts with.
    return bool(base) and (url == base or url.startswith(base + "/"))
