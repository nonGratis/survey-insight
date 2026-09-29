"""Guard against st.cache_data entries shared between users.

Streamlit excludes parameters whose name starts with an underscore from the
cache key. An identity parameter (token/session/credentials) named that way
silently turns a per-user cache into a global one.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDENTITY_NAMES = {"cache_token", "session_token", "token", "session_id"}
# Pure computations over their (hashed) arguments; no user data is fetched.
PURE_CACHED_FUNCTIONS = {"_cached_forecast"}


def _cached_functions() -> list[tuple[str, ast.FunctionDef]]:
    found: list[tuple[str, ast.FunctionDef]] = []
    for path in sorted((ROOT / "ui").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and any(
                "cache_data" in ast.unparse(dec) for dec in node.decorator_list
            ):
                found.append((str(path.relative_to(ROOT)), node))
    return found


def test_cached_functions_are_discovered() -> None:
    assert len(_cached_functions()) >= 10


def test_identity_parameters_are_part_of_the_cache_key() -> None:
    hidden = [
        f"{path}:{fn.name}({arg.arg})"
        for path, fn in _cached_functions()
        for arg in fn.args.args
        if arg.arg.startswith("_") and arg.arg.lstrip("_") in IDENTITY_NAMES
    ]
    assert hidden == []


def test_every_user_data_cache_is_keyed_by_identity() -> None:
    unkeyed = [
        f"{path}:{fn.name}"
        for path, fn in _cached_functions()
        if fn.name not in PURE_CACHED_FUNCTIONS
        and not any(arg.arg in IDENTITY_NAMES for arg in fn.args.args)
    ]
    assert unkeyed == []


def _saas_state(monkeypatch, state: dict) -> None:
    import ui.google_data as google_data

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("API_BASE_URL", "https://api.example.com")
    monkeypatch.setattr(google_data.st, "session_state", state)


def test_cache_key_isolation(monkeypatch) -> None:
    from ui.google_data import cache_token

    _saas_state(monkeypatch, {"saas_session_id": "sid", "user": {"id": "user_a"}})
    token_a = cache_token()
    _saas_state(monkeypatch, {"saas_session_id": "sid", "user": {"id": "user_b"}})
    token_b = cache_token()

    assert token_a != token_b


def test_cache_token_is_stable_for_the_same_session_and_user(monkeypatch) -> None:
    from ui.google_data import cache_token

    state = {"saas_session_id": "sid", "user": {"id": "user_a"}}
    _saas_state(monkeypatch, state)

    assert cache_token() == cache_token()


def test_cache_token_never_contains_raw_identifiers(monkeypatch) -> None:
    from ui.google_data import cache_token

    _saas_state(
        monkeypatch,
        {"saas_session_id": "raw-session", "user": {"id": "user_a", "email": "a@example.com"}},
    )

    token = cache_token()

    assert "raw-session" not in token
    assert "user_a" not in token
    assert "a@example.com" not in token


def test_anonymous_state_never_reuses_a_cache_entry(monkeypatch) -> None:
    from ui.google_data import cache_token

    _saas_state(monkeypatch, {"saas_session_id": "sid"})

    assert cache_token() != cache_token()
