"""Structured logging для survey-insight.

JSON-формат — для prod (Cloud Run / GCP Cloud Logging автоматично парсить).
Human-формат — для local dev.

У web StreamlitContextFilter додає session_ref і user_id до записів, зроблених під час
запуску сторінки; у потоках, які сторінка запускає сама, контексту немає, і це ОЧІКУВАНО.
``severity`` — поле, з якого Cloud Logging бере рівень запису.

Event-name convention:
- api_call_ok               — low-level через log_call() на .execute() сайтах
- auth_login_ok / auth_callback_failed / oauth_userinfo_failed
- ui_<page>_load_failed
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Literal

from core.saas.security import log_user_ref

# RESERVED: усе, що `LogRecord` ставить сам, виключаємо з extras.
# Авто-derive із порожнього запису — щоб не пропустити нові поля у Python 3.12+.
_SAMPLE = logging.LogRecord("", 0, "", 0, "", None, None)
RESERVED: set[str] = set(_SAMPLE.__dict__.keys()) | {"message"}

_SAFE_TYPES: tuple = (str, int, float, bool, type(None), list, dict, tuple)


def _safe_extras(record: logging.LogRecord) -> dict[str, object]:
    """Дістати все, що не RESERVED і безпечно серіалізується."""
    out: dict[str, object] = {}
    for k, v in record.__dict__.items():
        if k in RESERVED or k.startswith("_"):
            continue
        if isinstance(v, _SAFE_TYPES):
            out[k] = v
        else:
            # Непідтриманий тип (datetime, custom object) -> repr щоб не падати.
            out[k] = repr(v)
    return out


class JSONFormatter(logging.Formatter):
    """GCP Cloud Logging автоматично парсить однорядковий JSON зі stdout."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            # Cloud Logging reads the entry's level from this field (not from "level"),
            # so severity filters in Logs Explorer and Log Analytics work.
            "severity": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        payload.update(_safe_extras(record))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """Лаконічний формат для локального dev — у консолі streamlit run."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        base = f"{ts} {record.levelname:<7} {record.name}: {record.getMessage()}"
        extras = _safe_extras(record)
        if extras:
            base += "  [" + " ".join(f"{k}={v}" for k, v in extras.items()) + "]"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


# Browser session id -> log ref of the user signed in there, kept by bind_session_user.
_SESSION_USERS: dict[str, str] = {}
_SESSION_USERS_LOCK = threading.Lock()
_SESSION_USERS_MAX = 10_000


def bind_session_user(session_id: str, user_id: str | None) -> None:
    """Remember who is signed in to a browser session (None: nobody) for the web logs."""
    with _SESSION_USERS_LOCK:
        _SESSION_USERS.pop(session_id, None)
        if user_id:
            _SESSION_USERS[session_id] = log_user_ref(user_id)
            if len(_SESSION_USERS) > _SESSION_USERS_MAX:
                # Sessions closed without a sign-out are never unbound: drop the oldest.
                _SESSION_USERS.pop(next(iter(_SESSION_USERS)))


class StreamlitContextFilter(logging.Filter):
    """Adds ``session_ref`` and ``user_id`` to records written while a page script runs.

    Streamlit runs every page script in a thread of its own, never the main one, so the
    context comes from the script run context; threads the page starts itself (catalog
    batches) have none and their records go out without these fields. The user comes from
    bind_session_user, not from st.session_state: reading it is where Streamlit raises a
    pending stop or rerun, which would drop the record and could swallow a rerun.
    ``user_id`` is the same digest the API logs, so both services join on it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
        except ImportError:
            return True

        ctx = get_script_run_ctx(suppress_warning=True)
        if ctx is None:
            return True

        record.session_ref = hashlib.sha256(ctx.session_id.encode("utf-8")).hexdigest()[:12]
        user_ref = _SESSION_USERS.get(ctx.session_id)
        if user_ref:
            record.user_id = user_ref
        return True


def hash_email(email: str) -> str:
    """SHA-256 → перші 16 hex (достатньо для кореляції, нуль PII у logs)."""
    return hashlib.sha256(email.encode("utf-8")).hexdigest()[:16]


def _resolve_format() -> Literal["json", "human"]:
    """K_SERVICE (Cloud Run автоматично виставляє) → json; LOG_FORMAT — override."""
    override = os.getenv("LOG_FORMAT", "").lower().strip()
    if override in ("json", "human"):
        return override  # type: ignore[return-value]
    if os.getenv("K_SERVICE"):
        return "json"
    return "human"


_HANDLER_MARKER = "_survey_insight_logging_handler"


def setup_logging(force: bool = False) -> None:
    """Idempotent. Не чіпає чужі handler'и (Streamlit/uvicorn ставлять свої).

    Викликати ОДИН РАЗ у entry-point: app.py (після import streamlit, до імпортів core/),
    api/main.py, worker/main.py.
    """
    root = logging.getLogger()
    if not force and any(getattr(h, _HANDLER_MARKER, False) for h in root.handlers):
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level_name, logging.INFO))

    formatter: logging.Formatter
    formatter = JSONFormatter() if _resolve_format() == "json" else HumanFormatter()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    # Контекст сесії є лише у web: app.py імпортує Streamlit ще до цього виклику. API й worker
    # Streamlit не імпортують, а фільтр затягнув би в них увесь UI-стек на першому ж записі.
    if "streamlit" in sys.modules:
        handler.addFilter(StreamlitContextFilter())
    setattr(handler, _HANDLER_MARKER, True)
    root.addHandler(handler)

    # Притишити шумні бібліотеки — їх власні INFO нам не цікаві.
    for noisy in (
        "urllib3",
        "googleapiclient.discovery_cache",
        "googleapiclient.http",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Стандартний паттерн: logger = get_logger(__name__) у кожному модулі."""
    return logging.getLogger(name)


@contextmanager
def log_call(
    label: str,
    *,
    level: int = logging.DEBUG,
    logger: logging.Logger | None = None,
    **extras: object,
) -> Iterator[None]:
    """Context manager для duration-логу УСПІШНИХ викликів.

    На винятку — ТІЛЬКИ re-raise (нічого не логує). Логування з
    контекстом і traceback'ом — справа catching сайту (UI / boundary),
    щоб уникнути дубліката того самого exception у кількох шарах.
    """
    assert isinstance(level, int), f"level must be int, got {type(level).__name__}"
    log = logger or logging.getLogger(__name__)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        raise
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        log.log(level, label, extra={"duration_ms": duration_ms, **extras})
