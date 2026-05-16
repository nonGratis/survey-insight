"""Structured logging для survey-insight.

JSON-формат — для prod (Cloud Run / GCP Cloud Logging автоматично парсить).
Human-формат — для local dev.

Контекст (session_id, user_hash, page) додається через StreamlitContextFilter
тільки у main thread'і — у worker-thread'ах parallel_map контексту немає
і це ОЧІКУВАНО (логи з API workers просто матимуть менше полів).

Event-name convention:
- api_op_{ok,retry,failed}  — high-level через call_with_backoff
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
            "level": record.levelname,
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


class StreamlitContextFilter(logging.Filter):
    """Injects session_id, user_hash, page — лише у main thread.

    У ThreadPoolExecutor worker'ах Streamlit-context недоступний за дизайном
    (нема script_run_ctx, session_state ламається). Skip-аємо явно через
    main-thread check — без try/except, що замилює інші баги.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if threading.current_thread() is not threading.main_thread():
            return True

        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
        except ImportError:
            return True

        ctx = get_script_run_ctx()
        if ctx is None:
            return True

        record.session_id = (getattr(ctx, "session_id", "") or "")[:12]
        record.page = getattr(ctx, "page_script_hash", "") or "—"

        try:
            import streamlit as st

            user = st.session_state.get("user") or {}
        except Exception:
            return True

        email = user.get("email")
        if email:
            record.user_hash = hash_email(email)
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

    Викликати ОДИН РАЗ у entry-point (app.py) до будь-яких імпортів core/.
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
