"""Версія застосунку: яку збірку розгорнуто.

Скрипт деплою вшиває в образ короткий хеш коміту й дату збірки через змінні
`APP_VERSION` і `APP_BUILD_DATE` (див. `Dockerfile` і `deploy/cloud-run/cloudbuild.yaml`).
Без них (локальний запуск, тести, CI) версія — ``dev``. Значення, схожі на щось
інше, ніж мітка коду, відкидаємо: версія потрапляє в UI і у відповідь `/health`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

DEV_VERSION = "dev"
_VERSION_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]{0,39}")


@dataclass(frozen=True)
class AppVersion:
    """Мітка коду (короткий хеш коміту) і дата збірки, якщо відома."""

    version: str = DEV_VERSION
    build_date: date | None = None

    @property
    def label(self) -> str:
        """Рядок для людей: ``cd3b096 · 03.10.2026`` або просто ``dev``."""
        if self.build_date is None:
            return self.version
        return f"{self.version} · {self.build_date:%d.%m.%Y}"


def current_version(environ: Mapping[str, str] | None = None) -> AppVersion:
    """Прочитати версію з оточення процесу (або з переданого словника в тестах)."""
    env = os.environ if environ is None else environ
    raw_version = env.get("APP_VERSION", "").strip()
    version = raw_version if _VERSION_RE.fullmatch(raw_version) else DEV_VERSION
    return AppVersion(version=version, build_date=_parse_date(env.get("APP_BUILD_DATE", "")))


def _parse_date(raw: str) -> date | None:
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        return None
