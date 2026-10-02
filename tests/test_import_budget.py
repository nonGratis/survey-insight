"""Бюджет імпортів: що сервіс не має завантажувати на старті.

Cloud Run зупиняє сервіси без трафіку, тож перший запит після паузи чекає на
запуск процесу, а його час визначають імпорти. API та worker не рахують
статистику і не малюють UI: якщо в їхній граф імпортів потрапляє pandas чи
streamlit, це випадкова залежність через спільний модуль у ``core/``. Тест
ловить її одразу, а не на холодному старті в продакшні.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Аналітичний стек і UI: потрібні лише сервісу web.
ANALYTICS_AND_UI = frozenset(
    {
        "altair",
        "extra_streamlit_components",
        "numpy",
        "pandas",
        "plotly",
        "pyarrow",
        "reportlab",
        "ruptures",
        "scipy",
        "statsmodels",
        "streamlit",
    }
)


def _packages_loaded_by(entry_module: str) -> set[str]:
    """Імпортувати модуль у чистому процесі й повернути завантажені пакети."""
    code = (
        "import importlib, json, sys\n"
        f"importlib.import_module({entry_module!r})\n"
        "print(json.dumps(sorted({name.partition('.')[0] for name in sys.modules})))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO_ROOT,
        # in-memory адаптери: імпорт не звертається до GCP, хоч би що стояло в оточенні
        env={**os.environ, "APP_ENV": "test"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return set(json.loads(result.stdout.strip().splitlines()[-1]))


@pytest.mark.parametrize("entry_module", ["api.main", "worker.main"])
def test_service_start_does_not_load_analytics_or_ui_stack(entry_module: str):
    loaded = _packages_loaded_by(entry_module)

    assert entry_module.partition(".")[0] in loaded
    assert sorted(loaded & ANALYTICS_AND_UI) == []
