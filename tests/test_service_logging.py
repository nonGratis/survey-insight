"""API і worker пишуть структуровані логи, як і web.

Web налаштовує логування в ``app.py``, а API й worker запускає uvicorn, тож до цього тесту
корінь логування в них лишався без обробника: Python відкидав усі INFO-події (час запитів,
телеметрію каталогу, оновлення токенів), а WARNING і вище друкував голим текстом без полів.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICES = ["api.main", "worker.main"]


def _log_one_event_in(entry_module: str) -> tuple[list[dict], set[str]]:
    """Імпортувати сервіс у чистому процесі, як на Cloud Run, і записати одну INFO-подію.

    Повертає JSON-рядки з stderr і пакети, завантажені після запису.
    """
    code = (
        "import importlib, json, logging, sys\n"
        f"importlib.import_module({entry_module!r})\n"
        "logging.getLogger('probe').info('probe_event', extra={'probe_count': 3})\n"
        "print(json.dumps(sorted({name.partition('.')[0] for name in sys.modules})))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO_ROOT,
        # K_SERVICE ставить Cloud Run; APP_ENV=test тримає адаптери в пам'яті.
        env={**os.environ, "APP_ENV": "test", "K_SERVICE": "survey-insight-test"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in result.stderr.splitlines() if line.startswith("{")]
    return events, set(json.loads(result.stdout.strip().splitlines()[-1]))


@pytest.mark.parametrize("entry_module", SERVICES)
def test_service_writes_info_events_as_json(entry_module: str) -> None:
    events, _ = _log_one_event_in(entry_module)

    probe = [event for event in events if event.get("msg") == "probe_event"]
    assert len(probe) == 1
    assert probe[0]["severity"] == "INFO"
    assert probe[0]["probe_count"] == 3


@pytest.mark.parametrize("entry_module", SERVICES)
def test_service_logging_does_not_load_streamlit(entry_module: str) -> None:
    _, loaded = _log_one_event_in(entry_module)

    assert "streamlit" not in loaded
