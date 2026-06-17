from __future__ import annotations

from ui.components.action_bar import ActionBarStatus


def test_action_bar_status_renders_parts() -> None:
    status = ActionBarStatus(responses=47, updated="2 хв тому", note="кеш 60с")
    assert status.render() == "47 відповідей · оновлено 2 хв тому · кеш 60с"


def test_action_bar_status_omits_empty_parts() -> None:
    assert ActionBarStatus(note="готово").render() == "готово"
