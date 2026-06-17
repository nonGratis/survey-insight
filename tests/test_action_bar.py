from __future__ import annotations

from ui.components.action_bar import ActionBarStatus, render_action_status


class FakeContainer:
    def __init__(self) -> None:
        self.captions: list[str] = []

    def caption(self, body: str) -> None:
        self.captions.append(body)


def test_action_bar_status_renders_parts() -> None:
    status = ActionBarStatus(responses=47, updated="2 хв тому", note="кеш 60с")
    assert status.render() == "47 відповідей · оновлено 2 хв тому · кеш 60с"


def test_action_bar_status_omits_empty_parts() -> None:
    assert ActionBarStatus(note="готово").render() == "готово"


def test_render_action_status_outputs_caption() -> None:
    fake = FakeContainer()
    render_action_status(ActionBarStatus(responses=12, note="аналіз"), container=fake)
    assert fake.captions == ["12 відповідей · аналіз"]


def test_render_action_status_ignores_empty_status() -> None:
    fake = FakeContainer()
    render_action_status(None, container=fake)
    render_action_status(ActionBarStatus(), container=fake)
    assert fake.captions == []
