from __future__ import annotations

from dataclasses import dataclass, field

from ui.components.page_shell import (
    format_form_caption,
    render_empty_state,
    render_error_state,
    render_form_caption,
    render_page_header,
    render_state,
)


@dataclass
class FakeContainer:
    calls: list[tuple[str, str, str | None]] = field(default_factory=list)

    def title(self, body: str) -> None:
        self.calls.append(("title", body, None))

    def caption(self, body: str) -> None:
        self.calls.append(("caption", body, None))

    def info(self, body: str, *, icon: str | None = None) -> None:
        self.calls.append(("info", body, icon))

    def success(self, body: str, *, icon: str | None = None) -> None:
        self.calls.append(("success", body, icon))

    def warning(self, body: str, *, icon: str | None = None) -> None:
        self.calls.append(("warning", body, icon))

    def error(self, body: str, *, icon: str | None = None) -> None:
        self.calls.append(("error", body, icon))


def test_format_form_caption_falls_back_to_dash() -> None:
    assert format_form_caption("") == "Форма: **—**"
    assert format_form_caption(None) == "Форма: **—**"


def test_format_form_caption_uses_custom_label() -> None:
    assert format_form_caption("Супергерої КПІ", label="Опитування") == (
        "Опитування: **Супергерої КПІ**"
    )


def test_render_page_header_title_and_caption() -> None:
    fake = FakeContainer()
    render_page_header("Динаміка", "Короткий опис", container=fake)
    assert fake.calls == [
        ("title", "Динаміка", None),
        ("caption", "Короткий опис", None),
    ]


def test_render_form_caption() -> None:
    fake = FakeContainer()
    render_form_caption("Форма 1", container=fake)
    assert fake.calls == [("caption", "Форма: **Форма 1**", None)]


def test_render_state_appends_details_and_icon() -> None:
    fake = FakeContainer()
    render_state("Не вдалося завантажити форму.", kind="warning", details="HTTP 403", container=fake)
    assert fake.calls == [
        ("warning", "Не вдалося завантажити форму.\n\nHTTP 403", ":material/warning:")
    ]


def test_empty_and_error_helpers_use_expected_kinds() -> None:
    fake = FakeContainer()
    render_empty_state("Поки немає відповідей.", container=fake)
    render_error_state("Помилка.", container=fake)
    assert fake.calls == [
        ("info", "Поки немає відповідей.", ":material/info:"),
        ("error", "Помилка.", ":material/error:"),
    ]
