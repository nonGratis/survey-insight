from __future__ import annotations

from dataclasses import dataclass, field

from ui.components.metric_bar import MetricItem, render_metric_bar


@dataclass
class FakeColumn:
    calls: list[tuple[str, object, object | None, str, str | None]] = field(default_factory=list)

    def metric(
        self,
        label: str,
        value: object,
        delta: object | None = None,
        *,
        delta_color: str = "normal",
        help: str | None = None,
    ) -> None:
        self.calls.append((label, value, delta, delta_color, help))


@dataclass
class FakeContainer:
    created_specs: list[tuple[int, str]] = field(default_factory=list)
    columns_created: list[FakeColumn] = field(default_factory=list)

    def columns(self, spec: int, gap: str = "small") -> list[FakeColumn]:
        self.created_specs.append((spec, gap))
        self.columns_created = [FakeColumn() for _ in range(spec)]
        return self.columns_created


def test_render_metric_bar_uses_item_count_by_default() -> None:
    fake = FakeContainer()
    render_metric_bar(
        [
            MetricItem("Зараз", 47),
            MetricItem("Прогноз", 53, delta="±4", delta_color="off", help="95% CI"),
        ],
        container=fake,
    )
    assert fake.created_specs == [(2, "small")]
    assert fake.columns_created[0].calls == [("Зараз", 47, None, "normal", None)]
    assert fake.columns_created[1].calls == [("Прогноз", 53, "±4", "off", "95% CI")]


def test_render_metric_bar_accepts_explicit_column_count_and_gap() -> None:
    fake = FakeContainer()
    render_metric_bar([MetricItem("A", "1")], columns=3, gap="large", container=fake)
    assert fake.created_specs == [(3, "large")]
    assert fake.columns_created[0].calls == [("A", "1", None, "normal", None)]


def test_render_metric_bar_empty_is_noop() -> None:
    fake = FakeContainer()
    render_metric_bar([], container=fake)
    assert fake.created_specs == []
