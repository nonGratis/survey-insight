"""Tests for core.report — universal PDF renderer."""

from __future__ import annotations

import pytest

from core.report import (
    BarChart,
    FlowChart,
    FlowChartEdge,
    FlowChartNode,
    Heading,
    Metric,
    Metrics,
    PageBreak,
    Para,
    Report,
    TableBlock,
    render_pdf,
)


def _is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-" and b"%%EOF" in data[-1024:]


def test_render_minimal_report_is_valid_pdf():
    pdf = render_pdf(Report(title="Звіт", subtitle="підзаголовок"))
    assert _is_pdf(pdf)
    assert len(pdf) > 500


def test_render_all_block_types_cyrillic():
    report = Report(
        title="Звіт про репрезентативність",
        subtitle="Форма: Опитування · χ² DEFF ≈ 1,31",
        blocks=[
            Heading("Показники", level=2),
            Metrics(
                [Metric("DEFF", "1,31"), Metric("n_eff", "380"), Metric("MoE", "4,4%")], columns=3
            ),
            Para("Зважування коригує перекоси за <b>підрозділом</b> і курсом (їєґ)."),
            TableBlock(headers=["Страта", "Вага"], rows=[["ФІОТ", "0,499"], ["ФБМІ", "3,552"]]),
        ],
    )
    pdf = render_pdf(report)
    assert _is_pdf(pdf)


def test_render_large_table_multipage():
    rows = [[f"страта {i}", f"{i / 7:.3f}", f"{i}"] for i in range(120)]
    report = Report(
        title="Велика таблиця",
        blocks=[TableBlock(headers=["Назва", "Вага", "n"], rows=rows, col_widths=[0.6, 0.2, 0.2])],
    )
    pdf = render_pdf(report)
    assert _is_pdf(pdf)
    assert len(pdf) > 3000  # кілька сторінок → більший файл


def test_unknown_block_raises():
    with pytest.raises(TypeError):
        render_pdf(Report(title="x", blocks=[object()]))


def test_empty_blocks_ok():
    pdf = render_pdf(Report(title="Порожній"))
    assert _is_pdf(pdf)


def test_page_break_renders():
    rep = Report(title="T", blocks=[Para("перша"), PageBreak(), Para("друга")])
    assert _is_pdf(render_pdf(rep))


def test_barchart_renders():
    rep = Report(
        title="Діаграма",
        blocks=[
            BarChart(
                labels=["ФІОТ", "ФЕА", "дуже довга назва підрозділу понад тридцять символів"],
                values=[137, 50, 12],
            )
        ],
    )
    assert _is_pdf(render_pdf(rep))


def test_flowchart_renders():
    rep = Report(
        title="Карта переходів",
        blocks=[
            FlowChart(
                nodes=[
                    FlowChartNode("__start__", "Старт\nПитання маршруту", kind="start"),
                    FlowChartNode("sec_1", "Секція 1", kind="section"),
                    FlowChartNode("__submit__", "Надіслати", kind="submit"),
                ],
                edges=[
                    FlowChartEdge("__start__", "sec_1", "Так", kind="conditional"),
                    FlowChartEdge("sec_1", "__submit__", "надіслати", kind="default"),
                ],
            )
        ],
    )
    assert _is_pdf(render_pdf(rep))
