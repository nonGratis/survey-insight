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
    _barchart_flowables,
    _flowchart_flowable,
    _wrap_lines,
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
                value_labels=["68,5 % · 137", "25,0 % · 50", "6,0 % · 12"],
            )
        ],
    )
    assert _is_pdf(render_pdf(rep))


def test_barchart_label_wraps_instead_of_single_line_truncation():
    long_label = "дуже довга назва підрозділу понад тридцять символів для графіка"
    lines = _wrap_lines(long_label, max_chars=24, max_lines=3)

    assert len(lines) > 1
    assert lines[0] != long_label[:24] + "…"
    assert all(len(line) <= 25 for line in lines)


def test_large_barchart_splits_to_fit_pdf_pages():
    chart = BarChart(
        labels=[f"дуже довга текстова мітка варіанту відповіді номер {i}" for i in range(80)],
        values=list(range(80, 0, -1)),
        value_labels=[f"{i},0 % · {80 - i}" for i in range(80)],
    )

    assert len(_barchart_flowables(chart)) > 1
    assert _is_pdf(render_pdf(Report(title="Великий графік", blocks=[chart])))


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


def test_fit_page_flowchart_renders_large_graph_without_layout_error():
    nodes = [
        FlowChartNode(f"sec_{index}", f"Section {index}\nQuestion with long routing text")
        for index in range(36)
    ]
    edges = [
        FlowChartEdge(f"sec_{index}", f"sec_{index + 1}", "next", dashed=False)
        for index in range(len(nodes) - 1)
    ]
    chart = FlowChart(nodes=nodes, edges=edges, fit_page=True)
    flowable = _flowchart_flowable(chart)
    wrapped_width, wrapped_height = flowable.wrap(400, 600)

    assert wrapped_width <= 400
    assert wrapped_height <= 600

    rep = Report(
        title="Large flow",
        blocks=[
            PageBreak(),
            chart,
        ],
    )

    assert _is_pdf(render_pdf(rep))


def test_fit_page_flowchart_renders_routed_edges():
    nodes = [
        FlowChartNode("__start__", "Start", kind="start"),
        FlowChartNode("a", "A"),
        FlowChartNode("b", "B"),
        FlowChartNode("c", "C"),
        FlowChartNode("__submit__", "Submit", kind="submit"),
    ]
    chart = FlowChart(
        nodes=nodes,
        edges=[
            FlowChartEdge("__start__", "c", "jump", kind="conditional", dashed=False),
            FlowChartEdge("a", "__submit__", "finish", kind="default", dashed=True),
            FlowChartEdge("b", "a", "back", kind="conditional", dashed=False),
        ],
        fit_page=True,
    )

    assert _is_pdf(render_pdf(Report(title="Routed flow", blocks=[chart])))
