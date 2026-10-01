"""Tests for core.report — universal PDF renderer."""

from __future__ import annotations

import pytest

from core.report import (
    BarChart,
    FlowChart,
    FlowChartEdge,
    FlowChartNode,
    Heading,
    Markup,
    Metric,
    Metrics,
    PageBreak,
    Para,
    Report,
    ReportTheme,
    TableBlock,
    _barchart_flowables,
    _collect_heading_entries,
    _ensure_fonts,
    _flowchart_flowable,
    _paragraph,
    _styles,
    _wrap_lines,
    markup,
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
            Para(markup("Зважування коригує перекоси за <b>{}</b> і курсом (їєґ).", "підрозділом")),
            TableBlock(headers=["Страта", "Вага"], rows=[["ФІОТ", "0,499"], ["ФБМІ", "3,552"]]),
        ],
    )
    pdf = render_pdf(report)
    assert _is_pdf(pdf)


# --- текст із форми друкується буквально, а не як розмітка ReportLab ---------

# Відповідь респондента чи назва питання може містити будь-які символи. Раніше
# незакритий тег або «<» перед літерою валили весь експорт, а <img> змушував
# сервер відкривати вказаний ресурс.
_MARKUP_LIKE_TEXTS = [
    "<b>жирний",
    "18<x<25",
    "Tom & Jerry <i>курсив</i>",
    '<img src="file:///no/such/image.png" width="10" height="10"/>',
]

_TEXT_SLOTS = {
    "title": lambda text: Report(title=text),
    "subtitle": lambda text: Report(title="T", subtitle=text),
    "heading-1": lambda text: Report(title="T", blocks=[Heading(text, level=1)]),
    "heading-2": lambda text: Report(title="T", blocks=[Heading(text, level=2)]),
    "paragraph": lambda text: Report(title="T", blocks=[Para(text)]),
    "metric-label": lambda text: Report(title="T", blocks=[Metrics([Metric(text, "1")])]),
    "metric-value": lambda text: Report(title="T", blocks=[Metrics([Metric("n", text)])]),
    "table-header": lambda text: Report(
        title="T", blocks=[TableBlock(headers=[text, "N"], rows=[["a", "1"]])]
    ),
    "table-cell": lambda text: Report(
        title="T", blocks=[TableBlock(headers=["Відповідь", "N"], rows=[[text, "1"]])]
    ),
    "barchart-label": lambda text: Report(
        title="T", blocks=[BarChart(labels=[text], values=[1], value_labels=[text])]
    ),
    "flowchart-label": lambda text: Report(
        title="T",
        blocks=[
            FlowChart(
                nodes=[FlowChartNode("a", text), FlowChartNode("b", "B")],
                edges=[FlowChartEdge("a", "b", text)],
            )
        ],
    ),
}


@pytest.mark.parametrize("slot", sorted(_TEXT_SLOTS))
@pytest.mark.parametrize("text", _MARKUP_LIKE_TEXTS)
def test_markup_like_text_renders_in_every_text_slot(slot: str, text: str):
    assert _is_pdf(render_pdf(_TEXT_SLOTS[slot](text)))


@pytest.mark.parametrize("text", _MARKUP_LIKE_TEXTS)
def test_plain_text_is_printed_literally(text: str):
    _ensure_fonts()
    paragraph = _paragraph(text, _styles()["body"])

    assert paragraph.getPlainText() == text


def test_markup_keeps_our_tags_and_escapes_the_values():
    _ensure_fonts()
    text = markup("Бракує <b>{}</b> у {name}", "<5", name="A&B")

    assert isinstance(text, Markup)
    assert text == "Бракує <b>&lt;5</b> у A&amp;B"
    # Теги шаблону спрацювали як розмітка, значення надруковані буквально.
    assert _paragraph(text, _styles()["body"]).getPlainText() == "Бракує <5 у A&B"


def test_report_theme_can_be_customized():
    theme = ReportTheme(
        primary="#0f766e",
        primary_dark="#115e59",
        table_header_bg="#115e59",
        chart_bar="#0f766e",
        title_bg="#ecfdf5",
        title_border="#99f6e4",
    )
    report = Report(
        title="Theme",
        subtitle="custom",
        theme=theme,
        blocks=[
            Heading("Metrics", level=2),
            Metrics([Metric("n", "42")]),
            TableBlock(headers=["A", "B"], rows=[["x", "y"]]),
            BarChart(labels=["A"], values=[1]),
        ],
    )

    assert _is_pdf(render_pdf(report))


def test_collect_heading_entries_includes_all_headings_and_clamps_levels():
    entries = _collect_heading_entries(
        [
            Heading("Top", level=1),
            Heading("Deep without parent", level=3),
            Heading("Question", level=2),
        ]
    )

    assert [entry.text for entry in entries] == ["Top", "Deep without parent", "Question"]
    assert [entry.level for entry in entries] == [0, 1, 1]
    assert len({entry.key for entry in entries}) == 3


def test_pdf_has_interactive_toc_and_outline():
    fitz = pytest.importorskip("fitz")
    pdf = render_pdf(
        Report(
            title="Report",
            subtitle="with toc",
            blocks=[
                Heading("Overview", level=2),
                Para("Body"),
                Heading("Question 1", level=2),
                Para("Answer"),
                Heading("Details", level=3),
                Para("More"),
            ],
        )
    )
    doc = fitz.open(stream=pdf, filetype="pdf")

    toc = doc.get_toc()
    assert [row[1] for row in toc] == ["Overview", "Question 1", "Details"]
    assert [row[0] for row in toc] == [1, 1, 2]
    assert doc[1].get_links()


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
