"""Tests for core.reports — domain report builders."""

from __future__ import annotations

import pandas as pd

from core.form_flow import (
    START_ID,
    SUBMIT_ID,
    flow_edge_style,
    flow_node_label,
    parse_form_flow,
)
from core.report import (
    BarChart,
    FlowChart,
    Metric,
    Metrics,
    PageBreak,
    Report,
    TableBlock,
    render_pdf,
)
from core.reports import (
    DescriptiveConfig,
    OverviewConfig,
    associations_section,
    descriptive_section,
    dynamics_section,
    full_report,
    overview_section,
    questions_report,
    representativeness_report,
)
from core.weighting import Dimension, compute_weighting


def _q(qid, title, question):
    return {"title": title, "questionItem": {"question": {"questionId": qid, **question}}}


_STRUCTURE = {
    "items": [
        _q(
            "q1",
            "Стать?",
            {"choiceQuestion": {"type": "RADIO", "options": [{"value": "Ч"}, {"value": "Ж"}]}},
        ),
        _q("q2", "Коментар", {"textQuestion": {"paragraph": True}}),
    ]
}
_RESPONSES = [
    {
        "answers": {
            "q1": {"textAnswers": {"answers": [{"value": "Ч"}]}},
            "q2": {"textAnswers": {"answers": [{"value": "добре"}]}},
        }
    },
    {"answers": {"q1": {"textAnswers": {"answers": [{"value": "Ж"}]}}}},
]


def _sample_result():
    df = pd.DataFrame({"Підрозділ": ["A", "A", "B", "B", "B", "C"]})
    dims = [Dimension("Підрозділ", "Підрозділ", {"A": 100, "B": 100, "C": 100})]
    return compute_weighting(df, dims)


def test_builds_report_structure():
    rep = representativeness_report(_sample_result(), form_title="Тест")
    assert isinstance(rep, Report)
    assert rep.title == "Звіт про репрезентативність вибірки"
    assert "Тест" in rep.subtitle
    assert any(isinstance(b, Metrics) for b in rep.blocks)
    table = next(b for b in rep.blocks if isinstance(b, TableBlock))
    assert table.headers[0] == "Вимір"
    assert len(table.rows) == 3  # 3 страти A/B/C


def test_table_sorted_by_lack_desc():
    rep = representativeness_report(_sample_result())
    table = next(b for b in rep.blocks if isinstance(b, TableBlock))
    lacks = [float(row[-1].replace(",", ".")) for row in table.rows]
    assert lacks == sorted(lacks, reverse=True)


def test_report_renders_to_pdf():
    pdf = render_pdf(representativeness_report(_sample_result(), form_title="Демо"))
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 800


def test_questions_report_page_per_question():
    rep = questions_report(_STRUCTURE, _RESPONSES, form_title="Анкета")
    assert rep.title == "Звіт за результатами опитування"
    # один розрив сторінки перед кожним питанням (2 питання)
    assert sum(isinstance(b, PageBreak) for b in rep.blocks) == 2
    assert any(isinstance(b, Metrics) for b in rep.blocks)  # титульні характеристики


def test_questions_report_renders_to_pdf():
    pdf = render_pdf(questions_report(_STRUCTURE, _RESPONSES))
    assert pdf[:5] == b"%PDF-"


def test_questions_report_empty_responses():
    rep = questions_report(_STRUCTURE, [])
    assert render_pdf(rep)[:5] == b"%PDF-"


# --- секційні білдери + конфіг ----------------------------------------------

_OPEN_STRUCTURE = {
    "items": [
        _q(
            "q1",
            "Звідки дізнались?",
            {"choiceQuestion": {"type": "RADIO", "options": [{"value": "Сайт"}]}},
        )
    ]
}
_OPEN_RESPONSES = [
    {"answers": {"q1": {"textAnswers": {"answers": [{"value": v}]}}}}
    for v in ["Сайт", "знайомі", "випадково", "знайомі"]
]


def test_descriptive_chart_mode_emits_barchart():
    blocks = descriptive_section(_STRUCTURE, _RESPONSES, DescriptiveConfig(render_mode="chart"))
    chart = next(b for b in blocks if isinstance(b, BarChart))
    assert chart.value_labels == ["50,0 % · 1", "50,0 % · 1"]
    assert not any(isinstance(b, TableBlock) for b in blocks)


def test_descriptive_table_mode_emits_table():
    blocks = descriptive_section(_STRUCTURE, _RESPONSES, DescriptiveConfig(render_mode="table"))
    assert any(isinstance(b, TableBlock) for b in blocks)
    assert not any(isinstance(b, BarChart) for b in blocks)


def test_descriptive_anonymize_collapses_open_values():
    # «знайомі»/«випадково» поза варіантами → згортаються в «Інше*».
    blocks = descriptive_section(
        _OPEN_STRUCTURE, _OPEN_RESPONSES, DescriptiveConfig(anonymize=True, render_mode="table")
    )
    table = next(b for b in blocks if isinstance(b, TableBlock))
    labels = {row[0] for row in table.rows}
    assert "знайомі" not in labels and "Інше*" in labels


def test_descriptive_hide_only_other_skips_question():
    blocks = descriptive_section(
        {
            "items": [
                _q("q1", "Q", {"choiceQuestion": {"type": "RADIO", "options": [{"value": "X"}]}})
            ]
        },
        [{"answers": {"q1": {"textAnswers": {"answers": [{"value": "не з варіантів"}]}}}}],
        DescriptiveConfig(anonymize=True, hide_only_other=True),
    )
    assert blocks == []  # питання приховано


def test_descriptive_section_can_weight_question_results():
    structure = {
        "items": [
            _q(
                "dept",
                "Підрозділ",
                {"choiceQuestion": {"type": "RADIO", "options": [{"value": "A"}, {"value": "B"}]}},
            ),
            _q(
                "support",
                "Підтримка?",
                {
                    "choiceQuestion": {
                        "type": "RADIO",
                        "options": [{"value": "Так"}, {"value": "Ні"}],
                    }
                },
            ),
        ]
    }
    responses = [
        {
            "answers": {
                "dept": {"textAnswers": {"answers": [{"value": "A"}]}},
                "support": {"textAnswers": {"answers": [{"value": "Так"}]}},
            }
        },
        {
            "answers": {
                "dept": {"textAnswers": {"answers": [{"value": "B"}]}},
                "support": {"textAnswers": {"answers": [{"value": "Ні"}]}},
            }
        },
        {
            "answers": {
                "dept": {"textAnswers": {"answers": [{"value": "B"}]}},
                "support": {"textAnswers": {"answers": [{"value": "Ні"}]}},
            }
        },
    ]

    blocks = descriptive_section(
        structure,
        responses,
        DescriptiveConfig(
            render_mode="table",
            weighting_dimensions=(Dimension("Підрозділ", "dept", {"A": 100, "B": 100}),),
        ),
    )

    assert any("зважено" in getattr(block, "text", "") for block in blocks)
    support_table = [block for block in blocks if isinstance(block, TableBlock)][1]
    assert support_table.rows == [["Ні", "1,5", "50,0 %"], ["Так", "1,5", "50,0 %"]]


def test_descriptive_section_weighted_chart_labels_include_percent_and_count():
    structure = {
        "items": [
            _q(
                "dept",
                "Підрозділ",
                {"choiceQuestion": {"type": "RADIO", "options": [{"value": "A"}, {"value": "B"}]}},
            ),
            _q(
                "support",
                "Підтримка?",
                {
                    "choiceQuestion": {
                        "type": "RADIO",
                        "options": [{"value": "Так"}, {"value": "Ні"}],
                    }
                },
            ),
        ]
    }
    responses = [
        {
            "answers": {
                "dept": {"textAnswers": {"answers": [{"value": "A"}]}},
                "support": {"textAnswers": {"answers": [{"value": "Так"}]}},
            }
        },
        {
            "answers": {
                "dept": {"textAnswers": {"answers": [{"value": "B"}]}},
                "support": {"textAnswers": {"answers": [{"value": "Ні"}]}},
            }
        },
        {
            "answers": {
                "dept": {"textAnswers": {"answers": [{"value": "B"}]}},
                "support": {"textAnswers": {"answers": [{"value": "Ні"}]}},
            }
        },
    ]

    blocks = descriptive_section(
        structure,
        responses,
        DescriptiveConfig(
            render_mode="chart",
            weighting_dimensions=(Dimension("Підрозділ", "dept", {"A": 100, "B": 100}),),
        ),
    )

    support_chart = [block for block in blocks if isinstance(block, BarChart)][1]
    assert support_chart.value_labels == ["50,0 % · 1,5", "50,0 % · 1,5"]


def test_overview_section_can_include_flagged_questions_table():
    blocks = overview_section(
        {
            "items": [
                _q(
                    "q1",
                    "Дуже довге питання " + "x" * 130,
                    {"choiceQuestion": {"type": "RADIO", "options": [{"value": "Так"}]}},
                )
            ]
        },
        _RESPONSES,
        OverviewConfig(question_table_mode="flagged", include_flow_map=False),
    )

    assert any(getattr(block, "text", "") == "Огляд форми" for block in blocks)
    table = next(block for block in blocks if isinstance(block, TableBlock))
    assert table.headers == ("Запитання", "Тип", "Опцій", "Обов'язк.", "Прапори")
    assert len(table.rows) == 1


def test_overview_section_can_include_all_questions_table():
    blocks = overview_section(
        _STRUCTURE,
        _RESPONSES,
        OverviewConfig(question_table_mode="all", include_flow_map=False),
    )

    table = next(block for block in blocks if isinstance(block, TableBlock))
    assert len(table.rows) == 2
    assert {row[0] for row in table.rows} == {"Стать?", "Коментар"}


def test_overview_section_settings_can_hide_audit_table():
    blocks = overview_section(
        _STRUCTURE,
        _RESPONSES,
        OverviewConfig(question_table_mode="none", include_flow_map=False),
    )

    assert any(isinstance(block, Metrics) for block in blocks)
    assert not any(isinstance(block, TableBlock) for block in blocks)


def test_overview_section_can_include_flow_map():
    structure = {
        "items": [
            _q(
                "q1",
                "Маршрут?",
                {
                    "choiceQuestion": {
                        "type": "RADIO",
                        "options": [{"value": "Далі", "goToSectionId": "sec_2"}],
                    }
                },
            ),
            {"itemId": "sec_2", "title": "Друга секція", "pageBreakItem": {}},
        ]
    }
    blocks = overview_section(
        structure, [], OverviewConfig(question_table_mode="none", include_flow_map=True)
    )

    chart_index = next(index for index, block in enumerate(blocks) if isinstance(block, FlowChart))
    assert isinstance(blocks[chart_index - 2], PageBreak)
    assert getattr(blocks[chart_index - 1], "level", None) == 3
    assert blocks[chart_index].fit_page is True


def test_overview_section_does_not_create_flow_map_page_when_absent():
    blocks = overview_section(
        _STRUCTURE,
        _RESPONSES,
        OverviewConfig(question_table_mode="none", include_flow_map=True),
    )

    assert not any(isinstance(block, FlowChart) for block in blocks)
    assert not any(isinstance(block, PageBreak) for block in blocks)


def test_overview_flow_map_reuses_form_flow_labels_and_styles():
    structure = {
        "items": [
            _q(
                "q1",
                "Перше питання маршруту",
                {
                    "choiceQuestion": {
                        "type": "RADIO",
                        "options": [{"value": "До другої", "goToSectionId": "sec_2"}],
                    }
                },
            ),
            _q(
                "q2",
                "Друге питання маршруту",
                {
                    "choiceQuestion": {
                        "type": "RADIO",
                        "options": [{"value": "Фініш", "goToAction": "SUBMIT_FORM"}],
                    }
                },
            ),
            {"itemId": "sec_2", "title": "Друга секція", "pageBreakItem": {}},
        ]
    }

    flow = parse_form_flow(structure)
    blocks = overview_section(
        structure, [], OverviewConfig(question_table_mode="none", include_flow_map=True)
    )
    chart = next(block for block in blocks if isinstance(block, FlowChart))
    chart_nodes = {node.id: node for node in chart.nodes}
    chart_edges = {(edge.source, edge.target, edge.label): edge for edge in chart.edges}

    start = next(node for node in flow.nodes if node.id == START_ID)
    assert chart_nodes[START_ID].label == flow_node_label(start)
    assert "Перше питання маршруту" in chart_nodes[START_ID].label
    assert "Друге питання маршруту" in chart_nodes[START_ID].label

    submit_flow_edge = next(edge for edge in flow.edges if edge.target == SUBMIT_ID)
    submit_style = flow_edge_style(submit_flow_edge, flow)
    submit_chart_edge = chart_edges[
        (submit_flow_edge.source, submit_flow_edge.target, submit_style.label)
    ]
    assert submit_chart_edge.color == submit_style.color
    assert submit_chart_edge.font_color == submit_style.font_color
    assert submit_chart_edge.pen_width == submit_style.pen_width
    assert submit_chart_edge.dashed is True


def test_associations_section_empty_and_filled():
    assert any("Недостатньо" in getattr(b, "text", "") for b in associations_section([]))
    rows = [["A", "B", "Cramér's V", "0,42", "помірний", "0,001"]]
    assert any(isinstance(b, TableBlock) for b in associations_section(rows))


def test_full_report_composes_sections_with_breaks():
    sec1 = [Metrics(items=[Metric("n", "3")])]
    sec2 = dynamics_section([Metric("Зараз", "100")], note="без нової агітації")
    rep = full_report("Повний звіт", "Форма: X", [sec1, sec2])
    assert isinstance(rep, Report)
    assert sum(isinstance(b, PageBreak) for b in rep.blocks) == 1  # один розрив між 2 секціями
    assert render_pdf(rep)[:5] == b"%PDF-"
