"""Tests for core.reports — domain report builders."""

from __future__ import annotations

import pandas as pd

from core.report import BarChart, Metric, Metrics, PageBreak, Report, TableBlock, render_pdf
from core.reports import (
    DescriptiveConfig,
    associations_section,
    descriptive_section,
    dynamics_section,
    full_report,
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
    assert any(isinstance(b, BarChart) for b in blocks)
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
