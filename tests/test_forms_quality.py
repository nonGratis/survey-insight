"""Tests for core.forms_quality (design linter + response stats)."""

from __future__ import annotations

from core.forms_quality import (
    LONG_QUESTION_CHARS,
    SORT_ALPHA,
    SORT_FORM_ORDER,
    analyze_form_design,
    analyze_responses,
    anonymize_distribution,
    normalize_label,
    sort_distribution,
)


def _q(qid, title, question, *, required=False):
    q = {"questionId": qid, "required": required, **question}
    return {"title": title, "questionItem": {"question": q}}


_LONG = "Я" * (LONG_QUESTION_CHARS + 5)

FORM = {
    "items": [
        _q(
            "q1",
            "Стать?",
            {"choiceQuestion": {"type": "RADIO", "options": [{"value": "Ч"}, {"value": "Ж"}]}},
            required=True,
        ),
        _q(
            "q2",
            "Оцініть якість та швидкість",
            {"choiceQuestion": {"type": "RADIO", "options": [{"value": "1"}, {"value": "2"}]}},
        ),
        _q(
            "q3",
            _LONG,
            {
                "choiceQuestion": {
                    "type": "DROP_DOWN",
                    "options": [{"value": str(i)} for i in range(13)],
                }
            },
        ),
        _q("q4", "Коментар", {"textQuestion": {"paragraph": True}}),
        _q(
            "q5",
            "Джерела",
            {
                "choiceQuestion": {
                    "type": "CHECKBOX",
                    "options": [{"value": "A"}, {"value": "B"}, {"isOther": True}],
                }
            },
        ),
        {"title": "Секція", "pageBreakItem": {}},  # not a question → skipped
    ]
}


def _design_by_id():
    return {d.question_id: d for d in analyze_form_design(FORM)}


def test_design_parses_only_questions():
    d = _design_by_id()
    assert set(d) == {"q1", "q2", "q3", "q4", "q5"}  # section skipped


def test_design_types_and_options():
    d = _design_by_id()
    assert d["q1"].qtype == "radio" and d["q1"].n_options == 2 and d["q1"].required
    assert d["q3"].qtype == "dropdown" and d["q3"].n_options == 13
    assert d["q4"].qtype == "paragraph" and d["q4"].n_options is None
    assert d["q5"].qtype == "checkbox" and d["q5"].has_other


def test_design_flag_long():
    assert "задовге" in _design_by_id()["q3"].flags


def test_design_flag_double_barreled():
    assert any("подвійне" in f for f in _design_by_id()["q2"].flags)


def test_design_flag_too_many_options():
    assert any("забагато опцій" in f for f in _design_by_id()["q3"].flags)


def test_design_clean_question_no_flags():
    assert _design_by_id()["q1"].flags == []


RESPONSES = [
    {
        "answers": {
            "q1": {"textAnswers": {"answers": [{"value": "Ч"}]}},
            "q4": {"textAnswers": {"answers": [{"value": "добре"}]}},
        }
    },
    {"answers": {"q1": {"textAnswers": {"answers": [{"value": "Ж"}]}}}},
    {"answers": {"q5": {"textAnswers": {"answers": [{"value": "A"}, {"value": "B"}]}}}},
]


def test_responses_distribution_and_nonresponse():
    s = analyze_responses(FORM, RESPONSES)
    assert s["q1"].n_answered == 2 and s["q1"].n_total == 3
    assert s["q1"].distribution == {"Ч": 1, "Ж": 1}
    assert s["q1"].non_response_pct == 33.3


def test_responses_checkbox_multi_values():
    s = analyze_responses(FORM, RESPONSES)
    assert s["q5"].distribution == {"A": 1, "B": 1}  # one respondent, two values


def test_responses_text_median_length():
    s = analyze_responses(FORM, RESPONSES)
    assert s["q4"].is_text
    assert s["q4"].n_answered == 1
    assert s["q4"].text_median_len == float(len("добре"))


def test_responses_empty_list():
    s = analyze_responses(FORM, [])
    assert s["q1"].n_answered == 0 and s["q1"].non_response_pct == 0.0


def test_empty_form_design():
    assert analyze_form_design({}) == []


# --- спільні хелпери розподілу (екран + PDF-звіт) ---------------------------


def test_normalize_label_collapses_space_and_case():
    assert normalize_label("  ФІ  ОТ ") == normalize_label("фі от")


def test_anonymize_collapses_unknown_values():
    dist = {"Ч": 3, "Ж": 2, "вільний текст": 1, "ще інше": 1}
    out = anonymize_distribution(dist, ["Ч", "Ж"], "Інше*")
    assert out == {"Ч": 3, "Ж": 2, "Інше*": 2}


def test_sort_by_count_desc():
    assert [k for k, _ in sort_distribution({"a": 1, "b": 3, "c": 2}, "За величиною")] == [
        "b",
        "c",
        "a",
    ]


def test_sort_alpha_keeps_label_last():
    out = sort_distribution(
        {"Б": 1, "А": 2, "Інше*": 9}, SORT_ALPHA, keep_label_last=True, label_last_value="Інше*"
    )
    assert [k for k, _ in out] == ["А", "Б", "Інше*"]


def test_sort_form_order():
    out = sort_distribution({"z": 1, "a": 1, "m": 1}, SORT_FORM_ORDER, form_options=["m", "z", "a"])
    assert [k for k, _ in out] == ["m", "z", "a"]
