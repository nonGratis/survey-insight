"""Tests for core.crosstab_frame — чиста аналітика вкладки «Крос-таби».

Покриває те, що раніше жило в `ui/pages/questions.py` і не тестувалося:
побудова кадру «респондент × змінна», класифікація типів питань і вибір
міри зв'язку для пари змінних.
"""

from __future__ import annotations

import math

from core.crosstab_frame import (
    Var,
    answer_values,
    build_analysis_frame,
    pair_association,
    short_label,
    to_float,
)


def _q(qid, title, question):
    return {"title": title, "questionItem": {"question": {"questionId": qid, **question}}}


def _radio(qid, title, options):
    return _q(qid, title, {"choiceQuestion": {"type": "RADIO", "options": options}})


def _checkbox(qid, title, options):
    return _q(qid, title, {"choiceQuestion": {"type": "CHECKBOX", "options": options}})


def _scale(qid, title):
    return _q(qid, title, {"scaleQuestion": {"low": 1, "high": 5}})


def _text(qid, title):
    return _q(qid, title, {"textQuestion": {}})


def _opts(*values):
    return [{"value": v} for v in values]


def _ans(value):
    """Відповідь з одним значенням (RADIO/SCALE/TEXT)."""
    return {"textAnswers": {"answers": [{"value": value}]}}


def _multi(*values):
    """Відповідь з кількома значеннями (CHECKBOX)."""
    return {"textAnswers": {"answers": [{"value": v} for v in values]}}


def _resp(answers):
    return {"answers": answers}


# --- дрібні чисті хелпери ---------------------------------------------------


def test_to_float_parses_comma_decimal():
    assert to_float("3,5") == 3.5
    assert to_float(" 10 ") == 10.0


def test_to_float_non_number_is_nan():
    assert math.isnan(to_float("так"))
    assert math.isnan(to_float(""))


def test_short_label_truncates_only_long_text():
    assert short_label("коротко") == "коротко"
    long = "ц" * 60
    out = short_label(long)
    assert out.endswith("…") and len(out) == 46


def test_answer_values_handles_missing_and_multi():
    r = _resp({"q1": _ans("A"), "q2": _multi("X", "Y")})
    assert answer_values(r, "q1") == ["A"]
    assert answer_values(r, "q2") == ["X", "Y"]
    assert answer_values(r, "missing") == []


# --- класифікація типів питань у build_analysis_frame -----------------------


def test_nominal_choice_classified_nominal():
    form = {"items": [_radio("q1", "Стать", _opts("Ч", "Ж"))]}
    responses = [_resp({"q1": _ans("Ч")}), _resp({"q1": _ans("Ж")})]
    _frame, variables = build_analysis_frame(form, responses)
    assert [v.kind for v in variables] == ["nominal"]
    assert variables[0].key == "q1"


def test_numeric_choice_classified_ordinal():
    # RADIO, де всі варіанти числові (≥ NUMERIC_FRACTION) → порядкове.
    form = {"items": [_radio("q1", "Оцінка", _opts("1", "2", "3"))]}
    responses = [_resp({"q1": _ans("1")}), _resp({"q1": _ans("2")}), _resp({"q1": _ans("3")})]
    _frame, variables = build_analysis_frame(form, responses)
    assert variables[0].kind == "ordinal"


def test_scale_classified_ordinal():
    form = {"items": [_scale("q1", "Задоволеність")]}
    responses = [_resp({"q1": _ans("4")}), _resp({"q1": _ans("2")})]
    _frame, variables = build_analysis_frame(form, responses)
    assert variables[0].kind == "ordinal"


def test_numeric_text_classified_numeric():
    form = {"items": [_text("q1", "Вік")]}
    responses = [_resp({"q1": _ans(str(v))}) for v in (18, 25, 33, 41, 52)]
    _frame, variables = build_analysis_frame(form, responses)
    assert variables[0].kind == "numeric"


def test_free_text_excluded():
    # Текст без чисел — поза аналізом (вільний коментар).
    form = {"items": [_text("q1", "Коментар")]}
    responses = [_resp({"q1": _ans("добре")}), _resp({"q1": _ans("погано")})]
    _frame, variables = build_analysis_frame(form, responses)
    assert variables == []


def test_checkbox_expands_to_binary_indicators():
    form = {"items": [_checkbox("q1", "Джерела", _opts("Сайт", "Друзі"))]}
    responses = [
        _resp({"q1": _multi("Сайт")}),
        _resp({"q1": _multi("Сайт", "Друзі")}),
        _resp({"q1": _multi("Друзі")}),
    ]
    frame, variables = build_analysis_frame(form, responses)
    keys = {v.key for v in variables}
    assert keys == {"q1::Сайт", "q1::Друзі"}
    assert all(v.kind == "nominal" for v in variables)
    # «так»/«ні» індикатор по обраній опції.
    assert frame["q1::Сайт"].tolist() == ["так", "так", "ні"]
    assert frame["q1::Друзі"].tolist() == ["ні", "так", "так"]


def test_single_level_variable_dropped():
    # Усі відповіли однаково → <2 рівнів → змінна відсіюється.
    form = {"items": [_radio("q1", "Стать", _opts("Ч", "Ж"))]}
    responses = [_resp({"q1": _ans("Ч")}), _resp({"q1": _ans("Ч")})]
    _frame, variables = build_analysis_frame(form, responses)
    assert variables == []


def test_unanswered_question_skipped():
    form = {"items": [_radio("q1", "Стать", _opts("Ч", "Ж"))]}
    responses = [_resp({}), _resp({})]
    _frame, variables = build_analysis_frame(form, responses)
    assert variables == []


def test_date_question_excluded():
    form = {
        "items": [
            _radio("q1", "Стать", _opts("Ч", "Ж")),
            _q("q2", "Дата", {"dateQuestion": {}}),
        ]
    }
    responses = [
        _resp({"q1": _ans("Ч"), "q2": _ans("2026-01-01")}),
        _resp({"q1": _ans("Ж"), "q2": _ans("2026-02-01")}),
    ]
    _frame, variables = build_analysis_frame(form, responses)
    assert [v.key for v in variables] == ["q1"]


# --- вибір міри зв'язку у pair_association ----------------------------------


def _frame_with(meta_kinds, columns):
    import pandas as pd

    meta = {key: Var(key, key, kind) for key, kind in meta_kinds.items()}
    return pd.DataFrame(columns), meta


def test_pair_numeric_uses_pearson():
    frame, meta = _frame_with(
        {"a": "numeric", "b": "numeric"},
        {"a": ["1", "2", "3", "4", "5"], "b": ["2", "4", "6", "8", "10"]},
    )
    pa = pair_association(frame, meta, "a", "b")
    assert pa.measure == "pearson"
    assert pa.effect == 1.0  # ідеальна лінійна залежність
    assert pa.direction == 1.0


def test_pair_numeric_negative_direction():
    frame, meta = _frame_with(
        {"a": "numeric", "b": "numeric"},
        {"a": ["1", "2", "3", "4", "5"], "b": ["10", "8", "6", "4", "2"]},
    )
    pa = pair_association(frame, meta, "a", "b")
    assert pa.measure == "pearson"
    assert pa.direction == -1.0


def test_pair_ordinal_uses_spearman():
    frame, meta = _frame_with(
        {"a": "ordinal", "b": "ordinal"},
        {"a": ["1", "2", "3", "4"], "b": ["1", "2", "3", "4"]},
    )
    pa = pair_association(frame, meta, "a", "b")
    assert pa.measure == "spearman"


def test_pair_nominal_uses_cramers_v():
    frame, meta = _frame_with(
        {"a": "nominal", "b": "nominal"},
        {"a": ["Ч", "Ч", "Ж", "Ж"], "b": ["так", "ні", "так", "ні"]},
    )
    pa = pair_association(frame, meta, "a", "b")
    assert pa.measure == "cramers_v"
    assert pa.direction == 0.0


def test_pair_mixed_nominal_ordinal_uses_cramers_v():
    # Номінальне × порядкове не зводиться до кореляції → χ²/Cramér's V.
    frame, meta = _frame_with(
        {"a": "nominal", "b": "ordinal"},
        {"a": ["Ч", "Ж", "Ч", "Ж"], "b": ["1", "2", "3", "4"]},
    )
    pa = pair_association(frame, meta, "a", "b")
    assert pa.measure == "cramers_v"
