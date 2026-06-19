from __future__ import annotations

from core.forms_api import parse_question_types


def _grid(title, rows, columns, *, qtype="RADIO"):
    return {
        "title": title,
        "questionGroupItem": {
            "grid": {
                "columns": {
                    "type": qtype,
                    "options": [{"value": value} for value in columns],
                }
            },
            "questions": [
                {"questionId": qid, "rowQuestion": {"title": row_title}} for qid, row_title in rows
            ],
        },
    }


def test_parse_question_types_includes_radio_grid_rows_with_options():
    form = {
        "items": [
            _grid(
                "Матриця",
                [("q1", "Рядок 1"), ("q2", "Рядок 2")],
                ["Так", "Ні"],
            )
        ]
    }

    questions = parse_question_types(form)

    assert [(q.id, q.title, q.type, q.options) for q in questions] == [
        ("q1", "Матриця — Рядок 1", "MULTIPLE_CHOICE", ["Так", "Ні"]),
        ("q2", "Матриця — Рядок 2", "MULTIPLE_CHOICE", ["Так", "Ні"]),
    ]


def test_parse_question_types_includes_checkbox_grid_rows_with_options():
    form = {
        "items": [
            _grid(
                "Матриця чекбоксів",
                [("q1", "Рядок")],
                ["A", "B"],
                qtype="CHECKBOX",
            )
        ]
    }

    questions = parse_question_types(form)

    assert len(questions) == 1
    assert questions[0].id == "q1"
    assert questions[0].type == "CHECKBOX"
    assert questions[0].options == ["A", "B"]
