from __future__ import annotations

from core.forms_api import (
    RESPONSE_TIMESTAMPS_FIELDS,
    list_response_timestamps,
    parse_question_types,
)


class _FakeExecute:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class _FakeResponses:
    def __init__(self):
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeExecute({"responses": [{"createTime": "2026-06-01T10:00:00Z"}]})


class _FakeForms:
    def __init__(self, responses):
        self._responses = responses

    def responses(self):
        return self._responses


class _FakeFormsService:
    def __init__(self):
        self.responses_resource = _FakeResponses()

    def forms(self):
        return _FakeForms(self.responses_resource)


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


def test_list_response_timestamps_requests_only_create_time_fields(monkeypatch):
    service = _FakeFormsService()
    monkeypatch.setattr("core.forms_api.build", lambda *args, **kwargs: service)

    timestamps = list_response_timestamps(object(), "form_1")

    assert len(timestamps) == 1
    assert service.responses_resource.calls == [
        {
            "formId": "form_1",
            "pageToken": None,
            "fields": RESPONSE_TIMESTAMPS_FIELDS,
        }
    ]
