from __future__ import annotations

from core.form_flow import START_ID, SUBMIT_ID, flow_to_dot, parse_form_flow


def _page(section_id: str, title: str) -> dict:
    return {"itemId": section_id, "title": title, "pageBreakItem": {}}


def _radio(qid: str, title: str, options: list[dict]) -> dict:
    return {
        "title": title,
        "questionItem": {
            "question": {
                "questionId": qid,
                "choiceQuestion": {"type": "RADIO", "options": options},
            }
        },
    }


def test_parse_form_flow_for_linear_form_has_start_and_submit() -> None:
    flow = parse_form_flow({"items": [_radio("q1", "Q", [{"value": "A"}])]})

    assert [(node.id, node.title, node.kind) for node in flow.nodes] == [
        (START_ID, "Старт", "start"),
        (SUBMIT_ID, "Надіслати", "submit"),
    ]
    assert [(edge.source, edge.target, edge.label, edge.kind) for edge in flow.edges] == [
        (START_ID, SUBMIT_ID, "далі", "default")
    ]
    assert flow.section_count == 1
    assert flow.conditional_edge_count == 0
    assert flow.unreachable_section_ids == []


def test_parse_form_flow_reads_choice_navigation() -> None:
    form = {
        "items": [
            _radio(
                "q1",
                "Куди йти?",
                [
                    {"value": "До другої", "goToSectionId": "sec_2"},
                    {"value": "Завершити", "goToAction": "SUBMIT_FORM"},
                ],
            ),
            _page("sec_2", "Друга секція"),
        ]
    }

    flow = parse_form_flow(form)
    conditional = [edge for edge in flow.edges if edge.kind == "conditional"]

    assert flow.section_count == 2
    assert flow.conditional_edge_count == 2
    start = next(node for node in flow.nodes if node.id == START_ID)
    assert start.detail_lines == ("Куди йти?",)
    assert {(edge.source, edge.target, edge.label) for edge in conditional} == {
        (START_ID, "sec_2", "До другої"),
        (START_ID, SUBMIT_ID, "Завершити"),
    }
    assert flow.terminal_section_ids == [START_ID, "sec_2"]


def test_parse_form_flow_marks_section_unreachable_when_branch_submits_before_it() -> None:
    form = {
        "items": [
            _radio(
                "q1",
                "Завершити одразу?",
                [
                    {"value": "Так", "goToAction": "SUBMIT_FORM"},
                    {"value": "Також так", "goToAction": "SUBMIT_FORM"},
                ],
            ),
            _page("sec_2", "Недосяжна секція"),
        ]
    }

    flow = parse_form_flow(form)

    assert "sec_2" in flow.unreachable_section_ids
    assert (START_ID, "sec_2", "далі", "default") not in [
        (edge.source, edge.target, edge.label, edge.kind) for edge in flow.edges
    ]


def test_parse_form_flow_keeps_default_path_for_unrouted_choice_options() -> None:
    form = {
        "items": [
            _radio(
                "q1",
                "Куди йти?",
                [
                    {"value": "Завершити", "goToAction": "SUBMIT_FORM"},
                    {"value": "Продовжити"},
                ],
            ),
            _page("sec_2", "Досяжна секція"),
        ]
    }

    flow = parse_form_flow(form)

    assert "sec_2" not in flow.unreachable_section_ids
    assert (START_ID, "sec_2", "далі", "default") in [
        (edge.source, edge.target, edge.label, edge.kind) for edge in flow.edges
    ]


def test_parse_form_flow_detects_restart_cycle() -> None:
    form = {
        "items": [
            _radio("q0", "Старт?", [{"value": "До секції", "goToSectionId": "sec_2"}]),
            _page("sec_2", "Друга секція"),
            _radio("q1", "Повторити?", [{"value": "Назад", "goToAction": "RESTART_FORM"}]),
        ]
    }

    flow = parse_form_flow(form)

    assert flow.has_cycles is True


def test_parse_form_flow_does_not_infer_hidden_section_as_reachable() -> None:
    form = {
        "items": [
            _radio("q1", "Твій курс", [{"value": "1 курс"}, {"value": "2 курс"}]),
            _page("utm", "UTM"),
            {
                "title": "param",
                "questionItem": {
                    "question": {
                        "questionId": "q2",
                        "textQuestion": {"paragraph": True},
                    }
                },
            },
        ]
    }

    flow = parse_form_flow(form)

    assert "utm" in flow.unreachable_section_ids
    assert (START_ID, "utm", "далі", "default") not in [
        (edge.source, edge.target, edge.label, edge.kind) for edge in flow.edges
    ]
    assert (START_ID, SUBMIT_ID, "надіслати", "default") in [
        (edge.source, edge.target, edge.label, edge.kind) for edge in flow.edges
    ]


def test_flow_to_dot_uses_compact_rectangles_and_edge_styles() -> None:
    flow = parse_form_flow(
        {
            "items": [
                _radio(
                    "q1",
                    "Q",
                    [
                        {"value": "Завершити", "goToAction": "SUBMIT_FORM"},
                        {"value": "Далі"},
                    ],
                ),
                _page("sec_2", "Друга"),
            ]
        }
    )

    dot = flow_to_dot(flow)

    assert "rankdir=LR" in dot
    assert "shape=rect" in dot
    assert 'style="rounded,filled"' in dot
    assert 'style="dashed"' in dot
    assert 'style="solid"' in dot


def test_flow_to_dot_marks_unreachable_sections() -> None:
    flow = parse_form_flow(
        {
            "items": [
                _radio("q1", "Q", [{"value": "End", "goToAction": "SUBMIT_FORM"}]),
                _page("sec_2", "Dead end"),
            ]
        }
    )

    dot = flow_to_dot(flow)

    assert "#fff1f2" in dot
    assert 'style="rounded,filled,dashed"' in dot


def test_flow_to_dot_uses_graphviz_line_breaks_not_literal_slash_n() -> None:
    flow = parse_form_flow(
        {
            "items": [
                _radio(
                    "q1",
                    "Дуже довге запитання для перевірки переносу",
                    [
                        {
                            "value": "Дуже довгий варіант відповіді",
                            "goToAction": "SUBMIT_FORM",
                        }
                    ],
                )
            ]
        }
    )

    dot = flow_to_dot(flow)

    assert "\\n" in dot
    assert "\\\\n" not in dot
