from __future__ import annotations

import pytest

from core.response_weights import (
    compute_configured_response_weights,
    weighted_response_distribution,
)
from core.weighting import Dimension


def _response(**answers: list[str]) -> dict:
    return {
        "answers": {
            qid: {"textAnswers": {"answers": [{"value": value} for value in values]}}
            for qid, values in answers.items()
        }
    }


def test_compute_configured_weights_excludes_current_question_dimension() -> None:
    responses = [
        _response(dept=["A"], course=["1"]),
        _response(dept=["B"], course=["1"]),
        _response(dept=["B"], course=["2"]),
    ]
    dims = [
        Dimension("Підрозділ", "dept", {"A": 100, "B": 100}),
        Dimension("Курс", "course", {"1": 150, "2": 50}),
    ]

    weights = compute_configured_response_weights(responses, dims, exclude_column="dept")

    assert weights is not None
    assert len(weights) == len(responses)
    assert weights != compute_configured_response_weights(responses, dims)


def test_compute_configured_weights_returns_none_when_only_self_dimension_left() -> None:
    responses = [_response(dept=["A"]), _response(dept=["B"])]
    dims = [Dimension("Підрозділ", "dept", {"A": 100, "B": 100})]

    assert compute_configured_response_weights(responses, dims, exclude_column="dept") is None


def test_weighted_distribution_normalizes_mean_weight_to_one_for_answered_rows() -> None:
    responses = [
        _response(q=["A"]),
        _response(q=["B"]),
        _response(other=["ignored"]),
    ]

    dist = weighted_response_distribution(responses, "q", [2.0, 4.0, 100.0])

    assert dist is not None
    assert dist.n_answered == 2
    assert dist.denominator == 2.0
    assert dist.distribution["A"] == pytest.approx(2 / 3)
    assert dist.distribution["B"] == pytest.approx(4 / 3)
    assert sum(dist.distribution.values()) == pytest.approx(2.0)


def test_weighted_distribution_checkbox_uses_respondent_denominator() -> None:
    responses = [
        _response(q=["A", "B"]),
        _response(q=["A"]),
    ]

    dist = weighted_response_distribution(responses, "q", [1.0, 1.0])

    assert dist is not None
    assert dist.denominator == 2.0
    assert dist.distribution == {"A": 2.0, "B": 1.0}


def test_weighted_distribution_anonymizes_non_official_options() -> None:
    responses = [
        _response(q=["Так"]),
        _response(q=["ручний текст"]),
    ]

    dist = weighted_response_distribution(
        responses,
        "q",
        [1.0, 1.0],
        allowed_options=["Так", "Ні"],
        anonymize=True,
        anonymized_label="Інше*",
    )

    assert dist is not None
    assert dist.distribution == {"Так": 1.0, "Інше*": 1.0}


def test_weighted_distribution_canonicalizes_free_text_variants() -> None:
    responses = [
        _response(q=["Не знаю"]),
        _response(q=[" не   знаю "]),
    ]

    dist = weighted_response_distribution(responses, "q", [1.0, 1.0])

    assert dist is not None
    assert dist.distribution == {"Не знаю": 2.0}
