"""Helpers for applying post-stratification weights to per-question responses."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from core.crosstab_frame import answer_values
from core.forms_quality import normalize_label
from core.weighting import RID_COLUMN, Dimension, compute_weighting


@dataclass(frozen=True)
class WeightedDistribution:
    """Weighted response distribution normalized to the answered respondent count."""

    distribution: dict[str, float]
    denominator: float
    n_answered: int


def response_weight_frame(responses: list[dict], qids: Sequence[str]) -> pd.DataFrame:
    """Build one-answer-per-respondent frame for weighting dimensions."""
    rows: list[dict[str, object]] = []
    for i, resp in enumerate(responses, start=1):
        row: dict[str, object] = {RID_COLUMN: i}
        for qid in qids:
            values = answer_values(resp, qid)
            row[qid] = values[0].strip() if values else ""
        rows.append(row)
    return pd.DataFrame(rows)


def compute_configured_response_weights(
    responses: list[dict],
    dimensions: Sequence[Dimension],
    *,
    exclude_column: str | None = None,
    cap_value: float = 0.0,
    moe_pct: float = 5.0,
) -> list[float] | None:
    """Compute per-respondent weights from saved weighting dimensions.

    ``exclude_column`` implements leave-one-dimension-out: when rendering the
    distribution for the same question used as a stratum, that dimension is
    dropped so the question does not weight itself.
    """
    active_dimensions = [dim for dim in dimensions if dim.column != exclude_column]
    if not active_dimensions:
        return None

    frame = response_weight_frame(responses, [dim.column for dim in active_dimensions])
    caps = None
    if cap_value > 0:
        caps = {
            dim.name: {stratum: cap_value for stratum in dim.population}
            for dim in active_dimensions
        }
    result = compute_weighting(
        frame,
        active_dimensions,
        moe=moe_pct / 100.0,
        caps=caps,
    )
    return [_safe_weight(value) for value in result.frame["w"]]


def weighted_response_distribution(
    responses: list[dict],
    qid: str,
    weights: Sequence[float],
    *,
    allowed_options: Sequence[str] = (),
    anonymize: bool = False,
    anonymized_label: str = "Інше*",
) -> WeightedDistribution | None:
    """Weighted distribution for one question, normalized among answered rows.

    For checkbox questions a respondent's normalized weight is added to every
    selected option. Percent denominators should still use ``denominator``,
    which equals the actual number of respondents who answered the question.
    """
    answered_indices: list[int] = []
    answers_by_index: dict[int, list[str]] = {}
    for index, resp in enumerate(responses):
        values = [value for value in answer_values(resp, qid) if str(value).strip()]
        if not values:
            continue
        answered_indices.append(index)
        answers_by_index[index] = values

    if not answered_indices:
        return None

    raw_answered_weight = sum(_safe_weight(weights[index]) for index in answered_indices)
    if raw_answered_weight <= 0:
        return None

    multiplier = len(answered_indices) / raw_answered_weight
    allowed = {normalize_label(option): option for option in allowed_options}
    grouped: dict[str, dict[str, float | str]] = {}

    for index in answered_indices:
        normalized_weight = _safe_weight(weights[index]) * multiplier
        for raw_value in answers_by_index[index]:
            group_key, label = _weighted_label(
                raw_value,
                allowed,
                anonymize=anonymize,
                anonymized_label=anonymized_label,
            )
            if not group_key or not label:
                continue
            bucket = grouped.setdefault(group_key, {"label": label, "total": 0.0})
            bucket["total"] = float(bucket["total"]) + normalized_weight

    return WeightedDistribution(
        distribution={str(bucket["label"]): float(bucket["total"]) for bucket in grouped.values()},
        denominator=float(len(answered_indices)),
        n_answered=len(answered_indices),
    )


def _weighted_label(
    raw_value: object,
    allowed: dict[str, str],
    *,
    anonymize: bool,
    anonymized_label: str,
) -> tuple[str, str]:
    normalized = normalize_label(str(raw_value))
    if not normalized:
        return "", ""
    canonical = allowed.get(normalized)
    if canonical is not None:
        return normalized, canonical
    if anonymize:
        return normalize_label(anonymized_label), anonymized_label
    return normalized, " ".join(str(raw_value).split())


def _safe_weight(value: object) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 1.0
    return weight if math.isfinite(weight) and weight > 0 else 1.0
