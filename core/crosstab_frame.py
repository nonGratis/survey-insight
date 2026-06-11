"""Підготовка даних крос-аналізу: кадр «респондент × змінна» + типізація.

Чиста (без Streamlit та I/O) частина вкладки «Крос-таби»: із сирих відповідей
Forms API будує таблицю «респондент × змінна», класифікує кожну змінну
(номінальна / порядкова / числова / бінарні індикатори CHECKBOX) і рахує міру
зв'язку для пари змінних. Власне рендер (метрики, графіки) лишається у
`ui/pages/questions.py`; статистика — у `core.crosstab`.

Типи питань беремо з канонічного парсера `core.forms_api.parse_question_types`,
аби не дублювати розбір структури форми (DRY).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.crosstab import (
    PairAssociation,
    crosstab,
    numeric_correlation,
    ordinal_correlation,
)
from core.forms_api import parse_question_types

NUMERIC_FRACTION = 0.8  # частка числових відповідей, аби питання вважати числовим
MAX_LABEL = 45  # обрізання довгих формулювань у підписах змінних

# Типи питань, придатні для крос-аналізу (решта — DATE/TIME/UNKNOWN — поза ним).
_ANALYZABLE = ("MULTIPLE_CHOICE", "CHECKBOX", "LINEAR_SCALE", "SHORT_ANSWER")


@dataclass
class Var:
    """Аналітична змінна крос-аналізу (питання або бінарна опція checkbox)."""

    key: str  # унікальний ключ колонки у frame
    label: str  # людська назва
    kind: str  # "nominal" | "ordinal" | "numeric"


def short_label(text: str) -> str:
    """Обрізати довге формулювання до MAX_LABEL символів (для підписів)."""
    return (text[:MAX_LABEL] + "…") if len(text) > MAX_LABEL else text


def to_float(value: str) -> float:
    """Число з відповіді (кома як десятковий роздільник) або NaN, якщо не число."""
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        return math.nan


def answer_values(resp: dict[str, Any], qid: str) -> list[str]:
    """Усі текстові значення відповіді респондента на питання.

    Для CHECKBOX їх кілька (по одному на обрану опцію), для решти типів — одне.
    """
    ans = resp.get("answers", {}).get(qid, {})
    return [a.get("value", "") for a in ans.get("textAnswers", {}).get("answers", [])]


def build_analysis_frame(
    form: dict[str, Any], responses: list[dict[str, Any]]
) -> tuple[pd.DataFrame, list[Var]]:
    """Кадр «респондент × змінна» + типізація для крос-аналізу.

    Числові choice/шкали → порядкові; інші choice → номінальні; CHECKBOX →
    бінарні індикатори по кожній опції; текст із числами → числове. Вільний
    текст і дати — поза аналізом. Змінні з <2 рівнями відсіюються.
    """
    cols: dict[str, list[str]] = {}
    variables: list[Var] = []
    for q in parse_question_types(form):
        if q.type not in _ANALYZABLE:
            continue
        per_resp = [answer_values(r, q.id) for r in responses]
        answered = [bool(v) for v in per_resp]
        if not any(answered):
            continue

        if q.type == "CHECKBOX":
            for opt in q.options:
                if not opt:
                    continue
                key = f"{q.id}::{opt}"
                cols[key] = [
                    ("так" if opt in vals else "ні") if ans else ""
                    for vals, ans in zip(per_resp, answered, strict=True)
                ]
                variables.append(Var(key, f"{short_label(q.title)} → {opt}", "nominal"))
            continue

        first = [vals[0] if vals else "" for vals in per_resp]
        non_empty = [v for v in first if v.strip()]
        if not non_empty:
            continue
        numeric_share = sum(not math.isnan(to_float(v)) for v in non_empty) / len(non_empty)

        if q.type == "LINEAR_SCALE" or (
            q.type == "MULTIPLE_CHOICE" and numeric_share >= NUMERIC_FRACTION
        ):
            kind = "ordinal"
        elif q.type == "SHORT_ANSWER":
            if numeric_share < NUMERIC_FRACTION:
                continue  # вільний текст — поза аналізом
            kind = "numeric"
        else:
            kind = "nominal"
        cols[q.id] = first
        variables.append(Var(q.id, short_label(q.title), kind))

    frame = pd.DataFrame(cols)
    variables = [v for v in variables if frame[v.key].replace("", pd.NA).nunique() >= 2]
    return frame, variables


def pair_association(
    frame: pd.DataFrame,
    meta: dict[str, Var],
    k1: str,
    k2: str,
    weights: Sequence[float] | None = None,
) -> PairAssociation:
    """Міра зв'язку для пари змінних; тип міри обирається за kind обох.

    числове × числове → Pearson r; (порядкове/числове)² → Spearman ρ;
    решта → χ²/Cramér's V (із дизайн-скоригованим p). Знак кореляції несе
    напрям; для номінальних напряму немає (0).
    """
    t1, t2 = meta[k1].kind, meta[k2].kind
    if t1 == "numeric" and t2 == "numeric":
        cr = numeric_correlation(frame[k1].map(to_float), frame[k2].map(to_float), weights)
        return PairAssociation(
            k1, k2, "pearson", abs(cr.coef), math.copysign(1, cr.coef), cr.n, cr.p_value
        )
    if t1 in ("ordinal", "numeric") and t2 in ("ordinal", "numeric"):
        cr = ordinal_correlation(frame[k1].map(to_float), frame[k2].map(to_float), weights)
        return PairAssociation(
            k1, k2, "spearman", abs(cr.coef), math.copysign(1, cr.coef), cr.n, cr.p_value
        )
    ct = crosstab(frame[k1], frame[k2], weights)
    return PairAssociation(
        k1,
        k2,
        "cramers_v",
        ct.cramers_v,
        0.0,
        ct.n,
        ct.p_value_design,
        low_expected=ct.low_expected,
    )
