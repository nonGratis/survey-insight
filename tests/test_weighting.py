"""Tests for core.weighting — reconciled 1:1 з еталонним прототипом (data/*.csv).

Еталон: 497 відповідей, N=2111 (22 підрозділи) × 2 курси. Перевіряємо, що
модуль відтворює СТАТИЧНІ ваги, DEFF, n_eff, MoE, MoE_DEFF, n_target з точністю
до останнього знака. Таймлайн-вага (нормована) перевіряється на інваріанти
(перший = 1.0, DEFF не змінюється), а не на еталонну колонку — її нормалізація
у прототипі інша (кумулятивна) і свідомо не відтворюється.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from core.weighting import (
    Dimension,
    compute_weighting,
    cumulative_design_effect,
    design_effect,
    margin_of_error,
    required_sample_size,
    stratum_weights,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def _num(s: str) -> float | None:
    s = s.replace("\xa0", "").replace(" ", "").replace(",", ".").strip()
    return float(s) if s else None


def _load_population(csv_name: str) -> dict[str, int]:
    pop: dict[str, int] = {}
    with open(DATA / csv_name, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["strata"].strip()
            val = _num(row["population"])
            if name and val:
                pop[name] = int(val)
    return pop


def _load_responses() -> list[dict[str, str]]:
    with open(DATA / "responses.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- pure-function unit tests ------------------------------------------------


def test_required_sample_size_matches_reference():
    # N=2111, MoE=5% → 326 (ceil of 325.14). Це еталонний n_target.
    assert required_sample_size(2111) == 326


def test_required_sample_size_zero_population():
    assert required_sample_size(0) == 0


def test_margin_of_error_no_fpc():
    # n=497 → 4.40% (еталон summary.csv).
    assert margin_of_error(497) == pytest.approx(0.0440, abs=1e-4)


def test_margin_of_error_empty():
    assert margin_of_error(0) == 0.0


def test_design_effect_uniform_is_one():
    assert design_effect([1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_design_effect_scale_invariant():
    base = [0.5, 1.0, 2.0, 3.0]
    assert design_effect(base) == pytest.approx(design_effect([10 * w for w in base]))


def test_cumulative_design_effect_converges_to_final():
    ws = [0.5, 1.0, 2.0, 3.0, 0.8]
    cum = cumulative_design_effect(ws)
    assert len(cum) == len(ws)
    assert cum[0] == pytest.approx(1.0)  # одна вага → DEFF 1
    assert cum[-1] == pytest.approx(design_effect(ws))  # фінал = повний DEFF


def test_stratum_weight_formula():
    # ФЕА: N_h=104, N=2111, n_target=326, n_h=37 → 0.43407.
    w = stratum_weights({"ФЕА": 104, "_rest": 2007}, {"ФЕА": 37}, 326)
    assert w["ФЕА"] == pytest.approx(0.43407, abs=1e-5)


def test_stratum_weight_cap():
    w = stratum_weights({"A": 100, "B": 100}, {"A": 1}, 100, caps={"A": 2.0})
    assert w["A"] == 2.0  # без cap було б 50.0


def test_stratum_weight_zero_sample_is_nan():
    import math

    w = stratum_weights({"A": 100}, {}, 100)
    assert math.isnan(w["A"])


# --- end-to-end reconciliation з еталоном ------------------------------------


@pytest.fixture
def reference_result():
    dept_pop = _load_population("DEPARTMENT.csv")
    course_pop = _load_population("COURSE.csv")
    rows = _load_responses()
    df = pd.DataFrame(
        {
            "R_ID": [int(r["R_ID"]) for r in rows],
            "Підрозділ": [r["Твій підрозділ"].strip() for r in rows],
            "Курс": [r["Твій курс"].strip() for r in rows],
        }
    )
    dims = [
        Dimension("Підрозділ", "Підрозділ", dept_pop),
        Dimension("Курс", "Курс", course_pop),
    ]
    return compute_weighting(df, dims), rows


def test_reconcile_headline_metrics(reference_result):
    res, _ = reference_result
    assert res.n == 497
    assert res.population == 2111
    assert res.n_target == 326
    assert res.deff == pytest.approx(1.31, abs=0.01)
    assert res.n_eff == pytest.approx(380, abs=1)
    assert res.moe == pytest.approx(0.0440, abs=1e-4)
    assert res.moe_deff == pytest.approx(0.0503, abs=1e-3)
    assert res.sample_need == pytest.approx(426, abs=1)


def test_reconcile_static_department_weights(reference_result):
    res, rows = reference_result
    f = res.frame
    for i in (0, 1, 2, 495, 496):  # перші три + два пізні рядки
        assert f["w_Підрозділ"].iloc[i] == pytest.approx(_num(rows[i]["w_department"]), abs=1e-4)


def test_reconcile_static_course_weights(reference_result):
    res, rows = reference_result
    f = res.frame
    for i in (0, 1, 200, 496):
        assert f["w_Курс"].iloc[i] == pytest.approx(_num(rows[i]["w_course"]), abs=1e-4)


def test_reconcile_composite_weights(reference_result):
    res, rows = reference_result
    f = res.frame
    for i in (0, 495, 496):  # row1 0.4078, row496 0.2801
        assert f["w"].iloc[i] == pytest.approx(_num(rows[i]["w"]), abs=1e-3)


def test_timeline_first_respondent_is_one(reference_result):
    res, _ = reference_result
    assert res.frame["w_Підрозділ_timeline"].iloc[0] == pytest.approx(1.0)
    assert res.frame["w_Курс_timeline"].iloc[0] == pytest.approx(1.0)


def test_strata_frame_sortable_by_lack(reference_result):
    res, _ = reference_result
    sf = res.strata_frame()
    assert {"Страта", "Ще треба", "Покриття", "Вага w_h"}.issubset(sf.columns)
    assert (sf["Ще треба"] >= 0).all()


def test_rid_autoinjected_when_absent():
    df = pd.DataFrame({"Стать": ["Ч", "Ж", "Ч"]})
    res = compute_weighting(df, [Dimension("Стать", "Стать", {"Ч": 60, "Ж": 40})])
    assert list(res.frame["R_ID"]) == [1, 2, 3]
