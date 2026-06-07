"""Tests for core.crosstab — звірено зі scipy (χ², Spearman) та інваріантами."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from core.crosstab import (
    PairAssociation,
    association_scan,
    benjamini_hochberg,
    crosstab,
    numeric_correlation,
    odds_ratio_2x2,
    ordinal_correlation,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def _expand(counts: list[list[int]], rows: list[str], cols: list[str]):
    """Розгорнути матрицю частот у дві паралельні послідовності категорій."""
    a, b = [], []
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            a.extend([r] * counts[i][j])
            b.extend([c] * counts[i][j])
    return a, b


# --- χ² / Cramér's V проти scipy (золотий стандарт) -------------------------


def test_chi2_matches_scipy_unweighted():
    counts = [[10, 20, 30], [30, 20, 10]]
    a, b = _expand(counts, ["x", "y"], ["p", "q", "r"])
    res = crosstab(a, b)
    chi2_ref, p_ref, dof_ref, _ = stats.chi2_contingency(np.array(counts), correction=False)
    assert res.chi2 == pytest.approx(chi2_ref, rel=1e-9)
    assert res.dof == dof_ref
    assert res.p_value == pytest.approx(p_ref, rel=1e-9)


def test_cramers_v_formula():
    counts = [[10, 20, 30], [30, 20, 10]]
    a, b = _expand(counts, ["x", "y"], ["p", "q", "r"])
    res = crosstab(a, b)
    n = sum(sum(r) for r in counts)
    expected_v = (res.chi2 / (n * min(2 - 1, 3 - 1))) ** 0.5
    assert res.cramers_v == pytest.approx(expected_v)
    assert 0.0 <= res.cramers_v <= 1.0


def test_expected_matches_scipy():
    counts = [[10, 20, 30], [30, 20, 10]]
    a, b = _expand(counts, ["x", "y"], ["p", "q", "r"])
    res = crosstab(a, b)
    _, _, _, exp_ref = stats.chi2_contingency(np.array(counts), correction=False)
    assert np.allclose(res.expected.to_numpy(), exp_ref)


# --- зважування ------------------------------------------------------------


def test_weighted_reduces_to_unweighted_when_uniform():
    counts = [[10, 20], [30, 15]]
    a, b = _expand(counts, ["x", "y"], ["p", "q"])
    res_u = crosstab(a, b)
    res_w = crosstab(a, b, weights=[1.0] * len(a))
    assert res_w.chi2 == pytest.approx(res_u.chi2)
    assert res_w.deff == pytest.approx(1.0)
    assert res_w.p_value_design == pytest.approx(res_u.p_value)


def test_design_correction_inflates_p_value():
    # Нерівні ваги → DEFF>1 → дизайн-скоригований p НЕ менший за наївний.
    rng = np.random.default_rng(0)
    a = ["x" if v else "y" for v in rng.integers(0, 2, 300)]
    b = ["p" if v else "q" for v in rng.integers(0, 2, 300)]
    w = rng.uniform(0.2, 3.0, 300)
    res = crosstab(a, b, weights=w)
    assert res.deff > 1.0
    assert res.p_value_design >= res.p_value - 1e-12


def test_fisher_exact_on_2x2():
    counts = [[8, 2], [1, 9]]
    a, b = _expand(counts, ["x", "y"], ["p", "q"])
    res = crosstab(a, b)
    _, p_ref = stats.fisher_exact(np.array(counts))
    assert res.fisher_p == pytest.approx(p_ref, rel=1e-6)


def test_low_expected_flag_on_sparse_table():
    counts = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # дуже розріджена
    a, b = _expand(counts, ["x", "y", "z"], ["p", "q", "r"])
    res = crosstab(a, b)
    assert res.low_expected is True


def test_crosstab_empty_raises():
    with pytest.raises(ValueError):
        crosstab(["", ""], ["", ""])


# --- кореляції -------------------------------------------------------------


def test_spearman_matches_scipy_unweighted():
    x = [1, 2, 2, 3, 4, 5, 5, 6]
    y = [2, 1, 3, 4, 3, 5, 6, 5]
    res = ordinal_correlation(x, y)
    rho_ref, _ = stats.spearmanr(x, y)
    assert res.coef == pytest.approx(rho_ref, rel=1e-9)
    assert res.method == "spearman"


def test_pearson_matches_scipy_unweighted():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.1, 5.9, 8.0, 10.2]
    res = numeric_correlation(x, y)
    r_ref, _ = stats.pearsonr(x, y)
    assert res.coef == pytest.approx(r_ref, rel=1e-9)


def test_weighted_pearson_shifts_with_weights():
    x = [1.0, 2.0, 3.0, 4.0, 100.0]
    y = [1.0, 2.0, 3.0, 4.0, -100.0]
    unw = numeric_correlation(x, y).coef
    # Притлумлюємо вагою останню (аномальну) точку → кореляція стає додатною.
    w = [1.0, 1.0, 1.0, 1.0, 0.001]
    wtd = numeric_correlation(x, y, weights=w).coef
    assert wtd > unw


# --- Odds Ratio ------------------------------------------------------------


def test_odds_ratio_2x2():
    counts = [[10, 20], [30, 40]]  # OR = (10*40)/(20*30) = 0.6667
    a, b = _expand(counts, ["exposed", "not"], ["case", "control"])
    res = crosstab(a, b)
    orr = odds_ratio_2x2(res)
    assert orr is not None
    assert orr.odds_ratio == pytest.approx((10 * 40) / (20 * 30))
    assert orr.ci_low < orr.odds_ratio < orr.ci_high


def test_odds_ratio_none_for_non_2x2():
    counts = [[10, 20, 5], [30, 40, 5]]
    a, b = _expand(counts, ["x", "y"], ["p", "q", "r"])
    assert odds_ratio_2x2(crosstab(a, b)) is None


# --- FDR + scan ------------------------------------------------------------


def test_benjamini_hochberg_orders_and_adjusts():
    p_adj, rejected = benjamini_hochberg([0.001, 0.5, 0.04, 0.2])
    assert all(0.0 <= p <= 1.0 for p in p_adj)
    assert rejected[0] is True  # найменше p лишається значущим
    assert p_adj[1] >= 0.001  # велике p тільки зростає


def test_association_scan_sorts_by_effect_and_applies_fdr():
    pairs = [
        PairAssociation("q1", "q2", "cramers_v", effect=0.15, direction=0, n=100, p_raw=0.04),
        PairAssociation("q1", "q3", "cramers_v", effect=0.55, direction=0, n=100, p_raw=0.001),
        PairAssociation("q2", "q3", "spearman", effect=0.30, direction=1, n=100, p_raw=0.2),
    ]
    out = association_scan(pairs)
    assert [pr.effect for pr in out] == [0.55, 0.30, 0.15]  # за спаданням сили
    assert all(not np.isnan(pr.p_fdr) for pr in out)


# --- валідація на реальних даних -------------------------------------------


def _num(s: str) -> float:
    return (
        float(s.replace("\xa0", "").replace(" ", "").replace(",", "."))
        if s.strip()
        else float("nan")
    )


def test_real_data_weighted_crosstab_sane():
    with open(DATA / "responses.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    dept = [r["Твій підрозділ"].strip() for r in rows]
    course = [r["Твій курс"].strip() for r in rows]
    w = [_num(r["w"]) for r in rows]
    res = crosstab(dept, course, weights=w)
    assert res.n == len(rows)
    assert res.chi2 >= 0.0
    assert 0.0 <= res.cramers_v <= 1.0
    assert 0.0 <= res.p_value_design <= 1.0
    assert res.deff > 1.0  # реальні ваги нерівномірні
    assert res.p_value_design >= res.p_value - 1e-12
