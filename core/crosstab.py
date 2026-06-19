"""Крос-табуляції та міри зв'язку між питаннями анкети.

Призначення: для пари питань побудувати таблицю спряженості й оцінити, чи є
між ними зв'язок та якої сили — з урахуванням пост-стратифікаційних ваг.

Типозалежний вибір міри (категоріальні дані Google Forms — переважно
номінальні варіанти, шкали Лікерта, мультивибір):

  номінальне × номінальне   → χ² Пірсона + Cramér's V (+ Fisher для 2×2)
  порядкове × порядкове      → Spearman ρ (рангова кореляція)
  бінарне × бінарне (2×2)    → Odds Ratio
  числове × числове          → Pearson r

Зважування (Rao-Scott першого порядку): частоти в клітинках — зважені частки,
а статистика χ² ділиться на дизайн-ефект DEFF (Kish), щоб не занижувати
p-значення через нерівні ваги. Без ваг (w≡1) DEFF=1 і всі формули зводяться
до класичних — єдиний кодовий шлях.

Сканування всіх пар (`association_scan`) застосовує поправку Бенджаміні-Хохберга
(FDR): при ~190 парах без неї ~10 «значущих» зв'язків були б випадковими.

Акцент на РОЗМІР ЕФЕКТУ, не на p: при n≈500 майже все «значуще», тож пари
ранжуються за Cramér's V / |ρ|, а не за p-значенням.

Чисті функції без I/O. Категорії приходять уже як вирівняні послідовності
значень (один елемент на респондента, пропуски — порожній рядок).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from core.weighting import design_effect

# --- іменовані пороги (Cochran, Cohen/Cramér — не «магічні») ----------------
MIN_EXPECTED = 5.0  # очікувана частота в клітинці, нижче якої χ² ненадійний
LOW_EXPECTED_FRACTION = 0.2  # частка клітинок з E<5, за якої таблиця «розріджена»
MIN_EXPECTED_HARD = 1.0  # жодна клітинка не має мати E<1 (правило Кокрена)
HALDANE_CORRECTION = 0.5  # поправка нульових клітинок для Odds Ratio
FDR_ALPHA = 0.05  # рівень значущості для BH-FDR
Z_95 = 1.959963985  # квантиль N(0,1) для 95% CI
IMPORTANT_EFFECT_THRESHOLD = 0.15
STRONG_EFFECT_THRESHOLD = 0.30
ASSOCIATION_FILTER_MODES = ("Важливі", "Значущі", "Сильні", "Ненадійні", "Усі")

# Межі сили зв'язку (конвенційні; названі, з примітками у звіті).
_EFFECT_BANDS = ((0.1, "немає"), (0.3, "слабкий"), (0.5, "помірний"), (1.01, "сильний"))


def _effect_label(value: float) -> str:
    """Якісна оцінка сили зв'язку за абсолютним значенням міри (0..1)."""
    v = abs(value)
    for threshold, label in _EFFECT_BANDS:
        if v < threshold:
            return label
    return "сильний"


@dataclass
class CrosstabResult:
    """Результат аналізу однієї пари категоріальних питань."""

    table: pd.DataFrame  # рядки × стовпці: зважені (або сирі) частоти
    row_labels: list[str]
    col_labels: list[str]
    n: int  # к-сть респондентів з відповіддю на ОБИДВА питання
    n_weighted: float  # сума ваг цих респондентів
    chi2: float  # статистика χ² (на зважених частках × n)
    dof: int
    cramers_v: float  # розмір ефекту 0..1
    p_value: float  # p без урахування дизайну (χ²)
    deff: float  # дизайн-ефект ваг (Kish); 1.0 якщо незважено
    p_value_design: float  # p за Rao-Scott 1-го порядку (χ²/DEFF)
    expected: pd.DataFrame  # очікувані частоти за незалежності
    low_expected: bool  # таблиця розріджена → χ² ненадійний
    fisher_p: float | None = None  # точний тест Фішера (лише 2×2)

    @property
    def effect_label(self) -> str:
        return _effect_label(self.cramers_v)

    @property
    def significant(self) -> bool:
        """Значущий за дизайн-скоригованим p (рекомендований criterion)."""
        return self.p_value_design < FDR_ALPHA


def _clean_pairs(
    a: Sequence[str], b: Sequence[str], weights: Sequence[float] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Лишити лише респондентів з непорожньою відповіддю на ОБИДВА питання."""
    a_arr = np.array([str(x).strip() for x in a], dtype=object)
    b_arr = np.array([str(x).strip() for x in b], dtype=object)
    w_arr = np.ones(len(a_arr)) if weights is None else np.asarray(weights, dtype=float)
    mask = (a_arr != "") & (b_arr != "") & np.isfinite(w_arr)
    return a_arr[mask], b_arr[mask], w_arr[mask]


def crosstab(
    a: Sequence[str],
    b: Sequence[str],
    weights: Sequence[float] | None = None,
) -> CrosstabResult:
    """Таблиця спряженості пари питань + χ², Cramér's V, Rao-Scott p.

    Зважена статистика рахується на зважених частках, помножених на фактичне
    (незважене) n, тож при w≡1 точно зводиться до класичного χ² Пірсона.
    Дизайн-скоригований p = chi2.sf(χ²/DEFF, dof).
    """
    a_arr, b_arr, w_arr = _clean_pairs(a, b, weights)
    n = len(a_arr)
    if n == 0:
        raise ValueError("немає респондентів з відповіддю на обидва питання")

    # Зважена таблиця частот (рядки × стовпці).
    table = (
        pd.DataFrame({"a": a_arr, "b": b_arr, "w": w_arr})
        .pivot_table(index="a", columns="b", values="w", aggfunc="sum", fill_value=0.0)
        .sort_index()
        .sort_index(axis=1)
    )
    row_labels = [str(x) for x in table.index]
    col_labels = [str(x) for x in table.columns]
    obs = table.to_numpy(dtype=float)
    n_weighted = float(obs.sum())

    # Частки та очікувані частки за незалежності.
    p = obs / n_weighted
    row_marg = p.sum(axis=1, keepdims=True)
    col_marg = p.sum(axis=0, keepdims=True)
    exp_p = row_marg @ col_marg
    r, c = obs.shape
    dof = (r - 1) * (c - 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(exp_p > 0, (p - exp_p) ** 2 / exp_p, 0.0)
    chi2 = float(n * terms.sum())

    denom = n * max(min(r - 1, c - 1), 1)
    cramers_v = math.sqrt(chi2 / denom) if denom > 0 and dof > 0 else 0.0
    p_value = float(stats.chi2.sf(chi2, dof)) if dof > 0 else 1.0

    deff = design_effect(w_arr.tolist())
    chi2_rs = chi2 / deff if deff > 0 else chi2
    p_value_design = float(stats.chi2.sf(chi2_rs, dof)) if dof > 0 else 1.0

    expected = pd.DataFrame(exp_p * n, index=table.index, columns=table.columns)
    exp_vals = expected.to_numpy()
    low_expected = bool(
        (exp_vals < MIN_EXPECTED_HARD).any()
        or ((exp_vals < MIN_EXPECTED).mean() > LOW_EXPECTED_FRACTION)
    )

    fisher_p: float | None = None
    if r == 2 and c == 2:
        # Фішер на ОКРУГЛЕНИХ зважених частотах (точний тест вимагає цілих).
        counts = np.rint(obs).astype(int)
        try:
            _, fisher_p = stats.fisher_exact(counts)
            fisher_p = float(fisher_p)
        except ValueError:
            fisher_p = None

    return CrosstabResult(
        table=table,
        row_labels=row_labels,
        col_labels=col_labels,
        n=n,
        n_weighted=n_weighted,
        chi2=chi2,
        dof=dof,
        cramers_v=cramers_v,
        p_value=p_value,
        deff=deff,
        p_value_design=p_value_design,
        expected=expected,
        low_expected=low_expected,
        fisher_p=fisher_p,
    )


@dataclass
class CorrelationResult:
    """Рангова/лінійна кореляція пари числових або порядкових питань."""

    method: str  # "spearman" | "pearson"
    coef: float  # коефіцієнт −1..1 (знак = напрям)
    p_value: float
    n: int

    @property
    def effect_label(self) -> str:
        return _effect_label(self.coef)


def _weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Зважений коефіцієнт кореляції Пірсона."""
    wsum = w.sum()
    mx = (w * x).sum() / wsum
    my = (w * y).sum() / wsum
    cov = (w * (x - mx) * (y - my)).sum()
    vx = (w * (x - mx) ** 2).sum()
    vy = (w * (y - my) ** 2).sum()
    denom = math.sqrt(vx * vy)
    return float(cov / denom) if denom > 0 else 0.0


def _corr_p_value(coef: float, n: int) -> float:
    """Двостороннє p для коефіцієнта кореляції (t-апроксимація, df=n−2)."""
    if n < 3 or abs(coef) >= 1.0:
        return 0.0 if abs(coef) >= 1.0 and n >= 3 else 1.0
    t = coef * math.sqrt((n - 2) / (1.0 - coef * coef))
    return float(2.0 * stats.t.sf(abs(t), df=n - 2))


def ordinal_correlation(
    a_codes: Sequence[float],
    b_codes: Sequence[float],
    weights: Sequence[float] | None = None,
) -> CorrelationResult:
    """Зважена рангова кореляція Спірмена для порядкових (Лікерт) питань.

    Спірмен = Пірсон на рангах. Коди — числові позиції категорій (напр. 1..5);
    пропуски передають як NaN і відсіюються попарно. p — t-апроксимація.
    """
    x = np.asarray(a_codes, dtype=float)
    y = np.asarray(b_codes, dtype=float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[mask], y[mask], w[mask]
    n = len(x)
    if n < 3:
        return CorrelationResult("spearman", 0.0, 1.0, n)
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    coef = _weighted_pearson(rx, ry, w)
    return CorrelationResult("spearman", coef, _corr_p_value(coef, n), n)


def numeric_correlation(
    a_values: Sequence[float],
    b_values: Sequence[float],
    weights: Sequence[float] | None = None,
) -> CorrelationResult:
    """Зважений Pearson r для справді числових питань."""
    x = np.asarray(a_values, dtype=float)
    y = np.asarray(b_values, dtype=float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[mask], y[mask], w[mask]
    n = len(x)
    if n < 3:
        return CorrelationResult("pearson", 0.0, 1.0, n)
    coef = _weighted_pearson(x, y, w)
    return CorrelationResult("pearson", coef, _corr_p_value(coef, n), n)


@dataclass
class OddsRatioResult:
    """Відношення шансів для таблиці 2×2 (зважені частоти)."""

    odds_ratio: float
    ci_low: float
    ci_high: float
    n: int


def odds_ratio_2x2(result: CrosstabResult) -> OddsRatioResult | None:
    """Відношення шансів OR=(a·d)/(b·c) із 95% CI (Woolf, log-OR).

    Лише для таблиць 2×2. Нульові клітинки коригуються Холдейном-Анскомбом.
    SE на зважених частотах — наближена. None, якщо таблиця не 2×2.
    """
    obs = result.table.to_numpy(dtype=float)
    if obs.shape != (2, 2):
        return None
    a, b, c, d = obs[0, 0], obs[0, 1], obs[1, 0], obs[1, 1]
    if min(a, b, c, d) == 0:
        a, b, c, d = (
            a + HALDANE_CORRECTION,
            b + HALDANE_CORRECTION,
            c + HALDANE_CORRECTION,
            d + HALDANE_CORRECTION,
        )
    or_value = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    log_or = math.log(or_value)
    return OddsRatioResult(
        odds_ratio=or_value,
        ci_low=math.exp(log_or - Z_95 * se),
        ci_high=math.exp(log_or + Z_95 * se),
        n=result.n,
    )


@dataclass
class PairAssociation:
    """Один рядок матриці зв'язків (одна пара питань)."""

    q1: str
    q2: str
    measure: str  # "cramers_v" | "spearman" | "pearson"
    effect: float  # розмір ефекту 0..1 (для кореляцій — |coef|)
    direction: float  # знак для кореляцій (±1), 0 для номінальних
    n: int
    p_raw: float
    p_fdr: float = field(default=math.nan)
    significant: bool = False
    low_expected: bool = False

    @property
    def effect_label(self) -> str:
        return _effect_label(self.effect)


def benjamini_hochberg(p_values: Sequence[float]) -> tuple[list[float], list[bool]]:
    """BH-FDR поправка: повертає (скориговані p, чи відхилено H0) при α=0.05."""
    if not p_values:
        return [], []
    rejected, p_adj, _, _ = multipletests(list(p_values), alpha=FDR_ALPHA, method="fdr_bh")
    return [float(x) for x in p_adj], [bool(x) for x in rejected]


def association_scan(pairs: Sequence[PairAssociation]) -> list[PairAssociation]:
    """Застосувати BH-FDR до набору попередньо порахованих пар та посортувати.

    Кожна пара вже містить `measure`, `effect`, `p_raw`. Тут лише коригуємо
    p (множинні порівняння) і сортуємо за спаданням сили ефекту.
    """
    if not pairs:
        return []
    p_adj, rejected = benjamini_hochberg([pr.p_raw for pr in pairs])
    for pr, padj, rej in zip(pairs, p_adj, rejected, strict=True):
        pr.p_fdr = padj
        pr.significant = rej
    return sorted(pairs, key=lambda pr: pr.effect, reverse=True)


def classify_association(pr: PairAssociation) -> str:
    """Human-facing priority label for an already scanned association."""
    if pr.low_expected:
        return "ненадійний"
    if pr.significant and pr.effect >= STRONG_EFFECT_THRESHOLD:
        return "ключовий"
    if pr.significant and pr.effect >= IMPORTANT_EFFECT_THRESHOLD:
        return "важливий"
    if pr.significant:
        return "слабкий"
    return "непідтверджений"


def filter_associations(
    scanned: Sequence[PairAssociation],
    mode: str,
    min_effect: float = IMPORTANT_EFFECT_THRESHOLD,
    hide_sparse: bool = False,
    measures: Sequence[str] | None = None,
) -> list[PairAssociation]:
    """Filter scanned associations for the overview table without changing statistics."""
    allowed_measures = set(measures) if measures is not None else None
    out: list[PairAssociation] = []
    for pr in scanned:
        if allowed_measures is not None and pr.measure not in allowed_measures:
            continue
        if hide_sparse and pr.low_expected:
            continue
        if mode == "Важливі" and not (pr.significant and pr.effect >= min_effect):
            continue
        if mode == "Значущі" and not pr.significant:
            continue
        if mode == "Сильні" and pr.effect < STRONG_EFFECT_THRESHOLD:
            continue
        if mode == "Ненадійні" and not pr.low_expected:
            continue
        out.append(pr)
    return out
