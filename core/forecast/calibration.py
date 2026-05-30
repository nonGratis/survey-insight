"""Empirical CI calibration: multiplier розширює CI до близького до 95% coverage.

Baseline NHPP дає coverage 30.9% при заявлених 95% (research/02 backtest).
Корінь — параметрична uncertainty недо-оцінює варіансу між формами і
shape-категоріями. Multiplier k розширює CI bands навколо point estimate.

research/06_calibration_sweep.py показав:
    k=1.0:   30.9% coverage (no calibration)
    k=5.0:   67.0% coverage, median width 65
    k=10.0:  72.6% coverage, median width 130  ← обрано
    k=20.0:  76.4% coverage, width 260
    k=100.0: 79.2% coverage, width 1300

Coverage асимптотує на 79% — це через width=0 cases на великих формах
(sim_cum schlopується у reality_cap). Ці форми потребують окремої
обробки (empirical Bayes priors → P9), які не покривається масштабуванням.

CALIBRATION_MULTIPLIER = 10.0 — компроміс coverage / width. Перебільшення
не сильно покращує (diminishing returns), а width росте лінійно.

Reference: Roulston & Smith 2002, "Evaluating Probabilistic Forecasts".
"""

from __future__ import annotations

import numpy as np

# Підібрано sweepом у research/06_calibration_sweep.py.
CALIBRATION_MULTIPLIER = 10.0
DERIVED_COVERAGE = 0.726  # empirical coverage на 288 backtest-точках

# Per-form-type multipliers (P11). Підстава:
# research/11_multi_level_reliability.py показав що різні form_type мають
# радикально різну RAW-калібровку NHPP (raw cov@95% варіюється 15-52%),
# тому глобальний x10 multiplier пере-калібрує одні типи і недо-калібрує інші.
# research/12_prod_realistic_horizons.py показав calibrated cov per type:
#   event_registration: 89.6%, creative_submission: 90.5% (близько до 95%)
#   recruitment: 81%, volunteer_donor: 88%, event_feedback: 85% (недо-калібровка)
#   holiday: 78%, political: 80% (помітна недо-калібровка)
#   service: 67%, survey: 53% (сильна недо-калібровка)
#   other: 96% (легка над-калібровка)
# Multipliers обрані щоб таргетувати ~92% emp coverage на кожному типі без
# над-роздуття CI. Капи [5, 30] для запобігання absurd width.
PER_TYPE_MULTIPLIER: dict[str, float] = {
    "event_registration": 11.0,
    "event_feedback": 12.0,
    "survey": 28.0,
    "recruitment": 14.0,
    "service": 20.0,
    "volunteer_donor": 12.0,
    "political": 14.0,
    "creative_submission": 11.0,
    "holiday": 16.0,
    "other": 8.0,
    "unknown": 13.0,  # fallback для форм поза катологом
}


def get_calibration_multiplier(form_type: str | None) -> float:
    """Multiplier для form_type. None → глобальний дефолт (backward compat)."""
    if form_type is None:
        return CALIBRATION_MULTIPLIER
    return PER_TYPE_MULTIPLIER.get(form_type, CALIBRATION_MULTIPLIER)


# Sample-size-залежне розширення CI на малих N. Підібрано у
# research/10_variance_reduction_ab.py: на повному датасеті 141 форм × 5
# cutoffs це додає +1.9pp coverage глобально без зміни MAPE/bias, і
# +18.2pp coverage на late_burst shape на ранніх cutoffs (0.1, 0.2).
# Логіка: при малих n_train (≤15) curve-fit нестійкий, базовий CI занадто
# вузький → coverage gap (85% vs 95% nominal). Лінійне додаткове
# розширення CI half-width закриває цей gap.
CI_SCALE_LOW_N = 15  # n_train ≤ цього → max розширення
CI_SCALE_HIGH_N = 30  # n_train ≥ цього → без розширення (×1.0)
CI_SCALE_MAX = 1.5  # макс. множник на CI half-width при n_train ≤ LOW_N


def apply_calibration_arrays(
    point_arr: np.ndarray,
    ci_lower_arr: np.ndarray,
    ci_upper_arr: np.ndarray,
    multiplier: float = CALIBRATION_MULTIPLIER,
) -> tuple[np.ndarray, np.ndarray]:
    """Розширити CI bands навколо point_arr (element-wise).

    Args:
        point_arr: точкові оцінки на horizon-grid.
        ci_lower_arr / ci_upper_arr: оригінальні CI межі.
        multiplier: масштабний коефіцієнт (default = CALIBRATION_MULTIPLIER).

    Returns:
        (calibrated_lower, calibrated_upper) — element-wise розширені межі.
        Гарантовано: lower ≤ point ≤ upper після калібрування.
    """
    lo_half = np.maximum(point_arr - ci_lower_arr, 0.0)
    hi_half = np.maximum(ci_upper_arr - point_arr, 0.0)
    cal_lower = np.maximum(point_arr - multiplier * lo_half, 0.0)
    cal_upper = point_arr + multiplier * hi_half
    return cal_lower, cal_upper


def sample_size_ci_scale(n_train: int) -> float:
    """Множник на CI half-width в залежності від n_train.

    Лінійна інтерполяція між CI_SCALE_MAX (на n_train ≤ LOW_N) і 1.0
    (на n_train ≥ HIGH_N). Для n_train в зоні зрілих фітів — без впливу.

    Reference: research/10_variance_reduction_ab.py
    """
    if n_train <= CI_SCALE_LOW_N:
        return CI_SCALE_MAX
    if n_train >= CI_SCALE_HIGH_N:
        return 1.0
    frac = (n_train - CI_SCALE_LOW_N) / (CI_SCALE_HIGH_N - CI_SCALE_LOW_N)
    return CI_SCALE_MAX - frac * (CI_SCALE_MAX - 1.0)


def apply_sample_size_scaling(
    point_arr: np.ndarray,
    ci_lower_arr: np.ndarray,
    ci_upper_arr: np.ndarray,
    n_train: int,
    last_observed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Розширити CI на малих N — post-hoc масштабування half-width.

    Викликається ПІСЛЯ `apply_calibration_arrays`. Не торкає point estimate;
    точка вже зафіксована — лише CI bands розширюються коли вибірка мала.

    Args:
        point_arr: точкові оцінки (НЕ модифікуються).
        ci_lower_arr / ci_upper_arr: межі ПІСЛЯ глобальної калібровки.
        n_train: кількість тренувальних точок.
        last_observed: cumulative-floor на нижню межу.

    Returns:
        (lo_arr, hi_arr) — розширені межі. lower ≥ last_observed,
        upper ≥ point (з захистом від накладання).
    """
    scale = sample_size_ci_scale(n_train)
    if scale == 1.0:
        return ci_lower_arr, ci_upper_arr
    lo_half = np.maximum(point_arr - ci_lower_arr, 0.0)
    hi_half = np.maximum(ci_upper_arr - point_arr, 0.0)
    new_lower = np.maximum(point_arr - scale * lo_half, float(last_observed))
    new_upper = point_arr + scale * hi_half
    return new_lower, new_upper
