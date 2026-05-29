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
