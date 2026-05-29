"""Shape classifier — швидка категоризація кумулятивних кривих.

Public version of `research/01_dataset_overview._classify`. Використовується
для emp. Bayes priors (P9): обрати prior для нового timeline'у на основі
його shape-категорії (logarithmic / logistic / late_burst / ...).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SHAPES = ("insufficient", "linear", "logarithmic", "logistic", "late_burst")


def classify_timeline(timestamps: pd.Series) -> str:
    """Класифікувати timeline за shape-категорією.

    Швидка евристика на t50/t90 + auc_excess (без curve-fit).

    Returns:
        Один з: "insufficient", "linear", "logarithmic", "logistic", "late_burst".
    """
    n = len(timestamps)
    if n < 5:
        return "insufficient"
    ts_sorted = timestamps.sort_values().reset_index(drop=True)
    first = ts_sorted.iloc[0]
    last = ts_sorted.iloc[-1]
    span_seconds = (last - first).total_seconds()
    if span_seconds <= 0:
        return "insufficient"

    t_frac = (ts_sorted - first).dt.total_seconds().to_numpy() / span_seconds
    y_cum = np.arange(1, n + 1, dtype=float)
    y_norm = (y_cum - y_cum[0]) / max(y_cum[-1] - y_cum[0], 1.0)

    t50 = float(np.interp(0.5 * n, y_cum, t_frac))
    t90 = float(np.interp(0.9 * n, y_cum, t_frac))
    auc_excess = float(np.mean(y_norm - t_frac))

    # Late-burst: convex (більшість відповідей у кінці)
    if auc_excess < -0.20:
        return "late_burst"
    # Logistic: S-shape (плато досягнуто рано)
    if t90 < 0.75 and t50 < 0.50:
        return "logistic"
    # Linear: рівномірний темп
    if 0.40 < t50 < 0.60 and 0.80 < t90 < 0.95:
        return "linear"
    # Default: logarithmic (типове опитування з насиченням)
    return "logarithmic"
