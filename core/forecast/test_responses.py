"""Detect leading creator-test responses to skip before forecasting.

Типовий патерн: автор форми робить 1-кілька тестових відповідей при
створенні, потім — велика пауза до старту реальної кампанії. Такі лідери
спотворюють фіт першої хвилі.

Безпечна евристика (conservative): рахуємо ТІЛЬКИ суміжні ізольовані
лідер-відповіді — кожна відділена від наступної аномально великим розривом
(>> типового). Як тільки йде burst (малий розрив) — зупиняємось. Так НЕ
зрізаємо справжню першу хвилю агітації (вона — щільний burst, не ізольовані
точки).

Користувач завжди може скоригувати вручну (слайдер вікна в UI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_DEFAULT_MAX_SKIP = 10
_GAP_MULT = 20.0  # розрив > 20× типового = ізольована (тестова) точка
_MIN_GAP_H = 1.0  # і не менше 1 год (захист від дрібних коливань)
_MIN_N = 10  # на коротких рядах не вгадуємо


def detect_test_responses(
    timestamps: list | pd.Series,
    max_skip: int = _DEFAULT_MAX_SKIP,
) -> int:
    """К-сть провідних тестових відповідей до пропуску (0 якщо немає ознак).

    Args:
        timestamps: усі timestamps відповідей.
        max_skip: верхня межа на к-сть пропущених (захист).

    Returns:
        Скільки перших відповідей виглядають як ізольовані тестові.
    """
    ts = pd.Series(pd.to_datetime(timestamps)).sort_values().reset_index(drop=True)
    n = len(ts)
    if n < _MIN_N:
        return 0

    secs = np.array([t.timestamp() for t in ts])
    gaps_h = np.diff(secs) / 3600.0  # n-1 розривів
    if len(gaps_h) <= max_skip:
        return 0

    # Типовий розрив у "тілі" ряду (поза першими max_skip).
    tail = gaps_h[max_skip:]
    tail = tail[tail > 0]
    if len(tail) == 0:
        return 0
    later_median = float(np.median(tail))
    if later_median <= 0:
        return 0

    threshold = max(_GAP_MULT * later_median, _MIN_GAP_H)
    skip = 0
    for k in range(1, max_skip + 1):
        # розрив ПІСЛЯ k-ї відповіді (індекс k-1 у gaps_h)
        if gaps_h[k - 1] > threshold:
            skip = k
        else:
            break  # пішов burst — реальна хвиля почалась
    return skip
