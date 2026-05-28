"""Detection: changepoint detection + sub-daily ETL для rate-серій.

Модуль додає те, чого не вистачає core/forecast/ — детекцію хвиль агітації
через PELT і ETL-утиліти для роботи з sub-daily rate-серіями.

Workflow інтеграції з forecast:
    timestamps
        → to_rate_series → median_smooth → detect_changepoints
        → якщо є CP → беремо timestamps після останнього CP
        → build_timeline_from_timestamps(post_cp_subset)
        → forecast_responses(...)  # все, що вже маємо

Це addresses late_burst (9 форм, MAPE=31%) і ill_fit (6 форм, MAPE=39%)
з research/02 — обидві категорії — це multi-wave траєкторії, де single-
curve fit систематично провалюється.

Публічний API:
    - to_rate_series(timestamps, freq)   — ETL з timestamps у per-bucket counts
    - to_cumulative(rate)                — cumsum зі збереженням індексу
    - median_smooth(series, window)      — медіанний фільтр проти разових spike'ів
    - detect_changepoints(rate, ...)     — PELT через ruptures
    - InsufficientDataError              — недостатньо даних для детекції
    - Changepoint                        — frozen dataclass з timestamp/index
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd
import ruptures as rpt

# Мінімальний розмір сегменту для PELT — менше і одиничний сплеск
# рахуватиметься як changepoint. 5 годинних бакетів = реалістична
# мінімальна тривалість "хвилі" поширення.
_DEFAULT_MIN_SEGMENT: Final[int] = 5

# Дефолтна cost-функція PELT. "l2" виявляє зміни середнього значення —
# саме те, що нам потрібно для виявлення хвиль (зміна базового rate).
# Альтернативи: "rbf" (загальніша, у т.ч. зміна дисперсії), "l1" (стійка
# до викидів). Передається параметром, якщо знадобиться інший варіант.
_DEFAULT_COST_MODEL: Final[str] = "l2"


class InsufficientDataError(ValueError):
    """Замало точок для надійного фіту або детекції."""


@dataclass(frozen=True)
class Changepoint:
    """Виявлена точка структурного розриву у швидкості надходження.

    Attributes:
        timestamp: момент часу changepoint-у.
        index: позиція у вхідній серії (0-based).
    """

    timestamp: pd.Timestamp
    index: int


# ---------- ETL utilities ---------------------------------------------------


def to_rate_series(timestamps: pd.Series, freq: str = "1h") -> pd.Series:
    """Ресемплити окремі тайстемпи у швидкість надходжень за бакет.

    Args:
        timestamps: Series окремих DateTime подій (момент сабміту анкети).
        freq: pandas-офсет ("1h", "1D", "15min").

    Returns:
        Series ``int64`` (responses per bucket) із DatetimeIndex,
        обрізана з обох боків до першого та останнього ненульового бакета.

    Raises:
        InsufficientDataError: серія порожня, всі значення невалідні, або
            після ресемплу немає жодного ненульового бакета.
    """
    if timestamps.empty:
        raise InsufficientDataError("Порожня серія тайстемпів.")
    parsed = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values()
    if parsed.empty:
        raise InsufficientDataError("Усі тайстемпи не парсяться у datetime.")

    indicator = pd.Series(1, index=pd.DatetimeIndex(parsed))
    rate = indicator.resample(freq).sum().astype("int64")

    nonzero = rate[rate > 0]
    if nonzero.empty:
        raise InsufficientDataError("Після ресемплу немає ненульових бакетів.")
    return rate.loc[nonzero.index[0] : nonzero.index[-1]]


def to_cumulative(rate: pd.Series) -> pd.Series:
    """Кумулятивна сума зі збереженням індексу."""
    return rate.cumsum().astype("int64")


def median_smooth(series: pd.Series, window: int = 5) -> pd.Series:
    """Медіанний фільтр зі симетричним вікном (нейтралізує одиничні викиди).

    На відміну від ``rolling.mean``, медіанний фільтр стійкий до spike-ів
    (приклад: дубльований сабмішн форми, що дав +500 відповідей за бакет).
    ``min_periods=1`` гарантує відсутність NaN у крайніх позиціях.

    Args:
        series: вхідна числова Series.
        window: розмір симетричного вікна (рекомендовано непарне 3-7).

    Returns:
        Series тих самих shape, dtype та index, що й на вході.
    """
    if window < 1:
        raise ValueError("window має бути >= 1.")
    smoothed = series.rolling(window=window, center=True, min_periods=1).median()
    return smoothed.astype(series.dtype)


# ---------- Changepoint detection -------------------------------------------


def detect_changepoints(
    rate: pd.Series,
    penalty: float = 10.0,
    min_segment: int = _DEFAULT_MIN_SEGMENT,
    cost_model: str = _DEFAULT_COST_MODEL,
) -> list[Changepoint]:
    """Знайти точки розриву у швидкості надходження відповідей через PELT.

    PELT (Pruned Exact Linear Time) — точний алгоритм за лінійний час, який
    мінімізує суму cost-функцій по сегментах плюс штраф ``penalty`` за
    кожен новий сегмент. Більший ``penalty`` → менше точок розриву.

    Args:
        rate: Series швидкості (responses per bucket) із DatetimeIndex.
        penalty: штраф за нову точку розриву. Для погодинних бакетів
            орієнтовно 5-20. Підбирається по конкретному датасету.
        min_segment: мінімальний розмір сегменту в бакетах (відкидає
            короткі сплески, які не вважаємо хвилями).
        cost_model: cost-функція ``ruptures`` — ``"l2"`` (зміна середнього),
            ``"rbf"`` (загальніша), ``"l1"`` (стійка до викидів).

    Returns:
        Список ``Changepoint``, впорядкований у часі. Порожній, якщо точок
        розриву не виявлено або серія коротша за ``2 * min_segment``.
    """
    if len(rate) < 2 * min_segment:
        return []
    signal = rate.to_numpy(dtype="float64").reshape(-1, 1)
    algo = rpt.Pelt(model=cost_model, min_size=min_segment).fit(signal)
    raw_indices = algo.predict(pen=penalty)
    # ruptures повертає індекси кінців сегментів; останній — len(signal),
    # це не справжня changepoint, а штучний sentinel — відкидаємо.
    cp_indices = [i for i in raw_indices if i < len(rate)]
    return [Changepoint(timestamp=rate.index[i], index=i) for i in cp_indices]
