"""Public types for forecast package: result dataclasses and error."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .models import SaturationModel


class ForecastError(RuntimeError):
    """Прогнозна модель не змогла знайти збіжний фіт."""


@dataclass(frozen=True)
class FittedModel:
    """Зафіксований результат фіту однієї моделі.

    Attributes:
        model: екземпляр SaturationModel (тримає predict + n_params).
        params: оптимальні параметри (порядок як у model.predict).
        aicc: AICc на тренувальних даних (менше — краще).
        rmse: RMSE на тренувальних даних.
        r_squared: R² на тренувальних даних.
    """

    model: SaturationModel
    params: tuple[float, ...]
    aicc: float
    rmse: float
    r_squared: float


@dataclass(frozen=True)
class ForecastResult:
    """Результат прогнозу cumulative на горизонт.

    Attributes:
        model: ідентифікатор моделі (наприклад, "asymptotic_exp").
        future_dates: дати після останнього відомого факту, включно з горизонтом.
        future_cum: модельний cumulative на future_dates.
        ci_lower: нижня межа 95% інтервалу.
        ci_upper: верхня межа 95% інтервалу.
        final_estimate: цілочислова точкова оцінка cumulative на кінці горизонту.
        final_ci: (lower, upper) на кінці горизонту.
        rmse: RMSE моделі на тренувальних даних.
        r_squared: R² на тренувальних даних.
    """

    model: str
    future_dates: pd.DatetimeIndex
    future_cum: pd.Series
    ci_lower: pd.Series
    ci_upper: pd.Series
    final_estimate: int
    final_ci: tuple[int, int]
    rmse: float
    r_squared: float
