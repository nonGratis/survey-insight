"""Public types for forecast package: result dataclass and error."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class ForecastError(RuntimeError):
    """Прогнозна модель не змогла знайти збіжний фіт."""


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
