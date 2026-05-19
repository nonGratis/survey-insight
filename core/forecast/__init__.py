"""Прогноз кумулятивного потоку відповідей.

Public API:
    asymptotic_exp_forecast: оркестратор прогнозу.
    ForecastResult: результат прогнозу (dataclass).
    ForecastError: помилка фіту або валідації.
"""

from .service import asymptotic_exp_forecast
from .types import ForecastError, ForecastResult

__all__ = ["ForecastError", "ForecastResult", "asymptotic_exp_forecast"]
