"""Прогноз кумулятивного потоку відповідей.

Public API:
    forecast_responses: оркестратор прогнозу (multi-model + NHPP CI).
    ForecastResult: результат прогнозу (dataclass).
    ForecastError: помилка фіту або валідації.
"""

from .service import forecast_responses
from .types import ForecastError, ForecastResult

__all__ = [
    "ForecastError",
    "ForecastResult",
    "forecast_responses",
]
