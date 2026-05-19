"""Прогноз кумулятивного потоку відповідей.

Public API:
    forecast_responses: оркестратор прогнозу (multi-model + NHPP CI).
    ForecastResult: результат прогнозу (dataclass).
    ForecastError: помилка фіту або валідації.
"""

from .service import asymptotic_exp_forecast, forecast_responses
from .types import ForecastError, ForecastResult

__all__ = [
    "ForecastError",
    "ForecastResult",
    "asymptotic_exp_forecast",  # deprecated alias; буде прибрано в commit 6/7
    "forecast_responses",
]
