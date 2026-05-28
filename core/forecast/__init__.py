"""Прогноз кумулятивного потоку відповідей.

Public API:
    forecast_responses: оркестратор прогнозу (multi-model + NHPP CI).
    forecast_with_segmentation: CP-aware wrapper (PELT changepoint detection).
    ForecastResult: результат прогнозу (dataclass).
    ForecastError: помилка фіту або валідації.
"""

from .segmented import forecast_with_segmentation
from .service import forecast_responses
from .types import ForecastError, ForecastResult

__all__ = [
    "ForecastError",
    "ForecastResult",
    "forecast_responses",
    "forecast_with_segmentation",
]
