"""Прогноз кумулятивного потоку відповідей.

Public API:
    forecast_responses: оркестратор прогнозу (multi-model + NHPP CI).
    forecast_with_segmentation: CP-aware wrapper (PELT changepoint detection).
    ForecastResult: результат прогнозу (dataclass).
    ForecastError: помилка фіту або валідації.
"""

from .form_type import classify_form_type
from .segmented import forecast_with_segmentation
from .service import forecast_responses
from .types import ForecastError, ForecastResult
from .wave_service import forecast_current_wave

__all__ = [
    "ForecastError",
    "ForecastResult",
    "classify_form_type",
    "forecast_current_wave",
    "forecast_responses",
    "forecast_with_segmentation",
]
