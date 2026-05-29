"""Вибір найкращої моделі за AICc.

Селектор отримує список моделей через DI (Sequence[SaturationModel]),
фітить кожну, ранжує за AICc (з поправкою на малий N), повертає переможця.

Якщо жодна модель не зійшлася — ForecastError.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from core.logger import get_logger

from .metrics import aicc, r_squared, rmse
from .models import DEFAULT_MODELS, SaturationModel, fit_model
from .priors import ShapePrior, narrow_bounds_with_prior
from .types import FittedModel, ForecastError

log = get_logger(__name__)


def select_best_model(
    t: np.ndarray,
    y: np.ndarray,
    target: int | None = None,
    models: Sequence[SaturationModel] = DEFAULT_MODELS,
    priors: dict[tuple[str, str], ShapePrior] | None = None,
    shape: str | None = None,
) -> FittedModel:
    """Фіт кожну модель з `models`, повернути найкращу за AICc.

    Args:
        t: тренувальні моменти часу (1D, монотонно зростаючі).
        y: тренувальні cumulative-значення (1D, та сама довжина, що й t).
        target: цільова кількість відповідей; передається в bounds моделей
            як soft prior. None — широкі дефолти.
        models: реалізації SaturationModel. Default — Logistic + Gompertz
            + AsymptoticExp; для тестів можна передати інший набір.

    Returns:
        FittedModel переможця з заповненими aicc, rmse, r_squared.

    Raises:
        ForecastError: якщо жодна модель не зійшлася.
    """
    candidates: list[FittedModel] = []
    failures: list[tuple[str, str]] = []
    for model in models:
        # P9: звужуємо bounds через emp. Bayes prior, якщо переданий.
        bounds_override = None
        if priors and shape:
            prior = priors.get((model.name, shape))
            if prior is not None:
                default_bounds = model.bounds(y, target)
                bounds_override = narrow_bounds_with_prior(
                    default_bounds[0], default_bounds[1], prior
                )
        try:
            params, pcov = fit_model(model, t, y, target, bounds_override=bounds_override)
        except ForecastError as exc:
            failures.append((model.name, str(exc)))
            continue
        y_fitted = model.predict(t, *params)
        candidates.append(
            FittedModel(
                model=model,
                params=params,
                aicc=aicc(y, y_fitted, model.n_params),
                rmse=rmse(y, y_fitted),
                r_squared=r_squared(y, y_fitted),
                pcov=pcov,
            )
        )

    if not candidates:
        # Friendly message: список імен моделей, що не зійшлися; деталі — у log.
        log.warning(
            "forecast_all_models_failed",
            extra={"failures": [{"model": n, "reason": e} for n, e in failures]},
        )
        names = ", ".join(n for n, _ in failures)
        raise ForecastError(
            f"Модель не змогла знайти стійкий фіт ({names}). "
            f"Можливі причини: занадто мало точок, ряд не має тренду насичення, "
            f"або всі timestamps співпадають."
        )

    candidates.sort(key=lambda c: c.aicc)
    best = candidates[0]
    log.info(
        "forecast_model_selected",
        extra={
            "best": best.model.name,
            "aicc": round(best.aicc, 2),
            "candidates": [{"name": c.model.name, "aicc": round(c.aicc, 2)} for c in candidates],
            "failures": [n for n, _ in failures],
        },
    )
    return best
