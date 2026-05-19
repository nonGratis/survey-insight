"""Параметричні моделі насиченої кривої cumulative-відповідей.

Кожна модель — окремий клас, що реалізує `SaturationModel` Protocol.
Це дозволяє додавати нові моделі (Weibull, Bass, ...) без правки селектора
чи сервісу (Open/Closed). Селектор приймає список моделей через
залежність-інжекцію (Dependency Inversion), а не імпортує конкретні класи.

Bounds кожної моделі м'яко враховують `target` — цільову кількість
відповідей, задану користувачем. Якщо target=None, межі — широкі дефолти,
прив'язані до останнього спостереження.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy.optimize import curve_fit

from .types import ForecastError

CURVE_FIT_MAX_NFEV = 500


@runtime_checkable
class SaturationModel(Protocol):
    """Контракт для параметричної моделі кумулятивної кривої.

    Реалізації мусять бути stateless — усі параметри передаються в predict.
    """

    name: str
    n_params: int

    def predict(self, t: np.ndarray, *params: float) -> np.ndarray: ...

    def initial_guess(
        self, t: np.ndarray, y: np.ndarray, target: int | None
    ) -> tuple[float, ...]: ...

    def bounds(
        self, y: np.ndarray, target: int | None
    ) -> tuple[tuple[float, ...], tuple[float, ...]]: ...


def _capacity_bounds(y: np.ndarray, target: int | None) -> tuple[float, float]:
    """K_min, K_max — м'який prior на стелю (асимптоту) кумулятивної кривої.

    Якщо `target` задано — обмежуємо [0.3·target, 3·target] (з floor на
    останній факт). Без target — широкі дефолти [last, 10·last].
    """
    last = float(y[-1])
    if target is not None and target > 0:
        return max(last, 0.3 * target), max(last * 1.05, 3.0 * target)
    return max(last, 1.0), max(last * 10.0, 10.0)


class LogisticModel:
    """y = K / (1 + exp(-r·(t - t0))).

    Підходить для опитувань з різкою хвилею в середині (агітаційний пік):
    повільний старт → крутий ріст → плато на K.
    """

    name = "logistic"
    n_params = 3

    def predict(self, t: np.ndarray, K: float, r: float, t0: float) -> np.ndarray:  # noqa: N803
        return K / (1.0 + np.exp(-r * (t - t0)))

    def initial_guess(
        self, t: np.ndarray, y: np.ndarray, target: int | None
    ) -> tuple[float, float, float]:
        k_min, k_max = _capacity_bounds(y, target)
        k0 = float(np.clip(target if target else y[-1] * 1.5, k_min, k_max))
        t0_guess = float((t[0] + t[-1]) / 2.0)
        return k0, 0.3, t0_guess

    def bounds(
        self, y: np.ndarray, target: int | None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        k_min, k_max = _capacity_bounds(y, target)
        return (k_min, 1e-3, -1e6), (k_max, 5.0, 1e6)


class GompertzModel:
    """y = K · exp(-exp(-r·(t - t0))).

    Альтернатива логістичній: асиметрична S-крива з повільнішим затуханням
    після перегину. Часто краще описує "довгий хвіст" відповідей.
    """

    name = "gompertz"
    n_params = 3

    def predict(self, t: np.ndarray, K: float, r: float, t0: float) -> np.ndarray:  # noqa: N803
        return K * np.exp(-np.exp(-r * (t - t0)))

    def initial_guess(
        self, t: np.ndarray, y: np.ndarray, target: int | None
    ) -> tuple[float, float, float]:
        k_min, k_max = _capacity_bounds(y, target)
        k0 = float(np.clip(target if target else y[-1] * 1.5, k_min, k_max))
        t0_guess = float((t[0] + t[-1]) / 2.0)
        return k0, 0.2, t0_guess

    def bounds(
        self, y: np.ndarray, target: int | None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        k_min, k_max = _capacity_bounds(y, target)
        return (k_min, 1e-3, -1e6), (k_max, 5.0, 1e6)


class AsymptoticExpModel:
    """y = a · (1 - exp(-b·t)) + c.

    Експоненційне наближення до асимптоти `a + c`. Підходить, коли пік
    припадає на самий старт і далі — лише затухаюче поповнення.
    """

    name = "asymptotic_exp"
    n_params = 3

    def predict(self, t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * (1.0 - np.exp(-b * t)) + c

    def initial_guess(
        self, t: np.ndarray, y: np.ndarray, target: int | None
    ) -> tuple[float, float, float]:
        a_min, a_max = _capacity_bounds(y, target)
        a0 = float(np.clip(max(y[-1] - y[0], 1.0), a_min, a_max))
        return a0, 0.05, float(y[0])

    def bounds(
        self, y: np.ndarray, target: int | None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        a_min, a_max = _capacity_bounds(y, target)
        return (a_min * 0.5, 1e-6, 0.0), (a_max, 5.0, max(float(y[0]) + 1.0, 1.0))


DEFAULT_MODELS: tuple[SaturationModel, ...] = (
    LogisticModel(),
    GompertzModel(),
    AsymptoticExpModel(),
)


def fit_model(
    model: SaturationModel,
    t: np.ndarray,
    y: np.ndarray,
    target: int | None,
) -> tuple[float, ...]:
    """Знайти параметри моделі curve_fit'ом. ForecastError при non-convergence."""
    p0 = model.initial_guess(t, y, target)
    bounds = model.bounds(y, target)
    try:
        popt, _ = curve_fit(
            model.predict,
            t,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=CURVE_FIT_MAX_NFEV,
        )
    except (RuntimeError, ValueError) as exc:
        raise ForecastError(f"{model.name} не зійшовся: {exc}") from exc
    return tuple(float(p) for p in popt)
