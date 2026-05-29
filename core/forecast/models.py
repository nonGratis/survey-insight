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


K_MIN_RELAXATION = 1.30
# Bias-correction factor для нижньої межі K. Емпірично підібрано на
# 288 backtest-точках (96 форм × 3 cutoff'и) — див.
# research/reports/03_bias_investigation.md + 03b_relaxation_sweep.md.
#
# З K_min = last модель форсована до плато на поточному факті, що дає
# систематичний bias -15% (модель занижує майбутнє cumulative). Sweep
# по {1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0} показав:
#
#   factor=1.0:   bias=-15.0%  cov=47.6%  MAPE=25.6%  (baseline)
#   factor=1.3:   bias=-11.6%  cov=50.0%  MAPE=24.6%  ← обраний sweet spot
#   factor=2.0:   bias=-10.7%  cov=54.9%  MAPE=25.0%
#
# 1.30 — найкращий компроміс bias↓ і MAPE↓ без overshoot до позитивного
# bias. Залишковий bias має інші джерела (вибір моделі, t_future grid)
# і потребує окремої investigation.


def _capacity_bounds(y: np.ndarray, target: int | None) -> tuple[float, float]:
    """K_min, K_max — м'який prior на стелю (асимптоту) кумулятивної кривої.

    K_min = last·1.30 (а не просто last) — говорить моделі "будь готова,
    що буде ще принаймні 30% росту понад те, що бачимо". Це дозволяє
    curve_fit'у знайти кращу інтерпретацію траєкторії на префіксах,
    які ще не досягли плато.

    Якщо `target` задано — bound [0.3·target, 3·target] з floor'ом
    на last·1.30. Без target — [last·1.30, 10·last].
    """
    last = float(y[-1])
    last_relaxed = last * K_MIN_RELAXATION
    if target is not None and target > 0:
        return max(last_relaxed, 0.3 * target), max(last * 1.05, 3.0 * target)
    return max(last_relaxed, 1.0), max(last * 10.0, 10.0)


class LogisticModel:
    """y = K / (1 + exp(-r·(t - t0))).

    Підходить для опитувань з різкою хвилею в середині (агітаційний пік):
    повільний старт → крутий ріст → плато на K.
    """

    name = "logistic"
    n_params = 3

    def predict(self, t: np.ndarray, K: float, r: float, t0: float) -> np.ndarray:  # noqa: N803
        # Clip аргумент exp щоб запобігти overflow при extreme param-samples з NHPP.
        return K / (1.0 + np.exp(np.clip(-r * (t - t0), -500.0, 500.0)))

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
        # Подвійний exp може overflow'ити; clip обох рівнів.
        inner = np.clip(-r * (t - t0), -500.0, 500.0)
        return K * np.exp(-np.clip(np.exp(inner), 0.0, 1e150))

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

# Поріг "достатньо даних, щоб довіряти всім трьом моделям". Нижче нього
# Logistic і Gompertz часто фітять випадковий шум (особливо чутливі до
# розташування t0 на дуже короткому ряді), а AsymptoticExp залишається
# стабільним: 3 параметри, монотонно зростає від c до a+c. Тому при малому N
# обмежуємо набір лише ним.
SMALL_SAMPLE_THRESHOLD = 10


def models_for_n_points(n: int) -> tuple[SaturationModel, ...]:
    """Tiered набір моделей залежно від кількості тренувальних точок.

    - n < SMALL_SAMPLE_THRESHOLD: лише AsymptoticExp (найстабільніший на малому N).
    - n >= SMALL_SAMPLE_THRESHOLD: усі три моделі через AICc-селектор.
    """
    if n < SMALL_SAMPLE_THRESHOLD:
        return (AsymptoticExpModel(),)
    return DEFAULT_MODELS


def fit_model(
    model: SaturationModel,
    t: np.ndarray,
    y: np.ndarray,
    target: int | None,
    bounds_override: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
) -> tuple[tuple[float, ...], np.ndarray | None]:
    """Знайти параметри моделі curve_fit'ом і повернути (params, pcov).

    Args:
        model, t, y, target: standard fit inputs.
        bounds_override: якщо задано — використати ці bounds замість
            model.bounds(y, target). Використовується для emp. Bayes priors
            (P9), які звужують bounds до prior_median ± Nσ.

    Raises ForecastError при non-convergence.
    """
    p0 = model.initial_guess(t, y, target)
    bounds = bounds_override if bounds_override is not None else model.bounds(y, target)
    # Sanitize p0 → всередині bounds.
    low, high = bounds
    p0 = tuple(min(max(p, low[i]), high[i]) for i, p in enumerate(p0))
    try:
        popt, pcov = curve_fit(
            model.predict,
            t,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=CURVE_FIT_MAX_NFEV,
        )
    except (RuntimeError, ValueError) as exc:
        raise ForecastError(f"{model.name} не зійшовся: {exc}") from exc
    params = tuple(float(p) for p in popt)
    if pcov is None or not np.all(np.isfinite(pcov)):
        return params, None
    return params, pcov
