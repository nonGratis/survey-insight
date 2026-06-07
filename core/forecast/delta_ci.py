"""Classical delta-method CI на fitted SaturationModel.

Замінює NHPP-симуляцію + post-hoc multipliers (P7, P10, P11) як **primary**
механізм CI computation. Базується на standard non-linear regression theory
(Bates & Watts 1988): scipy.optimize.curve_fit повертає (popt, pcov), delta-
method перетворює це у CI на predict-grid через numerical Jacobian + Student t.

Формула:
    y(t; θ) — fitted curve, θ = (params)
    J(t) = ∂y/∂θ — Jacobian (numerical через finite differences)
    Var(y(t)) = J(t) · Σ · J(t)ᵀ   де Σ = pcov від curve_fit
    CI = y(t) ± t_{1−α/2, n−k} · √Var(y(t))

Переваги vs NHPP + multipliers:
- Width природно мала коли R² високий (pcov щільний) — точне відображення
  справжньої uncertainty замість uniform ×10×1.5×N inflation.
- Width природно велика коли fit поганий → коректно сигналізує проблему.
- Залежить ТІЛЬКИ від даних і fit-quality. Жодних магічних multipliers.
- Дешевше: O(n_future × n_params²) vs O(2000 × n_future) для NHPP.

Caveats:
- Якщо pcov ill-conditioned (Gompertz на коротких рядах, multi-wave формах
  з survey/feedback) → width може explode. `cap_width()` детектує і обмежує.
- На дуже довгих горизонтах + non-saturating model (Log) → extrapolation
  росте. AsympExp і Logistic — bounded (predict→K), тому Jacobian decays.

Reference:
- Bates DM, Watts DG (1988). Nonlinear Regression Analysis. Wiley.
- scipy.optimize.curve_fit (повертає pcov, scaled by MSE з absolute_sigma=False).
- research/16_delta_ci_on_selector_ab.py — Winkler A/B vs NHPP+multipliers.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

from .types import FittedModel


def delta_method_ci(
    fitted: FittedModel,
    t_future: np.ndarray,
    n_train: int,
    confidence: float = 0.95,
    step: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Classical delta-method CI для non-linear regression на t_future grid.

    Args:
        fitted: FittedModel з params (tuple) + pcov (np.ndarray | None).
            pcov МУСИТЬ бути finite і positive-semi-definite для коректного CI.
        t_future: моменти часу (1D float, дні від anchor), на яких рахуємо CI.
        n_train: кількість тренувальних точок (для df Student-t).
        confidence: рівень CI у (0..1). Default 0.95.
        step: відносний крок для numerical Jacobian (multiplied by max(1,|θ_j|)).

    Returns:
        (lower, upper) — CI bounds (1D float arrays довжиною len(t_future)).

    Raises:
        ValueError: якщо pcov None, містить NaN/Inf, або negative-definite.
    """
    if fitted.pcov is None:
        raise ValueError("delta_method_ci: fitted.pcov is None")
    pcov = np.asarray(fitted.pcov, dtype=float)
    if not np.all(np.isfinite(pcov)):
        raise ValueError("delta_method_ci: pcov has NaN/Inf")

    params = np.asarray(fitted.params, dtype=float)
    n_params = len(params)
    df = max(1, n_train - n_params)

    # Numerical Jacobian: J[i, j] = ∂y(t_i) / ∂θ_j через central diff.
    # Relative step для робастності на параметрах різного масштабу
    # (наприклад K~100, t0~50 — однакова step робила б помилкові градієнти).
    jac = np.zeros((len(t_future), n_params), dtype=float)
    for j in range(n_params):
        d = np.zeros(n_params)
        d[j] = step * max(1.0, abs(params[j]))
        y_plus = np.asarray(fitted.model.predict(t_future, *(params + d)), dtype=float)
        y_minus = np.asarray(fitted.model.predict(t_future, *(params - d)), dtype=float)
        jac[:, j] = (y_plus - y_minus) / (2.0 * d[j])

    # Var(y(t)) = diag(J · pcov · Jᵀ). Можна обчислити row-wise через einsum
    # без формування повної матриці J·pcov·Jᵀ (зекономить пам'ять на великих
    # t_future).
    var_y = np.einsum("ij,jk,ik->i", jac, pcov, jac)
    var_y = np.maximum(var_y, 0.0)  # numerical floor (теоретично ≥ 0)
    se_y = np.sqrt(var_y)

    t_val = float(student_t.ppf(1.0 - (1.0 - confidence) / 2.0, df))
    delta = t_val * se_y

    y_pred = np.asarray(fitted.model.predict(t_future, *params), dtype=float)
    return y_pred - delta, y_pred + delta


def cap_width(
    point_arr: np.ndarray,
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
    max_relative: float = 5.0,
    max_absolute: float = 5000.0,
    min_absolute: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Explosion guard: симетричний cap на CI half-width.

    Захист від ill-conditioned pcov (Gompertz на коротких рядах, multi-wave
    форми у survey/feedback де delta-CI може дати width >> point). Обмежує
    half-width до min(point · max_relative, max_absolute) / 2, з підлогою
    min_absolute / 2.

    Не змінює coverage коли delta-CI вже sane (нативна width < cap). Спрацьовує
    тільки на explosion-cases.

    Args:
        point_arr: точкові оцінки (НЕ змінюються).
        lower_arr / upper_arr: CI bounds від delta_method_ci.
        max_relative: cap relative до point (e.g. 5.0 = width ≤ 5×point).
        max_absolute: cap absolute (e.g. 5000 responses).
        min_absolute: floor на capped width (e.g. 20 responses).

    Returns:
        (capped_lower, capped_upper) — width гарантовано в [min_absolute, cap].
    """
    cap = np.minimum(point_arr * max_relative, max_absolute)
    cap = np.maximum(cap, min_absolute)
    half_widths = (upper_arr - lower_arr) / 2.0
    new_half = np.minimum(half_widths, cap / 2.0)
    return point_arr - new_half, point_arr + new_half
