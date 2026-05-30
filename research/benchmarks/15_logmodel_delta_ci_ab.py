"""15_logmodel_delta_ci_ab.py — pivot до classical delta-method CI.

Контекст: користувач показав скриншот де прод-CI [62, 818] на формі з
truth=60 і point=70. R² fit-у 0.923 (excellent), але CI безкорисний.
Причина: NHPP-симуляція + CALIBRATION_MULTIPLIER × per-type widening
блокують прод-практичність. Coverage metric маскує цю проблему.

Цей бенчмарк тестує **principle pivot**: bypass NHPP+multiplier повністю,
використати класичні delta-method CI на curve_fit-fitted parameters
(стандарт у scipy/statsmodels для non-linear regression).

Методи (на тих самих 3710 prod-realistic backtest points що 12_/13_):

- **A · prod** — поточний `forecast_responses` (NHPP + P7 calibration ×10 +
  P10 sample-size scaling + P11 per-type multiplier). Контроль.
- **B · log_delta** — fit `y = a·ln(t+1) + b` через scipy curve_fit,
  delta-method CI через numerical Jacobian + pcov + Student t-quantile.
  Без NHPP. Без множників. Без post-hoc розширення.

Primary метрика: **Winkler interval score** (Winkler 1972):
    W = (U − L) + (2/α)·max(L−y, 0) + (2/α)·max(y−U, 0)
Покаранує і ширину, і miss-rate. Lower = better. Це **proper scoring rule**,
оптимальний для nominal coverage. Покриття + sharpness разом, не окремо.

Додаткові метрики: MAPE_p50, coverage@95, sharpness_p50, sharpness_p90,
absolute width median (не relative — користувач скаржився саме на
absolute width).

Якщо B виграє A на Winkler global і на більшості form_type → структурний
сигнал що treba revertити P10/P11 і перехід до delta-method CI як основу.

Запуск:
    .venv/Scripts/python.exe research/benchmarks/15_logmodel_delta_ci_ab.py
    .venv/Scripts/python.exe research/benchmarks/15_logmodel_delta_ci_ab.py --limit 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import t as student_t

warnings.simplefilter("ignore", OptimizeWarning)
warnings.simplefilter("ignore", RuntimeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.forecast import ForecastError, forecast_responses  # noqa: E402
from core.timeline import build_timeline_from_timestamps  # noqa: E402
from research.benchmarks._common import (  # noqa: E402
    build_eligible_forms,
    df_to_md,
    file_sha256_short,
    idx_for_horizon,
    load_dataset,
    load_form_types,
    load_shapes,
)

logging.getLogger().setLevel(logging.WARNING)

N_TRAIN_VALUES = (5, 10, 15, 20, 25, 30)
HORIZON_HOURS = (2, 6, 24, 72, 168)
MIN_N_FOR_BACKTEST = 5
CONFIDENCE = 0.95
ALPHA = 1.0 - CONFIDENCE


# ---------- LogModel + delta-method CI -------------------------------------


def _log_predict(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """y = a · ln(t + 1) + b. +1 для уникнення log(0) at t=0."""
    return a * np.log(np.maximum(t + 1.0, 1.0)) + b


def _fit_log_with_delta_ci(
    t_train: np.ndarray,
    y_train: np.ndarray,
    t_future: np.ndarray,
    confidence: float = CONFIDENCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Fit log model + delta-method CI.

    Returns (y_pred, lower, upper, r_squared, mse).
    Raises ForecastError якщо fit падає або CI degenerate.
    """
    n_train = len(t_train)
    n_params = 2
    if n_train < n_params:
        raise ForecastError("log_delta: n_train < n_params")

    # Initial guess: linear-ish in ln(t+1)
    log_t = np.log(t_train + 1.0)
    if log_t[-1] - log_t[0] > 1e-6:
        a0 = (y_train[-1] - y_train[0]) / (log_t[-1] - log_t[0])
    else:
        a0 = 1.0
    b0 = float(y_train[0]) - a0 * float(log_t[0])

    try:
        popt, pcov = curve_fit(_log_predict, t_train, y_train, p0=(a0, b0), maxfev=5000)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        raise ForecastError(f"log_delta_fit_failed: {exc}") from exc

    if pcov is None or not np.all(np.isfinite(pcov)):
        raise ForecastError("log_delta: non-finite pcov")

    # Point estimate at future grid + on training (for R²)
    y_fit_train = _log_predict(t_train, *popt)
    y_pred = _log_predict(t_future, *popt)

    # R² for diagnostic only.
    ss_res = float(np.sum((y_train - y_fit_train) ** 2))
    ss_tot = float(np.sum((y_train - np.mean(y_train)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    mse = ss_res / max(1, n_train - n_params)

    # Delta-method: numerical Jacobian J[i, j] = d y(t_i) / d theta_j.
    step = 1e-6
    J = np.zeros((len(t_future), n_params))
    for j in range(n_params):
        d = np.zeros(n_params)
        d[j] = step
        y_plus = _log_predict(t_future, *(popt + d))
        y_minus = _log_predict(t_future, *(popt - d))
        J[:, j] = (y_plus - y_minus) / (2.0 * step)

    # var(y(t)) = J(t) · pcov · J(t)^T (для кожного t — діагональ).
    # pcov scipy curve_fit (absolute_sigma=False default) уже включає mse,
    # тому var_y вже у правильних одиницях.
    var_y = np.einsum("ij,jk,ik->i", J, pcov, J)
    var_y = np.maximum(var_y, 0.0)
    se_y = np.sqrt(var_y)

    # Student t-quantile.
    df = max(1, n_train - n_params)
    t_val = float(student_t.ppf(1.0 - (1.0 - confidence) / 2.0, df))
    delta = t_val * se_y

    lower = y_pred - delta
    upper = y_pred + delta

    return y_pred, lower, upper, r2, mse


# ---------- Winkler score --------------------------------------------------


def _winkler_score(truth: float, lower: float, upper: float, alpha: float = ALPHA) -> float:
    """Winkler interval score. Lower = better. Proper scoring rule.

    W = (U − L) + (2/α)·max(L−y, 0) + (2/α)·max(y−U, 0)

    Reference: Winkler RL (1972), "A Decision-Theoretic Approach to
    Interval Estimation", JASA 67(337):187-191.
    """
    width = max(0.0, upper - lower)
    penalty_lo = (2.0 / alpha) * max(lower - truth, 0.0)
    penalty_hi = (2.0 / alpha) * max(truth - upper, 0.0)
    return width + penalty_lo + penalty_hi


# ---------- A/B core -------------------------------------------------------


@dataclass(frozen=True)
class ABPoint:
    form_id: str
    shape: str
    form_type: str
    n_total: int
    n_train: int
    horizon_hours: float
    truth: int
    # A: prod
    a_point: int
    a_lo: int
    a_hi: int
    a_error: str | None
    # B: log + delta CI
    b_point: int
    b_lo: int
    b_hi: int
    b_r2: float
    b_error: str | None


def _backtest_one(form: dict, n_train: int, horizon_hours: float) -> ABPoint | None:
    ts = form["timestamps"]
    n_total = len(ts)
    if n_total < n_train:
        return None
    ts_sorted = ts.sort_values().reset_index(drop=True)
    cutoff_ts = ts_sorted.iloc[n_train - 1]
    cutoff_span_seconds = (cutoff_ts - ts_sorted.iloc[0]).total_seconds()
    if cutoff_span_seconds <= 0:
        return None
    horizon_seconds = horizon_hours * 3600.0
    horizon_end = cutoff_ts + pd.Timedelta(seconds=horizon_seconds)
    truth = int((ts_sorted <= horizon_end).sum())

    prefix = ts_sorted.iloc[:n_train].reset_index(drop=True)
    prefix_dt = [t.to_pydatetime() for t in prefix.tolist()]
    timeline = build_timeline_from_timestamps(prefix_dt)
    horizon_until = pd.Timestamp(horizon_end)

    # --- A: prod ---
    a_p, a_lo, a_hi, a_err = -1, -1, -1, None
    try:
        fc_a = forecast_responses(
            timeline, horizon_until=horizon_until, form_type=form["form_type"]
        )
        idx_a = idx_for_horizon(fc_a.future_dates, horizon_end)
        a_p = int(round(float(fc_a.future_cum.iloc[idx_a])))
        a_lo = int(round(float(fc_a.ci_lower.iloc[idx_a])))
        a_hi = int(round(float(fc_a.ci_upper.iloc[idx_a])))
    except ForecastError as e:
        a_err = str(e)

    # --- B: LogModel + delta CI ---
    b_p, b_lo, b_hi, b_r2, b_err = -1, -1, -1, 0.0, None
    try:
        # Continuous-time training: t (days from first), y = 1..n_train.
        t_train = (prefix - prefix.iloc[0]).dt.total_seconds().to_numpy() / 86400.0
        y_train = np.arange(1, n_train + 1, dtype=float)

        # horizon_end_days = days from first response to horizon_end
        horizon_end_days = (horizon_end - prefix.iloc[0]).total_seconds() / 86400.0
        # Запит на single point (the horizon end). Для consistency з A.
        t_future = np.array([horizon_end_days], dtype=float)

        y_pred, lo_arr, hi_arr, r2, _ = _fit_log_with_delta_ci(t_train, y_train, t_future)
        # Cumulative monotonicity floor: lo ≥ last_observed.
        b_p = int(round(max(float(y_pred[0]), float(n_train))))
        b_lo = int(round(max(float(lo_arr[0]), float(n_train))))
        b_hi = int(round(max(float(hi_arr[0]), float(b_p))))
        b_r2 = float(r2)
    except ForecastError as e:
        b_err = str(e)
    except Exception as e:  # numerical edge case
        b_err = f"numeric: {type(e).__name__}: {e}"

    return ABPoint(
        form_id=form["form_id"],
        shape=form["shape"],
        form_type=form["form_type"],
        n_total=n_total,
        n_train=n_train,
        horizon_hours=float(horizon_hours),
        truth=truth,
        a_point=a_p,
        a_lo=a_lo,
        a_hi=a_hi,
        a_error=a_err,
        b_point=b_p,
        b_lo=b_lo,
        b_hi=b_hi,
        b_r2=b_r2,
        b_error=b_err,
    )


# ---------- metrics --------------------------------------------------------


def _per_method_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        if r["truth"] <= 0:
            continue
        for prefix, method in [("a", "a_prod"), ("b", "b_log_delta")]:
            p = r[f"{prefix}_point"]
            lo = r[f"{prefix}_lo"]
            hi = r[f"{prefix}_hi"]
            if p < 0:
                continue
            truth = float(r["truth"])
            ape = abs(truth - p) / truth
            hit = bool(lo <= truth <= hi)
            sharp_rel = (hi - lo) / truth
            width_abs = float(hi - lo)
            signed = (p - truth) / truth
            winkler = _winkler_score(truth, float(lo), float(hi))
            rows.append(
                {
                    "form_id": r["form_id"],
                    "shape": r["shape"],
                    "form_type": r["form_type"],
                    "n_train": r["n_train"],
                    "horizon_hours": r["horizon_hours"],
                    "truth": truth,
                    "method": method,
                    "point": p,
                    "lo": lo,
                    "hi": hi,
                    "ape": ape,
                    "hit_95": hit,
                    "sharpness": sharp_rel,
                    "width_abs": width_abs,
                    "signed_err": signed,
                    "winkler": winkler,
                }
            )
    return pd.DataFrame(rows)


def _agg_with_winkler(df: pd.DataFrame, group_cols) -> pd.DataFrame:
    agg = df.groupby(group_cols, observed=True).agg(
        n=("ape", "size"),
        mape_p50=("ape", "median"),
        coverage=("hit_95", "mean"),
        sharpness_p50=("sharpness", "median"),
        width_p50=("width_abs", "median"),
        width_p90=("width_abs", lambda s: s.quantile(0.90)),
        winkler_p50=("winkler", "median"),
        winkler_mean=("winkler", "mean"),
        bias=("signed_err", "median"),
    )
    agg["n"] = agg["n"].astype(int)
    agg["mape_p50"] = (agg["mape_p50"] * 100).round(1)
    agg["coverage"] = (agg["coverage"] * 100).round(1)
    agg["sharpness_p50"] = agg["sharpness_p50"].round(2)
    agg["width_p50"] = agg["width_p50"].round(1)
    agg["width_p90"] = agg["width_p90"].round(1)
    agg["winkler_p50"] = agg["winkler_p50"].round(1)
    agg["winkler_mean"] = agg["winkler_mean"].round(1)
    agg["bias"] = (agg["bias"] * 100).round(1)
    return agg


# ---------- figures --------------------------------------------------------


def _fig_winkler_per_type(metrics: pd.DataFrame) -> go.Figure:
    agg = metrics.groupby(["form_type", "method"])["winkler"].median().reset_index()
    fig = px.bar(
        agg,
        x="form_type",
        y="winkler",
        color="method",
        barmode="group",
        title="Winkler interval score per form_type (lower = better)",
        labels={"winkler": "Median Winkler score"},
    )
    return fig


def _fig_width_per_type(metrics: pd.DataFrame) -> go.Figure:
    agg = metrics.groupby(["form_type", "method"])["width_abs"].median().reset_index()
    fig = px.bar(
        agg,
        x="form_type",
        y="width_abs",
        color="method",
        barmode="group",
        title="Absolute CI width median per form_type",
        labels={"width_abs": "Median |upper − lower|"},
    )
    return fig


def _fig_coverage_per_type(metrics: pd.DataFrame) -> go.Figure:
    agg = metrics.groupby(["form_type", "method"])["hit_95"].mean().reset_index()
    agg["coverage_pct"] = agg["hit_95"] * 100
    fig = px.bar(
        agg,
        x="form_type",
        y="coverage_pct",
        color="method",
        barmode="group",
        title="Empirical coverage at 95% nominal per form_type",
        labels={"coverage_pct": "Coverage (%)"},
    )
    fig.add_hline(y=95, line_dash="dash", line_color="gray", annotation_text="Nominal 95%")
    return fig


# ---------- main -----------------------------------------------------------


def main(
    input_path,
    features_csv,
    form_types_csv,
    output_md,
    figures_dir,
    n_train_values,
    horizon_hours_values,
    min_n,
    limit,
):
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_path)
    input_hash = file_sha256_short(input_path)
    shapes = load_shapes(features_csv)
    form_types = load_form_types(form_types_csv)
    eligible, skipped = build_eligible_forms(df, shapes, form_types, min_n)
    if limit:
        eligible = eligible[:limit]

    print(f"Eligible forms: {len(eligible)} (of {df['FORM_ID'].nunique()})")
    print(f"Skipped: {skipped}")
    print(f"n_train: {n_train_values}, horizons: {horizon_hours_values}")

    points: list[ABPoint] = []
    a_failed = 0
    b_failed = 0
    for i, form in enumerate(eligible, 1):
        for n_train in n_train_values:
            for h in horizon_hours_values:
                p = _backtest_one(form, n_train, float(h))
                if p is None:
                    continue
                if p.a_error is not None:
                    a_failed += 1
                if p.b_error is not None:
                    b_failed += 1
                points.append(p)
        if i % 10 == 0 or i == len(eligible):
            print(f"  processed {i}/{len(eligible)} forms (a_fail={a_failed}, b_fail={b_failed})")

    points_df = pd.DataFrame([p.__dict__ for p in points])
    points_df.to_csv(figures_dir / "15_ab_points.csv", index=False)
    metrics = _per_method_metrics(points_df)
    metrics.to_csv(figures_dir / "15_ab_metrics.csv", index=False)

    by_method = _agg_with_winkler(metrics, "method")
    by_method_type = _agg_with_winkler(metrics, ["method", "form_type"])
    by_method_n_train = _agg_with_winkler(metrics, ["method", "n_train"])
    by_method_horizon = _agg_with_winkler(metrics, ["method", "horizon_hours"])

    # Win count: на скільки backtest points B має ЛОВІШИЙ Winkler.
    pivot = points_df.copy()
    pivot["a_winkler"] = [
        _winkler_score(t, lo, hi) if p >= 0 and t > 0 else float("inf")
        for t, lo, hi, p in zip(
            pivot["truth"], pivot["a_lo"], pivot["a_hi"], pivot["a_point"], strict=True
        )
    ]
    pivot["b_winkler"] = [
        _winkler_score(t, lo, hi) if p >= 0 and t > 0 else float("inf")
        for t, lo, hi, p in zip(
            pivot["truth"], pivot["b_lo"], pivot["b_hi"], pivot["b_point"], strict=True
        )
    ]
    valid = pivot[(pivot["a_point"] >= 0) & (pivot["b_point"] >= 0) & (pivot["truth"] > 0)]
    b_wins = int((valid["b_winkler"] < valid["a_winkler"]).sum())
    a_wins = int((valid["a_winkler"] < valid["b_winkler"]).sum())
    ties = int(len(valid) - b_wins - a_wins)
    win_rate_pct = round(b_wins / max(1, len(valid)) * 100, 1)

    figs = {
        "winkler_per_type": _fig_winkler_per_type(metrics),
        "width_per_type": _fig_width_per_type(metrics),
        "coverage_per_type": _fig_coverage_per_type(metrics),
    }
    fig_paths = {}
    for name, fig in figs.items():
        path = figures_dir / f"15_{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        fig_paths[name] = path.relative_to(output_md.parent)

    md = _render_markdown(
        n_forms_total=df["FORM_ID"].nunique(),
        n_forms_eligible=len(eligible),
        n_points=len(points_df),
        skipped=skipped,
        n_train_values=n_train_values,
        horizon_hours_values=horizon_hours_values,
        min_n=min_n,
        input_path=input_path,
        input_hash=input_hash,
        by_method=by_method,
        by_method_type=by_method_type,
        by_method_n_train=by_method_n_train,
        by_method_horizon=by_method_horizon,
        b_wins=b_wins,
        a_wins=a_wins,
        ties=ties,
        win_rate_pct=win_rate_pct,
        a_failed=a_failed,
        b_failed=b_failed,
        fig_paths=fig_paths,
    )
    output_md.write_text(md, encoding="utf-8")
    print(f"\nReport: {output_md}")
    print(
        f"B wins (Winkler): {b_wins}/{len(valid)} = {win_rate_pct}%, A wins: {a_wins}, ties: {ties}"
    )


def _render_markdown(
    *,
    n_forms_total,
    n_forms_eligible,
    n_points,
    skipped,
    n_train_values,
    horizon_hours_values,
    min_n,
    input_path,
    input_hash,
    by_method,
    by_method_type,
    by_method_n_train,
    by_method_horizon,
    b_wins,
    a_wins,
    ties,
    win_rate_pct,
    a_failed,
    b_failed,
    fig_paths,
) -> str:
    return f"""# 15 - LogModel + delta-method CI vs prod (Winkler A/B)

**Generated:** {datetime.now().isoformat(timespec="seconds")}
**Source:** `{input_path}` (sha256:`{input_hash}`)
**Forms:** {n_forms_total} -> {n_forms_eligible} eligible (N >= {min_n})
**n_train:** {n_train_values} - **horizons:** {horizon_hours_values}
**Backtest points:** {n_points}
**Skipped:** {skipped}
**Fit failures:** A={a_failed}, B={b_failed}

## Контекст pivot-у

Користувач показав скриншот де прод-CI [62, 818] на формі з truth ~60.
Точка добре (point ~70), але CI безкорисний. R²=0.92, fit excellent —
але NHPP + CALIBRATION_MULTIPLIER + per-type widening роздуває CI без
залежності від якості fit.

Цей бенчмарк тестує **structural alternative**: bypass NHPP+multiplier,
використати класичну delta-method CI (curve_fit + numerical Jacobian +
Student t quantile) на простій log model.

## Методи

- **A · prod** — `forecast_responses(timeline, form_type=ft)` з усіма
  калібровками (P7+P10+P11). Контроль.
- **B · log_delta** — fit `y = a·ln(t+1) + b` через scipy curve_fit,
  delta-method CI без post-hoc widening.

Primary metric — **Winkler interval score** (proper scoring rule):
`W = (U − L) + (2/α)·max(L−y, 0) + (2/α)·max(y−U, 0)`. Lower = better.

## Win count (Winkler, per backtest point)

| | Кількість | % |
|---|---:|---:|
| **B (log_delta) виграє** | {b_wins} | {win_rate_pct}% |
| A (prod) виграє | {a_wins} | {round(a_wins / max(1, b_wins + a_wins + ties) * 100, 1)}% |
| Ties | {ties} | {round(ties / max(1, b_wins + a_wins + ties) * 100, 1)}% |

## Global per method

{df_to_md(by_method)}

## Per form_type (decisive)

{df_to_md(by_method_type)}

## Per n_train

{df_to_md(by_method_n_train)}

## Per horizon_hours

{df_to_md(by_method_horizon)}

## Figures

- [Winkler score per type]({fig_paths["winkler_per_type"]})
- [Absolute CI width per type]({fig_paths["width_per_type"]})
- [Coverage per type]({fig_paths["coverage_per_type"]})

## Як читати

- **width_p50, width_p90** — медіанна/p90 АБСОЛЮТНА ширина CI у units of
  responses. Це те, що бачить користувач на скріншоті. ↓ = краще.
- **winkler_p50** — proper score. ↓ = краще. Виграш на цій метриці означає
  CI одночасно вужче І capture truth.
- **coverage** — традиційна. Метрика КОЛИШНЬОЇ оптимізації.

Якщо B має нижче winkler І нижче width при coverage >= 80% → структурний
сигнал що delta-CI прод-практичніший. Тоді треба:
1. Revertити P10/P11 (multipliers).
2. Перенести LogModel і delta-CI у `core/forecast/` як основний механізм.
3. Додати інші моделі (Logistic, AsympExp) з delta-CI як альтернативи.

## Артефакти

- `figures/15_ab_points.csv` — wide-format raw результати.
- `figures/15_ab_metrics.csv` — long-format per-method метрики з Winkler.
"""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    p.add_argument(
        "--input", type=Path, default=repo_root / "data" / "Form Timestamp Collection.csv"
    )
    p.add_argument(
        "--features-csv",
        type=Path,
        default=repo_root / "research" / "reports" / "figures" / "01_per_form_features.csv",
    )
    p.add_argument(
        "--form-types-csv",
        type=Path,
        default=repo_root / "research" / "reports" / "figures" / "07_form_types.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=repo_root / "research" / "reports" / "15_logmodel_delta_ci_ab.md",
    )
    p.add_argument(
        "--figures-dir", type=Path, default=repo_root / "research" / "reports" / "figures"
    )
    p.add_argument("--n-train", type=str, default=",".join(str(v) for v in N_TRAIN_VALUES))
    p.add_argument("--horizons", type=str, default=",".join(str(v) for v in HORIZON_HOURS))
    p.add_argument("--min-n", type=int, default=MIN_N_FOR_BACKTEST)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    n_train_values = tuple(int(v) for v in args.n_train.split(","))
    horizon_hours_values = tuple(float(v) for v in args.horizons.split(","))
    main(
        args.input,
        args.features_csv,
        args.form_types_csv,
        args.output,
        args.figures_dir,
        n_train_values,
        horizon_hours_values,
        args.min_n,
        args.limit,
    )
