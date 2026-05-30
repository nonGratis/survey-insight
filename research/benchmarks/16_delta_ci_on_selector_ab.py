"""16_delta_ci_on_selector_ab.py — ізолюємо CI метод.

15_ показав що log model + delta-CI виграє на 2h horizon (Winkler -62%),
але програє на довгих горизонтах через log divergence. Висновок:
**проблема у виборі моделі, а не у delta-method**.

Цей бенчмарк ізолює "CI method" як єдину змінну:

- **A · prod** — поточний `forecast_responses` (NHPP+P7+P10+P11).
- **B · selector_delta** — selector обирає TУ САМУ модель (asymp_exp/
  logistic/gompertz). Точка identical. CI — delta-method на тій же моделі
  через її pcov + numerical Jacobian + Student t.

Якщо B виграє по Winkler — це структурний argument для революції в
calibration інфраструктурі. Power: тестується тільки CI computation,
point estimate і модель идентичні.

Запуск:
    .venv/Scripts/python.exe research/benchmarks/16_delta_ci_on_selector_ab.py
    .venv/Scripts/python.exe research/benchmarks/16_delta_ci_on_selector_ab.py --limit 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import OptimizeWarning
from scipy.stats import t as student_t

warnings.simplefilter("ignore", OptimizeWarning)
warnings.simplefilter("ignore", RuntimeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.forecast import ForecastError, forecast_responses  # noqa: E402
from core.forecast.models import models_for_n_points  # noqa: E402
from core.forecast.selector import select_best_model  # noqa: E402
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
MIN_DURATION_DAYS = 1.0 / 24.0


def _delta_method_ci(
    model,
    params: tuple[float, ...],
    pcov: np.ndarray,
    t_future: np.ndarray,
    n_train: int,
    confidence: float = CONFIDENCE,
    step: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classical delta-method CI на fitted SaturationModel.

    Returns (y_pred, lower, upper).
    pcov від scipy curve_fit (absolute_sigma=False) вже включає MSE-scaling.
    """
    params_arr = np.array(params, dtype=float)
    n_params = len(params_arr)
    df = max(1, n_train - n_params)

    y_pred = np.asarray(model.predict(t_future, *params_arr), dtype=float)

    # Numerical Jacobian: J[i, j] = d y(t_i) / d theta_j.
    J = np.zeros((len(t_future), n_params))
    for j in range(n_params):
        d = np.zeros(n_params)
        # Relative step для робастності навіть на параметрах різного scale.
        d[j] = step * max(1.0, abs(params_arr[j]))
        y_plus = np.asarray(model.predict(t_future, *(params_arr + d)), dtype=float)
        y_minus = np.asarray(model.predict(t_future, *(params_arr - d)), dtype=float)
        J[:, j] = (y_plus - y_minus) / (2.0 * d[j])

    # var(y(t)) = J pcov J^T (per-row).
    var_y = np.einsum("ij,jk,ik->i", J, pcov, J)
    var_y = np.maximum(var_y, 0.0)
    se_y = np.sqrt(var_y)

    t_val = float(student_t.ppf(1.0 - (1.0 - confidence) / 2.0, df))
    delta = t_val * se_y

    lower = y_pred - delta
    upper = y_pred + delta
    return y_pred, lower, upper


def _winkler_score(truth: float, lower: float, upper: float, alpha: float = ALPHA) -> float:
    width = max(0.0, upper - lower)
    return width + (2.0 / alpha) * max(lower - truth, 0.0) + (2.0 / alpha) * max(truth - upper, 0.0)


@dataclass(frozen=True)
class ABPoint:
    form_id: str
    shape: str
    form_type: str
    n_total: int
    n_train: int
    horizon_hours: float
    truth: int
    selected_model: str
    # A: prod (NHPP + multipliers)
    a_point: int
    a_lo: int
    a_hi: int
    a_error: str | None
    # B: same model, delta-CI
    b_point: int
    b_lo: int
    b_hi: int
    b_error: str | None


def _delta_forecast_at_horizon(
    timestamps: pd.Series,
    horizon_end: pd.Timestamp,
) -> tuple[str, int, int, int]:
    """Manually replicate selector's fit + delta-CI at horizon_end.

    Returns (model_name, point, lower, upper).
    """
    ts = pd.to_datetime(timestamps).sort_values().reset_index(drop=True)
    n = len(ts)
    if n < 5:
        raise ForecastError("delta: n_train < 5")
    first_ts = ts.iloc[0].to_pydatetime()
    last_ts = ts.iloc[-1].to_pydatetime()

    t_train = ((ts - pd.Timestamp(first_ts)).dt.total_seconds() / 86400.0).to_numpy(dtype=float)
    y_train = np.arange(1, n + 1, dtype=float)
    last_observed = n

    models = models_for_n_points(n)
    fitted = select_best_model(t_train, y_train, target=None, models=models)

    # Horizon as days from first_ts. select_best_model returned FittedModel
    # with params + pcov. Compute delta-CI at single point: horizon_end.
    last_known_day = pd.Timestamp(last_ts.date())
    future_start = last_known_day + timedelta(days=1)
    horizon_days = max(
        int(np.ceil((pd.Timestamp(horizon_end) - pd.Timestamp(last_ts)).total_seconds() / 86400.0)),
        1,
    )
    future_dates = pd.date_range(start=future_start, periods=horizon_days, freq="D")
    t_future_all = ((future_dates - pd.Timestamp(first_ts)).total_seconds() / 86400.0).to_numpy(
        dtype=float
    )
    idx = idx_for_horizon(future_dates, horizon_end)
    t_future_point = np.array([float(t_future_all[idx])])

    if fitted.pcov is None or not np.all(np.isfinite(fitted.pcov)):
        raise ForecastError("delta: pcov not available")

    y_pred, lower, upper = _delta_method_ci(
        fitted.model, fitted.params, fitted.pcov, t_future_point, n_train=n
    )

    # Cumulative monotonicity floor: lo >= last_observed; point >= last_observed.
    p = max(float(y_pred[0]), float(last_observed))
    lo = max(float(lower[0]), float(last_observed))
    hi = max(float(upper[0]), p)
    return fitted.model.name, int(round(p)), int(round(lo)), int(round(hi))


def _backtest_one(form: dict, n_train: int, horizon_hours: float) -> ABPoint | None:
    ts = form["timestamps"]
    n_total = len(ts)
    if n_total < n_train:
        return None
    ts_sorted = ts.sort_values().reset_index(drop=True)
    cutoff_ts = ts_sorted.iloc[n_train - 1]
    if (cutoff_ts - ts_sorted.iloc[0]).total_seconds() <= 0:
        return None
    horizon_seconds = horizon_hours * 3600.0
    horizon_end = cutoff_ts + pd.Timedelta(seconds=horizon_seconds)
    truth = int((ts_sorted <= horizon_end).sum())

    prefix = ts_sorted.iloc[:n_train].reset_index(drop=True)
    prefix_dt = [t.to_pydatetime() for t in prefix.tolist()]
    timeline = build_timeline_from_timestamps(prefix_dt)
    horizon_until = pd.Timestamp(horizon_end)

    # A: prod
    a_p, a_lo, a_hi, a_err = -1, -1, -1, None
    selected = "?"
    try:
        fc_a = forecast_responses(
            timeline, horizon_until=horizon_until, form_type=form["form_type"]
        )
        idx_a = idx_for_horizon(fc_a.future_dates, horizon_end)
        a_p = int(round(float(fc_a.future_cum.iloc[idx_a])))
        a_lo = int(round(float(fc_a.ci_lower.iloc[idx_a])))
        a_hi = int(round(float(fc_a.ci_upper.iloc[idx_a])))
        selected = fc_a.model
    except ForecastError as e:
        a_err = str(e)

    # B: delta-CI on selector model
    b_p, b_lo, b_hi, b_err = -1, -1, -1, None
    try:
        sel_b, b_p, b_lo, b_hi = _delta_forecast_at_horizon(prefix, horizon_end)
        if selected == "?":
            selected = sel_b
    except ForecastError as e:
        b_err = str(e)
    except Exception as e:
        b_err = f"numeric: {type(e).__name__}: {e}"

    return ABPoint(
        form_id=form["form_id"],
        shape=form["shape"],
        form_type=form["form_type"],
        n_total=n_total,
        n_train=n_train,
        horizon_hours=float(horizon_hours),
        truth=truth,
        selected_model=selected,
        a_point=a_p,
        a_lo=a_lo,
        a_hi=a_hi,
        a_error=a_err,
        b_point=b_p,
        b_lo=b_lo,
        b_hi=b_hi,
        b_error=b_err,
    )


def _per_method_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        if r["truth"] <= 0:
            continue
        for prefix, method in [("a", "a_prod"), ("b", "b_selector_delta")]:
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
                    "selected_model": r["selected_model"],
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
        width_p50=("width_abs", "median"),
        width_p90=("width_abs", lambda s: s.quantile(0.90)),
        winkler_p50=("winkler", "median"),
        winkler_mean=("winkler", "mean"),
        bias=("signed_err", "median"),
    )
    agg["n"] = agg["n"].astype(int)
    agg["mape_p50"] = (agg["mape_p50"] * 100).round(1)
    agg["coverage"] = (agg["coverage"] * 100).round(1)
    agg["width_p50"] = agg["width_p50"].round(1)
    agg["width_p90"] = agg["width_p90"].round(1)
    agg["winkler_p50"] = agg["winkler_p50"].round(1)
    agg["winkler_mean"] = agg["winkler_mean"].round(1)
    agg["bias"] = (agg["bias"] * 100).round(1)
    return agg


def _fig_winkler_per_horizon(metrics: pd.DataFrame) -> go.Figure:
    agg = metrics.groupby(["horizon_hours", "method"])["winkler"].median().reset_index()
    fig = px.line(
        agg,
        x="horizon_hours",
        y="winkler",
        color="method",
        markers=True,
        log_x=True,
        title="Winkler score per horizon: A (prod) vs B (selector + delta-CI)",
        labels={"winkler": "Median Winkler score", "horizon_hours": "Horizon (h)"},
    )
    return fig


def _fig_width_per_horizon(metrics: pd.DataFrame) -> go.Figure:
    agg = metrics.groupby(["horizon_hours", "method"])["width_abs"].median().reset_index()
    fig = px.line(
        agg,
        x="horizon_hours",
        y="width_abs",
        color="method",
        markers=True,
        log_x=True,
        title="Median CI width per horizon (absolute)",
        labels={"width_abs": "Median |U − L|", "horizon_hours": "Horizon (h)"},
    )
    return fig


def _fig_coverage_per_horizon(metrics: pd.DataFrame) -> go.Figure:
    agg = metrics.groupby(["horizon_hours", "method"])["hit_95"].mean().reset_index()
    agg["cov_pct"] = agg["hit_95"] * 100
    fig = px.line(
        agg,
        x="horizon_hours",
        y="cov_pct",
        color="method",
        markers=True,
        log_x=True,
        title="Coverage per horizon",
        labels={"cov_pct": "Coverage (%)"},
    )
    fig.add_hline(y=95, line_dash="dash", line_color="gray", annotation_text="Nominal 95%")
    return fig


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
    print(f"n_train: {n_train_values}, horizons: {horizon_hours_values}")

    points: list[ABPoint] = []
    a_fail = 0
    b_fail = 0
    for i, form in enumerate(eligible, 1):
        for n_train in n_train_values:
            for h in horizon_hours_values:
                p = _backtest_one(form, n_train, float(h))
                if p is None:
                    continue
                if p.a_error is not None:
                    a_fail += 1
                if p.b_error is not None:
                    b_fail += 1
                points.append(p)
        if i % 10 == 0 or i == len(eligible):
            print(f"  processed {i}/{len(eligible)} forms (a_fail={a_fail}, b_fail={b_fail})")

    points_df = pd.DataFrame([p.__dict__ for p in points])
    points_df.to_csv(figures_dir / "16_ab_points.csv", index=False)
    metrics = _per_method_metrics(points_df)
    metrics.to_csv(figures_dir / "16_ab_metrics.csv", index=False)

    by_method = _agg_with_winkler(metrics, "method")
    by_method_horizon = _agg_with_winkler(metrics, ["method", "horizon_hours"])
    by_method_type = _agg_with_winkler(metrics, ["method", "form_type"])
    by_method_n_train = _agg_with_winkler(metrics, ["method", "n_train"])
    by_method_selected = _agg_with_winkler(metrics, ["method", "selected_model"])

    # Win count per horizon (хочемо побачити горизонт-залежність).
    valid = points_df[
        (points_df["a_point"] >= 0) & (points_df["b_point"] >= 0) & (points_df["truth"] > 0)
    ].copy()
    valid["a_winkler"] = [
        _winkler_score(t, lo, hi)
        for t, lo, hi in zip(valid["truth"], valid["a_lo"], valid["a_hi"], strict=True)
    ]
    valid["b_winkler"] = [
        _winkler_score(t, lo, hi)
        for t, lo, hi in zip(valid["truth"], valid["b_lo"], valid["b_hi"], strict=True)
    ]
    win_rates = []
    for h in horizon_hours_values:
        sub = valid[valid["horizon_hours"] == h]
        if len(sub) == 0:
            continue
        b_wins = int((sub["b_winkler"] < sub["a_winkler"]).sum())
        win_rates.append(
            {
                "horizon_h": h,
                "n": len(sub),
                "b_wins": b_wins,
                "win_rate_pct": round(b_wins / len(sub) * 100, 1),
            }
        )
    win_rates_df = pd.DataFrame(win_rates)

    figs = {
        "winkler_per_horizon": _fig_winkler_per_horizon(metrics),
        "width_per_horizon": _fig_width_per_horizon(metrics),
        "coverage_per_horizon": _fig_coverage_per_horizon(metrics),
    }
    fig_paths = {}
    for name, fig in figs.items():
        path = figures_dir / f"16_{name}.html"
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
        by_method_horizon=by_method_horizon,
        by_method_type=by_method_type,
        by_method_n_train=by_method_n_train,
        by_method_selected=by_method_selected,
        win_rates_df=win_rates_df,
        a_fail=a_fail,
        b_fail=b_fail,
        fig_paths=fig_paths,
    )
    output_md.write_text(md, encoding="utf-8")
    print(f"\nReport: {output_md}")
    print("\nWin rates per horizon:")
    print(win_rates_df.to_string(index=False))


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
    by_method_horizon,
    by_method_type,
    by_method_n_train,
    by_method_selected,
    win_rates_df,
    a_fail,
    b_fail,
    fig_paths,
) -> str:
    return f"""# 16 - Selector model + delta-CI vs prod (ізолюємо CI method)

**Generated:** {datetime.now().isoformat(timespec="seconds")}
**Source:** `{input_path}` (sha256:`{input_hash}`)
**Forms:** {n_forms_total} -> {n_forms_eligible} eligible (N >= {min_n})
**n_train:** {n_train_values} - **horizons:** {horizon_hours_values}
**Backtest points:** {n_points}
**Skipped:** {skipped}
**Fit failures:** A={a_fail}, B={b_fail}

## Контекст

15_ показав що log+delta виграє Winkler на 2h horizon (-62%), але loses
на 168h (log diverges). Висновок: проблема НЕ у delta-method, а у виборі
моделі.

Цей бенчмарк ізолює "CI method" як змінну:
- Обидва методи використовують ТУ САМУ модель з AICc selector
  (asymp_exp / logistic / gompertz)
- Точка identical в обох
- Різниця: A використовує NHPP+P7+P10+P11 multipliers, B використовує
  classical delta-method CI на pcov моделі

Якщо B виграє Winkler глобально — це аргумент revert P7/P10/P11.

## Win rates по horizon

{df_to_md(win_rates_df.set_index("horizon_h"))}

## Global per method

{df_to_md(by_method)}

## Per horizon (decisive view)

{df_to_md(by_method_horizon)}

## Per form_type

{df_to_md(by_method_type)}

## Per n_train

{df_to_md(by_method_n_train)}

## Per selected model

{df_to_md(by_method_selected)}

## Figures

- [Winkler per horizon (A vs B)]({fig_paths["winkler_per_horizon"]})
- [Width per horizon]({fig_paths["width_per_horizon"]})
- [Coverage per horizon]({fig_paths["coverage_per_horizon"]})

## Інтерпретація

- **B wins Winkler globally** + не падає на довгих горизонтах (бо
  saturating моделі не diverge) → revert P7/P10/P11 і впровадити
  delta-CI як стандарт. Це **революція** у calibration модулі.
- **B wins на 2h-6h, loses на >24h** → реалізувати hybrid: delta-CI для
  short horizon, NHPP для long. Менш агресивно.
- **B loses глобально** → 15_ результат був специфічний до log model,
  delta-CI само по собі не виграє. Документувати negative і повернутися
  до Tier 2 (LinearModel, Bass).

## Артефакти

- `figures/16_ab_points.csv` — wide raw.
- `figures/16_ab_metrics.csv` — long per-method метрики з Winkler.
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
        default=repo_root / "research" / "reports" / "16_delta_ci_on_selector_ab.md",
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
