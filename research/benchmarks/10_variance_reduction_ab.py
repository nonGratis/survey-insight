"""10_variance_reduction_ab.py — A/B/C/D на ранніх передбаченнях.

Після того як 09_ спростувала Poisson-naive blend, лишилися дві
гіпотези про зниження варіансу без зміни inductive bias:

- **B · Akaike averaging** — замість selection-by-AICc усереднюємо
  усі fitted моделі за вагами `wᵢ = exp(-ΔAICcᵢ/2) / Σ exp(-ΔAICcⱼ/2)`
  (Burnham & Anderson 2002). Якщо AICc гепи між моделями малі —
  selection це додаткове джерело шуму, averaging його прибирає.
- **C · CI scaling** — post-hoc масштабування CI half-width на ранніх
  cutoffs: ×1.5 при n_train ≤ 15, лінійно до ×1.0 на n_train = 30.
  Не торкає point estimate; ціль — закрити coverage gap (85% → ~95%).

Методи:

- **A · baseline** — поточний `forecast_responses` (контроль, реплікує 08_).
- **B · averaged** — Akaike-weighted ensemble (без CI scaling).
- **C · scaled** — A + post-hoc CI scaling.
- **D · averaged + scaled** — B + post-hoc CI scaling.

ЖОДНИХ змін у `core/forecast/` під час A/B. Якщо ≥1 з B/C/D виграє за
обома критеріями (MAPE↓ AND coverage→95%) — promote окремими атомарними
коміттами.

Запуск:
    .venv/Scripts/python.exe research/benchmarks/10_variance_reduction_ab.py
    .venv/Scripts/python.exe research/benchmarks/10_variance_reduction_ab.py --limit 5
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import OptimizeWarning

warnings.simplefilter("ignore", OptimizeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.forecast import ForecastError, forecast_responses  # noqa: E402
from core.forecast.calibration import apply_calibration_arrays  # noqa: E402
from core.forecast.intervals import nhpp_prediction_interval  # noqa: E402
from core.forecast.metrics import aicc, r_squared, rmse  # noqa: E402
from core.forecast.models import fit_model, models_for_n_points  # noqa: E402
from core.forecast.types import FittedModel  # noqa: E402
from core.timeline import build_timeline_from_timestamps  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)

# ---------- config ---------------------------------------------------------

MIN_N_FOR_BACKTEST = 10
CUTOFFS_DEFAULT = (0.1, 0.2, 0.3, 0.5, 0.7)
HORIZON_FRACTION = 0.25
MIN_TRAIN_POINTS = 5
MIN_DURATION_DAYS = 1.0 / 24.0
N_SIMS = 2000
RANDOM_SEED = 42

# CI scaling thresholds (method C).
CI_SCALE_LOW_N = 15  # ×CI_SCALE_MAX below this
CI_SCALE_HIGH_N = 30  # ×1.0 above this
CI_SCALE_MAX = 1.5

METHODS = ("baseline", "averaged", "scaled", "avg_scaled")


@dataclass(frozen=True)
class ABPoint:
    form_id: str
    shape: str
    n_total: int
    cutoff_frac: float
    n_train: int
    horizon_days: float
    truth: int
    # per method
    baseline_point: int
    baseline_lo: int
    baseline_hi: int
    averaged_point: int
    averaged_lo: int
    averaged_hi: int
    scaled_point: int
    scaled_lo: int
    scaled_hi: int
    avg_scaled_point: int
    avg_scaled_lo: int
    avg_scaled_hi: int
    # diagnostic
    n_models_avg: int
    ci_scale: float
    error: str | None


# ---------- shape lookup ---------------------------------------------------


def _load_shapes(features_csv: Path) -> dict[str, str]:
    if not features_csv.exists():
        raise FileNotFoundError(f"Run 01_dataset_overview.py first to generate {features_csv}")
    df = pd.read_csv(features_csv)
    return dict(zip(df["form_id"], df["shape"], strict=True))


# ---------- CI scaling (method C) ------------------------------------------


def _ci_scale_factor(n_train: int) -> float:
    if n_train <= CI_SCALE_LOW_N:
        return CI_SCALE_MAX
    if n_train >= CI_SCALE_HIGH_N:
        return 1.0
    # linear interp from CI_SCALE_MAX at LOW_N → 1.0 at HIGH_N
    frac = (n_train - CI_SCALE_LOW_N) / (CI_SCALE_HIGH_N - CI_SCALE_LOW_N)
    return CI_SCALE_MAX - frac * (CI_SCALE_MAX - 1.0)


def _scale_ci(
    point: float, lo: float, hi: float, n_train: int, last_observed: int
) -> tuple[float, float]:
    scale = _ci_scale_factor(n_train)
    half_lo = max(point - lo, 0.0)
    half_hi = max(hi - point, 0.0)
    new_lo = point - scale * half_lo
    new_hi = point + scale * half_hi
    # Cumulative floor: ci_lower не нижче last_observed.
    new_lo = max(new_lo, float(last_observed))
    return new_lo, new_hi


# ---------- Akaike averaging (method B) ------------------------------------


def _akaike_averaged_forecast(
    timestamps: pd.Series,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, int]:
    """Репліка inner-logic service.py з Akaike model averaging замість selection.

    Returns:
        (future_dates, future_cum_arr, ci_lower_arr, ci_upper_arr, n_models_used).

    Raises ForecastError якщо жодна модель не зійшлася.
    """
    ts = pd.to_datetime(timestamps).sort_values().reset_index(drop=True)
    n = len(ts)
    first_ts = ts.iloc[0].to_pydatetime()
    last_ts = ts.iloc[-1].to_pydatetime()

    t_train = ((ts - pd.Timestamp(first_ts)).dt.total_seconds() / 86400.0).to_numpy(dtype=float)
    y_train = np.arange(1, n + 1, dtype=float)
    last_observed = n

    duration_days = max((last_ts - first_ts).total_seconds() / 86400.0, MIN_DURATION_DAYS)
    horizon_days = max(int(np.ceil(duration_days * HORIZON_FRACTION)), 1)

    models = models_for_n_points(n)

    # Fit each model independently (НЕ використовуємо select_best_model, бо він
    # сортує і повертає одного переможця).
    fitted_list: list[FittedModel] = []
    for model in models:
        try:
            params, pcov = fit_model(model, t_train, y_train, None)
        except ForecastError:
            continue
        y_fitted = model.predict(t_train, *params)
        fitted_list.append(
            FittedModel(
                model=model,
                params=params,
                aicc=aicc(y_train, y_fitted, model.n_params),
                rmse=rmse(y_train, y_fitted),
                r_squared=r_squared(y_train, y_fitted),
                pcov=pcov,
            )
        )

    # Відсіюємо моделі з NaN AICc — це нестабільні fit-и на дуже малих N
    # (зазвичай SSE→0 або residuals degenerate). Чесно: вони ламали б і selector,
    # просто там через sort() NaN потрапляє у кінець.
    fitted_list = [f for f in fitted_list if np.isfinite(f.aicc)]
    if not fitted_list:
        raise ForecastError("no_model_converged_in_averaging")

    # Akaike weights з poprawkою на overflow: subtract min AICc.
    aiccs = np.array([f.aicc for f in fitted_list], dtype=float)
    delta = aiccs - aiccs.min()
    raw = np.exp(-delta / 2.0)
    weights = raw / raw.sum()

    # Future grid (same logic as service.py).
    last_known_day = pd.Timestamp(last_ts.date())
    future_dates = pd.date_range(
        start=last_known_day + timedelta(days=1), periods=horizon_days, freq="D"
    )
    t_future = ((future_dates - pd.Timestamp(first_ts)).total_seconds() / 86400.0).to_numpy(
        dtype=float
    )

    # For each model: run NHPP simulation, get mean + CI bands.
    means: list[np.ndarray] = []
    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    for fitted in fitted_list:
        # Свіжий rng з тим самим seed для відтворюваності.
        rng = np.random.default_rng(RANDOM_SEED)
        mean_arr, _median_arr, lo_arr, hi_arr = nhpp_prediction_interval(
            fitted, t_future, last_observed=last_observed, n_sims=N_SIMS, rng=rng
        )
        means.append(mean_arr)
        lowers.append(lo_arr)
        uppers.append(hi_arr)

    # Weighted ensemble.
    mean_arr = np.zeros_like(means[0])
    lower_arr = np.zeros_like(lowers[0])
    upper_arr = np.zeros_like(uppers[0])
    for w, m, lo, hi in zip(weights, means, lowers, uppers, strict=True):
        mean_arr += w * m
        lower_arr += w * lo
        upper_arr += w * hi

    # Apply same post-processing as service.py.
    future_cum_arr = np.maximum.accumulate(np.maximum(mean_arr, float(last_observed)))
    lower_arr, upper_arr = apply_calibration_arrays(future_cum_arr, lower_arr, upper_arr)
    min_half_width = np.maximum(future_cum_arr * 0.10, 5.0)
    lower_arr = np.minimum(lower_arr, future_cum_arr - min_half_width)
    upper_arr = np.maximum(upper_arr, future_cum_arr + min_half_width)
    lower_arr = np.maximum(lower_arr, float(last_observed))

    return future_dates, future_cum_arr, lower_arr, upper_arr, len(fitted_list)


# ---------- A/B core -------------------------------------------------------


def _ab_one(
    form_id: str,
    timestamps: pd.Series,
    shape: str,
    cutoff_frac: float,
) -> ABPoint | None:
    n_total = len(timestamps)
    n_train = int(round(cutoff_frac * n_total))
    if n_train < MIN_TRAIN_POINTS:
        return None

    ts_sorted = timestamps.sort_values().reset_index(drop=True)
    first_ts = ts_sorted.iloc[0]
    last_ts = ts_sorted.iloc[-1]
    if (last_ts - first_ts).total_seconds() <= 0:
        return None

    cutoff_ts = ts_sorted.iloc[n_train - 1]
    cutoff_span_seconds = (cutoff_ts - first_ts).total_seconds()
    if cutoff_span_seconds <= 0:
        return None

    horizon_seconds = max(cutoff_span_seconds * HORIZON_FRACTION, 86400.0)
    horizon_end = cutoff_ts + pd.Timedelta(seconds=horizon_seconds)
    truth = int((ts_sorted <= horizon_end).sum())

    prefix = ts_sorted.iloc[:n_train].tolist()
    prefix_dt = [t.to_pydatetime() for t in prefix]
    timeline = build_timeline_from_timestamps(prefix_dt)

    err: str | None = None

    # --- A: baseline ---
    try:
        fc_a = forecast_responses(timeline)
        idx_a = _idx_for_horizon(fc_a.future_dates, horizon_end)
        a_p = float(fc_a.future_cum.iloc[idx_a])
        a_lo = float(fc_a.ci_lower.iloc[idx_a])
        a_hi = float(fc_a.ci_upper.iloc[idx_a])
    except ForecastError as e:
        return ABPoint(
            form_id=form_id,
            shape=shape,
            n_total=n_total,
            cutoff_frac=cutoff_frac,
            n_train=n_train,
            horizon_days=horizon_seconds / 86400.0,
            truth=truth,
            baseline_point=-1,
            baseline_lo=-1,
            baseline_hi=-1,
            averaged_point=-1,
            averaged_lo=-1,
            averaged_hi=-1,
            scaled_point=-1,
            scaled_lo=-1,
            scaled_hi=-1,
            avg_scaled_point=-1,
            avg_scaled_lo=-1,
            avg_scaled_hi=-1,
            n_models_avg=0,
            ci_scale=1.0,
            error=f"baseline_failed: {e}",
        )

    # --- B: averaged ---
    try:
        future_dates_b, cum_b, lo_b_arr, hi_b_arr, n_models = _akaike_averaged_forecast(
            pd.Series(prefix_dt)
        )
        idx_b = _idx_for_horizon(future_dates_b, horizon_end)
        b_p = float(cum_b[idx_b])
        b_lo = float(lo_b_arr[idx_b])
        b_hi = float(hi_b_arr[idx_b])
    except ForecastError:
        # Якщо averaging падає, дублюємо baseline (degenerate).
        b_p, b_lo, b_hi = a_p, a_lo, a_hi
        n_models = 0
        err = "averaging_fallback_to_baseline"

    # --- C: baseline + scaled CI ---
    scale = _ci_scale_factor(n_train)
    c_lo, c_hi = _scale_ci(a_p, a_lo, a_hi, n_train, last_observed=n_train)

    # --- D: averaged + scaled CI ---
    d_lo, d_hi = _scale_ci(b_p, b_lo, b_hi, n_train, last_observed=n_train)

    return ABPoint(
        form_id=form_id,
        shape=shape,
        n_total=n_total,
        cutoff_frac=cutoff_frac,
        n_train=n_train,
        horizon_days=horizon_seconds / 86400.0,
        truth=truth,
        baseline_point=int(round(a_p)),
        baseline_lo=int(round(a_lo)),
        baseline_hi=int(round(a_hi)),
        averaged_point=int(round(b_p)),
        averaged_lo=int(round(b_lo)),
        averaged_hi=int(round(b_hi)),
        scaled_point=int(round(a_p)),
        scaled_lo=int(round(c_lo)),
        scaled_hi=int(round(c_hi)),
        avg_scaled_point=int(round(b_p)),
        avg_scaled_lo=int(round(d_lo)),
        avg_scaled_hi=int(round(d_hi)),
        n_models_avg=n_models,
        ci_scale=scale,
        error=err,
    )


def _idx_for_horizon(future_dates: pd.DatetimeIndex, horizon_end: pd.Timestamp) -> int:
    future_dates = pd.DatetimeIndex(future_dates)
    target_date = pd.Timestamp(horizon_end.normalize())
    if target_date <= future_dates[0]:
        return 0
    if target_date >= future_dates[-1]:
        return len(future_dates) - 1
    idx = int(np.searchsorted(future_dates, target_date, side="left"))
    return min(idx, len(future_dates) - 1)


# ---------- metrics --------------------------------------------------------


def _per_method_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        if r["truth"] <= 0:
            continue
        for m in METHODS:
            p = r[f"{m}_point"]
            lo = r[f"{m}_lo"]
            hi = r[f"{m}_hi"]
            if p < 0:
                continue
            ape = abs(r["truth"] - p) / r["truth"]
            hit = lo <= r["truth"] <= hi
            sharp = (hi - lo) / r["truth"]
            signed = (p - r["truth"]) / r["truth"]
            rows.append(
                {
                    "form_id": r["form_id"],
                    "shape": r["shape"],
                    "n_train": r["n_train"],
                    "cutoff_frac": r["cutoff_frac"],
                    "method": m,
                    "truth": r["truth"],
                    "point": p,
                    "lo": lo,
                    "hi": hi,
                    "ape": ape,
                    "hit_95": hit,
                    "sharpness": sharp,
                    "signed_err": signed,
                }
            )
    return pd.DataFrame(rows)


def _agg(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    agg = df.groupby(by, observed=True).agg(
        n=("ape", "size"),
        mape_p50=("ape", "median"),
        mape_p90=("ape", lambda s: s.quantile(0.90)),
        coverage=("hit_95", "mean"),
        sharpness_p50=("sharpness", "median"),
        bias=("signed_err", "median"),
    )
    agg["n"] = agg["n"].astype(int)
    agg["mape_p50"] = (agg["mape_p50"] * 100).round(1)
    agg["mape_p90"] = (agg["mape_p90"] * 100).round(1)
    agg["coverage"] = (agg["coverage"] * 100).round(1)
    agg["sharpness_p50"] = agg["sharpness_p50"].round(2)
    agg["bias"] = (agg["bias"] * 100).round(1)
    return agg


# ---------- figures --------------------------------------------------------


def _fig_per_cutoff(metrics: pd.DataFrame, value: str, ylabel: str, title: str) -> go.Figure:
    pivot = (
        metrics.groupby(["cutoff_frac", "method"])
        .agg(
            v=(
                "hit_95" if value == "coverage" else "ape",
                "mean" if value == "coverage" else "median",
            )
        )
        .reset_index()
    )
    pivot["v"] = pivot["v"] * 100
    fig = go.Figure()
    colors = {
        "baseline": "#1f77b4",
        "averaged": "#ff7f0e",
        "scaled": "#2ca02c",
        "avg_scaled": "#d62728",
    }
    for m in METHODS:
        sub = pivot[pivot["method"] == m]
        fig.add_bar(name=m, x=sub["cutoff_frac"].astype(str), y=sub["v"], marker_color=colors[m])
    fig.update_layout(
        title=title, barmode="group", xaxis_title="Cutoff fraction", yaxis_title=ylabel
    )
    if value == "coverage":
        fig.add_hline(y=95, line_dash="dash", line_color="gray", annotation_text="Nominal 95%")
    return fig


# ---------- main -----------------------------------------------------------


def _build_eligible(df: pd.DataFrame, shapes: dict[str, str], min_n: int):
    eligible = []
    skipped = {"too_few": 0, "insufficient_shape": 0, "no_span": 0}
    for form_id, group in df.groupby("FORM_ID"):
        ts = group["TIMESTAMP"].sort_values().reset_index(drop=True)
        n = len(ts)
        if n < min_n:
            skipped["too_few"] += 1
            continue
        shape = shapes.get(form_id, "unknown")
        if shape == "insufficient":
            skipped["insufficient_shape"] += 1
            continue
        if (ts.iloc[-1] - ts.iloc[0]).total_seconds() <= 0:
            skipped["no_span"] += 1
            continue
        eligible.append({"form_id": form_id, "timestamps": ts, "shape": shape})
    return eligible, skipped


def main(input_path, features_csv, output_md, figures_dir, cutoffs, min_n, limit):
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df = df.drop_duplicates(subset=["FORM_ID", "TIMESTAMP"]).reset_index(drop=True)
    input_hash = _file_sha256_short(input_path)

    shapes = _load_shapes(features_csv)
    eligible, skipped = _build_eligible(df, shapes, min_n)
    if limit:
        eligible = eligible[:limit]

    print(f"Eligible forms: {len(eligible)} (of {df['FORM_ID'].nunique()})")
    print(f"Skipped: {skipped}")
    print(f"Cutoffs: {cutoffs}")

    points: list[ABPoint] = []
    for i, form in enumerate(eligible, 1):
        for cutoff in cutoffs:
            p = _ab_one(form["form_id"], form["timestamps"], form["shape"], cutoff)
            if p is not None:
                points.append(p)
        if i % 10 == 0 or i == len(eligible):
            print(f"  processed {i}/{len(eligible)} forms...")

    points_df = pd.DataFrame([p.__dict__ for p in points])
    points_df.to_csv(figures_dir / "10_ab_points.csv", index=False)
    metrics = _per_method_metrics(points_df)
    metrics.to_csv(figures_dir / "10_ab_metrics.csv", index=False)

    by_method = _agg(metrics, ["method"]).reindex(list(METHODS))
    by_method_cutoff = _agg(metrics, ["method", "cutoff_frac"])
    early = metrics[metrics["cutoff_frac"].isin([0.1, 0.2])]
    by_method_shape_early = _agg(early, ["method", "shape"])

    figs = {
        "mape_per_cutoff": _fig_per_cutoff(
            metrics, "ape", "MAPE p50 (%)", "MAPE p50 за cutoff × method"
        ),
        "coverage_per_cutoff": _fig_per_cutoff(
            metrics, "coverage", "Coverage (%)", "Coverage за cutoff × method"
        ),
    }
    fig_paths = {}
    for name, fig in figs.items():
        path = figures_dir / f"10_{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        fig_paths[name] = path.relative_to(output_md.parent)

    md = _render_markdown(
        n_forms_total=df["FORM_ID"].nunique(),
        n_forms_eligible=len(eligible),
        n_points=int(len(metrics) // max(1, metrics["method"].nunique())),
        skipped=skipped,
        cutoffs=cutoffs,
        min_n=min_n,
        input_path=input_path,
        input_hash=input_hash,
        by_method=by_method,
        by_method_cutoff=by_method_cutoff,
        by_method_shape_early=by_method_shape_early,
        fig_paths=fig_paths,
    )
    output_md.write_text(md, encoding="utf-8")
    print(f"\nReport written to: {output_md}")


def _render_markdown(
    *,
    n_forms_total,
    n_forms_eligible,
    n_points,
    skipped,
    cutoffs,
    min_n,
    input_path,
    input_hash,
    by_method,
    by_method_cutoff,
    by_method_shape_early,
    fig_paths,
) -> str:
    return f"""# 10 — Variance-reduction A/B/C/D

**Generated:** {datetime.now().isoformat(timespec="seconds")}
**Source:** `{input_path}` (sha256:`{input_hash}`)
**Forms:** {n_forms_total} → {n_forms_eligible} eligible (N ≥ {min_n})
**Cutoffs:** {cutoffs} · **Horizon fraction:** {HORIZON_FRACTION}
**Backtest points / method:** {n_points}
**Skipped:** {skipped}

## Методи

- **A · baseline** — поточний `forecast_responses` (контроль).
- **B · averaged** — Akaike model averaging (refit усі 3, weight `exp(-ΔAICc/2)`).
- **C · scaled** — A + CI scale `×{CI_SCALE_MAX}` при n_train≤{CI_SCALE_LOW_N}, лінійно до ×1.0 на n_train={CI_SCALE_HIGH_N}.
- **D · avg_scaled** — B + той самий CI scale.

## Глобально (усі cutoffs)

{_df_to_md(by_method)}

## Decisive view: method × cutoff

{_df_to_md(by_method_cutoff)}

## Per-shape на ранніх cutoffs (0.1, 0.2)

{_df_to_md(by_method_shape_early)}

## Фігури

- [MAPE p50 за cutoff × method]({fig_paths["mape_per_cutoff"]})
- [Coverage за cutoff × method]({fig_paths["coverage_per_cutoff"]})

## Критерії promote

| Метрика | Поріг promote |
|---|---|
| MAPE@0.1, 0.2 | НЕ гірше за baseline (≤ +1pp) |
| Coverage@0.1, 0.2 | Ближче до 95% за baseline |
| MAPE@0.5, 0.7 | НЕ гірше за baseline (≤ +1pp) |
| Coverage@0.5, 0.7 | НЕ гірше за baseline (≥ −3pp) |

Якщо ≥1 з B/C/D відповідає — promote окремим коміттом. Інакше документуємо
як negative result.

## Артефакти

- `figures/10_ab_points.csv` — wide raw результати.
- `figures/10_ab_metrics.csv` — long per-method метрики.
"""


def _df_to_md(df: pd.DataFrame) -> str:
    df = df.reset_index()
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in headers) + " |",
        "|" + "|".join("---:" if i > 0 else "---" for i in range(len(headers))) + "|",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _file_sha256_short(path: Path, length: int = 12) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


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
        "--output",
        type=Path,
        default=repo_root / "research" / "reports" / "10_variance_reduction_ab.md",
    )
    p.add_argument(
        "--figures-dir", type=Path, default=repo_root / "research" / "reports" / "figures"
    )
    p.add_argument("--cutoffs", type=str, default=",".join(str(c) for c in CUTOFFS_DEFAULT))
    p.add_argument("--min-n", type=int, default=MIN_N_FOR_BACKTEST)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cutoffs = tuple(float(x) for x in args.cutoffs.split(","))
    main(
        args.input,
        args.features_csv,
        args.output,
        args.figures_dir,
        cutoffs,
        args.min_n,
        args.limit,
    )
