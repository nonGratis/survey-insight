"""03_bias_investigation.py — глибока декомпозиція -15% bias.

Backtest 02 показав systematic median bias = -15% (модель занижує). Це
скрипт перевіряє 9 гіпотез, локалізує root cause, і вимірює effect-size
від конкретних фіксів.

Гіпотези (H1–H9):
  H1: K_min=last_observed змушує плато → модель не вірить у зростання.
  H2: Initial guess b0=0.05 / r=0.3 занадто повільний.
  H3: AICc bias toward concave моделей → predispose to saturation.
  H4: Bias залежить від cutoff_frac (більше даних → менший bias).
  H5: Bias росте з horizon_days (compound error).
  H6: Per-shape: різні bias на різних shape-категоріях.
  H7: "K hit floor" → модель forced до плато.
  H8: (skipped — truth-counting checked у 02).
  H9: Жодна модель не вловлює прискорення → fundamental limit.

Контрольні re-fit experiments:
  E1: K_min relaxation — change `_capacity_bounds` floor from `last`
      to `last * 1.5`, re-run backtest, compare bias.
  E2: Force linear baseline — fit просто y=a*t+b на префікс, compare bias.

Output: research/reports/03_bias_investigation.md з cells, графіками,
і конкретним recommendation для P4 фіксу в production-коді.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import OptimizeWarning

warnings.simplefilter("ignore", OptimizeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.forecast.intervals import nhpp_prediction_interval  # noqa: E402
from core.forecast.models import (  # noqa: E402
    AsymptoticExpModel,
    GompertzModel,
    LogisticModel,
)
from core.forecast.selector import select_best_model  # noqa: E402
from core.forecast.types import FittedModel  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)


# ---------- experiments ----------------------------------------------------


def _backtest_one(
    timestamps: list[pd.Timestamp],
    cutoff_frac: float,
    fit_fn,
) -> dict | None:
    """Один backtest-point з custom fit_fn для контрольних experiments.

    fit_fn(t_train, y_train) → (point_estimate_at_horizon, ci_lower, ci_upper,
                                 winner_name, k_estimate, k_floor)
    """
    n_total = len(timestamps)
    n_train = int(round(cutoff_frac * n_total))
    if n_train < 5:
        return None

    ts_sorted = sorted(timestamps)
    first_ts = ts_sorted[0]
    last_ts = ts_sorted[-1]
    span_seconds = (last_ts - first_ts).total_seconds()
    if span_seconds <= 0:
        return None

    cutoff_ts = ts_sorted[n_train - 1]
    cutoff_span_seconds = (cutoff_ts - first_ts).total_seconds()
    if cutoff_span_seconds <= 0:
        return None

    horizon_seconds = max(cutoff_span_seconds * 0.25, 86400.0)
    horizon_end = cutoff_ts + pd.Timedelta(seconds=horizon_seconds)
    truth = sum(1 for t in ts_sorted if t <= horizon_end)

    t_train = np.array([(t - first_ts).total_seconds() / 86400.0 for t in ts_sorted[:n_train]])
    y_train = np.arange(1, n_train + 1, dtype=float)

    try:
        result = fit_fn(t_train, y_train, horizon_seconds)
        if result is None:
            return None
        point, ci_lo, ci_hi, winner, k_est, k_floor = result
        return {
            "n_total": n_total,
            "cutoff_frac": cutoff_frac,
            "n_train": n_train,
            "horizon_days": horizon_seconds / 86400.0,
            "truth": truth,
            "point_estimate": int(round(point)),
            "ci_lower": int(round(ci_lo)),
            "ci_upper": int(round(ci_hi)),
            "winner": winner,
            "k_estimate": float(k_est),
            "k_floor": float(k_floor),
            "k_at_floor": bool(abs(k_est - k_floor) < 0.01),
        }
    except Exception:  # noqa: BLE001
        return None


def _fit_default(t_train, y_train, horizon_seconds):
    """Default fit — як у production."""
    fitted = select_best_model(t_train, y_train, target=None)
    return _run_fitted(fitted, t_train, y_train, horizon_seconds)


def _fit_relaxed_k(t_train, y_train, horizon_seconds):
    """E1: K_min relaxed to last·1.5. Тимчасово підмінимо bounds-функції."""
    fitted = _select_with_relaxed_bounds(t_train, y_train)
    return _run_fitted(fitted, t_train, y_train, horizon_seconds)


def _fit_linear_baseline(t_train, y_train, horizon_seconds):
    """E2: проста linear regression як baseline."""
    a, b = np.polyfit(t_train, y_train, 1)
    horizon_days = horizon_seconds / 86400.0
    t_future = np.arange(t_train[-1] + 1.0, t_train[-1] + np.ceil(horizon_days) + 1.0)
    if len(t_future) == 0:
        return None
    # Point: simple linear forecast at end of horizon.
    point = a * t_future[-1] + b
    # CI: residual-based bootstrap (1000 resamples)
    residuals = y_train - (a * t_train + b)
    rng = np.random.default_rng(42)
    futures = []
    for _ in range(1000):
        idx = rng.integers(0, len(residuals), size=len(residuals))
        resampled = (a * t_train + b) + residuals[idx]
        a_b, b_b = np.polyfit(t_train, resampled, 1)
        futures.append(a_b * t_future[-1] + b_b)
    ci_lo = np.percentile(futures, 2.5)
    ci_hi = np.percentile(futures, 97.5)
    return point, ci_lo, ci_hi, "linear_baseline", point, 0.0


def _run_fitted(fitted: FittedModel, t_train, y_train, horizon_seconds):
    """Run NHPP on a fitted model — повертає (point, ci_lo, ci_hi, winner, K, K_floor)."""
    last_observed = int(y_train[-1])
    horizon_days = horizon_seconds / 86400.0
    t_future = np.arange(t_train[-1] + 1.0, t_train[-1] + np.ceil(horizon_days) + 1.0)
    if len(t_future) == 0:
        t_future = np.array([t_train[-1] + 1.0])
    rng = np.random.default_rng(42)
    mean_cum, ci_lo, ci_hi = nhpp_prediction_interval(
        fitted, t_future, last_observed=last_observed, n_sims=2000, rng=rng
    )
    point = max(mean_cum[-1], float(last_observed))
    # Extract K (asymptote) from params: для всіх 3 моделей перший param — K/a.
    k_est = fitted.params[0]
    # K_floor — нижня межа з bounds моделі.
    bounds_low, _ = fitted.model.bounds(y_train, target=None)
    k_floor = bounds_low[0]
    return point, ci_lo[-1], ci_hi[-1], fitted.model.name, k_est, k_floor


def _select_with_relaxed_bounds(t_train, y_train):
    """Той самий селектор, але з К_min := last·1.5 для всіх моделей."""

    class _RelaxedLogistic(LogisticModel):
        def bounds(self, y, target):
            (k_min, r_lo, t0_lo), (k_max, r_hi, t0_hi) = super().bounds(y, target)
            return (max(float(y[-1]) * 1.5, k_min), r_lo, t0_lo), (k_max, r_hi, t0_hi)

    class _RelaxedGompertz(GompertzModel):
        def bounds(self, y, target):
            (k_min, r_lo, t0_lo), (k_max, r_hi, t0_hi) = super().bounds(y, target)
            return (max(float(y[-1]) * 1.5, k_min), r_lo, t0_lo), (k_max, r_hi, t0_hi)

    class _RelaxedAsympExp(AsymptoticExpModel):
        def bounds(self, y, target):
            (a_min, b_lo, c_lo), (a_max, b_hi, c_hi) = super().bounds(y, target)
            return (max(float(y[-1]) * 1.5, a_min), b_lo, c_lo), (a_max, b_hi, c_hi)

    models = (_RelaxedLogistic(), _RelaxedGompertz(), _RelaxedAsympExp())
    return select_best_model(t_train, y_train, target=None, models=models)


# ---------- aggregation ----------------------------------------------------


def _compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["truth"] > 0]
    df["ape"] = (df["truth"] - df["point_estimate"]).abs() / df["truth"]
    df["signed_err"] = (df["point_estimate"] - df["truth"]) / df["truth"]
    df["hit_95"] = (df["ci_lower"] <= df["truth"]) & (df["truth"] <= df["ci_upper"])
    df["k_to_truth_ratio"] = df["k_estimate"] / df["truth"]
    return df


def _bias_summary(metrics: pd.DataFrame, group_col: str) -> pd.DataFrame:
    agg = metrics.groupby(group_col, observed=True).agg(
        n=("signed_err", "size"),
        bias_p50=("signed_err", "median"),
        bias_p25=("signed_err", lambda s: s.quantile(0.25)),
        bias_p75=("signed_err", lambda s: s.quantile(0.75)),
        mape_p50=("ape", "median"),
        coverage=("hit_95", "mean"),
    )
    agg["bias_p50"] = (agg["bias_p50"] * 100).round(1)
    agg["bias_p25"] = (agg["bias_p25"] * 100).round(1)
    agg["bias_p75"] = (agg["bias_p75"] * 100).round(1)
    agg["mape_p50"] = (agg["mape_p50"] * 100).round(1)
    agg["coverage"] = (agg["coverage"] * 100).round(1)
    return agg


# ---------- figures --------------------------------------------------------


def _fig_bias_per_winner(metrics: pd.DataFrame) -> go.Figure:
    fig = px.box(
        metrics,
        x="winner",
        y="signed_err",
        color="shape",
        title="Bias decomposition: per winning model × shape",
        labels={"signed_err": "(point - truth) / truth"},
        log_y=False,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig


def _fig_bias_vs_horizon(metrics: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        metrics,
        x="horizon_days",
        y="signed_err",
        color="shape",
        size="n_train",
        log_x=True,
        hover_data=[
            "form_id",
            "n_train",
            "truth",
            "point_estimate",
            "winner",
            "k_estimate",
            "k_floor",
        ],
        title="Bias vs horizon (logarithmic x-axis)",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig


def _fig_k_at_floor(metrics: pd.DataFrame) -> go.Figure:
    freq = metrics.groupby(["winner", "k_at_floor"]).size().unstack(fill_value=0)
    freq["floor_pct"] = freq.get(True, 0) / freq.sum(axis=1) * 100
    fig = px.bar(
        freq.reset_index(),
        x="winner",
        y="floor_pct",
        title="% of fits where K stuck at lower bound (saturation forced)",
        labels={"floor_pct": "% K-at-floor"},
        text=freq["floor_pct"].round(1).astype(str) + "%",
    )
    fig.update_traces(textposition="outside")
    return fig


def _fig_k_ratio_vs_bias(metrics: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        metrics,
        x="k_to_truth_ratio",
        y="signed_err",
        color="winner",
        hover_data=["form_id", "shape", "truth", "k_estimate", "horizon_days"],
        title="Forecast asymptote / truth → bias",
        labels={"k_to_truth_ratio": "K_estimate / truth", "signed_err": "(point - truth) / truth"},
        log_x=True,
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="green")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig


def _fig_experiment_comparison(
    default: pd.DataFrame, relaxed: pd.DataFrame, linear: pd.DataFrame
) -> go.Figure:
    rows = []
    for label, df in [("default", default), ("relaxed_K", relaxed), ("linear_baseline", linear)]:
        rows.append(
            {
                "experiment": label,
                "bias_p50": df["signed_err"].median() * 100,
                "mape_p50": df["ape"].median() * 100,
                "coverage": df["hit_95"].mean() * 100,
            }
        )
    df_cmp = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=df_cmp["experiment"], y=df_cmp["bias_p50"], name="Bias %", marker_color="#d62728")
    )
    fig.add_trace(
        go.Bar(x=df_cmp["experiment"], y=df_cmp["mape_p50"], name="MAPE %", marker_color="#ff7f0e")
    )
    fig.add_trace(
        go.Bar(
            x=df_cmp["experiment"], y=df_cmp["coverage"], name="Coverage %", marker_color="#2ca02c"
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_hline(y=95, line_dash="dot", line_color="green", annotation_text="Coverage target")
    fig.update_layout(
        title="Experiment comparison: default vs relaxed-K vs linear-baseline",
        barmode="group",
    )
    return fig


# ---------- main -----------------------------------------------------------


def main(input_path: Path, features_csv: Path, output_md: Path, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df = df.drop_duplicates(subset=["FORM_ID", "TIMESTAMP"]).reset_index(drop=True)
    shapes_df = pd.read_csv(features_csv)
    shapes = dict(zip(shapes_df["form_id"], shapes_df["shape"], strict=True))

    print("Running 3 experiments × all eligible forms × 3 cutoffs...")
    eligible = []
    for fid, g in df.groupby("FORM_ID"):
        n = len(g)
        if n < 30:
            continue
        shape = shapes.get(fid, "unknown")
        if shape == "insufficient":
            continue
        eligible.append((fid, shape, g["TIMESTAMP"].tolist()))
    print(f"Eligible forms: {len(eligible)}")

    rows_default, rows_relaxed, rows_linear = [], [], []
    for i, (fid, shape, ts) in enumerate(eligible, 1):
        for cutoff in (0.3, 0.5, 0.7):
            for label, fit_fn, container in [
                ("default", _fit_default, rows_default),
                ("relaxed_K", _fit_relaxed_k, rows_relaxed),
                ("linear", _fit_linear_baseline, rows_linear),
            ]:
                pt = _backtest_one(ts, cutoff, fit_fn)
                if pt is not None:
                    pt["form_id"] = fid
                    pt["shape"] = shape
                    pt["experiment"] = label
                    container.append(pt)
        if i % 20 == 0:
            print(f"  {i}/{len(eligible)} forms processed...")

    default_df = _compute_metrics(pd.DataFrame(rows_default))
    relaxed_df = _compute_metrics(pd.DataFrame(rows_relaxed))
    linear_df = _compute_metrics(pd.DataFrame(rows_linear))

    # Save raw experiment outputs
    default_df.to_csv(figures_dir / "03_default_metrics.csv", index=False)
    relaxed_df.to_csv(figures_dir / "03_relaxed_metrics.csv", index=False)
    linear_df.to_csv(figures_dir / "03_linear_metrics.csv", index=False)

    # ----- Hypothesis tests -----
    by_winner = _bias_summary(default_df, "winner")
    by_shape = _bias_summary(default_df, "shape")
    by_cutoff = _bias_summary(default_df, "cutoff_frac")
    default_df["horizon_bucket"] = pd.cut(
        default_df["horizon_days"],
        bins=[0, 1, 3, 7, 30, 1000],
        labels=["<1d", "1-3d", "3-7d", "7-30d", ">30d"],
    )
    by_horizon = _bias_summary(default_df, "horizon_bucket")
    default_df["n_bucket"] = pd.cut(
        default_df["n_train"],
        bins=[0, 15, 30, 100, 1000, 100000],
        labels=["<15", "15-30", "30-100", "100-1k", "1k+"],
    )
    by_nbucket = _bias_summary(default_df, "n_bucket")

    floor_freq = default_df.groupby("winner", observed=True)["k_at_floor"].mean() * 100

    # ----- Figures -----
    figs = {
        "bias_per_winner": _fig_bias_per_winner(default_df),
        "bias_vs_horizon": _fig_bias_vs_horizon(default_df),
        "k_at_floor": _fig_k_at_floor(default_df),
        "k_ratio_vs_bias": _fig_k_ratio_vs_bias(default_df),
        "experiment_comparison": _fig_experiment_comparison(default_df, relaxed_df, linear_df),
    }
    fig_paths: dict[str, Path] = {}
    for name, fig in figs.items():
        path = figures_dir / f"03_{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        fig_paths[name] = path.relative_to(output_md.parent)

    md = _render_markdown(
        input_path=input_path,
        input_hash=_file_sha256_short(input_path),
        n_forms=len(eligible),
        n_points=len(default_df),
        default_df=default_df,
        relaxed_df=relaxed_df,
        linear_df=linear_df,
        by_winner=by_winner,
        by_shape=by_shape,
        by_cutoff=by_cutoff,
        by_horizon=by_horizon,
        by_nbucket=by_nbucket,
        floor_freq=floor_freq,
        fig_paths=fig_paths,
    )
    output_md.write_text(md, encoding="utf-8")
    print(f"\nReport written to: {output_md}")


def _render_markdown(
    *,
    input_path,
    input_hash,
    n_forms,
    n_points,
    default_df,
    relaxed_df,
    linear_df,
    by_winner,
    by_shape,
    by_cutoff,
    by_horizon,
    by_nbucket,
    floor_freq,
    fig_paths,
) -> str:

    def _summary(df, name):
        return (
            f"- **{name}**: "
            f"bias_p50 = {df['signed_err'].median() * 100:+.1f}%, "
            f"MAPE = {df['ape'].median() * 100:.1f}%, "
            f"coverage = {df['hit_95'].mean() * 100:.1f}%"
        )

    default_bias = default_df["signed_err"].median() * 100
    relaxed_bias = relaxed_df["signed_err"].median() * 100
    linear_bias = linear_df["signed_err"].median() * 100

    # Statistical test: Wilcoxon signed-rank on paired (default vs relaxed)
    from scipy.stats import wilcoxon

    paired = default_df.merge(relaxed_df, on=["form_id", "cutoff_frac"], suffixes=("_def", "_rel"))
    if len(paired) >= 10:
        wstat, wpval = wilcoxon(paired["signed_err_def"], paired["signed_err_rel"])
        stat_line = f"Wilcoxon W={wstat:.0f}, p={wpval:.4g} (default vs relaxed-K, paired)"
    else:
        stat_line = "Wilcoxon: not enough paired points."

    # H1 verdict
    h1_delta = relaxed_bias - default_bias
    h1_verdict = (
        f"**H1 (K_min floor): {'✅ CONFIRMED' if abs(h1_delta) > 5 else '⚠️ INCONCLUSIVE'}** — "
        f"релаксація K_min → bias change of {h1_delta:+.1f}pp. {stat_line}"
    )

    # H9 verdict
    h9_delta = linear_bias - default_bias
    h9_verdict = (
        f"**H9 (model family limit): {'✅ CONFIRMED' if h9_delta > 5 else '❌ REJECTED'}** — "
        f"linear baseline bias = {linear_bias:+.1f}% vs default {default_bias:+.1f}% "
        f"(Δ = {h9_delta:+.1f}pp). "
        f"{'Linear краще → saturating-моделі за-saturate.' if h9_delta > 5 else 'Linear не виграє → проблема НЕ в model family.'}"
    )

    # H7 verdict
    avg_floor_freq = floor_freq.mean()
    h7_verdict = (
        f"**H7 (K hits floor): {'✅ CONFIRMED' if avg_floor_freq > 30 else '⚠️ PARTIAL'}** — "
        f"в середньому {avg_floor_freq:.1f}% фітів закінчуються з K = K_min (плато forced)."
    )

    return f"""# 03 — Bias Investigation (P4)

**Generated:** {datetime.now().isoformat(timespec="seconds")}
**Source:** `{input_path}` (sha256:`{input_hash}`)
**Eligible forms:** {n_forms} · **Backtest points (default):** {n_points}

## TL;DR

{_summary(default_df, "Default (production)")}
{_summary(relaxed_df, "Relaxed-K (E1: K_min = last·1.5)")}
{_summary(linear_df, "Linear baseline (E2)")}

{h1_verdict}

{h7_verdict}

{h9_verdict}

## Гіпотези й тести

### H3/H6: Bias per winning model

{_df_to_md(by_winner)}

### Per-shape bias

{_df_to_md(by_shape)}

### H4: Bias per cutoff_frac

{_df_to_md(by_cutoff)}

### H5: Bias per horizon-bucket

{_df_to_md(by_horizon)}

### Bias per N-bucket

{_df_to_md(by_nbucket)}

### H7: Frequency of K stuck at lower bound

{_series_to_md(floor_freq, "winner", "% K-at-floor")}

## Контрольні experiments

### E1 — Relaxed K_min (last → last·1.5)

| Метрика | Default | Relaxed-K | Δ |
|---|---:|---:|---:|
| Bias (median) | {default_bias:+.1f}% | {relaxed_bias:+.1f}% | {h1_delta:+.1f}pp |
| MAPE (median) | {default_df["ape"].median() * 100:.1f}% | {relaxed_df["ape"].median() * 100:.1f}% | {(relaxed_df["ape"].median() - default_df["ape"].median()) * 100:+.1f}pp |
| Coverage 95% | {default_df["hit_95"].mean() * 100:.1f}% | {relaxed_df["hit_95"].mean() * 100:.1f}% | {(relaxed_df["hit_95"].mean() - default_df["hit_95"].mean()) * 100:+.1f}pp |

### E2 — Linear baseline (просте y = a·t + b)

| Метрика | Default | Linear baseline | Δ |
|---|---:|---:|---:|
| Bias (median) | {default_bias:+.1f}% | {linear_bias:+.1f}% | {h9_delta:+.1f}pp |
| MAPE (median) | {default_df["ape"].median() * 100:.1f}% | {linear_df["ape"].median() * 100:.1f}% | {(linear_df["ape"].median() - default_df["ape"].median()) * 100:+.1f}pp |
| Coverage 95% | {default_df["hit_95"].mean() * 100:.1f}% | {linear_df["hit_95"].mean() * 100:.1f}% | {(linear_df["hit_95"].mean() - default_df["hit_95"].mean()) * 100:+.1f}pp |

## Графіки

- [Bias per winning model × shape (boxplot)]({fig_paths["bias_per_winner"]})
- [Bias vs horizon_days (scatter)]({fig_paths["bias_vs_horizon"]})
- [% K-at-floor per winner (bar)]({fig_paths["k_at_floor"]})
- [K/truth ratio vs bias (scatter)]({fig_paths["k_ratio_vs_bias"]})
- [Experiment comparison (bar)]({fig_paths["experiment_comparison"]})

## Recommendation для P4 фіксу в core/forecast/

На основі чисел вище — конкретний фікс випливає з:
- Якщо H1 ✅ і Δbias істотний → **relax K_min в `_capacity_bounds`** (production fix).
- Якщо H9 ✅ → треба **додати LinearModel у селектор** як 4-у модель.
- Якщо H7 ✅ і >30% фітів сидять на floor → це самостійна причина.

Якщо всі три ❌ — bias має іншу природу (initial guess, AICc-bias, multi-wave forms). Тоді потрібен наступний рівень investigation.

## Артефакти

- `figures/03_default_metrics.csv` — default backtest results
- `figures/03_relaxed_metrics.csv` — з релаксованим K_min
- `figures/03_linear_metrics.csv` — linear baseline
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


def _series_to_md(series: pd.Series, index_name: str, value_name: str) -> str:
    lines = [
        f"| {index_name} | {value_name} |",
        "|---|---:|",
    ]
    for idx, val in series.items():
        lines.append(f"| {idx} | {val:.1f}% |")
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
        default=repo_root / "research" / "reports" / "03_bias_investigation.md",
    )
    p.add_argument(
        "--figures-dir", type=Path, default=repo_root / "research" / "reports" / "figures"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.input, args.features_csv, args.output, args.figures_dir)
