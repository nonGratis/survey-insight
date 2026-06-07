"""08_full_dataset_backtest.py — діагностичний backtest на повному датасеті.

Розширює `02_rolling_origin_backtest.py` (не замінює його — той закріплений
як thesis-evidence baseline). Цілі:

- **Покриття:** `MIN_N_FOR_BACKTEST = 10` (було 30) — відкриває small-N зону,
  яка домінує prod-сценарій (свіжа форма у користувача).
- **Раннє передбачення:** cutoffs `(0.1, 0.2, 0.3, 0.5, 0.7)` — додає
  10%/20%-зрізи, де живуть нові форми.
- **Три нові осі категоризації** окрім shape:
    * `tempo` — burst / daily_flow / long_tail / sporadic (за median Δt + CV).
    * `n_class` — tiny / small / medium / large / huge.
    * `duration_class` — hours / days / weeks / months.
- **Cross-tab heat-maps:** shape × n_class, tempo × cutoff.
- **Per-cell failure dumps:** для (shape × n_class) з MAPE_p50 > 30% або
  coverage < 75% — HTML з 4–6 найгіршими формами (truth + forecast + CI).

Запуск:
    .venv/Scripts/python.exe research/benchmarks/08_full_dataset_backtest.py
    .venv/Scripts/python.exe research/benchmarks/08_full_dataset_backtest.py --limit 5

Артефакти:
    research/reports/08_full_dataset_backtest.md
    research/reports/figures/08_*.html
    research/reports/figures/08_backtest_points.csv
    research/reports/figures/08_backtest_metrics.csv
    research/reports/figures/08_failures/<shape>_<n_class>.html
"""

from __future__ import annotations

import argparse
import hashlib
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
from scipy.optimize import OptimizeWarning

warnings.simplefilter("ignore", OptimizeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.forecast import ForecastError, forecast_responses  # noqa: E402
from core.timeline import build_timeline_from_timestamps  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)

# ---------- config ---------------------------------------------------------

MIN_N_FOR_BACKTEST = 10
CUTOFFS_DEFAULT = (0.1, 0.2, 0.3, 0.5, 0.7)
HORIZON_FRACTION = 0.25
MIN_TRAIN_POINTS = 5  # mirror of service.py guard

# Failure dump thresholds.
FAIL_MAPE_P50 = 0.30
FAIL_COVERAGE = 0.75
FAIL_MAX_FORMS_PER_CELL = 6

SHAPE_ORDER = ["linear", "logarithmic", "logistic", "late_burst", "ill_fit", "unknown"]
N_CLASS_ORDER = ["tiny", "small", "medium", "large", "huge"]
TEMPO_ORDER = ["burst", "daily_flow", "long_tail", "sporadic"]
DURATION_ORDER = ["hours", "days", "weeks", "months"]


@dataclass(frozen=True)
class BacktestPoint:
    form_id: str
    shape: str
    n_class: str
    tempo: str
    duration_class: str
    n_total: int
    cutoff_frac: float
    n_train: int
    horizon_days: float
    truth: int
    point_estimate: int
    ci_lower: int
    ci_upper: int
    error: str | None


# ---------- shape lookup ---------------------------------------------------


def _load_shapes(features_csv: Path) -> dict[str, str]:
    if not features_csv.exists():
        raise FileNotFoundError(f"Run 01_dataset_overview.py first to generate {features_csv}")
    df = pd.read_csv(features_csv)
    return dict(zip(df["form_id"], df["shape"], strict=True))


# ---------- taxonomies -----------------------------------------------------


def _n_class(n_total: int) -> str:
    if n_total < 10:
        return "tiny"
    if n_total < 30:
        return "small"
    if n_total < 100:
        return "medium"
    if n_total < 1000:
        return "large"
    return "huge"


def _tempo_class(timestamps: pd.Series) -> str:
    """Класифікує форму за тиском подій у часі.

    - burst: median Δt < 0.5h (потік >2 відповідей/год).
    - daily_flow: 0.5h ≤ median Δt < 12h, CV(Δt) < 1.5 (рівномірний денний потік).
    - long_tail: median Δt ≥ 12h, CV(Δt) < 2.0 (повільно, рівномірно).
    - sporadic: CV(Δt) ≥ 2.0 — тиша + сплески, або CV високий взагалі.
    """
    ts = timestamps.sort_values().reset_index(drop=True)
    if len(ts) < 2:
        return "sporadic"
    deltas_sec = ts.diff().dropna().dt.total_seconds()
    deltas_sec = deltas_sec[deltas_sec > 0]
    if len(deltas_sec) == 0:
        return "burst"
    median_h = float(deltas_sec.median()) / 3600.0
    mean_s = float(deltas_sec.mean())
    std_s = float(deltas_sec.std()) if len(deltas_sec) > 1 else 0.0
    cv = std_s / mean_s if mean_s > 0 else 0.0
    if cv >= 2.0:
        return "sporadic"
    if median_h < 0.5:
        return "burst"
    if median_h < 12.0 and cv < 1.5:
        return "daily_flow"
    return "long_tail"


def _duration_class(timestamps: pd.Series) -> str:
    ts = timestamps.sort_values()
    span_h = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 3600.0
    if span_h < 24:
        return "hours"
    if span_h < 24 * 7:
        return "days"
    if span_h < 24 * 30:
        return "weeks"
    return "months"


# ---------- backtest core --------------------------------------------------


def _backtest_one(
    form_id: str,
    timestamps: pd.Series,
    shape: str,
    n_class: str,
    tempo: str,
    duration_class: str,
    cutoff_frac: float,
) -> BacktestPoint | None:
    n_total = len(timestamps)
    n_train = int(round(cutoff_frac * n_total))
    if n_train < MIN_TRAIN_POINTS:
        return None

    ts_sorted = timestamps.sort_values().reset_index(drop=True)
    first_ts = ts_sorted.iloc[0]
    last_ts = ts_sorted.iloc[-1]
    span_seconds = (last_ts - first_ts).total_seconds()
    if span_seconds <= 0:
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
    try:
        fc = forecast_responses(timeline)
        future_dates = pd.DatetimeIndex(fc.future_dates)
        target_date = pd.Timestamp(horizon_end.normalize())
        if target_date <= future_dates[0]:
            idx = 0
        elif target_date >= future_dates[-1]:
            idx = len(future_dates) - 1
        else:
            idx = int(np.searchsorted(future_dates, target_date, side="left"))
            idx = min(idx, len(future_dates) - 1)
        return BacktestPoint(
            form_id=form_id,
            shape=shape,
            n_class=n_class,
            tempo=tempo,
            duration_class=duration_class,
            n_total=n_total,
            cutoff_frac=cutoff_frac,
            n_train=n_train,
            horizon_days=horizon_seconds / 86400.0,
            truth=truth,
            point_estimate=int(round(fc.future_cum.iloc[idx])),
            ci_lower=int(round(fc.ci_lower.iloc[idx])),
            ci_upper=int(round(fc.ci_upper.iloc[idx])),
            error=None,
        )
    except ForecastError as e:
        return BacktestPoint(
            form_id=form_id,
            shape=shape,
            n_class=n_class,
            tempo=tempo,
            duration_class=duration_class,
            n_total=n_total,
            cutoff_frac=cutoff_frac,
            n_train=n_train,
            horizon_days=horizon_seconds / 86400.0,
            truth=truth,
            point_estimate=-1,
            ci_lower=-1,
            ci_upper=-1,
            error=str(e),
        )


# ---------- metrics --------------------------------------------------------


def _compute_metrics(points: pd.DataFrame) -> pd.DataFrame:
    """Per-row метрики; копія з 02_, узгоджена 1-в-1 для апле-ту-апле порівняння."""
    ok = points[points["error"].isna()].copy()
    ok = ok[ok["truth"] > 0]
    ok["ape"] = (ok["truth"] - ok["point_estimate"]).abs() / ok["truth"]
    ok["hit_95"] = (ok["ci_lower"] <= ok["truth"]) & (ok["truth"] <= ok["ci_upper"])
    ok["sharpness"] = (ok["ci_upper"] - ok["ci_lower"]) / ok["truth"]
    ok["signed_err"] = (ok["point_estimate"] - ok["truth"]) / ok["truth"]

    def _mode(row):
        if row["truth"] < row["ci_lower"]:
            return "overconfident_high"
        if row["truth"] > row["ci_upper"]:
            return "overconfident_low"
        return "in_ci"

    ok["mode"] = ok.apply(_mode, axis=1)
    return ok


def _agg_by(metrics: pd.DataFrame, group_cols: list[str] | str) -> pd.DataFrame:
    """Per-group метрики. Підтримує single col або list для cross-tab."""
    agg = metrics.groupby(group_cols, observed=True).agg(
        n_points=("ape", "size"),
        mape_p50=("ape", "median"),
        mape_p90=("ape", lambda s: s.quantile(0.90)),
        coverage=("hit_95", "mean"),
        sharpness_p50=("sharpness", "median"),
        bias=("signed_err", "median"),
    )
    agg["n_points"] = agg["n_points"].astype(int)
    agg["mape_p50"] = (agg["mape_p50"] * 100).round(1)
    agg["mape_p90"] = (agg["mape_p90"] * 100).round(1)
    agg["coverage"] = (agg["coverage"] * 100).round(1)
    agg["sharpness_p50"] = agg["sharpness_p50"].round(2)
    agg["bias"] = (agg["bias"] * 100).round(1)
    return agg


# ---------- figures --------------------------------------------------------


def _figure_heatmap(metrics: pd.DataFrame, row: str, col: str, value: str, title: str) -> go.Figure:
    pivot = metrics.pivot_table(
        index=row,
        columns=col,
        values=value,
        aggfunc=("median" if value in {"ape", "signed_err", "sharpness"} else "mean"),
    )
    if value == "hit_95":
        pivot = pivot * 100
        color_scale = "RdYlGn"
        zmid = 95.0
    elif value == "ape":
        pivot = pivot * 100
        color_scale = "RdYlGn_r"
        zmid = 25.0
    else:
        color_scale = "Viridis"
        zmid = None
    fig = px.imshow(
        pivot,
        text_auto=".1f",
        color_continuous_scale=color_scale,
        title=title,
        labels={"color": value},
        aspect="auto",
    )
    if zmid is not None:
        fig.update_traces(zmid=zmid)
    return fig


def _figure_per_axis_bars(metrics: pd.DataFrame, axis: str, order: list[str]) -> go.Figure:
    agg = _agg_by(metrics, axis).reindex([x for x in order if x in metrics[axis].unique()])
    fig = go.Figure()
    fig.add_bar(name="MAPE_p50 (%)", x=agg.index, y=agg["mape_p50"], yaxis="y1")
    fig.add_bar(name="Coverage (%)", x=agg.index, y=agg["coverage"], yaxis="y2")
    fig.update_layout(
        title=f"MAPE & coverage за {axis}",
        yaxis=dict(title="MAPE p50 (%)", side="left"),
        yaxis2=dict(title="Coverage (%)", overlaying="y", side="right", range=[0, 100]),
        barmode="group",
    )
    return fig


def _figure_failure_modes(metrics: pd.DataFrame) -> go.Figure:
    df = metrics.groupby(["shape", "mode"], observed=True).size().reset_index(name="count")
    fig = px.bar(
        df,
        x="shape",
        y="count",
        color="mode",
        title="Failure modes за shape",
        category_orders={
            "shape": SHAPE_ORDER,
            "mode": ["in_ci", "overconfident_low", "overconfident_high"],
        },
        color_discrete_map={
            "in_ci": "#2ca02c",
            "overconfident_low": "#d62728",
            "overconfident_high": "#ff7f0e",
        },
    )
    return fig


def _figure_form_curves(
    df_pts: pd.DataFrame,
    ts_by_form: dict[str, pd.Series],
    title: str,
) -> go.Figure:
    """Per-form panel: truth curve + cutoff + forecast point + CI band."""
    fig = go.Figure()
    palette = px.colors.qualitative.Set2
    for i, (_, row) in enumerate(df_pts.iterrows()):
        color = palette[i % len(palette)]
        form_id = row["form_id"]
        ts = ts_by_form[form_id].sort_values().reset_index(drop=True)
        cum_x = ts.tolist()
        cum_y = list(range(1, len(ts) + 1))
        cutoff_ts = ts.iloc[row["n_train"] - 1]
        horizon_seconds = (cutoff_ts - ts.iloc[0]).total_seconds() * HORIZON_FRACTION
        horizon_seconds = max(horizon_seconds, 86400.0)
        horizon_end = cutoff_ts + pd.Timedelta(seconds=horizon_seconds)
        short_id = form_id[:10] + "…"
        fig.add_trace(
            go.Scatter(
                x=cum_x,
                y=cum_y,
                mode="lines",
                name=f"{short_id} truth",
                line=dict(color=color, width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[cutoff_ts],
                y=[row["n_train"]],
                mode="markers",
                marker=dict(color=color, size=10, symbol="x"),
                name=f"{short_id} cutoff",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[horizon_end, horizon_end],
                y=[row["ci_lower"], row["ci_upper"]],
                mode="lines+markers",
                line=dict(color=color, width=3),
                marker=dict(size=8),
                name=f"{short_id} CI",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[horizon_end],
                y=[row["point_estimate"]],
                mode="markers",
                marker=dict(color=color, size=12, symbol="diamond"),
                name=f"{short_id} point (truth={row['truth']})",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Час",
        yaxis_title="Cumulative responses",
        height=500,
    )
    return fig


# ---------- main -----------------------------------------------------------


def _build_eligible_forms(
    df: pd.DataFrame, shapes: dict[str, str], min_n: int
) -> tuple[list[dict], dict[str, int]]:
    """Повертає список eligible форм + skip-stats."""
    eligible: list[dict] = []
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
        eligible.append(
            {
                "form_id": form_id,
                "timestamps": ts,
                "shape": shape,
                "n_class": _n_class(n),
                "tempo": _tempo_class(ts),
                "duration_class": _duration_class(ts),
            }
        )
    return eligible, skipped


def main(
    input_path: Path,
    features_csv: Path,
    output_md: Path,
    figures_dir: Path,
    cutoffs: tuple[float, ...],
    min_n: int,
    limit: int | None,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = figures_dir / "08_failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df = df.drop_duplicates(subset=["FORM_ID", "TIMESTAMP"]).reset_index(drop=True)
    input_hash = _file_sha256_short(input_path)

    shapes = _load_shapes(features_csv)
    eligible, skipped = _build_eligible_forms(df, shapes, min_n)

    if limit:
        eligible = eligible[:limit]

    print(f"Eligible forms: {len(eligible)} (of {df['FORM_ID'].nunique()})")
    print(f"Skipped: {skipped}")
    print(f"Cutoffs: {cutoffs}")
    print(f"Total backtests (upper bound): {len(eligible) * len(cutoffs)}")

    points: list[BacktestPoint] = []
    ts_by_form: dict[str, pd.Series] = {}
    for i, form in enumerate(eligible, 1):
        ts_by_form[form["form_id"]] = form["timestamps"]
        for cutoff in cutoffs:
            point = _backtest_one(
                form_id=form["form_id"],
                timestamps=form["timestamps"],
                shape=form["shape"],
                n_class=form["n_class"],
                tempo=form["tempo"],
                duration_class=form["duration_class"],
                cutoff_frac=cutoff,
            )
            if point is not None:
                points.append(point)
        if i % 25 == 0 or i == len(eligible):
            print(f"  processed {i}/{len(eligible)} forms...")

    points_df = pd.DataFrame([p.__dict__ for p in points])
    metrics = _compute_metrics(points_df)

    points_df.to_csv(figures_dir / "08_backtest_points.csv", index=False)
    metrics.to_csv(figures_dir / "08_backtest_metrics.csv", index=False)

    # Per-axis aggregations.
    by_shape = _agg_by(metrics, "shape")
    by_n_class = _agg_by(metrics, "n_class").reindex(
        [c for c in N_CLASS_ORDER if c in metrics["n_class"].unique()]
    )
    by_tempo = _agg_by(metrics, "tempo").reindex(
        [c for c in TEMPO_ORDER if c in metrics["tempo"].unique()]
    )
    by_duration = _agg_by(metrics, "duration_class").reindex(
        [c for c in DURATION_ORDER if c in metrics["duration_class"].unique()]
    )
    by_cutoff = _agg_by(metrics, "cutoff_frac")

    # Cross-tabs (also computed as DataFrames for the report).
    by_shape_n = _agg_by(metrics, ["shape", "n_class"])
    by_tempo_cutoff = _agg_by(metrics, ["tempo", "cutoff_frac"])

    # Baseline sanity: subset N>=30 + cutoffs in {0.3, 0.5, 0.7} should resemble 02_.
    baseline_mask = (metrics["n_total"] >= 30) & (metrics["cutoff_frac"].isin([0.3, 0.5, 0.7]))
    baseline_subset = metrics[baseline_mask]
    baseline_summary = {
        "n_points": int(len(baseline_subset)),
        "mape_p50": float(baseline_subset["ape"].median() * 100) if len(baseline_subset) else 0.0,
        "coverage": float(baseline_subset["hit_95"].mean() * 100) if len(baseline_subset) else 0.0,
        "bias": float(baseline_subset["signed_err"].median() * 100)
        if len(baseline_subset)
        else 0.0,
    }

    # Figures.
    figs: dict[str, go.Figure] = {
        "shape_bars": _figure_per_axis_bars(metrics, "shape", SHAPE_ORDER),
        "n_class_bars": _figure_per_axis_bars(metrics, "n_class", N_CLASS_ORDER),
        "tempo_bars": _figure_per_axis_bars(metrics, "tempo", TEMPO_ORDER),
        "duration_bars": _figure_per_axis_bars(metrics, "duration_class", DURATION_ORDER),
        "cutoff_bars": _figure_per_axis_bars(
            metrics, "cutoff_frac", sorted(metrics["cutoff_frac"].unique().tolist())
        ),
        "heat_mape_shape_n": _figure_heatmap(
            metrics, "shape", "n_class", "ape", "MAPE p50 (%) — shape × n_class"
        ),
        "heat_cov_shape_n": _figure_heatmap(
            metrics, "shape", "n_class", "hit_95", "Coverage (%) — shape × n_class"
        ),
        "heat_mape_tempo_cutoff": _figure_heatmap(
            metrics, "tempo", "cutoff_frac", "ape", "MAPE p50 (%) — tempo × cutoff"
        ),
        "heat_cov_tempo_cutoff": _figure_heatmap(
            metrics, "tempo", "cutoff_frac", "hit_95", "Coverage (%) — tempo × cutoff"
        ),
        "failure_modes": _figure_failure_modes(metrics),
    }
    fig_paths: dict[str, Path] = {}
    for name, fig in figs.items():
        path = figures_dir / f"08_{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        fig_paths[name] = path.relative_to(output_md.parent)

    # Per-cell failure dumps.
    failure_cells: list[dict] = []
    for (shape, n_class), cell in by_shape_n.iterrows():
        if cell["n_points"] < 3:
            continue
        bad = cell["mape_p50"] / 100.0 > FAIL_MAPE_P50 or cell["coverage"] / 100.0 < FAIL_COVERAGE
        if not bad:
            continue
        sub = metrics[(metrics["shape"] == shape) & (metrics["n_class"] == n_class)]
        worst = sub.sort_values("ape", ascending=False).head(FAIL_MAX_FORMS_PER_CELL)
        if worst.empty:
            continue
        fname = f"{shape}_{n_class}.html"
        fpath = failures_dir / fname
        title = f"Failure dump · shape={shape} · n_class={n_class} · MAPE_p50={cell['mape_p50']}% · cov={cell['coverage']}%"
        fig = _figure_form_curves(worst, ts_by_form, title)
        fig.write_html(fpath, include_plotlyjs="cdn")
        failure_cells.append(
            {
                "shape": shape,
                "n_class": n_class,
                "n_points": int(cell["n_points"]),
                "mape_p50": float(cell["mape_p50"]),
                "coverage": float(cell["coverage"]),
                "bias": float(cell["bias"]),
                "path": fpath.relative_to(output_md.parent),
            }
        )

    failure_cells.sort(key=lambda c: c["mape_p50"], reverse=True)

    md = _render_markdown(
        n_rows=len(df),
        n_forms_total=df["FORM_ID"].nunique(),
        n_forms_eligible=len(eligible),
        n_points=len(metrics),
        skipped=skipped,
        cutoffs=cutoffs,
        min_n=min_n,
        input_path=input_path,
        input_hash=input_hash,
        baseline_summary=baseline_summary,
        by_shape=by_shape,
        by_n_class=by_n_class,
        by_tempo=by_tempo,
        by_duration=by_duration,
        by_cutoff=by_cutoff,
        by_shape_n=by_shape_n,
        by_tempo_cutoff=by_tempo_cutoff,
        metrics=metrics,
        fig_paths=fig_paths,
        failure_cells=failure_cells,
    )
    output_md.write_text(md, encoding="utf-8")
    print(f"\nReport written to: {output_md}")
    print(f"Failure cells: {len(failure_cells)}")


def _render_markdown(
    *,
    n_rows: int,
    n_forms_total: int,
    n_forms_eligible: int,
    n_points: int,
    skipped: dict[str, int],
    cutoffs: tuple[float, ...],
    min_n: int,
    input_path: Path,
    input_hash: str,
    baseline_summary: dict,
    by_shape: pd.DataFrame,
    by_n_class: pd.DataFrame,
    by_tempo: pd.DataFrame,
    by_duration: pd.DataFrame,
    by_cutoff: pd.DataFrame,
    by_shape_n: pd.DataFrame,
    by_tempo_cutoff: pd.DataFrame,
    metrics: pd.DataFrame,
    fig_paths: dict[str, Path],
    failure_cells: list[dict],
) -> str:
    global_mape = metrics["ape"].median() * 100
    global_coverage = metrics["hit_95"].mean() * 100
    global_sharpness = metrics["sharpness"].median()
    global_bias = metrics["signed_err"].median() * 100

    failures_section = _render_failures_section(failure_cells)

    return f"""# 08 — Full-dataset Diagnostic Backtest

**Generated:** {datetime.now().isoformat(timespec="seconds")}
**Source:** `{input_path}` (sha256:`{input_hash}`)
**Forms:** {n_forms_total} total → {n_forms_eligible} eligible (N ≥ {min_n}, shape ≠ insufficient)
**Cutoffs:** {cutoffs} · **Horizon fraction:** {HORIZON_FRACTION}
**Backtest points:** {n_points}
**Skipped:** {skipped}

## Глобальні метрики

| Метрика | Значення | Орієнтир |
|---|---:|---|
| **MAPE (median)** | {global_mape:.1f}% | < 15% — добре, < 25% — прийнятно |
| **Coverage 95% PI** | {global_coverage:.1f}% | Має бути ≈ 95% |
| **Sharpness (median)** | {global_sharpness:.2f} | width / truth |
| **Bias (median)** | {global_bias:+.1f}% | 0% — unbiased |

## Sanity vs 02_ baseline (N≥30, cutoffs ∈ {{0.3, 0.5, 0.7}})

Очікуємо ≈ 87% coverage / 24.6% MAPE / -12% bias з [02_backtest.md](02_backtest.md).
Якщо суттєво різниться — bug у pipeline 08_, не в моделі.

| Метрика | 08_ на baseline-subset | 02_ baseline |
|---|---:|---:|
| n_points | {baseline_summary["n_points"]} | 288 |
| MAPE_p50 | {baseline_summary["mape_p50"]:.1f}% | 24.6% |
| Coverage | {baseline_summary["coverage"]:.1f}% | 87% |
| Bias | {baseline_summary["bias"]:+.1f}% | -12% |

## Per-axis breakdowns

### Shape
{_df_to_md(by_shape)}

### N-class (нова таксономія)
{_df_to_md(by_n_class)}

### Tempo (нова таксономія: burst/daily/long-tail/sporadic)
{_df_to_md(by_tempo)}

### Duration (нова таксономія: hours/days/weeks/months)
{_df_to_md(by_duration)}

### Cutoff (як метод деградує з 10%→70% життя форми)
{_df_to_md(by_cutoff)}

## Cross-tab heat-maps

- [MAPE p50 (%) — shape × n_class]({fig_paths["heat_mape_shape_n"]})
- [Coverage (%) — shape × n_class]({fig_paths["heat_cov_shape_n"]})
- [MAPE p50 (%) — tempo × cutoff]({fig_paths["heat_mape_tempo_cutoff"]})
- [Coverage (%) — tempo × cutoff]({fig_paths["heat_cov_tempo_cutoff"]})

### Shape × N-class (точні значення, MAPE/coverage)
{_df_to_md(by_shape_n)}

### Tempo × cutoff
{_df_to_md(by_tempo_cutoff)}

## Bar-chart фігури

- [Shape: MAPE & coverage]({fig_paths["shape_bars"]})
- [N-class: MAPE & coverage]({fig_paths["n_class_bars"]})
- [Tempo: MAPE & coverage]({fig_paths["tempo_bars"]})
- [Duration: MAPE & coverage]({fig_paths["duration_bars"]})
- [Cutoff: MAPE & coverage]({fig_paths["cutoff_bars"]})
- [Failure modes за shape]({fig_paths["failure_modes"]})

## Failure spotlight (cells з MAPE_p50 > {int(FAIL_MAPE_P50 * 100)}% або coverage < {int(FAIL_COVERAGE * 100)}%)

{failures_section}

## Висновки для prod (input для fix-сесії)

Перечитати таблиці вище і відповісти на питання:

1. **Де метод системно недопрогнозує** (bias більш ніж -15%)?
   Перевірити `by_shape`, `by_n_class`, `by_cutoff`. Малі N + ранні cutoffs —
   очікуваний негативний bias через K_min relaxation, що калібрований на N≥30.
2. **Де coverage сильно < 95%**? Це cells, що потребують локальної
   per-cell calibration multiplier'у (зараз глобальний ×10).
3. **Найгірші cells у failure spotlight** — це або (а) нова модель потрібна
   (наприклад LinearModel для shape=linear), (б) занижений MIN_TRAIN_POINTS
   для конкретного shape, (в) HORIZON_FRACTION надто агресивний для коротких
   форм. Обрати найбільш impactful напрямок.
4. **Тempo×cutoff heat-map**: чи burst-форми ламаються на ранніх 10–20%?
   Якщо так — це core prod-проблема (свіжа форма з потужним сплеском у
   перший день, користувач дивиться на прогноз і бачить unrealistic точку).

## Артефакти

- `figures/08_backtest_points.csv` — повний log усіх backtest-runs (з tempo, n_class, duration_class).
- `figures/08_backtest_metrics.csv` — обчислені метрики.
- `figures/08_failures/<shape>_<n_class>.html` — per-cell failure curves.
"""


def _render_failures_section(failure_cells: list[dict]) -> str:
    if not failure_cells:
        return "_Жодної cell не перетнуло thresholds — система робастна на цьому датасеті._"
    lines = [
        "| shape | n_class | n_pts | MAPE_p50 | Coverage | Bias | Дамп |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for c in failure_cells:
        lines.append(
            f"| {c['shape']} | {c['n_class']} | {c['n_points']} | {c['mape_p50']:.1f}% "
            f"| {c['coverage']:.1f}% | {c['bias']:+.1f}% | [HTML]({c['path']}) |"
        )
    return "\n".join(lines)


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
        "--input",
        type=Path,
        default=repo_root / "data" / "Form Timestamp Collection.csv",
    )
    p.add_argument(
        "--features-csv",
        type=Path,
        default=repo_root / "research" / "reports" / "figures" / "01_per_form_features.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=repo_root / "research" / "reports" / "08_full_dataset_backtest.md",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=repo_root / "research" / "reports" / "figures",
    )
    p.add_argument(
        "--cutoffs",
        type=str,
        default=",".join(str(c) for c in CUTOFFS_DEFAULT),
        help="Comma-separated, напр. '0.1,0.2,0.3,0.5,0.7'.",
    )
    p.add_argument(
        "--min-n",
        type=int,
        default=MIN_N_FOR_BACKTEST,
        help="Мінімальне N для включення форми у backtest.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Sanity-режим: обмежити кількість eligible форм.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cutoffs = tuple(float(x) for x in args.cutoffs.split(","))
    main(
        input_path=args.input,
        features_csv=args.features_csv,
        output_md=args.output,
        figures_dir=args.figures_dir,
        cutoffs=cutoffs,
        min_n=args.min_n,
        limit=args.limit,
    )
