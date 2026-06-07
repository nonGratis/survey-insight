"""02_rolling_origin_backtest.py — honest evaluation of forecast_responses.

Rolling-origin cross-validation (Hyndman & Athanasopoulos, Forecasting:
Principles and Practice, гл. 5.10):

  Для кожної форми з N >= MIN_N_FOR_BACKTEST (default 30):
    Для кожного cutoff ∈ {0.3, 0.5, 0.7}:
      prefix = timestamps[: int(cutoff * N)]
      forecast = forecast_responses(timeline(prefix))
      horizon_end_t = first_ts + (last_ts - first_ts) * (cutoff + horizon_fraction)
      truth = #responses with ts <= horizon_end_t (з повного датасету)

      Метрики:
        - point_err = |truth - point_estimate| / truth        (MAPE-component)
        - hit_95    = ci_lower <= truth <= ci_upper           (coverage)
        - sharpness = (ci_upper - ci_lower) / truth           (CI ширина)
        - crps     = continuous ranked probability score      (probabilistic)

Aggregation:
  - By shape category (з 01_dataset_overview features)
  - By N-bucket
  - Глобальна reliability diagram: nominal vs empirical coverage

Output: research/reports/02_backtest.md з конкретними цифрами для
diagnose'у failure modes (where MAPE > 20%, where coverage < 80%, тощо).

Запуск:
    .venv/Scripts/python.exe research/benchmarks/02_rolling_origin_backtest.py
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

# Дозволяємо імпортувати core.forecast з research-скриптів.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.forecast import ForecastError, forecast_responses  # noqa: E402
from core.timeline import build_timeline_from_timestamps  # noqa: E402

# Заглушити шумний core.logger під час бенчмарку.
logging.getLogger().setLevel(logging.WARNING)

MIN_N_FOR_BACKTEST = 30
CUTOFFS = (0.3, 0.5, 0.7)
HORIZON_FRACTION = 0.25  # частка span'у вперед від cutoff'у


@dataclass(frozen=True)
class BacktestPoint:
    form_id: str
    shape: str  # з 01_per_form_features
    n_total: int
    cutoff_frac: float
    n_train: int
    horizon_days: float
    truth: int
    point_estimate: int
    ci_lower: int
    ci_upper: int
    error: str | None  # ForecastError message або None


# ---------- shape lookup ---------------------------------------------------


def _load_shapes(features_csv: Path) -> dict[str, str]:
    """Form_id → shape category, із 01_per_form_features.csv."""
    if not features_csv.exists():
        raise FileNotFoundError(f"Run 01_dataset_overview.py first to generate {features_csv}")
    df = pd.read_csv(features_csv)
    return dict(zip(df["form_id"], df["shape"], strict=True))


# ---------- backtest core --------------------------------------------------


def _backtest_one(
    form_id: str,
    timestamps: pd.Series,
    shape: str,
    cutoff_frac: float,
) -> BacktestPoint | None:
    """Один cutoff-point для однієї форми. None якщо технічно неможливо."""
    n_total = len(timestamps)
    n_train = int(round(cutoff_frac * n_total))
    if n_train < 5:  # MIN_TRAIN_POINTS у service.py
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

    # Горизонт — частка cutoff-тривалості (так, як це робить service.py:
    # ceil(duration * horizon_fraction), мінімум 1 день).
    horizon_seconds = max(cutoff_span_seconds * HORIZON_FRACTION, 86400.0)
    horizon_end = cutoff_ts + pd.Timedelta(seconds=horizon_seconds)

    # Truth: скільки реальних timestamps у вікні [first .. horizon_end].
    truth = int((ts_sorted <= horizon_end).sum())

    # Запуск нашого прогнозу на prefix.
    prefix = ts_sorted.iloc[:n_train].tolist()
    prefix_dt = [t.to_pydatetime() for t in prefix]
    timeline = build_timeline_from_timestamps(prefix_dt)
    try:
        fc = forecast_responses(timeline)
        # forecast.future_cum індексується щоденно; шукаємо значення на
        # дату, найближчу до horizon_end. Якщо horizon_end < перша майбутня
        # дата (rare для коротких span'ів), беремо перший future-point.
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
    """Per-row метрики: ape, hit_95, sharpness."""
    ok = points[points["error"].isna()].copy()
    # Захист від truth=0
    ok = ok[ok["truth"] > 0]
    ok["ape"] = (ok["truth"] - ok["point_estimate"]).abs() / ok["truth"]
    ok["hit_95"] = (ok["ci_lower"] <= ok["truth"]) & (ok["truth"] <= ok["ci_upper"])
    ok["sharpness"] = (ok["ci_upper"] - ok["ci_lower"]) / ok["truth"]
    ok["signed_err"] = (ok["point_estimate"] - ok["truth"]) / ok["truth"]

    # Failure mode: under / in / over
    def _mode(row):
        if row["truth"] < row["ci_lower"]:
            return "overconfident_high"  # модель завищила, реальність нижче
        if row["truth"] > row["ci_upper"]:
            return "overconfident_low"  # модель занизила, реальність вище
        return "in_ci"

    ok["mode"] = ok.apply(_mode, axis=1)
    return ok


# ---------- aggregations ---------------------------------------------------


def _agg_by(metrics: pd.DataFrame, group_col: str) -> pd.DataFrame:
    agg = metrics.groupby(group_col).agg(
        n_points=("ape", "size"),
        mape_p50=("ape", "median"),
        mape_p90=("ape", lambda s: s.quantile(0.90)),
        coverage=("hit_95", "mean"),
        sharpness_p50=("sharpness", "median"),
        bias=("signed_err", "median"),
    )
    agg["mape_p50"] = (agg["mape_p50"] * 100).round(1)
    agg["mape_p90"] = (agg["mape_p90"] * 100).round(1)
    agg["coverage"] = (agg["coverage"] * 100).round(1)
    agg["sharpness_p50"] = agg["sharpness_p50"].round(2)
    agg["bias"] = (agg["bias"] * 100).round(1)
    return agg


# ---------- figures --------------------------------------------------------


def _figure_mape_box(metrics: pd.DataFrame) -> go.Figure:
    fig = px.box(
        metrics,
        x="shape",
        y="ape",
        category_orders={"shape": ["linear", "logarithmic", "logistic", "late_burst", "ill_fit"]},
        log_y=True,
        title="MAPE distribution by shape (log scale)",
        labels={"ape": "|truth - point| / truth"},
    )
    fig.add_hline(y=0.10, line_dash="dot", line_color="green", annotation_text="10% target")
    fig.add_hline(y=0.30, line_dash="dot", line_color="red", annotation_text="30% threshold")
    return fig


def _figure_coverage_bar(metrics: pd.DataFrame) -> go.Figure:
    coverage = metrics.groupby("shape")["hit_95"].mean() * 100
    fig = px.bar(
        x=coverage.index,
        y=coverage.values,
        labels={"x": "Shape", "y": "Empirical coverage (%)"},
        title="95% PI empirical coverage by shape",
        text=[f"{v:.1f}%" for v in coverage.values],
    )
    fig.update_traces(textposition="outside")
    fig.add_hline(y=95, line_dash="dash", line_color="green", annotation_text="Nominal 95%")
    return fig


def _figure_reliability_diagram(metrics: pd.DataFrame) -> go.Figure:
    """Як часто truth попадає у q-quantile CI, для q ∈ {50%, 80%, 90%, 95%}.

    Зараз ми маємо лише 95% CI — тож показуємо просту версію: empirical
    coverage vs nominal 95% по кожному shape. Повна reliability diagram
    потребує множинних CI-рівнів — TODO у наступних бенчмарках.
    """
    coverage_global = metrics["hit_95"].mean() * 100
    per_shape = metrics.groupby("shape")["hit_95"].mean() * 100
    df = per_shape.reset_index()
    df.columns = ["shape", "empirical_95"]
    df["nominal_95"] = 95.0
    fig = px.scatter(
        df,
        x="nominal_95",
        y="empirical_95",
        text="shape",
        title=f"Reliability: global coverage = {coverage_global:.1f}% (nominal 95%)",
        labels={"nominal_95": "Nominal coverage (%)", "empirical_95": "Empirical (%)"},
    )
    fig.update_traces(textposition="top center", marker=dict(size=12))
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="gray", dash="dash"))
    fig.update_xaxes(range=[80, 100])
    fig.update_yaxes(range=[0, 105])
    return fig


def _figure_failure_modes(metrics: pd.DataFrame) -> go.Figure:
    df = metrics.groupby(["shape", "mode"]).size().reset_index(name="count")
    fig = px.bar(
        df,
        x="shape",
        y="count",
        color="mode",
        title="Failure modes by shape",
        category_orders={"mode": ["in_ci", "overconfident_low", "overconfident_high"]},
        color_discrete_map={
            "in_ci": "#2ca02c",
            "overconfident_low": "#d62728",
            "overconfident_high": "#ff7f0e",
        },
    )
    return fig


def _figure_bias_by_n(metrics: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        metrics,
        x="n_train",
        y="signed_err",
        color="shape",
        log_x=True,
        title="Bias vs N_train (signed err = (point - truth) / truth)",
        labels={"signed_err": "Signed error", "n_train": "N at cutoff (log)"},
        hover_data=["form_id", "cutoff_frac", "truth", "point_estimate", "ci_lower", "ci_upper"],
    )
    fig.add_hline(y=0, line_dash="dot", line_color="green")
    return fig


# ---------- main -----------------------------------------------------------


def main(input_path: Path, features_csv: Path, output_md: Path, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    input_hash = _file_sha256_short(input_path)

    shapes = _load_shapes(features_csv)

    # Дедуплікуємо (form_id, timestamp) — same-second ties не повинні
    # роздувати backtest-результати.
    df = df.drop_duplicates(subset=["FORM_ID", "TIMESTAMP"]).reset_index(drop=True)

    points: list[BacktestPoint] = []
    skipped = {"insufficient": 0, "too_few": 0, "no_span": 0}

    eligible_forms = []
    for form_id, group in df.groupby("FORM_ID"):
        n = len(group)
        if n < MIN_N_FOR_BACKTEST:
            skipped["too_few"] += 1
            continue
        shape = shapes.get(form_id, "unknown")
        if shape == "insufficient":
            skipped["insufficient"] += 1
            continue
        eligible_forms.append((form_id, group["TIMESTAMP"], shape))

    print(f"Eligible forms: {len(eligible_forms)} (of {df['FORM_ID'].nunique()})")
    print(f"Cutoffs: {CUTOFFS}")
    print(f"Total backtests: {len(eligible_forms) * len(CUTOFFS)}")

    for i, (form_id, ts, shape) in enumerate(eligible_forms, 1):
        for cutoff in CUTOFFS:
            point = _backtest_one(form_id, ts, shape, cutoff)
            if point is not None:
                points.append(point)
        if i % 25 == 0:
            print(f"  processed {i}/{len(eligible_forms)} forms...")

    points_df = pd.DataFrame([p.__dict__ for p in points])
    metrics = _compute_metrics(points_df)

    # Save raw points + metrics for downstream
    points_df.to_csv(figures_dir / "02_backtest_points.csv", index=False)
    metrics.to_csv(figures_dir / "02_backtest_metrics.csv", index=False)

    by_shape = _agg_by(metrics, "shape")
    metrics["n_bucket"] = pd.cut(
        metrics["n_train"],
        bins=[0, 15, 30, 100, 1000, 100000],
        labels=["<15", "15-30", "30-100", "100-1k", "1k+"],
    )
    by_nbucket = _agg_by(metrics, "n_bucket")

    figs = {
        "mape_box": _figure_mape_box(metrics),
        "coverage_bar": _figure_coverage_bar(metrics),
        "reliability": _figure_reliability_diagram(metrics),
        "failure_modes": _figure_failure_modes(metrics),
        "bias_by_n": _figure_bias_by_n(metrics),
    }
    fig_paths: dict[str, Path] = {}
    for name, fig in figs.items():
        path = figures_dir / f"02_{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        fig_paths[name] = path.relative_to(output_md.parent)

    md = _render_markdown(
        n_rows=len(df),
        n_forms_total=df["FORM_ID"].nunique(),
        n_forms_eligible=len(eligible_forms),
        n_points=len(metrics),
        input_path=input_path,
        input_hash=input_hash,
        by_shape=by_shape,
        by_nbucket=by_nbucket,
        metrics=metrics,
        fig_paths=fig_paths,
    )
    output_md.write_text(md, encoding="utf-8")
    print(f"\nReport written to: {output_md}")


def _render_markdown(
    *,
    n_rows: int,
    n_forms_total: int,
    n_forms_eligible: int,
    n_points: int,
    input_path: Path,
    input_hash: str,
    by_shape: pd.DataFrame,
    by_nbucket: pd.DataFrame,
    metrics: pd.DataFrame,
    fig_paths: dict[str, Path],
) -> str:
    global_mape = metrics["ape"].median() * 100
    global_coverage = metrics["hit_95"].mean() * 100
    global_sharpness = metrics["sharpness"].median()
    global_bias = metrics["signed_err"].median() * 100

    return f"""# 02 — Rolling-Origin Backtest

**Generated:** {datetime.now().isoformat(timespec="seconds")}
**Source:** `{input_path}` (sha256:`{input_hash}`)
**Forms:** {n_forms_total} total → {n_forms_eligible} eligible (N ≥ {MIN_N_FOR_BACKTEST})
**Cutoffs:** {CUTOFFS} · **Horizon fraction:** {HORIZON_FRACTION}
**Backtest points:** {n_points}

## Глобальні метрики

| Метрика | Значення | Орієнтир |
|---|---:|---|
| **MAPE (median)** | {global_mape:.1f}% | < 15% — добре, < 25% — прийнятно |
| **Coverage 95% PI** | {global_coverage:.1f}% | Має бути ≈ 95% |
| **Sharpness (median)** | {global_sharpness:.2f} | width / truth |
| **Bias (median)** | {global_bias:+.1f}% | 0% — unbiased |

## По shape-категоріях

{_df_to_md(by_shape)}

## По N-buckets

{_df_to_md(by_nbucket)}

## Failure modes

| Mode | Опис |
|---|---|
| `in_ci` | Truth у [ci_lower, ci_upper] — успіх |
| `overconfident_low` | Truth > ci_upper (модель занизила) |
| `overconfident_high` | Truth < ci_lower (модель завищила) |

## Графіки

- [MAPE distribution by shape (boxplot)]({fig_paths["mape_box"]})
- [Empirical coverage vs nominal 95%]({fig_paths["coverage_bar"]})
- [Reliability diagram]({fig_paths["reliability"]})
- [Failure modes by shape]({fig_paths["failure_modes"]})
- [Bias vs N_train]({fig_paths["bias_by_n"]})

## Що з цього випливає

Дивись числа вище:

1. **Глобальний coverage**: якщо < 90% → CI занадто вузький
   (треба збільшити n_sims або врахувати додаткове джерело варіансу).
   Якщо > 98% → занадто широкий, sharpness страждає.

2. **Найгірший shape за MAPE**: кандидат на нову модель або
   shape-specific selector.

3. **Bias за N-bucket**: якщо bias систематично negative для малих N
   → модель консервативна на старті (засипана target-prior'ом?).
   Якщо positive — оптимістична. Це підказує напрямок calibration'у.

4. **Failure modes**: переважання `overconfident_low` означає, що CI
   треба зсунути вгору (або просто розширити). `overconfident_high` —
   зсунути вниз.

## Артефакти

- `figures/02_backtest_points.csv` — повний log усіх backtest-runs
- `figures/02_backtest_metrics.csv` — обчислені метрики
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
        default=repo_root / "research" / "reports" / "02_backtest.md",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=repo_root / "research" / "reports" / "figures",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.input, args.features_csv, args.output, args.figures_dir)
