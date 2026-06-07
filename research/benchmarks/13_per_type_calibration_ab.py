"""13_per_type_calibration_ab.py — A/B per-type calibration multiplier.

A: forecast_responses(timeline) — глобальний multiplier=10 (baseline)
B: forecast_responses(timeline, form_type=ft) — per-type multiplier з PER_TYPE_MULTIPLIER

Backtest mirrors 12_'s structure: 141 форм × {5, 10, 15, 20, 25, 30} n_train
× {2, 6, 24, 72, 168} hours horizon. Дві forecast-evaluation per cutoff =
~2x compute vs. 12_ (~30-90 хв).

Criteria для promote:
- Per-type coverage значно ближче до 95% (gap reduction > 5pp на проблемних типах)
- MAPE незмінний (point estimate не торкаємо)
- Sharpness НЕ зростає більш ніж +50% (over-widening захист)
- Global coverage не падає (gain ≥ 0)

Якщо B перемагає → промочуємо P11. Інакше — adjust multiplier-и і повтор.

Запуск:
    .venv/Scripts/python.exe research/benchmarks/13_per_type_calibration_ab.py
    .venv/Scripts/python.exe research/benchmarks/13_per_type_calibration_ab.py --limit 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import OptimizeWarning

warnings.simplefilter("ignore", OptimizeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.forecast import ForecastError, forecast_responses  # noqa: E402
from core.timeline import build_timeline_from_timestamps  # noqa: E402
from research.benchmarks._common import (  # noqa: E402
    agg_by,
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


@dataclass(frozen=True)
class ABPoint:
    form_id: str
    shape: str
    form_type: str
    n_class: str
    n_total: int
    n_train: int
    horizon_hours: float
    truth: int
    # A: baseline (no form_type)
    a_point: int
    a_lo: int
    a_hi: int
    # B: with form_type
    b_point: int
    b_lo: int
    b_hi: int
    multiplier_b: float
    error: str | None


def _backtest_one(form: dict, n_train: int, horizon_hours: float) -> ABPoint | None:
    from core.forecast.calibration import get_calibration_multiplier

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

    prefix_dt = [t.to_pydatetime() for t in ts_sorted.iloc[:n_train].tolist()]
    timeline = build_timeline_from_timestamps(prefix_dt)
    horizon_until = pd.Timestamp(horizon_end)

    err: str | None = None
    # --- A: baseline ---
    try:
        fc_a = forecast_responses(timeline, horizon_until=horizon_until)
        idx_a = idx_for_horizon(fc_a.future_dates, horizon_end)
        a_p = int(round(float(fc_a.future_cum.iloc[idx_a])))
        a_lo = int(round(float(fc_a.ci_lower.iloc[idx_a])))
        a_hi = int(round(float(fc_a.ci_upper.iloc[idx_a])))
    except ForecastError as e:
        a_p, a_lo, a_hi = -1, -1, -1
        err = f"a_failed: {e}"

    # --- B: with form_type ---
    try:
        fc_b = forecast_responses(
            timeline, horizon_until=horizon_until, form_type=form["form_type"]
        )
        idx_b = idx_for_horizon(fc_b.future_dates, horizon_end)
        b_p = int(round(float(fc_b.future_cum.iloc[idx_b])))
        b_lo = int(round(float(fc_b.ci_lower.iloc[idx_b])))
        b_hi = int(round(float(fc_b.ci_upper.iloc[idx_b])))
    except ForecastError as e:
        b_p, b_lo, b_hi = -1, -1, -1
        err = (err or "") + f"; b_failed: {e}"

    multiplier_b = get_calibration_multiplier(form["form_type"])
    return ABPoint(
        form_id=form["form_id"],
        shape=form["shape"],
        form_type=form["form_type"],
        n_class=form["n_class"],
        n_total=n_total,
        n_train=n_train,
        horizon_hours=float(horizon_hours),
        truth=truth,
        a_point=a_p,
        a_lo=a_lo,
        a_hi=a_hi,
        b_point=b_p,
        b_lo=b_lo,
        b_hi=b_hi,
        multiplier_b=multiplier_b,
        error=err,
    )


METHODS = ("a_baseline", "b_per_type")


def _per_method_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        if r["truth"] <= 0:
            continue
        for prefix, method in [("a", "a_baseline"), ("b", "b_per_type")]:
            p = r[f"{prefix}_point"]
            lo = r[f"{prefix}_lo"]
            hi = r[f"{prefix}_hi"]
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
                    "form_type": r["form_type"],
                    "n_class": r["n_class"],
                    "n_train": r["n_train"],
                    "horizon_hours": r["horizon_hours"],
                    "truth": r["truth"],
                    "method": method,
                    "point": p,
                    "lo": lo,
                    "hi": hi,
                    "ape": ape,
                    "hit_95": hit,
                    "sharpness": sharp,
                    "signed_err": signed,
                    "multiplier": r["multiplier_b"] if prefix == "b" else 10.0,
                }
            )
    return pd.DataFrame(rows)


def _fig_per_type_coverage(metrics: pd.DataFrame) -> go.Figure:
    """Bar: x=form_type, y=coverage, color=method."""
    agg = (
        metrics.groupby(["form_type", "method"])
        .agg(
            cov=("hit_95", "mean"),
            n=("hit_95", "size"),
        )
        .reset_index()
    )
    agg["cov"] = agg["cov"] * 100
    fig = px.bar(
        agg,
        x="form_type",
        y="cov",
        color="method",
        barmode="group",
        title="Per form_type coverage: A (global) vs B (per-type)",
        labels={"cov": "Empirical coverage (%)"},
        hover_data=["n"],
    )
    fig.add_hline(y=95, line_dash="dash", line_color="gray", annotation_text="Nominal 95%")
    return fig


def _fig_per_type_sharpness(metrics: pd.DataFrame) -> go.Figure:
    agg = metrics.groupby(["form_type", "method"])["sharpness"].median().reset_index()
    fig = px.bar(
        agg,
        x="form_type",
        y="sharpness",
        color="method",
        barmode="group",
        title="Per form_type sharpness (CI width / truth): A vs B",
        labels={"sharpness": "Median sharpness"},
    )
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
    print(f"Skipped: {skipped}")
    print(
        f"Backtests upper bound: {len(eligible) * len(n_train_values) * len(horizon_hours_values)}"
    )

    points: list[ABPoint] = []
    for i, form in enumerate(eligible, 1):
        for n_train in n_train_values:
            for h in horizon_hours_values:
                p = _backtest_one(form, n_train, float(h))
                if p is not None:
                    points.append(p)
        if i % 10 == 0 or i == len(eligible):
            print(f"  processed {i}/{len(eligible)} forms...")

    points_df = pd.DataFrame([p.__dict__ for p in points])
    points_df.to_csv(figures_dir / "13_ab_points.csv", index=False)
    metrics = _per_method_metrics(points_df)
    metrics.to_csv(figures_dir / "13_ab_metrics.csv", index=False)

    by_method = agg_by(metrics, "method", hit_col="hit_95")
    by_method_type = agg_by(metrics, ["method", "form_type"], hit_col="hit_95")
    by_method_n_train = agg_by(metrics, ["method", "n_train"], hit_col="hit_95")
    by_method_horizon = agg_by(metrics, ["method", "horizon_hours"], hit_col="hit_95")
    by_method_type_horizon = agg_by(
        metrics, ["method", "form_type", "horizon_hours"], hit_col="hit_95"
    )

    figs = {
        "per_type_coverage": _fig_per_type_coverage(metrics),
        "per_type_sharpness": _fig_per_type_sharpness(metrics),
    }
    fig_paths = {}
    for name, fig in figs.items():
        path = figures_dir / f"13_{name}.html"
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
        by_method_type_horizon=by_method_type_horizon,
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
    n_train_values,
    horizon_hours_values,
    min_n,
    input_path,
    input_hash,
    by_method,
    by_method_type,
    by_method_n_train,
    by_method_horizon,
    by_method_type_horizon,
    fig_paths,
) -> str:
    return f"""# 13 - Per-form-type calibration A/B (P11 candidate)

**Generated:** {datetime.now().isoformat(timespec="seconds")}
**Source:** `{input_path}` (sha256:`{input_hash}`)
**Forms:** {n_forms_total} -> {n_forms_eligible} eligible (N >= {min_n})
**n_train:** {n_train_values} - **horizon_hours:** {horizon_hours_values}
**Backtest points:** {n_points}
**Skipped:** {skipped}

## Методи

- **A · baseline** — `forecast_responses(timeline)` з глобальним CALIBRATION_MULTIPLIER=10.0.
- **B · per_type** — `forecast_responses(timeline, form_type=ft)` з PER_TYPE_MULTIPLIER:
  - survey 28, service 20, holiday 16, recruitment 14, political 14,
  - volunteer 12, feedback 12, event_reg 11, creative 11, other 8, unknown 13.

## Global

{df_to_md(by_method)}

## Per form_type (decisive view)

{df_to_md(by_method_type)}

## Per n_train

{df_to_md(by_method_n_train)}

## Per horizon

{df_to_md(by_method_horizon)}

## Per (form_type x horizon) (для перевірки чи покращення на ВСІХ горизонтах)

{df_to_md(by_method_type_horizon)}

## Figures

- [Per-type coverage A vs B]({fig_paths["per_type_coverage"]})
- [Per-type sharpness A vs B]({fig_paths["per_type_sharpness"]})

## Критерії promote

| Критерій | Поріг |
|---|---|
| Per-type cov: survey, service, holiday | > 80% (з 53/67/78 → ≥80) |
| Per-type cov: усі типи | gap до 95% ≤ 10pp |
| Sharpness | НЕ зростає більш ніж +50% global |
| MAPE | незмінне (point estimate не торкаємо) |
| Global cov | gain ≥ 0 |

## Артефакти

- `figures/13_ab_points.csv` — wide-format raw.
- `figures/13_ab_metrics.csv` — long-format per-method метрики.
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
        default=repo_root / "research" / "reports" / "13_per_type_calibration_ab.md",
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
