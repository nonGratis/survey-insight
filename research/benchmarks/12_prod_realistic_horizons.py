"""12_prod_realistic_horizons.py — точна прод-постановка.

Питання користувача: "якщо маємо перші 10-30 відповідей з форми, то які
рези через пару годин".

Замість fraction-based cutoffs + relative horizon (як у 08-11) — тут:
- **n_train** ∈ {5, 10, 15, 20, 25, 30} (абсолютні, не fraction від N).
- **horizon_abs** ∈ {2h, 6h, 24h, 72h, 168h (7d)} наперед від cutoff_ts.

Кожна точка backtest = (form, n_train, horizon_abs). Для кожної:
- Прогноз через прод-flow (`forecast_responses`) — фінальна готова система.
- truth = N відповідей у вікні [first_ts, cutoff_ts + horizon_abs].
- Метрики: MAPE, hit_95, bias, sharpness.

Per-(n_train × horizon × form_type) → де метод працює, де ні.

Запуск:
    .venv/Scripts/python.exe research/benchmarks/12_prod_realistic_horizons.py
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

# Configurable.
N_TRAIN_VALUES = (5, 10, 15, 20, 25, 30)
HORIZON_HOURS = (2, 6, 24, 72, 168)
MIN_N_FOR_BACKTEST = 5  # хочемо включити tiny (5-9)
RANDOM_SEED = 42


@dataclass(frozen=True)
class ProdPoint:
    form_id: str
    shape: str
    form_type: str
    n_class: str
    tempo: str
    duration_class: str
    n_total: int
    n_train: int
    horizon_hours: float
    truth: int
    point: int
    lo: int
    hi: int
    error: str | None


def _backtest_one(form: dict, n_train: int, horizon_hours: float) -> ProdPoint | None:
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
    try:
        # Use horizon_until to make forecast precisely reach horizon_end.
        fc = forecast_responses(timeline, horizon_until=pd.Timestamp(horizon_end))
        idx = idx_for_horizon(fc.future_dates, horizon_end)
        return ProdPoint(
            form_id=form["form_id"],
            shape=form["shape"],
            form_type=form["form_type"],
            n_class=form["n_class"],
            tempo=form["tempo"],
            duration_class=form["duration_class"],
            n_total=n_total,
            n_train=n_train,
            horizon_hours=float(horizon_hours),
            truth=truth,
            point=int(round(float(fc.future_cum.iloc[idx]))),
            lo=int(round(float(fc.ci_lower.iloc[idx]))),
            hi=int(round(float(fc.ci_upper.iloc[idx]))),
            error=None,
        )
    except ForecastError as e:
        return ProdPoint(
            form_id=form["form_id"],
            shape=form["shape"],
            form_type=form["form_type"],
            n_class=form["n_class"],
            tempo=form["tempo"],
            duration_class=form["duration_class"],
            n_total=n_total,
            n_train=n_train,
            horizon_hours=float(horizon_hours),
            truth=truth,
            point=-1,
            lo=-1,
            hi=-1,
            error=str(e),
        )


def _compute_metrics(points_df: pd.DataFrame) -> pd.DataFrame:
    ok = points_df[(points_df["point"] >= 0) & (points_df["truth"] > 0)].copy()
    ok["ape"] = (ok["truth"] - ok["point"]).abs() / ok["truth"]
    ok["hit_95"] = (ok["lo"] <= ok["truth"]) & (ok["truth"] <= ok["hi"])
    ok["sharpness"] = (ok["hi"] - ok["lo"]) / ok["truth"]
    ok["signed_err"] = (ok["point"] - ok["truth"]) / ok["truth"]

    def _mode(row):
        if row["truth"] < row["lo"]:
            return "overconfident_high"
        if row["truth"] > row["hi"]:
            return "overconfident_low"
        return "in_ci"

    ok["mode"] = ok.apply(_mode, axis=1)
    return ok


def _fig_heatmap(
    metrics: pd.DataFrame, row_col: str, col_col: str, value: str, title: str
) -> go.Figure:
    pivot = metrics.pivot_table(
        index=row_col,
        columns=col_col,
        values=value,
        aggfunc=("mean" if value == "hit_95" else "median"),
    )
    if value == "hit_95":
        pivot = pivot * 100
        scale, zmid = "RdYlGn", 95.0
    elif value == "ape":
        pivot = pivot * 100
        scale, zmid = "RdYlGn_r", 25.0
    else:
        scale, zmid = "Viridis", None
    fig = px.imshow(
        pivot,
        text_auto=".0f",
        color_continuous_scale=scale,
        title=title,
        aspect="auto",
        labels={"color": value},
    )
    if zmid is not None:
        fig.update_traces(zmid=zmid)
    return fig


def _fig_per_n_train_lines(metrics: pd.DataFrame, value: str, ylabel: str, title: str) -> go.Figure:
    """X = horizon_hours (log), Y = metric, color = n_train, для прод-readability."""
    agg = (
        metrics.groupby(["n_train", "horizon_hours"])
        .agg(
            mape_p50=("ape", "median"),
            coverage=("hit_95", "mean"),
        )
        .reset_index()
    )
    if value == "ape":
        agg["v"] = agg["mape_p50"] * 100
    else:
        agg["v"] = agg["coverage"] * 100
    fig = px.line(
        agg,
        x="horizon_hours",
        y="v",
        color="n_train",
        markers=True,
        log_x=True,
        labels={"v": ylabel, "horizon_hours": "Horizon (h)"},
        title=title,
    )
    if value == "hit_95":
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
    print(f"Skipped: {skipped}")
    print(f"n_train values: {n_train_values}")
    print(f"horizon_hours: {horizon_hours_values}")
    print(
        f"Total backtest points (upper bound): "
        f"{len(eligible) * len(n_train_values) * len(horizon_hours_values)}"
    )

    points: list[ProdPoint] = []
    for i, form in enumerate(eligible, 1):
        for n_train in n_train_values:
            for h in horizon_hours_values:
                p = _backtest_one(form, n_train, float(h))
                if p is not None:
                    points.append(p)
        if i % 10 == 0 or i == len(eligible):
            print(f"  processed {i}/{len(eligible)} forms...")

    points_df = pd.DataFrame([p.__dict__ for p in points])
    points_df.to_csv(figures_dir / "12_prod_points.csv", index=False)
    metrics = _compute_metrics(points_df)
    metrics.to_csv(figures_dir / "12_prod_metrics.csv", index=False)

    # Aggregations.
    by_n_train = agg_by(metrics, "n_train", hit_col="hit_95")
    by_horizon = agg_by(metrics, "horizon_hours", hit_col="hit_95")
    by_n_train_horizon = agg_by(metrics, ["n_train", "horizon_hours"], hit_col="hit_95")
    by_form_type = agg_by(metrics, "form_type", hit_col="hit_95")
    by_form_type_horizon = agg_by(metrics, ["form_type", "horizon_hours"], hit_col="hit_95")
    by_form_type_n_train = agg_by(metrics, ["form_type", "n_train"], hit_col="hit_95")
    by_shape = agg_by(metrics, "shape", hit_col="hit_95")

    # Figures.
    figs = {
        "heat_mape": _fig_heatmap(
            metrics, "n_train", "horizon_hours", "ape", "MAPE p50 (%) — n_train × horizon (hours)"
        ),
        "heat_coverage": _fig_heatmap(
            metrics,
            "n_train",
            "horizon_hours",
            "hit_95",
            "Coverage 95% (%) — n_train × horizon (hours)",
        ),
        "heat_mape_form_type": _fig_heatmap(
            metrics, "form_type", "horizon_hours", "ape", "MAPE p50 (%) — form_type × horizon"
        ),
        "heat_coverage_form_type": _fig_heatmap(
            metrics, "form_type", "horizon_hours", "hit_95", "Coverage (%) — form_type × horizon"
        ),
        "lines_mape": _fig_per_n_train_lines(
            metrics, "ape", "MAPE p50 (%)", "MAPE за horizon, фарбовано за n_train"
        ),
        "lines_coverage": _fig_per_n_train_lines(
            metrics, "hit_95", "Coverage (%)", "Coverage за horizon, фарбовано за n_train"
        ),
    }
    fig_paths = {}
    for name, fig in figs.items():
        path = figures_dir / f"12_{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        fig_paths[name] = path.relative_to(output_md.parent)

    md = _render_markdown(
        n_forms_total=df["FORM_ID"].nunique(),
        n_forms_eligible=len(eligible),
        n_points=int(len(points_df)),
        n_evaluable=int(len(metrics)),
        skipped=skipped,
        n_train_values=n_train_values,
        horizon_hours_values=horizon_hours_values,
        min_n=min_n,
        input_path=input_path,
        input_hash=input_hash,
        by_n_train=by_n_train,
        by_horizon=by_horizon,
        by_n_train_horizon=by_n_train_horizon,
        by_form_type=by_form_type,
        by_form_type_horizon=by_form_type_horizon,
        by_form_type_n_train=by_form_type_n_train,
        by_shape=by_shape,
        fig_paths=fig_paths,
    )
    output_md.write_text(md, encoding="utf-8")
    print(f"\nReport written to: {output_md}")


def _render_markdown(
    *,
    n_forms_total,
    n_forms_eligible,
    n_points,
    n_evaluable,
    skipped,
    n_train_values,
    horizon_hours_values,
    min_n,
    input_path,
    input_hash,
    by_n_train,
    by_horizon,
    by_n_train_horizon,
    by_form_type,
    by_form_type_horizon,
    by_form_type_n_train,
    by_shape,
    fig_paths,
) -> str:
    return f"""# 12 - Prod-realistic absolute scenarios

**Generated:** {datetime.now().isoformat(timespec="seconds")}
**Source:** `{input_path}` (sha256:`{input_hash}`)
**Forms:** {n_forms_total} -> {n_forms_eligible} eligible (N >= {min_n})
**n_train values:** {n_train_values}
**horizon_hours:** {horizon_hours_values}
**Backtest points:** {n_points} (з них evaluable з truth > 0: {n_evaluable})
**Skipped:** {skipped}

## Постановка

Це **точна прод-постановка** від користувача: "якщо маємо перші 10-30
відповідей з форми, які рези через пару годин".

Замість fraction cutoffs тестуємо абсолютні: n_train ∈ {n_train_values},
horizon ∈ {horizon_hours_values} hours.

Прод-flow `forecast_responses` (з P10 sample-size CI scaling).

## Global per n_train

{df_to_md(by_n_train)}

## Global per horizon (hours)

{df_to_md(by_horizon)}

## n_train × horizon (decisive view)

{df_to_md(by_n_train_horizon)}

## Per form_type

{df_to_md(by_form_type)}

## Per form_type × horizon

{df_to_md(by_form_type_horizon)}

## Per form_type × n_train

{df_to_md(by_form_type_n_train)}

## Per shape (sanity з 08_)

{df_to_md(by_shape)}

## Figures

- [Heatmap: MAPE p50 - n_train × horizon]({fig_paths["heat_mape"]})
- [Heatmap: Coverage - n_train × horizon]({fig_paths["heat_coverage"]})
- [Heatmap: MAPE p50 - form_type × horizon]({fig_paths["heat_mape_form_type"]})
- [Heatmap: Coverage - form_type × horizon]({fig_paths["heat_coverage_form_type"]})
- [Lines: MAPE per horizon, by n_train]({fig_paths["lines_mape"]})
- [Lines: Coverage per horizon, by n_train]({fig_paths["lines_coverage"]})

## Як читати

1. **n_train=5 vs n_train=30, horizon=2h:** наскільки сильно покращується точність
   з накопиченням даних? Це baseline прод-чутливість.
2. **horizon=2h vs horizon=168h:** як деградує точність з горизонтом.
3. **Form type x horizon:** перевірка гіпотези користувача:
   - event_registration -> очікувано sharp на коротких горизонтах (форма короткоживуча).
   - survey -> long-tail, погано на 2h, краще на 24-72h.
   - service -> steady, точність повинна бути найбільш передбачуваною.
4. **Failure cells:** комбінації (n_train, horizon, form_type) з MAPE > 50% або
   Cov < 80% - кандидати для targeted fix.

## Артефакти

- `figures/12_prod_points.csv` - один рядок на (form, n_train, horizon).
- `figures/12_prod_metrics.csv` - eval-готові метрики.
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
        default=repo_root / "research" / "reports" / "12_prod_realistic_horizons.md",
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
