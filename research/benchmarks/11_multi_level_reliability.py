"""11_multi_level_reliability.py — повна reliability-діаграма (50/80/90/95).

Замість бінарного "hit_95 чи ні" обчислює coverage на 4 рівнях нормально-
заявленої CI {50, 80, 90, 95}%. Дає proper reliability diagram замість
однієї точки.

Що тестується:
- RAW NHPP-симуляція без `apply_calibration_arrays` і без P10 scaling —
  щоб побачити, наскільки сам по собі sim-loop well-calibrated. Прод-flow
  додає калібровку зверху, але це маскує реальну якість моделі.
- Per-level coverage + width, per-shape, per-form-type, per-cutoff.

Reliability ⇒ якщо emp_coverage(L) лежить на діагоналі emp = L — модель
ідеально калібрована. Якщо emp_coverage(50%) = 30% → CI занадто вузька;
emp_coverage(95%) = 73% (з 02_) — також занадто вузька, але каліброване ×10
до 89% (P10).

Запуск:
    .venv/Scripts/python.exe research/benchmarks/11_multi_level_reliability.py
    .venv/Scripts/python.exe research/benchmarks/11_multi_level_reliability.py --limit 5
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

warnings.simplefilter("ignore", OptimizeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.forecast import ForecastError  # noqa: E402
from core.forecast.intervals import nhpp_prediction_multi_level  # noqa: E402
from core.forecast.metrics import aicc, r_squared, rmse  # noqa: E402
from core.forecast.models import fit_model, models_for_n_points  # noqa: E402
from core.forecast.types import FittedModel  # noqa: E402
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

MIN_N_FOR_BACKTEST = 10
CUTOFFS_DEFAULT = (0.1, 0.2, 0.3, 0.5, 0.7)
HORIZON_FRACTION = 0.25
MIN_TRAIN_POINTS = 5
MIN_DURATION_DAYS = 1.0 / 24.0
N_SIMS = 2000
RANDOM_SEED = 42

LEVELS = (0.50, 0.80, 0.90, 0.95)


@dataclass(frozen=True)
class MultiLevelPoint:
    form_id: str
    shape: str
    form_type: str
    n_class: str
    tempo: str
    duration_class: str
    n_total: int
    cutoff_frac: float
    n_train: int
    horizon_days: float
    truth: int
    point: int  # mean_cum at horizon
    # For each level — lo, hi.
    lo_50: int
    hi_50: int
    lo_80: int
    hi_80: int
    lo_90: int
    hi_90: int
    lo_95: int
    hi_95: int
    error: str | None


def _fit_and_predict_multi_level(
    timestamps: pd.Series, horizon_seconds: float
) -> tuple[pd.DatetimeIndex, np.ndarray, dict[float, tuple[np.ndarray, np.ndarray]]]:
    """Repл inner-logic service.py + multi-level CI без калібровки.

    Returns (future_dates, mean_cum, levels_dict).
    """
    ts = pd.to_datetime(timestamps).sort_values().reset_index(drop=True)
    n = len(ts)
    if n < MIN_TRAIN_POINTS:
        raise ForecastError("too_few_points")
    first_ts = ts.iloc[0].to_pydatetime()
    last_ts = ts.iloc[-1].to_pydatetime()

    t_train = ((ts - pd.Timestamp(first_ts)).dt.total_seconds() / 86400.0).to_numpy(dtype=float)
    y_train = np.arange(1, n + 1, dtype=float)
    last_observed = n

    # Span guard: 1h floor.
    _ = max((last_ts - first_ts).total_seconds() / 86400.0, MIN_DURATION_DAYS)
    horizon_days = max(int(np.ceil(horizon_seconds / 86400.0)), 1)

    models = models_for_n_points(n)
    fitted_best: FittedModel | None = None
    best_aicc = float("inf")
    for model in models:
        try:
            params, pcov = fit_model(model, t_train, y_train, None)
        except ForecastError:
            continue
        y_fitted = model.predict(t_train, *params)
        a = aicc(y_train, y_fitted, model.n_params)
        if np.isfinite(a) and a < best_aicc:
            best_aicc = a
            fitted_best = FittedModel(
                model=model,
                params=params,
                aicc=a,
                rmse=rmse(y_train, y_fitted),
                r_squared=r_squared(y_train, y_fitted),
                pcov=pcov,
            )
    if fitted_best is None:
        raise ForecastError("all_models_failed")

    last_known_day = pd.Timestamp(last_ts.date())
    future_dates = pd.date_range(
        start=last_known_day + timedelta(days=1), periods=horizon_days, freq="D"
    )
    t_future = ((future_dates - pd.Timestamp(first_ts)).total_seconds() / 86400.0).to_numpy(
        dtype=float
    )

    rng = np.random.default_rng(RANDOM_SEED)
    mean_cum, median_cum, levels = nhpp_prediction_multi_level(
        fitted_best, t_future, last_observed=last_observed, levels=LEVELS, n_sims=N_SIMS, rng=rng
    )
    # Monotonic point + floor (як у service.py, але БЕЗ calibration і P10 —
    # хочемо raw reliability).
    mean_cum = np.maximum.accumulate(np.maximum(mean_cum, float(last_observed)))
    return future_dates, mean_cum, levels


def _backtest_one(
    form: dict,
    cutoff_frac: float,
) -> MultiLevelPoint | None:
    ts = form["timestamps"]
    n_total = len(ts)
    n_train = int(round(cutoff_frac * n_total))
    if n_train < MIN_TRAIN_POINTS:
        return None

    ts_sorted = ts.sort_values().reset_index(drop=True)
    cutoff_ts = ts_sorted.iloc[n_train - 1]
    cutoff_span_seconds = (cutoff_ts - ts_sorted.iloc[0]).total_seconds()
    if cutoff_span_seconds <= 0:
        return None
    horizon_seconds = max(cutoff_span_seconds * HORIZON_FRACTION, 86400.0)
    horizon_end = cutoff_ts + pd.Timedelta(seconds=horizon_seconds)
    truth = int((ts_sorted <= horizon_end).sum())

    prefix = pd.Series(ts_sorted.iloc[:n_train].tolist())
    try:
        future_dates, mean_cum, levels = _fit_and_predict_multi_level(prefix, horizon_seconds)
        idx = idx_for_horizon(future_dates, horizon_end)
        point = int(round(float(mean_cum[idx])))
        per_level = {
            f"{int(L * 100)}": (
                int(round(float(lo[idx]))),
                int(round(float(hi[idx]))),
            )
            for L, (lo, hi) in levels.items()
        }
        return MultiLevelPoint(
            form_id=form["form_id"],
            shape=form["shape"],
            form_type=form["form_type"],
            n_class=form["n_class"],
            tempo=form["tempo"],
            duration_class=form["duration_class"],
            n_total=n_total,
            cutoff_frac=cutoff_frac,
            n_train=n_train,
            horizon_days=horizon_seconds / 86400.0,
            truth=truth,
            point=point,
            lo_50=per_level["50"][0],
            hi_50=per_level["50"][1],
            lo_80=per_level["80"][0],
            hi_80=per_level["80"][1],
            lo_90=per_level["90"][0],
            hi_90=per_level["90"][1],
            lo_95=per_level["95"][0],
            hi_95=per_level["95"][1],
            error=None,
        )
    except ForecastError as e:
        return MultiLevelPoint(
            form_id=form["form_id"],
            shape=form["shape"],
            form_type=form["form_type"],
            n_class=form["n_class"],
            tempo=form["tempo"],
            duration_class=form["duration_class"],
            n_total=n_total,
            cutoff_frac=cutoff_frac,
            n_train=n_train,
            horizon_days=horizon_seconds / 86400.0,
            truth=truth,
            point=-1,
            lo_50=-1,
            hi_50=-1,
            lo_80=-1,
            hi_80=-1,
            lo_90=-1,
            hi_90=-1,
            lo_95=-1,
            hi_95=-1,
            error=str(e),
        )


def _compute_per_level(points_df: pd.DataFrame) -> pd.DataFrame:
    """Long-format: один рядок на (point × level) з hit/sharpness/нескінченність."""
    rows = []
    for _, r in points_df.iterrows():
        if r["truth"] <= 0 or r["point"] < 0:
            continue
        for lvl in LEVELS:
            tag = f"{int(lvl * 100)}"
            lo, hi = r[f"lo_{tag}"], r[f"hi_{tag}"]
            ape = abs(r["truth"] - r["point"]) / r["truth"]
            hit = lo <= r["truth"] <= hi
            sharp = (hi - lo) / r["truth"]
            signed = (r["point"] - r["truth"]) / r["truth"]
            rows.append(
                {
                    "form_id": r["form_id"],
                    "shape": r["shape"],
                    "form_type": r["form_type"],
                    "n_class": r["n_class"],
                    "tempo": r["tempo"],
                    "cutoff_frac": r["cutoff_frac"],
                    "n_train": r["n_train"],
                    "level": lvl,
                    "level_pct": f"{int(lvl * 100)}",
                    "truth": r["truth"],
                    "point": r["point"],
                    "lo": lo,
                    "hi": hi,
                    "ape": ape,
                    "hit": hit,
                    "sharpness": sharp,
                    "signed_err": signed,
                }
            )
    return pd.DataFrame(rows)


# ---------- figures --------------------------------------------------------


def _fig_reliability(metrics: pd.DataFrame) -> go.Figure:
    """Reliability diagram: nominal vs empirical, per shape (separate trace)."""
    fig = go.Figure()
    # Diagonal reference.
    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="Perfect calibration",
        )
    )
    # Global.
    g_cov = metrics.groupby("level")["hit"].mean() * 100
    g_cov.index = g_cov.index * 100
    fig.add_trace(
        go.Scatter(
            x=g_cov.index,
            y=g_cov.values,
            mode="lines+markers",
            name="Global (all forms)",
            marker=dict(size=10),
            line=dict(width=3),
        )
    )
    # Per shape.
    for shape in metrics["shape"].unique():
        sub = metrics[metrics["shape"] == shape]
        if len(sub) < 5:
            continue
        cov = sub.groupby("level")["hit"].mean() * 100
        cov.index = cov.index * 100
        fig.add_trace(
            go.Scatter(
                x=cov.index,
                y=cov.values,
                mode="lines+markers",
                name=f"shape={shape}",
                line=dict(width=1.5),
            )
        )
    fig.update_layout(
        title="Reliability diagram: nominal CI level vs empirical coverage",
        xaxis_title="Nominal coverage (%)",
        yaxis_title="Empirical coverage (%)",
        xaxis_range=[0, 100],
        yaxis_range=[0, 100],
    )
    return fig


def _fig_reliability_per_form_type(metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="Perfect calibration",
        )
    )
    g_cov = metrics.groupby("level")["hit"].mean() * 100
    g_cov.index = g_cov.index * 100
    fig.add_trace(
        go.Scatter(
            x=g_cov.index,
            y=g_cov.values,
            mode="lines+markers",
            name="Global",
            marker=dict(size=10),
            line=dict(width=3),
        )
    )
    for ft in metrics["form_type"].unique():
        sub = metrics[metrics["form_type"] == ft]
        if len(sub) < 5:
            continue
        cov = sub.groupby("level")["hit"].mean() * 100
        cov.index = cov.index * 100
        fig.add_trace(
            go.Scatter(
                x=cov.index,
                y=cov.values,
                mode="lines+markers",
                name=f"type={ft}",
                line=dict(width=1.5),
            )
        )
    fig.update_layout(
        title="Reliability per form_type",
        xaxis_title="Nominal coverage (%)",
        yaxis_title="Empirical coverage (%)",
        xaxis_range=[0, 100],
        yaxis_range=[0, 100],
    )
    return fig


def _fig_sharpness_per_level(metrics: pd.DataFrame) -> go.Figure:
    sub = metrics.groupby("level")["sharpness"].median().reset_index()
    sub["level_pct"] = (sub["level"] * 100).astype(int)
    fig = px.bar(
        sub,
        x="level_pct",
        y="sharpness",
        labels={"level_pct": "CI level (%)", "sharpness": "Median (hi-lo)/truth"},
        title="Sharpness (raw, БЕЗ калібровки) за рівнем CI",
    )
    return fig


# ---------- main -----------------------------------------------------------


def main(input_path, features_csv, form_types_csv, output_md, figures_dir, cutoffs, min_n, limit):
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
    print(f"Cutoffs: {cutoffs}")
    print(
        f"Form type coverage: {sum(1 for f in eligible if f['form_type'] != 'unknown')}/{len(eligible)}"
    )

    points: list[MultiLevelPoint] = []
    for i, form in enumerate(eligible, 1):
        for cutoff in cutoffs:
            p = _backtest_one(form, cutoff)
            if p is not None:
                points.append(p)
        if i % 10 == 0 or i == len(eligible):
            print(f"  processed {i}/{len(eligible)} forms...")

    points_df = pd.DataFrame([p.__dict__ for p in points])
    points_df.to_csv(figures_dir / "11_multilevel_points.csv", index=False)

    metrics = _compute_per_level(points_df)
    metrics.to_csv(figures_dir / "11_multilevel_metrics.csv", index=False)

    # Aggregations per level.
    by_level = agg_by(metrics, "level")
    by_level_shape = agg_by(metrics, ["level", "shape"])
    by_level_form_type = agg_by(metrics, ["level", "form_type"])
    by_level_cutoff = agg_by(metrics, ["level", "cutoff_frac"])
    by_level_n_class = agg_by(metrics, ["level", "n_class"])

    # Figures.
    figs = {
        "reliability_shape": _fig_reliability(metrics),
        "reliability_form_type": _fig_reliability_per_form_type(metrics),
        "sharpness_per_level": _fig_sharpness_per_level(metrics),
    }
    fig_paths = {}
    for name, fig in figs.items():
        path = figures_dir / f"11_{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        fig_paths[name] = path.relative_to(output_md.parent)

    md = _render_markdown(
        n_forms_total=df["FORM_ID"].nunique(),
        n_forms_eligible=len(eligible),
        n_points=int(len(points_df)),
        n_eval=int(metrics["form_id"].nunique() * len(LEVELS)),
        skipped=skipped,
        cutoffs=cutoffs,
        min_n=min_n,
        input_path=input_path,
        input_hash=input_hash,
        by_level=by_level,
        by_level_shape=by_level_shape,
        by_level_form_type=by_level_form_type,
        by_level_cutoff=by_level_cutoff,
        by_level_n_class=by_level_n_class,
        fig_paths=fig_paths,
    )
    output_md.write_text(md, encoding="utf-8")
    print(f"\nReport written to: {output_md}")


def _render_markdown(
    *,
    n_forms_total,
    n_forms_eligible,
    n_points,
    n_eval,
    skipped,
    cutoffs,
    min_n,
    input_path,
    input_hash,
    by_level,
    by_level_shape,
    by_level_form_type,
    by_level_cutoff,
    by_level_n_class,
    fig_paths,
) -> str:
    return f"""# 11 - Multi-level Reliability (RAW, no calibration)

**Generated:** {datetime.now().isoformat(timespec="seconds")}
**Source:** `{input_path}` (sha256:`{input_hash}`)
**Forms:** {n_forms_total} -> {n_forms_eligible} eligible (N >= {min_n})
**Cutoffs:** {cutoffs} - **Horizon:** {HORIZON_FRACTION} of cutoff span
**Backtest points:** {n_points} - **Multi-level eval rows:** {len(LEVELS) * n_points}
**Skipped:** {skipped}

## Що тестується

Цей звіт обчислює coverage на 4 рівнях CI: 50%, 80%, 90%, 95% --
**БЕЗ** `apply_calibration_arrays` (multiplier x10), **БЕЗ** P10 sample-size scaling.
Це RAW NHPP-симуляція. Прод-flow застосовує калібровку, що маскує справжню
якість моделі. Reliability-діаграма показує наскільки сам NHPP калібрований.

Якщо emp_coverage(L) ~= L для всіх L (на діагоналі) - модель ідеально калібрована.
Якщо emp_coverage значно нижче L - CI занадто вузька (variance underestimation).

## Global per level

{df_to_md(by_level)}

## Per shape x level

{df_to_md(by_level_shape)}

## Per form_type x level

{df_to_md(by_level_form_type)}

## Per cutoff x level

{df_to_md(by_level_cutoff)}

## Per n_class x level

{df_to_md(by_level_n_class)}

## Figures

- [Reliability per shape]({fig_paths["reliability_shape"]})
- [Reliability per form_type]({fig_paths["reliability_form_type"]})
- [Median sharpness per level]({fig_paths["sharpness_per_level"]})

## Як читати

1. **Global per level**: на 50% nominal має бути ~50% emp. Дивишся куди тягне.
   З 02_ ми знаємо що на 95% raw це ~30% (до калібровки). Тому 50% rаw очікувано ~10-20%.
2. **Per form_type**: чи survey, event_registration, recruitment мають різну калібровку?
   Якщо так - per-type calibration matters.
3. **Per cutoff**: рання cutoff (0.1, 0.2) дає більше variance => CI відносно вже того
   що треба. Це підтвердить P10.

## Артефакти

- `figures/11_multilevel_points.csv` - один рядок на (form, cutoff) з усіма 4 рівнями.
- `figures/11_multilevel_metrics.csv` - long format (point x level).
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
        default=repo_root / "research" / "reports" / "11_multi_level_reliability.md",
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
        args.form_types_csv,
        args.output,
        args.figures_dir,
        cutoffs,
        args.min_n,
        args.limit,
    )
