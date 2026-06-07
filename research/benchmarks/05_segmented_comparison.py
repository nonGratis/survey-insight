"""05_segmented_comparison.py — AB-comparison default vs CP-aware forecast.

Прогоняє той самий backtest-grid, що 02 (96 форм × 3 cutoff'и), але
порівнює два режими:

  A. default        — forecast_responses на повному префіксі
  B. segmented      — forecast_with_segmentation: PELT детектить хвилі,
                      модель тренується на post-CP сегменті

Очікувані ефекти (гіпотеза з research/04 — Ljung-Box 87% rejection):
- late_burst:  значне покращення MAPE (агітаційна хвиля під дедлайн)
- ill_fit:     значне покращення (multi-wave траєкторії)
- logarithmic: без змін (PELT не знаходить CP на чистому ряду)
- logistic:    мінімальні зміни

Output: research/reports/05_segmented_comparison.md з ΔMAPE / ΔCoverage /
ΔBias per shape, статистичним тестом значущості (paired Wilcoxon).
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
import plotly.graph_objects as go
from scipy.optimize import OptimizeWarning
from scipy.stats import wilcoxon

warnings.simplefilter("ignore", OptimizeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.forecast import (  # noqa: E402
    ForecastError,
    forecast_responses,
    forecast_with_segmentation,
)
from core.timeline import build_timeline_from_timestamps  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)

MIN_N_FOR_BACKTEST = 30
CUTOFFS = (0.3, 0.5, 0.7)
HORIZON_FRACTION = 0.25


@dataclass(frozen=True)
class Point:
    form_id: str
    shape: str
    cutoff_frac: float
    n_train: int
    horizon_days: float
    truth: int
    mode: str  # "default" або "segmented"
    point_estimate: int
    ci_lower: int
    ci_upper: int
    n_changepoints: int
    error: str | None


def _backtest_one(form_id, shape, timestamps, cutoff, mode):
    n_total = len(timestamps)
    n_train = int(round(cutoff * n_total))
    if n_train < 5:
        return None
    ts_sorted = sorted(timestamps)
    first_ts = ts_sorted[0]
    cutoff_ts = ts_sorted[n_train - 1]
    cutoff_span = (cutoff_ts - first_ts).total_seconds()
    if cutoff_span <= 0:
        return None
    horizon_seconds = max(cutoff_span * HORIZON_FRACTION, 86400.0)
    horizon_end = cutoff_ts + pd.Timedelta(seconds=horizon_seconds)
    truth = sum(1 for t in ts_sorted if t <= horizon_end)
    if truth <= 0:
        return None
    prefix_dt = [t.to_pydatetime() for t in ts_sorted[:n_train]]
    timeline = build_timeline_from_timestamps(prefix_dt)
    try:
        if mode == "default":
            fc = forecast_responses(timeline)
            cps = []
        else:
            # auto_segment=True явно, бо default тепер False (research/05 result)
            fc, cps = forecast_with_segmentation(timeline, auto_segment=True)
        future_dates = pd.DatetimeIndex(fc.future_dates)
        target_date = pd.Timestamp(horizon_end.normalize())
        if target_date <= future_dates[0]:
            idx = 0
        elif target_date >= future_dates[-1]:
            idx = len(future_dates) - 1
        else:
            idx = min(
                int(np.searchsorted(future_dates, target_date, side="left")),
                len(future_dates) - 1,
            )
        return Point(
            form_id=form_id,
            shape=shape,
            cutoff_frac=cutoff,
            n_train=n_train,
            horizon_days=horizon_seconds / 86400.0,
            truth=truth,
            mode=mode,
            point_estimate=int(round(fc.future_cum.iloc[idx])),
            ci_lower=int(round(fc.ci_lower.iloc[idx])),
            ci_upper=int(round(fc.ci_upper.iloc[idx])),
            n_changepoints=len(cps),
            error=None,
        )
    except ForecastError as e:
        return Point(
            form_id=form_id,
            shape=shape,
            cutoff_frac=cutoff,
            n_train=n_train,
            horizon_days=horizon_seconds / 86400.0,
            truth=truth,
            mode=mode,
            point_estimate=-1,
            ci_lower=-1,
            ci_upper=-1,
            n_changepoints=0,
            error=str(e),
        )


def _metrics(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["error"].isna() & (df["truth"] > 0)].copy()
    ok["ape"] = (ok["truth"] - ok["point_estimate"]).abs() / ok["truth"]
    ok["hit_95"] = (ok["ci_lower"] <= ok["truth"]) & (ok["truth"] <= ok["ci_upper"])
    ok["signed_err"] = (ok["point_estimate"] - ok["truth"]) / ok["truth"]
    return ok


def _agg_by_shape(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Per (shape, mode) median MAPE / coverage / bias."""
    return metrics_df.groupby(["shape", "mode"], observed=True).agg(
        n=("ape", "size"),
        mape_p50=("ape", lambda s: round(s.median() * 100, 1)),
        coverage=("hit_95", lambda s: round(s.mean() * 100, 1)),
        bias=("signed_err", lambda s: round(s.median() * 100, 1)),
    )


def _delta_table(by_shape: pd.DataFrame) -> pd.DataFrame:
    """ΔMAPE / ΔCoverage per shape: segmented - default."""
    pivot = by_shape.unstack("mode")
    deltas = pd.DataFrame(
        {
            "n": pivot[("n", "default")],
            "mape_default": pivot[("mape_p50", "default")],
            "mape_segmented": pivot[("mape_p50", "segmented")],
            "delta_mape": pivot[("mape_p50", "segmented")] - pivot[("mape_p50", "default")],
            "cov_default": pivot[("coverage", "default")],
            "cov_segmented": pivot[("coverage", "segmented")],
            "delta_cov": pivot[("coverage", "segmented")] - pivot[("coverage", "default")],
            "bias_default": pivot[("bias", "default")],
            "bias_segmented": pivot[("bias", "segmented")],
        }
    )
    return deltas


def _paired_wilcoxon(metrics_df: pd.DataFrame) -> dict:
    """Paired Wilcoxon на APE: default vs segmented."""
    pivot = metrics_df.pivot_table(
        index=["form_id", "cutoff_frac"], columns="mode", values="ape"
    ).dropna()
    if len(pivot) < 10:
        return {"n_paired": len(pivot), "wstat": None, "pvalue": None}
    try:
        wstat, pval = wilcoxon(pivot["default"], pivot["segmented"])
        return {"n_paired": int(len(pivot)), "wstat": float(wstat), "pvalue": float(pval)}
    except Exception:  # noqa: BLE001
        return {"n_paired": int(len(pivot)), "wstat": None, "pvalue": None}


def _figure_delta_mape(deltas: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=deltas.index,
            y=deltas["delta_mape"],
            marker_color=["green" if v < 0 else "red" for v in deltas["delta_mape"]],
            text=[f"{v:+.1f}pp" for v in deltas["delta_mape"]],
            textposition="outside",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Δ MAPE per shape (negative = segmented краще)",
        yaxis_title="Δ MAPE % (segmented - default)",
    )
    return fig


def _figure_changepoint_freq(metrics_df: pd.DataFrame) -> go.Figure:
    seg = metrics_df[metrics_df["mode"] == "segmented"]
    cps_per_shape = seg.groupby("shape", observed=True)["n_changepoints"].mean()
    fig = go.Figure(
        go.Bar(
            x=cps_per_shape.index,
            y=cps_per_shape.values,
            text=[f"{v:.2f}" for v in cps_per_shape.values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Середня кількість виявлених CP per shape",
        yaxis_title="N changepoints",
    )
    return fig


def main(input_path: Path, features_csv: Path, output_md: Path, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df = df.drop_duplicates(subset=["FORM_ID", "TIMESTAMP"]).reset_index(drop=True)
    shapes_df = pd.read_csv(features_csv)
    shapes = dict(zip(shapes_df["form_id"], shapes_df["shape"], strict=True))

    eligible = []
    for fid, g in df.groupby("FORM_ID"):
        if len(g) < MIN_N_FOR_BACKTEST:
            continue
        shape = shapes.get(fid, "unknown")
        if shape == "insufficient":
            continue
        eligible.append((fid, shape, g["TIMESTAMP"].tolist()))
    print(f"Eligible forms: {len(eligible)}")
    print(f"Total backtests: {len(eligible) * len(CUTOFFS) * 2} (default + segmented)")

    points = []
    for i, (fid, shape, ts) in enumerate(eligible, 1):
        for cutoff in CUTOFFS:
            for mode in ("default", "segmented"):
                pt = _backtest_one(fid, shape, ts, cutoff, mode)
                if pt is not None:
                    points.append(pt)
        if i % 20 == 0:
            print(f"  processed {i}/{len(eligible)}...")

    raw_df = pd.DataFrame([p.__dict__ for p in points])
    raw_df.to_csv(figures_dir / "05_raw_points.csv", index=False)

    metrics_df = _metrics(raw_df)
    metrics_df.to_csv(figures_dir / "05_metrics.csv", index=False)
    by_shape = _agg_by_shape(metrics_df)
    deltas = _delta_table(by_shape)
    wilcoxon_result = _paired_wilcoxon(metrics_df)

    figs = {
        "delta_mape": _figure_delta_mape(deltas),
        "cp_freq": _figure_changepoint_freq(metrics_df),
    }
    fig_paths = {}
    for name, fig in figs.items():
        path = figures_dir / f"05_{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        fig_paths[name] = path.relative_to(output_md.parent)

    md = _render_md(
        input_path=input_path,
        input_hash=_file_sha256_short(input_path),
        n_forms=len(eligible),
        n_points=len(metrics_df),
        deltas=deltas,
        wilcoxon=wilcoxon_result,
        metrics_df=metrics_df,
        fig_paths=fig_paths,
    )
    output_md.write_text(md, encoding="utf-8")
    print(f"\nReport: {output_md}")


def _render_md(
    *, input_path, input_hash, n_forms, n_points, deltas, wilcoxon, metrics_df, fig_paths
) -> str:
    global_default = metrics_df[metrics_df["mode"] == "default"]
    global_seg = metrics_df[metrics_df["mode"] == "segmented"]
    mape_def = global_default["ape"].median() * 100
    mape_seg = global_seg["ape"].median() * 100
    cov_def = global_default["hit_95"].mean() * 100
    cov_seg = global_seg["hit_95"].mean() * 100
    cps_total = global_seg["n_changepoints"].sum()
    cps_mean = global_seg["n_changepoints"].mean()

    wilcoxon_line = (
        f"Wilcoxon W={wilcoxon['wstat']:.0f}, p={wilcoxon['pvalue']:.4g}, "
        f"n_paired={wilcoxon['n_paired']}"
        if wilcoxon["wstat"] is not None
        else "Wilcoxon: not enough paired points."
    )

    return f"""# 05 — Segmented vs Default Forecast (A/B Comparison)

**Generated:** {datetime.now().isoformat(timespec="seconds")}
**Source:** `{input_path}` (sha256:`{input_hash}`)
**Eligible forms:** {n_forms} · **Points per mode:** {len(global_default)} / {len(global_seg)}

## TL;DR

| Метрика | Default | Segmented | Δ |
|---|---:|---:|---:|
| **MAPE (median)** | {mape_def:.1f}% | {mape_seg:.1f}% | {mape_seg - mape_def:+.1f}pp |
| **Coverage 95%** | {cov_def:.1f}% | {cov_seg:.1f}% | {cov_seg - cov_def:+.1f}pp |
| **CP total** | — | {cps_total} | — |
| **CP per backtest (avg)** | — | {cps_mean:.2f} | — |

**Paired test:** {wilcoxon_line}

## Per-shape ΔMAPE

{_df_to_md(deltas)}

Інтерпретація:
- `delta_mape < 0` → segmented краще (модель захоплює структуру через CP).
- `delta_mape > 0` → segmented гірше (false-positive CP розрізали корисну
  криву на дрібні шматки).
- `delta_cov > 0` → segmented дає кращий PI calibration.

## Графіки

- [Δ MAPE per shape (bar)]({fig_paths["delta_mape"]})
- [Average CP count per shape]({fig_paths["cp_freq"]})

## Висновок для production

Дивись числа вище. Якщо segmented виграє у `late_burst` і `ill_fit` (як
гіпотеза з research/04), і не псує `logarithmic`/`logistic` — segmented
залишаємо як default (auto_segment=True у service).

Якщо програш у `logarithmic`/`logistic` істотний → варто тюнити
`cp_penalty` під ці категорії, або зробити shape-aware enable/disable.
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
        default=repo_root / "research" / "reports" / "05_segmented_comparison.md",
    )
    p.add_argument(
        "--figures-dir", type=Path, default=repo_root / "research" / "reports" / "figures"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.input, args.features_csv, args.output, args.figures_dir)
