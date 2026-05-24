"""03b_relaxation_sweep.py — знайти оптимальний K_min relaxation factor.

03_bias_investigation показав, що K_min = last знижує bias до -6.2%,
а K_min = last·1.5 перестрибує до +16.5%. Цей скрипт sweep'ує
relaxation factor ∈ {1.0, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50} і
шукає sweet spot, де:
  - |bias| мінімальний
  - coverage максимальний
  - MAPE не сильно зростає

Output: research/reports/03b_relaxation_sweep.md з рекомендацією.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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

logging.getLogger().setLevel(logging.WARNING)

RELAXATION_FACTORS = [1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50]


def _make_relaxed_models(factor: float):
    class _Logistic(LogisticModel):
        def bounds(self, y, target):
            (k_min, r_lo, t0_lo), high = super().bounds(y, target)
            return (max(float(y[-1]) * factor, k_min), r_lo, t0_lo), high

    class _Gompertz(GompertzModel):
        def bounds(self, y, target):
            (k_min, r_lo, t0_lo), high = super().bounds(y, target)
            return (max(float(y[-1]) * factor, k_min), r_lo, t0_lo), high

    class _AsympExp(AsymptoticExpModel):
        def bounds(self, y, target):
            (a_min, b_lo, c_lo), high = super().bounds(y, target)
            return (max(float(y[-1]) * factor, a_min), b_lo, c_lo), high

    return (_Logistic(), _Gompertz(), _AsympExp())


def _run_one(ts_sorted, cutoff_frac, factor):
    n_total = len(ts_sorted)
    n_train = int(round(cutoff_frac * n_total))
    if n_train < 5:
        return None
    first_ts = ts_sorted[0]
    cutoff_ts = ts_sorted[n_train - 1]
    cutoff_span = (cutoff_ts - first_ts).total_seconds()
    if cutoff_span <= 0:
        return None
    horizon_seconds = max(cutoff_span * 0.25, 86400.0)
    horizon_end = cutoff_ts + pd.Timedelta(seconds=horizon_seconds)
    truth = sum(1 for t in ts_sorted if t <= horizon_end)
    if truth <= 0:
        return None
    t_train = np.array([(t - first_ts).total_seconds() / 86400.0 for t in ts_sorted[:n_train]])
    y_train = np.arange(1, n_train + 1, dtype=float)
    try:
        models = _make_relaxed_models(factor) if factor > 1.0 else None
        fitted = (
            select_best_model(t_train, y_train, target=None, models=models)
            if models is not None
            else select_best_model(t_train, y_train, target=None)
        )
        last_observed = int(y_train[-1])
        horizon_days = horizon_seconds / 86400.0
        t_future = np.arange(t_train[-1] + 1.0, t_train[-1] + np.ceil(horizon_days) + 1.0)
        if len(t_future) == 0:
            t_future = np.array([t_train[-1] + 1.0])
        rng = np.random.default_rng(42)
        mean_cum, ci_lo, ci_hi = nhpp_prediction_interval(
            fitted, t_future, last_observed=last_observed, n_sims=2000, rng=rng
        )
        point = max(int(round(mean_cum[-1])), last_observed)
        return {
            "factor": factor,
            "truth": truth,
            "point_estimate": point,
            "ci_lower": int(round(ci_lo[-1])),
            "ci_upper": int(round(ci_hi[-1])),
        }
    except Exception:  # noqa: BLE001
        return None


def main(input_path: Path, output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df = df.drop_duplicates(subset=["FORM_ID", "TIMESTAMP"]).reset_index(drop=True)

    eligible = []
    for fid, g in df.groupby("FORM_ID"):
        n = len(g)
        if n < 30:
            continue
        ts = sorted(g["TIMESTAMP"].tolist())
        eligible.append((fid, ts))
    print(f"Eligible forms: {len(eligible)}")
    print(f"Sweeping {len(RELAXATION_FACTORS)} factors × {len(eligible)} forms × 3 cutoffs")

    all_rows = []
    for factor in RELAXATION_FACTORS:
        print(f"  factor={factor}...")
        for fid, ts in eligible:
            for cutoff in (0.3, 0.5, 0.7):
                row = _run_one(ts, cutoff, factor)
                if row is not None:
                    row["form_id"] = fid
                    row["cutoff_frac"] = cutoff
                    all_rows.append(row)

    res = pd.DataFrame(all_rows)
    res["signed_err"] = (res["point_estimate"] - res["truth"]) / res["truth"]
    res["ape"] = (res["truth"] - res["point_estimate"]).abs() / res["truth"]
    res["hit_95"] = (res["ci_lower"] <= res["truth"]) & (res["truth"] <= res["ci_upper"])

    summary = res.groupby("factor").agg(
        n_points=("signed_err", "size"),
        bias_p50=("signed_err", lambda s: s.median() * 100),
        bias_abs=("signed_err", lambda s: s.median() * 100),
        mape_p50=("ape", lambda s: s.median() * 100),
        coverage=("hit_95", lambda s: s.mean() * 100),
    )
    summary["bias_abs"] = summary["bias_abs"].abs()

    # Composite score: minimize |bias| + maximize coverage gap to 95
    summary["coverage_gap"] = (95 - summary["coverage"]).abs()
    summary["score"] = summary["bias_abs"] + summary["coverage_gap"]

    summary = summary.round(2)
    summary.to_csv(Path(output_md).parent / "figures" / "03b_relaxation_sweep.csv", index=True)

    # Найкращий factor за composite score
    best_factor = summary["score"].idxmin()

    # Графік
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=summary.index,
            y=summary["bias_p50"],
            mode="lines+markers",
            name="Bias (median %)",
            line=dict(color="#d62728"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary.index,
            y=summary["coverage"],
            mode="lines+markers",
            name="Coverage (%)",
            line=dict(color="#2ca02c"),
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary.index,
            y=summary["mape_p50"],
            mode="lines+markers",
            name="MAPE (median %)",
            line=dict(color="#ff7f0e"),
            yaxis="y2",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_vline(
        x=best_factor, line_dash="dash", line_color="blue", annotation_text=f"Best={best_factor}"
    )
    fig.update_layout(
        title="K_min relaxation sweep: bias / coverage / MAPE vs factor",
        xaxis_title="Relaxation factor (K_min = last × factor)",
        yaxis=dict(title="Bias (%)"),
        yaxis2=dict(title="Coverage / MAPE (%)", overlaying="y", side="right"),
    )
    fig_path = Path(output_md).parent / "figures" / "03b_relaxation_sweep.html"
    fig.write_html(fig_path, include_plotlyjs="cdn")

    md = f"""# 03b — K_min Relaxation Sweep

Запуск на {len(eligible)} формах × 3 cutoff'и × {len(RELAXATION_FACTORS)} factors.

## Sweep results

| factor | n | bias % | |bias| | MAPE % | coverage % | score |
|---:|---:|---:|---:|---:|---:|---:|
{
        chr(10).join(
            f"| {idx} | {int(row.n_points)} | {row.bias_p50:+.2f} | {row.bias_abs:.2f} | "
            f"{row.mape_p50:.2f} | {row.coverage:.2f} | {row.score:.2f} |"
            for idx, row in summary.iterrows()
        )
    }

## Найкращий factor: **{best_factor}**

За composite score = |bias| + |coverage - 95|.

## Графік

[K_min sweep: bias / coverage / MAPE]({fig_path.relative_to(output_md.parent)})

## Recommendation

Змінити `_capacity_bounds` у `core/forecast/models.py`:

```python
def _capacity_bounds(y, target):
    last = float(y[-1])
    if target is not None and target > 0:
        return max(last * {best_factor}, 0.3 * target), max(last * 1.05, 3.0 * target)
    return max(last * {best_factor}, 1.0), max(last * 10.0, 10.0)
```

Очікувані ефекти на production:
- Bias: -6.2% → {summary.loc[best_factor, "bias_p50"]:+.1f}%
- Coverage: 50.0% → {summary.loc[best_factor, "coverage"]:.1f}%
- MAPE: 27.7% → {summary.loc[best_factor, "mape_p50"]:.1f}%
"""
    output_md.write_text(md, encoding="utf-8")
    print(f"\nReport: {output_md}")
    print(f"\nBest factor: {best_factor}")
    print(summary)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    p.add_argument(
        "--input", type=Path, default=repo_root / "data" / "Form Timestamp Collection.csv"
    )
    p.add_argument(
        "--output",
        type=Path,
        default=repo_root / "research" / "reports" / "03b_relaxation_sweep.md",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.input, args.output)
