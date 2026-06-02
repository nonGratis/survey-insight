"""20_wave_vs_prod_figure.py — comparison figure for the thesis explanatory note.

Порівнює прод-методи прогнозу (для пояснювальної записки, НЕ для UI-тоглу):
  - OLD: forecast_responses (full-curve fit на весь timeline) — те що було в проді.
  - NEW: current-wave estimator (CUSUM + within-wave fit) — нове в проді.

Читає вже пораховані holdout-точки з 18_endtoend_points.csv (a_prod = OLD,
b_detected = NEW; той самий no-oracle протокол horizon-from-now). Не рахує
заново — single source of truth.

Output:
  - research/reports/20_wave_vs_prod.md — таблиця MAPE/coverage per horizon + global.
  - research/reports/figures/20_wave_vs_prod.html — групова стовпчикова діаграма.

Запуск:
    .venv/Scripts/python.exe research/benchmarks/20_wave_vs_prod_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

REPO = Path(__file__).resolve().parents[2]
POINTS_CSV = REPO / "research" / "reports" / "figures" / "18_endtoend_points.csv"


def _mape(pred: pd.Series, truth: pd.Series) -> float:
    return float((pred - truth).abs().div(truth.clip(lower=1)).median() * 100)


def _coverage(lo: pd.Series, hi: pd.Series, truth: pd.Series) -> float:
    return float(((lo <= truth) & (truth <= hi)).mean() * 100)


def main():
    if not POINTS_CSV.exists():
        print(f"Missing {POINTS_CSV}. Run 18_wave_endtoend_ab.py first.")
        sys.exit(1)
    pts = pd.read_csv(POINTS_CSV)
    old = pts[pts["a_prod"] >= 0]
    new = pts[pts["b_detected"] >= 0]

    lines = ["# 20 — OLD (full-curve) vs NEW (current-wave) на holdout", ""]
    lines.append("Дані: 18_endtoend_points.csv (no-oracle, horizon-from-now).")
    lines.append("")
    lines.append("| Горизонт | OLD MAPE | NEW MAPE | OLD cov95 | NEW cov95 | n |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    horizons = sorted(pts["horizon_h"].unique())
    fig_rows = []
    for h in [*horizons, "ALL"]:
        o = old if h == "ALL" else old[old["horizon_h"] == h]
        nw = new if h == "ALL" else new[new["horizon_h"] == h]
        o_mape = _mape(o["a_prod"], o["truth_cum"])
        n_mape = _mape(nw["b_detected"], nw["truth_cum"])
        o_cov = _coverage(o["a_lo"], o["a_hi"], o["truth_cum"])
        n_cov = _coverage(nw["b_lo"], nw["b_hi"], nw["truth_cum"])
        label = "усі" if h == "ALL" else f"{h} год"
        lines.append(
            f"| {label} | {o_mape:.0f}% | {n_mape:.0f}% | {o_cov:.0f}% | {n_cov:.0f}% | {len(nw)} |"
        )
        if h != "ALL":
            fig_rows.append((label, o_mape, n_mape))

    report = "\n".join(lines)
    (REPO / "research" / "reports" / "20_wave_vs_prod.md").write_text(report, encoding="utf-8")
    print(report)

    # Grouped bar chart: MAPE OLD vs NEW per horizon.
    labels = [r[0] for r in fig_rows]
    fig = go.Figure()
    fig.add_bar(
        name="OLD (повна крива)", x=labels, y=[r[1] for r in fig_rows], marker_color="#d62728"
    )
    fig.add_bar(
        name="NEW (поточна хвиля)", x=labels, y=[r[2] for r in fig_rows], marker_color="#2ca02c"
    )
    fig.update_layout(
        title="MAPE прогнозу: стара повна крива vs нова поточна хвиля (holdout, no-oracle)",
        xaxis_title="Горизонт прогнозу",
        yaxis_title="MAPE, % (менше = краще)",
        barmode="group",
        template="plotly_white",
    )
    out_html = REPO / "research" / "reports" / "figures" / "20_wave_vs_prod.html"
    fig.write_html(str(out_html))
    print("\nSaved: research/reports/20_wave_vs_prod.md + figures/20_wave_vs_prod.html")


if __name__ == "__main__":
    main()
