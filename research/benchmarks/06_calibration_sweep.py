"""06_calibration_sweep.py — sweep CI calibration multiplier до 95% coverage.

Поточний backtest показує coverage 30.9% при заявлених 95%. Sweep шукає
multiplier k такий, що inflated CI:
    ci_lower_calib = point - k · (point - ci_lower)
    ci_upper_calib = point + k · (ci_upper - point)
дає empirical coverage ≈ 95% на 288 точках.

Result: запис у `core/forecast/calibration.py` як CALIBRATION_MULTIPLIER.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning

warnings.simplefilter("ignore", OptimizeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.getLogger().setLevel(logging.WARNING)

MULTIPLIERS = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]
TARGET_COVERAGE = 0.95


def main(metrics_csv: Path, output_md: Path) -> None:
    df = pd.read_csv(metrics_csv)
    df = df[df["truth"] > 0]
    df["point"] = df["point_estimate"]
    df["lo_half"] = df["point"] - df["ci_lower"]
    df["hi_half"] = df["ci_upper"] - df["point"]

    rows = []
    for k in MULTIPLIERS:
        lo = df["point"] - k * df["lo_half"]
        hi = df["point"] + k * df["hi_half"]
        hits = ((lo <= df["truth"]) & (df["truth"] <= hi)).mean()
        avg_width = (hi - lo).median()
        rows.append({"multiplier": k, "coverage": hits, "median_width": avg_width})

    summary = pd.DataFrame(rows)
    # Найменший k, при якому coverage >= target.
    qualifying = summary[summary["coverage"] >= TARGET_COVERAGE]
    best = qualifying.iloc[0] if not qualifying.empty else summary.iloc[-1]

    summary = summary.round(3)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    md = f"""# 06 — CI Calibration Sweep

**Target coverage:** {TARGET_COVERAGE * 100:.0f}%
**Backtest points:** {len(df)}

## Sweep

| Multiplier | Coverage | Median Width |
|---:|---:|---:|
{
        chr(10).join(
            f"| {row.multiplier:.1f} | {row.coverage:.3f} | {row.median_width:.0f} |"
            for row in summary.itertuples()
        )
    }

## Recommended

**CALIBRATION_MULTIPLIER = {best["multiplier"]:.1f}** → emp.coverage = {best["coverage"]:.3f}

Записати у `core/forecast/calibration.py`.
"""
    output_md.write_text(md, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nBest: {best['multiplier']} (coverage {best['coverage']:.3f})")
    print(f"Report: {output_md}")


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metrics-csv",
        type=Path,
        default=repo_root / "research" / "reports" / "figures" / "02_backtest_metrics.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=repo_root / "research" / "reports" / "06_calibration.md",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.metrics_csv, args.output)
