"""calibrate_conformal.py — offline conformal quantile calibration.

Читає делта-CI residuals з повного backtest (12_prod_points.csv після P12)
і обчислює empirical 95-th quantile нормованого residual per-cell.
Результат → JSON артефакт, який runtime `core/forecast/conformal.py` завантажує.

Cell structure: `n_class × horizon_bucket` (15 cells).
Hierarchical fallback: exact cell → horizon_bucket → global.

Methodology (Vovk 2005, Romano 2019, "split conformal"):
- residual r = (truth − point) / max(half_delta, 0.5)
- quantile q_{1−α} = |r|.quantile(0.95) per cell
- Runtime: new_half = half_delta × q_cell
- Asymptotic guarantee: empirical coverage → 1−α якщо calibration set i.i.d.

Caveats:
- Зараз calibration set == test set (slight leakage 1/161 per form).
  Для проду це нормально (старі форми → quantile → застосовуй на нових).
- Cell threshold: 30 samples мінімум для exact cell. Інакше fallback.

Запуск:
    .venv/Scripts/python.exe research/benchmarks/calibrate_conformal.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Горизонт-бакети для часткового groupping. Coverage просідала найбільш на
# 24h і вище → треба окремий quantile для long-horizon.
HORIZON_BUCKETS = {2.0: "short", 6.0: "short", 24.0: "mid", 72.0: "mid", 168.0: "long"}

CELL_MIN_SAMPLES = 30  # мінімум для exact cell quantile
ALPHA = 0.05  # 95% confidence


def horizon_bucket(h: float) -> str:
    """Map horizon_hours → bucket."""
    if h <= 6:
        return "short"
    if h <= 72:
        return "mid"
    return "long"


def main(input_csv: Path, output_json: Path) -> None:
    df = pd.read_csv(input_csv)
    df = df[(df["point"] >= 0) & (df["truth"] > 0)].copy()
    print(f"Input: {len(df)} valid backtest points from {input_csv.name}")

    # Symmetric half-width assumed (delta-CI is symmetric by construction).
    df["half"] = (df["hi"] - df["lo"]) / 2.0
    df["abs_residual"] = (df["truth"] - df["point"]).abs() / np.maximum(df["half"], 0.5)
    df["horizon_bucket"] = df["horizon_hours"].apply(horizon_bucket)

    quantiles: dict[str, float] = {}
    cell_stats: list[dict] = []

    # Level 1: exact cells (n_class × horizon_bucket).
    for (nc, hb), sub in df.groupby(["n_class", "horizon_bucket"]):
        if len(sub) < CELL_MIN_SAMPLES:
            continue
        q = float(sub["abs_residual"].quantile(1.0 - ALPHA))
        quantiles[f"{nc}|{hb}"] = q
        cell_stats.append(
            {
                "key": f"{nc}|{hb}",
                "n": int(len(sub)),
                "q_95": round(q, 3),
                "median_r": round(float(sub["abs_residual"].median()), 3),
            }
        )

    # Level 2: horizon-bucket only (always present).
    for hb, sub in df.groupby("horizon_bucket"):
        q = float(sub["abs_residual"].quantile(1.0 - ALPHA))
        quantiles[f"_|{hb}"] = q
        cell_stats.append(
            {
                "key": f"_|{hb}",
                "n": int(len(sub)),
                "q_95": round(q, 3),
                "median_r": round(float(sub["abs_residual"].median()), 3),
            }
        )

    # Level 3: global fallback.
    q_global = float(df["abs_residual"].quantile(1.0 - ALPHA))
    quantiles["_|_"] = q_global
    cell_stats.append(
        {
            "key": "_|_",
            "n": int(len(df)),
            "q_95": round(q_global, 3),
            "median_r": round(float(df["abs_residual"].median()), 3),
        }
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1,
        "alpha": ALPHA,
        "confidence": 1.0 - ALPHA,
        "n_calibration_samples": len(df),
        "cell_schema": "n_class|horizon_bucket",
        "fallback_order": ["<n_class>|<hb>", "_|<hb>", "_|_"],
        "horizon_buckets": {"short": "<=6h", "mid": "<=72h", "long": ">72h"},
        "quantiles": quantiles,
        "cell_stats": sorted(cell_stats, key=lambda c: -c["n"]),
    }
    output_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(f"\nGlobal q_95 = {q_global:.3f}")
    print("\nCells (sorted by n):")
    for c in cell_stats[:20]:
        print(
            f"  {c['key']:<20}: n={c['n']:>5}, q_95={c['q_95']:.3f}, median_r={c['median_r']:.3f}"
        )
    print(f"\nWrote: {output_json}")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    p.add_argument(
        "--input",
        type=Path,
        default=repo / "research" / "reports" / "figures" / "12_prod_points.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=repo / "core" / "forecast" / "conformal_quantiles.json",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.input, args.output)
