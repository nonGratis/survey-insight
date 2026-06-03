"""calibrate_wave_ci.py — generate Mondrian conformal CI calibration.

Winner of 19_wave_ci_ab.py: M7 = per-pred-bucket conformal scaling of the
delta-CI half-width, then hard cap_width (prod policy). Holdout coverage 86%,
Winkler 13, zero absurd intervals.

This script:
  1. Validates the within-wave-bucketed calibration on a train/holdout split
     (must reproduce the ~86% from 19_), printing honest holdout coverage.
  2. Calibrates on the FULL annotated dataset (deployment) and writes the
     result into core/forecast/wave_priors.json under "ci_calibration".

Bucketing variable = within-wave point estimate (what estimate_wave produces;
pre_count shifts both bounds and truth equally so coverage is preserved).

No magic numbers: bucket edges = train terciles of the within-wave point,
per-bucket q = empirical CONFIDENCE-quantile of |err| / delta_half.

Запуск:
    .venv/Scripts/python.exe research/benchmarks/calibrate_wave_ci.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning

warnings.simplefilter("ignore", OptimizeWarning)
warnings.simplefilter("ignore", RuntimeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.forecast import ForecastError  # noqa: E402
from core.forecast.delta_ci import cap_width  # noqa: E402
from core.forecast.wave_detector import detect_wave_starts  # noqa: E402
from core.forecast.wave_estimator import estimate_wave  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PRIORS_PATH = REPO / "core" / "forecast" / "wave_priors.json"
N_TRAIN_VALUES = (5, 8, 10, 15, 20)
HORIZON_HOURS = (0.5, 1.0, 2.0, 6.0)
CONFIDENCE = 0.95
TEST_SIZE = 0.32
SEED = 42
CAP_MAX_RELATIVE = 2.0
CAP_MIN_ABSOLUTE = 10.0


def _wave_residuals(ids: set, valid: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Within-wave point, delta half-width, and truth — for calibration."""
    rows = []
    for _, row in valid[valid["form_id"].isin(ids)].iterrows():
        fid, ftype = row["form_id"], row["form_type"]
        ts_all = df[df["FORM_ID"] == fid]["TIMESTAMP"].sort_values().reset_index(drop=True)
        if len(ts_all) < 6:
            continue
        for n_train in N_TRAIN_VALUES:
            if len(ts_all) <= n_train:
                continue
            observed = ts_all.iloc[:n_train].reset_index(drop=True)
            t_c = observed.iloc[-1]
            if (t_c - observed.iloc[0]).total_seconds() <= 0:
                continue
            waves = detect_wave_starts(observed, form_type=ftype, test_skip=0)
            if not waves:
                continue
            ws = waves[-1].timestamp
            pre = int((observed < ws).sum())
            cur = observed[observed >= ws].reset_index(drop=True)
            if len(cur) < 5:
                continue
            for h in HORIZON_HOURS:
                horizon_end = t_c + pd.Timedelta(hours=h)
                hfw = (horizon_end - ws).total_seconds() / 3600.0
                if hfw <= 0:
                    continue
                truth_within = int((ts_all <= horizon_end).sum()) - pre
                try:
                    wf = estimate_wave(cur.tolist(), horizon_h=hfw, form_type=ftype)
                except ForecastError:
                    continue
                rows.append(
                    {
                        "point": wf.point,  # within-wave point
                        "half": max((wf.ci_upper - wf.ci_lower) / 2.0, 1.0),
                        "truth": truth_within,
                    }
                )
    return pd.DataFrame(rows)


def fit_calibration(tr: pd.DataFrame) -> tuple[list[float], list[float]]:
    """Return (bucket_edges, bucket_q) from training residuals."""
    edges = [float(np.quantile(tr["point"], q)) for q in (1 / 3, 2 / 3)]

    def b(p):
        return 0 if p < edges[0] else (1 if p < edges[1] else 2)

    tr = tr.assign(bk=tr["point"].apply(b))
    q = []
    for k in (0, 1, 2):
        g = tr[tr["bk"] == k]
        nr = (g["point"] - g["truth"]).abs() / g["half"]
        q.append(float(np.quantile(nr, CONFIDENCE)) if len(g) else 3.0)
    return edges, q


def coverage(ho: pd.DataFrame, edges: list[float], q: list[float]) -> dict:
    def b(p):
        return 0 if p < edges[0] else (1 if p < edges[1] else 2)

    hits, widths, caps = [], [], []
    for _, r in ho.iterrows():
        qq = q[b(r["point"])]
        lo = r["point"] - qq * r["half"]
        hi = r["point"] + qq * r["half"]
        clo, chi = cap_width(
            np.array([float(r["point"])]),
            np.array([lo]),
            np.array([hi]),
            max_relative=CAP_MAX_RELATIVE,
            min_absolute=CAP_MIN_ABSOLUTE,
        )
        lo, hi = float(clo[0]), float(chi[0])
        hits.append(lo <= r["truth"] <= hi)
        widths.append(hi - lo)
        caps.append((hi - lo) / 2.0 > CAP_MAX_RELATIVE * max(r["point"], 1))
    return {
        "coverage": float(np.mean(hits) * 100),
        "width_p50": float(np.median(widths)),
        "width_max": float(np.max(widths)),
        "cap_viol": float(np.mean(caps) * 100),
        "n": len(ho),
    }


def _stratified_split(forms: pd.DataFrame) -> tuple[set, set]:
    rng = np.random.RandomState(SEED)
    forms = forms.copy()
    counts = forms["ftype"].value_counts()
    forms["strat"] = forms["ftype"].apply(lambda x: x if counts[x] >= 6 else "rare")
    train, holdout = [], []
    for _, idx in forms.groupby("strat").groups.items():
        idx = list(idx)
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * TEST_SIZE))
        holdout.extend(idx[:n_test])
        train.extend(idx[n_test:])
    return set(forms.loc[train, "form_id"]), set(forms.loc[holdout, "form_id"])


def main():
    # Ідемпотентність: residual нормуємо на RAW delta-півширину (cap, без
    # conformal). Інакше estimate_wave застосував би вже-завантажений q →
    # калібрування залежало б від попереднього стану (не відтворювано).
    import core.forecast.wave_estimator as _we

    base = json.loads(PRIORS_PATH.read_text(encoding="utf-8"))
    base.pop("ci_calibration", None)
    _we._PRIORS_CACHE = base  # estimate_wave тепер дає raw-capped CI

    ann = pd.read_csv(REPO / "data" / "wave_annotations.csv")
    df = pd.read_csv(REPO / "data" / "Form Timestamp Collection.csv")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    valid = ann[ann["waves_iso"].notna() & (ann["waves_iso"] != "")].copy()
    valid["ftype"] = valid["form_type"].fillna("unknown")

    # 1. Validate on split (honesty check).
    train_ids, ho_ids = _stratified_split(valid[["form_id", "ftype"]])
    tr = _wave_residuals(train_ids, valid, df)
    ho = _wave_residuals(ho_ids, valid, df)
    edges_v, q_v = fit_calibration(tr)
    cov = coverage(ho, edges_v, q_v)
    print("=== VALIDATION (train->holdout) ===")
    print(f"  edges={[round(e, 1) for e in edges_v]}  q={[round(x, 2) for x in q_v]}")
    print(
        f"  holdout coverage={cov['coverage']:.1f}%  width_p50={cov['width_p50']:.0f}  "
        f"width_max={cov['width_max']:.0f}  cap_viol={cov['cap_viol']:.0f}%  n={cov['n']}"
    )

    # 2. Calibrate on FULL data (deployment).
    full = _wave_residuals(set(valid["form_id"]), valid, df)
    edges, q = fit_calibration(full)
    print("\n=== DEPLOYMENT (full data) ===")
    print(
        f"  edges={[round(e, 1) for e in edges]}  q={[round(x, 2) for x in q]}  n_calib={len(full)}"
    )

    priors = json.loads(PRIORS_PATH.read_text(encoding="utf-8"))
    priors["ci_calibration"] = {
        "schema": 1,
        "method": "mondrian_capped",
        "confidence": CONFIDENCE,
        "bucket_var": "within_wave_point",
        "bucket_edges": [round(e, 3) for e in edges],
        "bucket_q": [round(x, 4) for x in q],
        "cap_max_relative": CAP_MAX_RELATIVE,
        "cap_min_absolute": CAP_MIN_ABSOLUTE,
        "n_calib": int(len(full)),
        "holdout_coverage": round(cov["coverage"] / 100.0, 3),
    }
    PRIORS_PATH.write_text(json.dumps(priors, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWritten ci_calibration to {PRIORS_PATH}")


if __name__ == "__main__":
    main()
