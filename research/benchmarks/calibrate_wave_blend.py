"""calibrate_wave_blend.py — maturation-prior + shrinkage для within-wave (G1).

Переможець 22_: blend = w·prior + (1-w)·fit, w = n_prior/(n_prior+n),
prior = n / maturation_type(elapsed_h). Holdout: 44.4% → 32.4% MAPE, p<0.0001.

Пише у core/forecast/wave_priors.json["blend_calibration"]:
  - maturation per-type (+global) на knots,
  - n_prior (сила shrinkage),
  - holdout MAPE (для чесності/тези).

Калібрування — на ПОВНИХ даних (deployment); holdout-число — для звіту.

Запуск:
    .venv/Scripts/python.exe research/benchmarks/calibrate_wave_blend.py
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
from core.forecast.wave_estimator import estimate_wave  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PRIORS_PATH = REPO / "core" / "forecast" / "wave_priors.json"
N_OBS = (5, 8, 10, 15, 20)
WINDOW_H = 6.0
KNOTS_H = (0.25, 0.5, 1.0, 2.0, 3.0, 6.0)
FRAC_FLOOR = 0.05
N_PRIOR_GRID = (2, 3, 5, 8, 12, 20, 30)
TEST_SIZE = 0.32
SEED = 42


def _records(ann, df):
    rows = []
    for _, r in ann.iterrows():
        if not r["waves_iso"] or pd.isna(r["waves_iso"]):
            continue
        waves = [pd.Timestamp(w) for w in str(r["waves_iso"]).split("|")]
        ftype, fid = r["form_type"], r["form_id"]
        skip = int(r["test_resp"] or 0)
        grp = df[df["FORM_ID"] == fid]["TIMESTAMP"].sort_values().iloc[skip:].reset_index(drop=True)
        for i, wt in enumerate(waves):
            nxt = waves[i + 1] if i + 1 < len(waves) else wt + pd.Timedelta(hours=24)
            in_wave = grp[(grp >= wt) & (grp < min(nxt, wt + pd.Timedelta(hours=WINDOW_H)))]
            in_wave = in_wave.reset_index(drop=True)
            truth = len(in_wave)
            if truth < 8:
                continue
            mat = {kh: int((in_wave < wt + pd.Timedelta(hours=kh)).sum()) / truth for kh in KNOTS_H}
            for n in N_OBS:
                if n >= truth:
                    continue
                obs = in_wave.iloc[:n]
                span_h = (obs.iloc[-1] - wt).total_seconds() / 3600.0
                if span_h <= 0:
                    continue
                try:
                    wf = estimate_wave(obs.tolist(), horizon_h=WINDOW_H, form_type=ftype)
                except ForecastError:
                    continue
                rows.append(
                    {
                        "fid": fid,
                        "ftype": ftype,
                        "n": n,
                        "span_h": span_h,
                        "fit": wf.point,
                        "truth": truth,
                        "mat": mat,
                    }
                )
    return pd.DataFrame(rows)


def _maturation(train):
    out = {}
    for key, g in [("__global__", train), *train.groupby("ftype")]:
        if len(g) < 8:
            continue
        med = [float(np.median([m[kh] for m in g["mat"]])) for kh in KNOTS_H]
        out[key] = list(np.maximum.accumulate(med))
    return out


def _prior(mat, ftype, span_h, n):
    curve = mat.get(ftype, mat["__global__"])
    frac = float(np.interp(span_h, KNOTS_H, curve, left=curve[0], right=curve[-1]))
    return n / max(frac, FRAC_FLOOR)


def _calib_n_prior(tr, mat):
    pri = np.array([_prior(mat, r.ftype, r.span_h, r.n) for r in tr.itertuples()])
    best, best_m = N_PRIOR_GRID[0], 1e9
    truth = tr["truth"].to_numpy()
    for npri in N_PRIOR_GRID:
        w = npri / (npri + tr["n"].to_numpy())
        blend = w * pri + (1 - w) * tr["fit"].to_numpy()
        m = float(np.median(np.abs(blend - truth) / np.maximum(truth, 1)))
        if m < best_m:
            best_m, best = m, npri
    return best


def main():
    ann = pd.read_csv(REPO / "data" / "wave_annotations.csv")
    df = pd.read_csv(REPO / "data" / "Form Timestamp Collection.csv")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    ann = ann[ann["waves_iso"].notna() & (ann["waves_iso"] != "")].copy()
    ann["ftype"] = ann["form_type"].fillna("unknown")
    recs = _records(ann, df)

    # Holdout validation (honest number for report).
    rng = np.random.RandomState(SEED)
    uniq = ann[["form_id", "ftype"]].drop_duplicates()
    counts = uniq["ftype"].value_counts()
    strat = uniq["ftype"].apply(lambda x: x if counts[x] >= 6 else "rare")
    ho = []
    for _, idx in uniq.groupby(strat).groups.items():
        idx = list(idx)
        rng.shuffle(idx)
        ho.extend(idx[: max(1, int(len(idx) * TEST_SIZE))])
    ho_ids = set(uniq.loc[ho, "form_id"])
    tr_r, ho_r = recs[~recs["fid"].isin(ho_ids)], recs[recs["fid"].isin(ho_ids)]
    mat_tr = _maturation(tr_r)
    npri_tr = _calib_n_prior(tr_r, mat_tr)
    pri = np.array([_prior(mat_tr, r.ftype, r.span_h, r.n) for r in ho_r.itertuples()])
    w = npri_tr / (npri_tr + ho_r["n"].to_numpy())
    blend = w * pri + (1 - w) * ho_r["fit"].to_numpy()
    truth = ho_r["truth"].to_numpy()
    fit_mape = float(np.median(np.abs(ho_r["fit"].to_numpy() - truth) / np.maximum(truth, 1)) * 100)
    blend_mape = float(np.median(np.abs(blend - truth) / np.maximum(truth, 1)) * 100)
    print(f"HOLDOUT: fit={fit_mape:.1f}%  blend={blend_mape:.1f}%  (n_prior={npri_tr})")

    # Deployment calibration (full data).
    mat = _maturation(recs)
    npri = _calib_n_prior(recs, mat)
    print(f"DEPLOY: n_prior={npri}, types={sorted(k for k in mat if k != '__global__')}")

    priors = json.loads(PRIORS_PATH.read_text(encoding="utf-8"))
    priors["blend_calibration"] = {
        "schema": 1,
        "method": "maturation_shrinkage",
        "knots_h": list(KNOTS_H),
        "frac_floor": FRAC_FLOOR,
        "n_prior": int(npri),
        "maturation": {k: [round(x, 4) for x in v] for k, v in mat.items()},
        "holdout_fit_mape": round(fit_mape / 100, 3),
        "holdout_blend_mape": round(blend_mape / 100, 3),
        "n_calib": int(len(recs)),
    }
    PRIORS_PATH.write_text(json.dumps(priors, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Written blend_calibration to {PRIORS_PATH}")


if __name__ == "__main__":
    main()
