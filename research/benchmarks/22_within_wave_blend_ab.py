"""22_within_wave_blend_ab.py — G1: shrinkage-бленд для within-wave точності.

Діагноз 21_: within-wave MAPE падає від 62% (n=5) до 34% (n=20). Найгірше —
малі n (curve-fit на 5-8 точках нестабільний). Гіпотеза: на малих n підмішати
maturation-prior (скільки хвилі типово вже зібрано за elapsed-час), вага→prior
коли n малий (James-Stein shrinkage).

Методи (на ОРАКУЛ-хвилях, той самий протокол що 21_):
  A · fit    — поточний estimate_wave (curve-fit).
  B · blend  — w·prior + (1-w)·fit, w = n_prior/(n_prior+n),
               prior = n / maturation_type(elapsed_h).

Калібрування на TRAIN (без leakage):
  - maturation_type(τ): медіана (cum@τ / wave_total) на train-хвилях, монотонна.
  - n_prior: grid, мінімум train within-wave MAPE бленду.
Оцінка на HOLDOUT: MAPE by n × type, Wilcoxon (paired).

Запуск:
    .venv/Scripts/python.exe research/benchmarks/22_within_wave_blend_ab.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning
from scipy.stats import wilcoxon

warnings.simplefilter("ignore", OptimizeWarning)
warnings.simplefilter("ignore", RuntimeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.forecast import ForecastError  # noqa: E402
from core.forecast.wave_estimator import estimate_wave  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
N_OBS = (5, 8, 10, 15, 20)
WINDOW_H = 6.0
KNOTS_H = (0.25, 0.5, 1.0, 2.0, 3.0, 6.0)
FRAC_FLOOR = 0.05  # захист від ділення на ~0 (вибух prior)
N_PRIOR_GRID = (2, 3, 5, 8, 12, 20, 30)
TEST_SIZE = 0.32
SEED = 42


def _split(forms: pd.DataFrame) -> tuple[set, set]:
    rng = np.random.RandomState(SEED)
    counts = forms["ftype"].value_counts()
    strat = forms["ftype"].apply(lambda x: x if counts[x] >= 6 else "rare")
    ho = []
    for _, idx in forms.groupby(strat).groups.items():
        idx = list(idx)
        rng.shuffle(idx)
        ho.extend(idx[: max(1, int(len(idx) * TEST_SIZE))])
    ho_ids = set(forms.loc[ho, "form_id"])
    return set(forms["form_id"]) - ho_ids, ho_ids


def _wave_records(ann: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Один рядок на (хвиля × n_obs): fit-point, span, truth, maturation-внески."""
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
            wave_end = min(nxt, wt + pd.Timedelta(hours=WINDOW_H))
            in_wave = grp[(grp >= wt) & (grp < wave_end)].reset_index(drop=True)
            truth = len(in_wave)
            if truth < 8:
                continue
            # maturation: fraction reached at each knot (for calibration)
            mat = {
                f"k{kh}": int((in_wave < wt + pd.Timedelta(hours=kh)).sum()) / truth
                for kh in KNOTS_H
            }
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
                        **mat,
                    }
                )
    return pd.DataFrame(rows)


def _maturation(train: pd.DataFrame) -> dict:
    """Per-type (+global) монотонна maturation-крива на knots."""
    out = {}
    knot_cols = [f"k{kh}" for kh in KNOTS_H]
    for key, g in [("__global__", train), *train.groupby("ftype")]:
        if len(g) < 8:
            continue
        med = [float(g[c].median()) for c in knot_cols]
        med = list(np.maximum.accumulate(med))  # монотонність
        out[key] = med
    return out


def _prior_size(mat: dict, ftype: str, span_h: float, n: int) -> float:
    knots = np.array(KNOTS_H)
    curve = mat.get(ftype, mat["__global__"])
    frac = float(np.interp(span_h, knots, curve, left=curve[0], right=curve[-1]))
    return n / max(frac, FRAC_FLOOR)


def _blend(fit: float, prior: float, n: int, n_prior: float) -> float:
    w = n_prior / (n_prior + n)
    return w * prior + (1.0 - w) * fit


def _mape(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.median(np.abs(pred - truth) / np.maximum(truth, 1)) * 100)


def main():
    ann = pd.read_csv(REPO / "data" / "wave_annotations.csv")
    df = pd.read_csv(REPO / "data" / "Form Timestamp Collection.csv")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    ann = ann[ann["waves_iso"].notna() & (ann["waves_iso"] != "")].copy()
    ann["ftype"] = ann["form_type"].fillna("unknown")

    tr_ids, ho_ids = _split(ann[["form_id", "ftype"]].drop_duplicates())
    recs = _wave_records(ann, df)
    tr = recs[recs["fid"].isin(tr_ids)].reset_index(drop=True)
    ho = recs[recs["fid"].isin(ho_ids)].reset_index(drop=True)
    print(f"records: train={len(tr)} holdout={len(ho)}")

    mat = _maturation(tr)

    # Calibrate n_prior on TRAIN (min blend MAPE).
    tr_prior = np.array([_prior_size(mat, r.ftype, r.span_h, r.n) for r in tr.itertuples()])
    best_np, best_mape = N_PRIOR_GRID[0], 1e9
    for npri in N_PRIOR_GRID:
        blend = np.array(
            [_blend(f, p, n, npri) for f, p, n in zip(tr["fit"], tr_prior, tr["n"], strict=False)]
        )
        m = _mape(blend, tr["truth"].to_numpy())
        if m < best_mape:
            best_mape, best_np = m, npri
    print(f"calibrated n_prior={best_np} (train blend MAPE={best_mape:.1f}%)")
    print(f"maturation types: {sorted(k for k in mat if k != '__global__')}")
    print(f"global maturation @knots {KNOTS_H}: {[round(x, 2) for x in mat['__global__']]}\n")

    # Evaluate on HOLDOUT.
    ho_prior = np.array([_prior_size(mat, r.ftype, r.span_h, r.n) for r in ho.itertuples()])
    ho_blend = np.array(
        [_blend(f, p, n, best_np) for f, p, n in zip(ho["fit"], ho_prior, ho["n"], strict=False)]
    )
    truth = ho["truth"].to_numpy()
    fit_ape = np.abs(ho["fit"].to_numpy() - truth) / np.maximum(truth, 1)
    blend_ape = np.abs(ho_blend - truth) / np.maximum(truth, 1)

    print("=== HOLDOUT MAPE by n_obs (A fit vs B blend) ===")
    print(f"{'n':>5}{'fit':>8}{'blend':>8}{'n':>7}")
    for n in N_OBS:
        mask = ho["n"].to_numpy() == n
        if mask.sum() == 0:
            continue
        print(
            f"{n:>5}{np.median(fit_ape[mask]) * 100:>7.0f}%{np.median(blend_ape[mask]) * 100:>7.0f}%{mask.sum():>7}"
        )

    print("\n=== HOLDOUT MAPE by form_type ===")
    for ft in sorted(ho["ftype"].unique()):
        mask = ho["ftype"].to_numpy() == ft
        if mask.sum() < 8:
            continue
        print(
            f"  {ft:<22} fit {np.median(fit_ape[mask]) * 100:>3.0f}%  blend {np.median(blend_ape[mask]) * 100:>3.0f}%  (n={mask.sum()})"
        )

    stat, p = wilcoxon(fit_ape, blend_ape)
    winner = "blend" if np.median(blend_ape) < np.median(fit_ape) else "fit"
    print(
        f"\nGLOBAL: fit={np.median(fit_ape) * 100:.1f}%  blend={np.median(blend_ape) * 100:.1f}%  "
        f"Wilcoxon p={p:.4f} {'**SIG**' if p < 0.05 else 'ns'} winner={winner}"
    )


if __name__ == "__main__":
    main()
