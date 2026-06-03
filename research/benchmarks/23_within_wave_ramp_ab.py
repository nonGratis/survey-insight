"""23_within_wave_ramp_ab.py — landing-only ramp blend, END-TO-END (no oracle).

Попередній бленд (22_) рівномірно масштабував криву → переоцінка коротких
горизонтів → регрес end-to-end (14.9%→18.2%). Тут — landing-only ramp:
масштаб росте від 1 (останній факт) до blend/fit на НАДІЙНОМУ горизонті
(3×span), не на горизонті запиту. Короткі горизонти ≈ fit, посадка ≈ blend.

Захищувана метрика: end-to-end no-oracle (детектовані хвилі, horizon-from-now),
як у multiseed/18_. Калібрування maturation+n_prior на TRAIN (oracle waves),
оцінка на HOLDOUT (detected). A=pure fit, B=ramp-blend. Якщо B<=A скрізь
(без регресу коротких, виграш довгих) → ставимо в core.

Запуск:
    .venv/Scripts/python.exe research/benchmarks/23_within_wave_ramp_ab.py
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
from core.forecast.models import models_for_n_points  # noqa: E402
from core.forecast.selector import select_best_model  # noqa: E402
from core.forecast.service import HORIZON_SPAN_FACTOR  # noqa: E402
from core.forecast.wave_detector import detect_wave_starts  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
N_TRAIN = (5, 8, 10, 15, 20)
HORIZON_HOURS = (0.5, 1.0, 2.0, 6.0)
WINDOW_H = 6.0
KNOTS_H = (0.25, 0.5, 1.0, 2.0, 3.0, 6.0)
FRAC_FLOOR = 0.05
N_PRIOR_GRID = (2, 3, 5, 8, 12, 20, 30)
TEST_SIZE = 0.32
SEED = 42


def _fit_cum(t_train, n, t_eval_h):
    """Fit within-wave model, return model+params for predict; cum at scalar h."""
    y = np.arange(1, n + 1, dtype=float)
    fitted = select_best_model(t_train, y, target=None, models=models_for_n_points(n))

    def cum(h):
        v = float(fitted.model.predict(np.array([h]), *fitted.params)[0])
        return max(v, float(n))

    return cum


def _maturation(train_rows):
    out = {}
    df = pd.DataFrame(train_rows)
    for key, g in [("__global__", df), *df.groupby("ftype")]:
        if len(g) < 8:
            continue
        med = [float(np.median([m[kh] for m in g["mat"]])) for kh in KNOTS_H]
        out[key] = list(np.maximum.accumulate(med))
    return out


def _prior(mat, ftype, span_h, n):
    curve = mat.get(ftype, mat["__global__"])
    frac = float(np.interp(span_h, KNOTS_H, curve, left=curve[0], right=curve[-1]))
    return n / max(frac, FRAC_FLOOR)


def _collect(ann, df, ids):
    """Per (form × n_train × horizon) on DETECTED waves: fit-cum fn + context."""
    out = []
    for _, r in ann[ann["form_id"].isin(ids)].iterrows():
        if not r["waves_iso"] or pd.isna(r["waves_iso"]):
            continue
        ftype, fid = r["form_type"], r["form_id"]
        ts = df[df["FORM_ID"] == fid]["TIMESTAMP"].sort_values().reset_index(drop=True)
        if len(ts) < 6:
            continue
        for n_tr in N_TRAIN:
            if len(ts) <= n_tr:
                continue
            obs = ts.iloc[:n_tr].reset_index(drop=True)
            t_c = obs.iloc[-1]
            if (t_c - obs.iloc[0]).total_seconds() <= 0:
                continue
            waves = detect_wave_starts(obs, form_type=ftype, test_skip=0)
            ws = waves[-1].timestamp if waves else obs.iloc[0]
            pre = int((obs < ws).sum())
            cur = obs[obs >= ws].reset_index(drop=True)
            if len(cur) < 5:
                continue
            n = len(cur)
            span_h = (cur.iloc[-1] - ws).total_seconds() / 3600.0
            if span_h <= 0:
                continue
            t_train = np.array([(t - ws).total_seconds() / 3600.0 for t in cur])
            try:
                cum = _fit_cum(t_train, n, None)
            except ForecastError:
                continue
            for h in HORIZON_HOURS:
                he = t_c + pd.Timedelta(hours=h)
                h_ws = (he - ws).total_seconds() / 3600.0
                truth = int((ts <= he).sum())
                out.append(
                    {"ftype": ftype, "n": n, "span_h": span_h, "pre": pre,
                     "h_ws": h_ws, "horizon": h, "truth": truth, "cum": cum}
                )
    return out


def _train_rows(ann, df, ids):
    """Oracle-wave maturation records (for calibration), like 22_."""
    rows = []
    for _, r in ann[ann["form_id"].isin(ids)].iterrows():
        if not r["waves_iso"] or pd.isna(r["waves_iso"]):
            continue
        waves = [pd.Timestamp(w) for w in str(r["waves_iso"]).split("|")]
        ftype, fid = r["form_type"], r["form_id"]
        skip = int(r["test_resp"] or 0)
        grp = df[df["FORM_ID"] == fid]["TIMESTAMP"].sort_values().iloc[skip:].reset_index(drop=True)
        for i, wt in enumerate(waves):
            nxt = waves[i + 1] if i + 1 < len(waves) else wt + pd.Timedelta(hours=24)
            in_w = grp[(grp >= wt) & (grp < min(nxt, wt + pd.Timedelta(hours=WINDOW_H)))]
            in_w = in_w.reset_index(drop=True)
            truth = len(in_w)
            if truth < 8:
                continue
            mat = {kh: int((in_w < wt + pd.Timedelta(hours=kh)).sum()) / truth for kh in KNOTS_H}
            rows.append({"ftype": ftype, "mat": mat, "truth": truth})
    return rows


def main():
    ann = pd.read_csv(REPO / "data" / "wave_annotations.csv")
    df = pd.read_csv(REPO / "data" / "Form Timestamp Collection.csv")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    ann = ann[ann["waves_iso"].notna() & (ann["waves_iso"] != "")].copy()
    ann["ftype"] = ann["form_type"].fillna("unknown")

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
    tr_ids = set(uniq["form_id"]) - ho_ids

    # Calibrate maturation + n_prior on TRAIN (oracle), reuse 22_ n_prior logic
    # via landing target on train detected waves would be circular; use the
    # 22_-validated n_prior search on oracle landing.
    mat = _maturation(_train_rows(ann, df, tr_ids))

    # n_prior: grid on TRAIN end-to-end (detected) — minimise A/B target itself.
    tr_eval = _collect(ann, df, tr_ids)
    ho_eval = _collect(ann, df, ho_ids)
    print(f"train evals={len(tr_eval)} holdout evals={len(ho_eval)}")

    def predict(rec, n_prior, mode):
        cum, n, span_h, pre, h_ws = rec["cum"], rec["n"], rec["span_h"], rec["pre"], rec["h_ws"]
        fit_h = cum(h_ws)
        if mode == "fit":
            return pre + fit_h
        landing_h = max(HORIZON_SPAN_FACTOR * span_h, span_h + 1e-6)
        fit_land = cum(landing_h)
        if fit_land <= 0:
            return pre + fit_h
        prior = _prior(mat, rec["ftype"], span_h, n)
        w = n_prior / (n_prior + n)
        blend_land = w * prior + (1 - w) * fit_land
        scale_land = blend_land / fit_land
        prog = min(max((h_ws - span_h) / (landing_h - span_h), 0.0), 1.0)
        scale_h = 1.0 + (scale_land - 1.0) * prog
        return pre + max(fit_h * scale_h, float(n))

    def mape(evals, n_prior, mode):
        ape = [abs(predict(r, n_prior, mode) - r["truth"]) / max(r["truth"], 1) for r in evals]
        return float(np.median(ape) * 100)

    best_np, best_m = N_PRIOR_GRID[0], 1e9
    for npri in N_PRIOR_GRID:
        m = mape(tr_eval, npri, "blend")
        if m < best_m:
            best_m, best_np = m, npri
    print(f"calibrated n_prior={best_np} (train end-to-end blend MAPE={best_m:.1f}%)\n")

    # Holdout per-horizon A vs B.
    print(f"{'horizon':>8}{'fit':>8}{'ramp':>8}{'n':>7}")
    for h in HORIZON_HOURS:
        sub = [r for r in ho_eval if r["horizon"] == h]
        fa = [abs(predict(r, best_np, "fit") - r["truth"]) / max(r["truth"], 1) for r in sub]
        ba = [abs(predict(r, best_np, "blend") - r["truth"]) / max(r["truth"], 1) for r in sub]
        print(f"{h:>8}{np.median(fa) * 100:>7.0f}%{np.median(ba) * 100:>7.0f}%{len(sub):>7}")

    fa = np.array([abs(predict(r, best_np, "fit") - r["truth"]) / max(r["truth"], 1) for r in ho_eval])
    ba = np.array([abs(predict(r, best_np, "blend") - r["truth"]) / max(r["truth"], 1) for r in ho_eval])
    stat, p = wilcoxon(fa, ba)
    win = "ramp" if np.median(ba) < np.median(fa) else "fit"
    print(
        f"\nGLOBAL end-to-end: fit={np.median(fa) * 100:.1f}%  ramp={np.median(ba) * 100:.1f}%  "
        f"Wilcoxon p={p:.4f} {'**SIG**' if p < 0.05 else 'ns'} winner={win}"
    )


if __name__ == "__main__":
    main()
