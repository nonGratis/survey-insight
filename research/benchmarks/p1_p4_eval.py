"""P1 + P4: Clean train/test split evaluation + wave-level conformal CI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd


# sklearn not installed — manual stratified split
class StratifiedShuffleSplit:
    def __init__(self, n_splits=1, test_size=0.3, random_state=42):
        self.test_size = test_size
        self.rs = random_state

    def split(self, x, y):
        rng = np.random.RandomState(self.rs)
        y = np.array(y)
        classes = np.unique(y)
        test_idx, train_idx = [], []
        for c in classes:
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            n_test = max(1, int(len(idx) * self.test_size))
            test_idx.extend(idx[:n_test].tolist())
            train_idx.extend(idx[n_test:].tolist())
        yield np.array(train_idx), np.array(test_idx)


from core.forecast import forecast_responses  # noqa: E402
from core.forecast.wave_estimator import estimate_wave  # noqa: E402
from core.timeline import build_timeline_from_timestamps  # noqa: E402

np.random.seed(42)
HORIZON_H = 2.0

ann = pd.read_csv("data/wave_annotations.csv")
df = pd.read_csv("data/Form Timestamp Collection.csv")
df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

valid = ann[ann["waves_iso"].notna() & (ann["waves_iso"] != "")].copy()
valid["ftype"] = valid["form_type"].fillna("unknown")
counts = valid["ftype"].value_counts()
valid["ftype_strat"] = valid["ftype"].apply(lambda x: x if counts[x] >= 6 else "rare")

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.32, random_state=42)
idx_tr, idx_ho = next(sss.split(valid, valid["ftype_strat"]))
train_forms = set(valid.iloc[idx_tr]["form_id"])
holdout_forms = set(valid.iloc[idx_ho]["form_id"])
print(f"Train: {len(train_forms)} forms, Holdout: {len(holdout_forms)} forms")

train_ann = valid[valid["form_id"].isin(train_forms)]
ho_ann = valid[valid["form_id"].isin(holdout_forms)]


def within_wave_ts(row, df_ts, wt, n_max=20):
    fid = row["form_id"]
    test_skip = int(row["test_resp"] or 0)
    grp = (
        df_ts[df_ts["FORM_ID"] == fid]["TIMESTAMP"]
        .sort_values()
        .iloc[test_skip:]
        .reset_index(drop=True)
    )
    in_w = grp[(grp >= wt) & (grp < wt + pd.Timedelta(hours=3))].iloc[:n_max]
    return grp, in_w


def eval_row(row, df_ts, train_or_holdout="holdout"):
    recs = []
    waves = [pd.Timestamp(w) for w in str(row["waves_iso"]).split("|")]
    ftype = row["form_type"]
    grp_all, _ = within_wave_ts(row, df_ts, waves[0])
    if len(grp_all) < 15:
        return recs
    for wt in waves:
        horizon_end = wt + pd.Timedelta(hours=HORIZON_H)
        truth = int(grp_all[(grp_all >= wt) & (grp_all < horizon_end)].shape[0])
        if truth < 2:
            continue
        _, in_w = within_wave_ts(row, df_ts, wt)
        if len(in_w) < 5:
            continue
        train_span_h = (in_w.iloc[-1] - wt).total_seconds() / 3600

        # Wave estimator
        ape_w = hit_w = wid_w = wf_point = wf_half = None
        try:
            wf = estimate_wave(in_w.tolist(), horizon_h=HORIZON_H, form_type=ftype)
            ape_w = abs(wf.point - truth) / truth
            hit_w = wf.ci_lower <= truth <= wf.ci_upper
            wid_w = wf.ci_upper - wf.ci_lower
            wf_point = wf.point
            wf_half = max((wf.ci_upper - wf.ci_lower) / 2.0, 1.0)
        except Exception:
            pass

        # Naive baseline: rate × horizon
        rate_h = len(in_w) / max(train_span_h, 0.01)
        naive_pt = int(round(rate_h * HORIZON_H))
        ape_naive = abs(naive_pt - truth) / truth

        # Prod (P17)
        ape_p = hit_p = wid_p = None
        try:
            tl = build_timeline_from_timestamps([t.to_pydatetime() for t in in_w])
            fc = forecast_responses(tl, horizon_until=pd.Timestamp(horizon_end))
            idx_h = min(
                range(len(fc.future_dates)),
                key=lambda i: abs((fc.future_dates[i] - horizon_end).total_seconds()),
            )
            prod_pt = int(round(float(fc.future_cum.iloc[idx_h])))
            prod_lo = int(round(float(fc.ci_lower.iloc[idx_h])))
            prod_hi = int(round(float(fc.ci_upper.iloc[idx_h])))
            ape_p = abs(prod_pt - truth) / truth
            hit_p = prod_lo <= truth <= prod_hi
            wid_p = prod_hi - prod_lo
        except Exception:
            pass

        recs.append(
            {
                "ftype": ftype,
                "truth": truth,
                "ape_wave": ape_w,
                "hit_wave": hit_w,
                "wid_wave": wid_w,
                "wf_point": wf_point,
                "wf_half": wf_half,
                "ape_naive": ape_naive,
                "ape_prod": ape_p,
                "hit_prod": hit_p,
                "wid_prod": wid_p,
            }
        )
    return recs


# --- P4: Train conformal quantiles -------------------------------------------
print("Computing train conformal quantiles...")
train_conformal = []
for _, row in train_ann.iterrows():
    recs = eval_row(row, df)
    for rec in recs:
        if rec["wf_point"] is None or rec["wf_half"] is None:
            continue
        norm_r = abs(rec["wf_point"] - rec["truth"]) / rec["wf_half"]
        train_conformal.append({"ftype": rec["ftype"], "norm_resid": norm_r})
TC = pd.DataFrame(train_conformal)
q_global = float(TC["norm_resid"].quantile(0.95))
q_by_type = TC.groupby("ftype")["norm_resid"].quantile(0.95).to_dict()
print(f"Train conformal q_0.95 global: {q_global:.2f}")
print()

# --- P1: Evaluate on HOLDOUT -------------------------------------------------
print("Evaluating on holdout...")
ho_recs = []
for _, row in ho_ann.iterrows():
    ho_recs.extend(eval_row(row, df))

R = pd.DataFrame(ho_recs).dropna(subset=["ape_wave", "ape_prod", "ape_naive"])
print(f"Holdout wave-cutoffs: {len(R)}")
print()

print(f"{'':25} {'Wave':>12}  {'Prod (P17)':>12}  {'Naive r*h':>12}")
print(
    f"{'MAPE_p50':25} {R.ape_wave.median() * 100:>11.1f}%  {R.ape_prod.median() * 100:>11.1f}%  {R.ape_naive.median() * 100:>11.1f}%"
)
print(f"{'Coverage (raw)':25} {R.hit_wave.mean() * 100:>11.1f}%  {R.hit_prod.mean() * 100:>11.1f}%")
print(f"{'CI width p50':25} {R.wid_wave.median():>12.0f}  {R.wid_prod.median():>12.0f}")
print()

# Apply wave conformal on holdout
hits_conf = 0
n_eval = 0
for _, rec in R.iterrows():
    if rec["wf_half"] is None:
        continue
    q = q_by_type.get(rec["ftype"], q_global)
    new_lo = rec["wf_point"] - q * rec["wf_half"]
    new_hi = rec["wf_point"] + q * rec["wf_half"]
    hits_conf += new_lo <= rec["truth"] <= new_hi
    n_eval += 1

print(
    f"{'Coverage wave+conformal (P4)':25} {hits_conf / max(n_eval, 1) * 100:>11.1f}%  (target 95%)"
)
print()

print("=== Per form_type (MAPE / Coverage) ===")
for ft, g in R.groupby("ftype"):
    if len(g) < 5:
        continue
    q = q_by_type.get(ft, q_global)
    hits_c = sum(
        (r["wf_point"] - q * r["wf_half"] <= r["truth"] <= r["wf_point"] + q * r["wf_half"])
        for _, r in g.iterrows()
        if r["wf_half"] is not None
    )
    print(
        f"  {ft:<22}: wave {g.ape_wave.median() * 100:.0f}%  naive {g.ape_naive.median() * 100:.0f}%  "
        f"prod {g.ape_prod.median() * 100:.0f}%  cov_conf {hits_c / len(g) * 100:.0f}%  (n={len(g)})"
    )
