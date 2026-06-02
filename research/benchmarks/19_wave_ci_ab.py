"""19_wave_ci_ab.py — A/B of CI methods for the wave estimator.

P6 показав: точковий wave-прогноз (detected, no oracle) б'є prod/naive
значущо. АЛЕ CI зламаний: delta-raw дає 80% coverage, а глобальний
conformal на delta-half (q=6) дає 96% ціною width=5× точки і нерівного
покриття (86% на medium-формах). Це band-aid + порушення cap≤point.

Питання: який nonconformity score дає чесний 95% coverage без роздуття?

Невизначеність — у ПРИРОСТІ (майбутні відповіді); floor current_cum
відомий точно. Кандидати (split-conformal, Vovk 2005):

  M0 delta_raw     — pred ± t·se (baseline, без калібрування)
  M1 conf_delta    — pred ± q·delta_half, q = quantile_0.95(|err|/delta_half)
  M2 ratio         — current_cum + pred_inc·[q_lo, q_hi],
                     q = quantile(truth_inc/pred_inc)   (мультиплікативний)
  M3 poisson       — pred ± q·√pred, q = quantile((truth−pred)/√pred)
                     (variance≈mean для рахункових даних)
  M4 relative      — pred·(1 + [q_lo, q_hi]),
                     q = quantile((truth−pred)/pred)

Калібрування квантилів — ТІЛЬКИ train. Оцінка — holdout.
Переможець: max coverage-adequate (≥90%) з min Winkler і контролем
cap-violation (частка точок де півширина > точки — правило користувача).

Квантильні рівні (2.5/97.5 для 95%) — не магічні: це визначення інтервалу.

Запуск:
    .venv/Scripts/python.exe research/benchmarks/19_wave_ci_ab.py
"""

from __future__ import annotations

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

CAP_MAX_RELATIVE = 2.0  # prod policy: half-width <= 2x point (DELTA_CI_MAX_RELATIVE)
CAP_MIN_ABSOLUTE = 10.0

REPO = Path(__file__).resolve().parents[2]
N_TRAIN_VALUES = (5, 8, 10, 15, 20)
HORIZON_HOURS = (0.5, 1.0, 2.0, 6.0)
CONFIDENCE = 0.95
ALPHA = 1.0 - CONFIDENCE
LO_Q = ALPHA / 2.0  # 0.025
HI_Q = 1.0 - ALPHA / 2.0  # 0.975
TEST_SIZE = 0.32
SEED = 42


def _winkler(truth: float, lo: float, hi: float) -> float:
    width = max(0.0, hi - lo)
    return width + (2.0 / ALPHA) * max(lo - truth, 0.0) + (2.0 / ALPHA) * max(truth - hi, 0.0)


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


def _wave_points(ids: set, valid: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Compute wave (detected, no oracle) predictions over the cutoff grid."""
    rows = []
    for _, row in valid[valid["form_id"].isin(ids)].iterrows():
        fid = row["form_id"]
        ftype = row["form_type"]
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
            pre_count = int((observed < ws).sum())
            cur = observed[observed >= ws].reset_index(drop=True)
            if len(cur) < 5:
                continue
            for h in HORIZON_HOURS:
                horizon_end = t_c + pd.Timedelta(hours=h)
                horizon_from_ws = (horizon_end - ws).total_seconds() / 3600.0
                if horizon_from_ws <= 0:
                    continue
                truth_cum = int((ts_all <= horizon_end).sum())
                try:
                    wf = estimate_wave(cur.tolist(), horizon_h=horizon_from_ws, form_type=ftype)
                except ForecastError:
                    continue
                pred = pre_count + wf.point
                rows.append(
                    {
                        "ftype": ftype,
                        "n_train": n_train,
                        "horizon_h": h,
                        "current_cum": n_train,
                        "pred": pred,
                        "pred_inc": max(pred - n_train, 0),
                        "truth": truth_cum,
                        "truth_inc": truth_cum - n_train,
                        "delta_half": max((wf.ci_upper - wf.ci_lower) / 2.0, 1.0),
                        "delta_lo": pre_count + wf.ci_lower,
                        "delta_hi": pre_count + wf.ci_upper,
                    }
                )
    return pd.DataFrame(rows)


# ---------- CI methods: calibrate on train, return interval fn ---------------


def calibrate_methods(tr: pd.DataFrame) -> dict:
    """Return {name: callable(row) -> (lo, hi)} calibrated on train."""
    methods = {}

    # M0: delta raw (no calibration)
    methods["M0 delta_raw"] = lambda r: (r["delta_lo"], r["delta_hi"])

    # M1: conformal on delta half-width
    nr = ((tr["pred"] - tr["truth"]).abs() / tr["delta_half"]).to_numpy()
    q1 = float(np.quantile(nr, CONFIDENCE))
    methods["M1 conf_delta"] = lambda r, q=q1: (
        r["pred"] - q * r["delta_half"],
        r["pred"] + q * r["delta_half"],
    )

    # M2: multiplicative ratio on increment
    ratio = tr["truth_inc"].to_numpy() / np.maximum(tr["pred_inc"].to_numpy(), 1.0)
    r_lo, r_hi = float(np.quantile(ratio, LO_Q)), float(np.quantile(ratio, HI_Q))
    methods["M2 ratio"] = lambda r, lo=r_lo, hi=r_hi: (
        r["current_cum"] + r["pred_inc"] * lo,
        r["current_cum"] + r["pred_inc"] * hi,
    )

    # M3: Poisson sqrt scaling on total
    z = (tr["truth"].to_numpy() - tr["pred"].to_numpy()) / np.sqrt(
        np.maximum(tr["pred"].to_numpy(), 1.0)
    )
    z_lo, z_hi = float(np.quantile(z, LO_Q)), float(np.quantile(z, HI_Q))
    methods["M3 poisson"] = lambda r, lo=z_lo, hi=z_hi: (
        r["pred"] + lo * np.sqrt(max(r["pred"], 1.0)),
        r["pred"] + hi * np.sqrt(max(r["pred"], 1.0)),
    )

    # M4: relative error on total
    e = (tr["truth"].to_numpy() - tr["pred"].to_numpy()) / np.maximum(tr["pred"].to_numpy(), 1.0)
    e_lo, e_hi = float(np.quantile(e, LO_Q)), float(np.quantile(e, HI_Q))
    methods["M4 relative"] = lambda r, lo=e_lo, hi=e_hi: (
        r["pred"] * (1.0 + lo),
        r["pred"] * (1.0 + hi),
    )

    # --- Mondrian (group-conditional) conformal: quantile per pred-bucket ---
    # Bucket edges = train terciles of pred (data-driven, known at inference).
    edges = [float(np.quantile(tr["pred"], q)) for q in (1 / 3, 2 / 3)]

    def bucket(pred: float) -> int:
        return 0 if pred < edges[0] else (1 if pred < edges[1] else 2)

    tr_b = tr.assign(b=tr["pred"].apply(bucket))

    # M5: Mondrian on relative error (two-sided, per bucket).
    rel_q = {}
    for b, g in tr_b.groupby("b"):
        ee = (g["truth"].to_numpy() - g["pred"].to_numpy()) / np.maximum(g["pred"].to_numpy(), 1.0)
        rel_q[b] = (float(np.quantile(ee, LO_Q)), float(np.quantile(ee, HI_Q)))

    def m5(r, rel_q=rel_q):
        lo, hi = rel_q[bucket(r["pred"])]
        return r["pred"] * (1.0 + lo), r["pred"] * (1.0 + hi)

    methods["M5 mondrian_rel"] = m5

    # M6: Mondrian on conf_delta (|err|/delta_half, per bucket).
    delta_q = {}
    for b, g in tr_b.groupby("b"):
        nrb = (g["pred"] - g["truth"]).abs() / g["delta_half"]
        delta_q[b] = float(np.quantile(nrb, CONFIDENCE))

    def m6(r, delta_q=delta_q):
        q = delta_q[bucket(r["pred"])]
        return r["pred"] - q * r["delta_half"], r["pred"] + q * r["delta_half"]

    methods["M6 mondrian_delta"] = m6

    # M7: M6 calibration THEN hard cap_width (prod policy). Calibration where
    # it helps, cap where it would explode — guarantees no absurd interval.
    def m7(r, delta_q=delta_q):
        q = delta_q[bucket(r["pred"])]
        lo = r["pred"] - q * r["delta_half"]
        hi = r["pred"] + q * r["delta_half"]
        clo, chi = cap_width(
            np.array([float(r["pred"])]),
            np.array([lo]),
            np.array([hi]),
            max_relative=CAP_MAX_RELATIVE,
            min_absolute=CAP_MIN_ABSOLUTE,
        )
        return float(clo[0]), float(chi[0])

    methods["M7 mondrian_capped"] = m7

    params = {
        "M1 conf_delta": f"q={q1:.2f}",
        "M2 ratio": f"[{r_lo:.2f}, {r_hi:.2f}]",
        "M3 poisson": f"z=[{z_lo:.2f}, {z_hi:.2f}]",
        "M4 relative": f"[{e_lo:.2f}, {e_hi:.2f}]",
        "M5 mondrian_rel": f"edges={edges[0]:.0f}/{edges[1]:.0f} q={ {k: (round(v[0], 1), round(v[1], 1)) for k, v in rel_q.items()} }",
        "M6 mondrian_delta": f"edges={edges[0]:.0f}/{edges[1]:.0f} q={ {k: round(v, 1) for k, v in delta_q.items()} }",
        "M7 mondrian_capped": f"M6 + cap_width(max_rel={CAP_MAX_RELATIVE})",
    }
    return methods, params


def evaluate(methods: dict, ho: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, fn in methods.items():
        winks, relws, abswidths, caps1, caps2, hits = [], [], [], [], [], []
        for _, r in ho.iterrows():
            lo, hi = fn(r)
            # Monotonic floor: cumulative cannot drop below current_cum.
            lo = max(lo, r["current_cum"])
            hi = max(hi, r["pred"])
            hits.append(lo <= r["truth"] <= hi)
            winks.append(_winkler(r["truth"], lo, hi))
            relws.append((hi - lo) / max(r["truth"], 1.0))
            abswidths.append(hi - lo)
            half = (hi - lo) / 2.0
            caps1.append(half > r["pred"])  # half > point
            caps2.append(half > 2.0 * r["pred"])  # > prod policy DELTA_CI_MAX_RELATIVE
        out.append(
            {
                "method": name,
                "coverage": np.mean(hits) * 100,
                "winkler_p50": np.median(winks),
                "relwidth_p50": np.median(relws),
                "abswidth_p90": np.quantile(abswidths, 0.9),
                "abswidth_max": np.max(abswidths),
                "cap_gt1x": np.mean(caps1) * 100,
                "cap_gt2x": np.mean(caps2) * 100,
            }
        )
    return pd.DataFrame(out)


def main():
    ann = pd.read_csv(REPO / "data" / "wave_annotations.csv")
    df = pd.read_csv(REPO / "data" / "Form Timestamp Collection.csv")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    valid = ann[ann["waves_iso"].notna() & (ann["waves_iso"] != "")].copy()
    valid["ftype"] = valid["form_type"].fillna("unknown")

    train_ids, ho_ids = _stratified_split(valid[["form_id", "ftype"]])
    print(f"Train {len(train_ids)} forms / Holdout {len(ho_ids)} forms")
    tr = _wave_points(train_ids, valid, df)
    ho = _wave_points(ho_ids, valid, df)
    print(f"Train points {len(tr)} / Holdout points {len(ho)}")
    print()

    methods, params = calibrate_methods(tr)
    res = evaluate(methods, ho)

    lines = []
    lines.append("=== CI METHOD A/B (holdout, target coverage 95%) ===")
    lines.append(
        f"{'method':<18}{'coverage':>9}{'Winkler':>9}{'relW':>7}{'absW90':>8}{'absWmax':>9}{'>1x':>6}{'>2x':>6}"
    )
    for _, r in res.iterrows():
        lines.append(
            f"{r['method']:<18}{r['coverage']:>8.1f}%{r['winkler_p50']:>9.1f}"
            f"{r['relwidth_p50']:>7.2f}{r['abswidth_p90']:>8.0f}{r['abswidth_max']:>9.0f}"
            f"{r['cap_gt1x']:>5.0f}%{r['cap_gt2x']:>5.0f}%"
        )
    lines.append("")
    lines.append("params: " + "; ".join(f"{k}:{v}" for k, v in params.items()))
    lines.append("")

    # Winner: coverage >= 90 AND respects prod cap policy (cap_gt2x low),
    # then minimal Winkler.
    adequate = res[(res["coverage"] >= 90.0) & (res["cap_gt2x"] <= 10.0)]
    if len(adequate):
        win = adequate.sort_values("winkler_p50").iloc[0]
        lines.append(
            f"WINNER (cov>=90, cap_gt2x<=10%, min Winkler): {win['method']}  "
            f"cov={win['coverage']:.0f}%  Winkler={win['winkler_p50']:.1f}  "
            f"relW={win['relwidth_p50']:.2f}  cap>2x={win['cap_gt2x']:.0f}%"
        )
    else:
        cov_ok = res[res["coverage"] >= 90.0]
        if len(cov_ok):
            win = cov_ok.sort_values("winkler_p50").iloc[0]
            lines.append(
                f"None passed cap policy. Best covered by Winkler: {win['method']} "
                f"cov={win['coverage']:.0f}% Winkler={win['winkler_p50']:.1f} cap>2x={win['cap_gt2x']:.0f}%"
            )
        else:
            best = res.sort_values("coverage", ascending=False).iloc[0]
            lines.append(f"NO method reached 90%. Best: {best['method']} {best['coverage']:.0f}%")

    # Per-size breakdown of winner vs baselines
    lines.append("")
    lines.append("=== Coverage by truth-size bucket (key methods) ===")
    buckets = [(0, 30, "tiny<30"), (30, 100, "small"), (100, 500, "med"), (500, 1e12, "big500+")]
    key = ["M1 conf_delta", "M4 relative", "M5 mondrian_rel", "M6 mondrian_delta"]
    hdr = f"{'bucket':<10}{'n':>5}" + "".join(f"{k.split()[0]:>9}" for k in key)
    lines.append(hdr)
    for lo, hi, lab in buckets:
        sub = ho[(ho["truth"] >= lo) & (ho["truth"] < hi)]
        if not len(sub):
            continue
        cells = []
        for k in key:
            fn = methods[k]
            hits = []
            for _, r in sub.iterrows():
                a, b = fn(r)
                a = max(a, r["current_cum"])
                b = max(b, r["pred"])
                hits.append(a <= r["truth"] <= b)
            cells.append(f"{np.mean(hits) * 100:>8.0f}%")
        lines.append(f"{lab:<10}{len(sub):>5}" + "".join(cells))

    # --- Anchor cases: the two forms from the user's angry screenshots ---
    # Reproduce early-cutoff wave forecast + each CI to check absolute sanity.
    lines.append("")
    lines.append("=== ANCHOR CASES (user screenshots): interval sanity ===")
    anchors = [
        ("47-form (1p0ERtAe)", "1p0ERtAe-_c4J_EL0H-f3Ykbbc6GBbWmqY4-SX1r9I3Y"),
        ("7433-form (1GM-api8tg)", "1GM-api8tg1DaVEE_NJ203b9K01TA4_uCW_3c2gkkh0c"),
    ]
    for label, fid in anchors:
        ts_all = df[df["FORM_ID"] == fid]["TIMESTAMP"].sort_values().reset_index(drop=True)
        if len(ts_all) < 16:
            lines.append(f"  {label}: not enough data")
            continue
        ftype = valid[valid["form_id"] == fid]["form_type"]
        ftype = ftype.iloc[0] if len(ftype) else "unknown"
        observed = ts_all.iloc[:15].reset_index(drop=True)
        t_c = observed.iloc[-1]
        waves = detect_wave_starts(observed, form_type=ftype, test_skip=0)
        ws = waves[-1].timestamp if waves else observed.iloc[0]
        pre = int((observed < ws).sum())
        cur = observed[observed >= ws].reset_index(drop=True)
        if len(cur) < 5:
            lines.append(f"  {label}: wave<5")
            continue
        horizon_end = t_c + pd.Timedelta(hours=6.0)
        truth = int((ts_all <= horizon_end).sum())
        final = len(ts_all)
        try:
            wf = estimate_wave(
                cur.tolist(),
                horizon_h=(horizon_end - ws).total_seconds() / 3600.0,
                form_type=ftype,
            )
        except ForecastError:
            lines.append(f"  {label}: fit failed")
            continue
        pred = pre + wf.point
        row = {
            "ftype": ftype,
            "pred": pred,
            "pred_inc": max(pred - 15, 0),
            "current_cum": 15,
            "delta_half": max((wf.ci_upper - wf.ci_lower) / 2.0, 1.0),
            "delta_lo": pre + wf.ci_lower,
            "delta_hi": pre + wf.ci_upper,
            "truth": truth,
        }
        lines.append(f"  {label}: pred@6h={pred} truth@6h={truth} final={final}")
        for name in ["M0 delta_raw", "M6 mondrian_delta", "M7 mondrian_capped"]:
            lo, hi = methods[name](row)
            lo = max(lo, 15)
            hi = max(hi, pred)
            lines.append(f"      {name:<18} CI=[{lo:.0f}, {hi:.0f}]  half={(hi - lo) / 2:.0f}")

    report = "\n".join(lines)
    print(report)
    (REPO / "research" / "reports" / "19_wave_ci_ab.md").write_text(
        f"# 19 — Wave CI method A/B\n\n```\n{report}\n```\n", encoding="utf-8"
    )
    print("\nSaved: research/reports/19_wave_ci_ab.md")


if __name__ == "__main__":
    main()
