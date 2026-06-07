"""18_wave_endtoend_ab.py — production-realistic end-to-end evaluation.

ЧЕСНЕ питання: чи wave-estimator працює В ПРОДІ, де немає ручних міток
хвиль? P1 виміряв 19% MAPE, але ітерував по АНОТОВАНИХ стартах хвиль —
оракул. Тут детектор сам шукає хвилі (CUSUM, без міток).

Протокол (horizon-from-now — реальне UI-питання "скільки буде через H год"):
  - observe перші n_train відповідей, cutoff t_c = ts[n_train-1]
  - truth = cumulative count at t_c + horizon_h (ВСІ відповіді форми)
  - кожен метод прогнозує цей самий target (cumulative total)

Методи:
  A · prod          — forecast_responses (P17 delta+cap)
  B · wave_detected — CUSUM (БЕЗ оракула, test_skip=0) → current wave →
                      estimate_wave → total = pre_count + within_wave_pred
  C · wave_oracle   — анотований старт хвилі + оракул test_skip (верхня межа)
  D · naive_recent  — n_train + (rate за останню 1h) × horizon
  E · naive_overall — n_train + (n_train / span) × horizon

Calibration (priors + conformal q) — ТІЛЬКИ на train split.
Evaluation — ТІЛЬКИ на holdout.

Detection penalty = MAPE(B) − MAPE(C): скільки коштує відсутність оракула.

Метрики: cum MAPE, increment sMAPE, coverage@95, Winkler, CI width, bias.
Significance: Wilcoxon signed-rank (paired APE) — B vs D, B vs A, C vs B.

Запуск:
    .venv/Scripts/python.exe research/benchmarks/18_wave_endtoend_ab.py
    .venv/Scripts/python.exe research/benchmarks/18_wave_endtoend_ab.py --limit 10
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning
from scipy.stats import wilcoxon

warnings.simplefilter("ignore", OptimizeWarning)
warnings.simplefilter("ignore", RuntimeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.forecast import ForecastError, forecast_responses  # noqa: E402
from core.forecast.wave_detector import detect_wave_starts  # noqa: E402
from core.forecast.wave_estimator import estimate_wave  # noqa: E402
from core.timeline import build_timeline_from_timestamps  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
N_TRAIN_VALUES = (5, 8, 10, 15, 20)
HORIZON_HOURS = (0.5, 1.0, 2.0, 6.0)
CONFIDENCE = 0.95
ALPHA = 1.0 - CONFIDENCE
TEST_SIZE = 0.32
SEED = 42


# ---------- helpers ---------------------------------------------------------


def _winkler(truth: float, lo: float, hi: float, alpha: float = ALPHA) -> float:
    width = max(0.0, hi - lo)
    return width + (2.0 / alpha) * max(lo - truth, 0.0) + (2.0 / alpha) * max(truth - hi, 0.0)


def _smape(pred: float, truth: float) -> float:
    return abs(pred - truth) / ((abs(pred) + abs(truth)) / 2.0 + 1.0)


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
    return (
        set(forms.loc[train, "form_id"]),
        set(forms.loc[holdout, "form_id"]),
    )


@dataclass
class Point:
    form_id: str
    ftype: str
    n_train: int
    horizon_h: float
    truth_cum: int
    truth_inc: int
    # per-method cumulative point predictions (-1 = failed)
    a_prod: int
    b_detected: int
    c_oracle: int
    d_naive_recent: int
    e_naive_overall: int
    # CI (lo, hi) for A, B, C
    a_lo: int
    a_hi: int
    b_lo: int
    b_hi: int
    c_lo: int
    c_hi: int
    # wave half-width (for conformal) — B and C
    b_half: float
    c_half: float


# ---------- per-method forecasting ------------------------------------------


def _wave_total(
    observed_all: pd.Series,
    ws: pd.Timestamp,
    n_train: int,
    horizon_end: pd.Timestamp,
    ftype: str,
) -> tuple[int, int, int, float] | None:
    """Return (point, lo, hi, half) total form count via within-wave fit.

    ws — wave start. pre_count + within-wave prediction = total.
    """
    pre_count = int((observed_all < ws).sum())
    current_wave_ts = observed_all[observed_all >= ws].reset_index(drop=True)
    if len(current_wave_ts) < 5:
        return None
    horizon_from_ws = (horizon_end - ws).total_seconds() / 3600.0
    if horizon_from_ws <= 0:
        return None
    try:
        wf = estimate_wave(current_wave_ts.tolist(), horizon_h=horizon_from_ws, form_type=ftype)
    except ForecastError:
        return None
    point = pre_count + wf.point
    lo = pre_count + wf.ci_lower
    hi = pre_count + wf.ci_upper
    half = max((wf.ci_upper - wf.ci_lower) / 2.0, 1.0)
    return int(point), int(lo), int(hi), float(half)


def _backtest_point(
    ts_all: pd.Series,
    n_train: int,
    horizon_h: float,
    ftype: str,
    oracle_starts: list[pd.Timestamp],
    oracle_test_skip: int,
) -> Point | None:
    n_total = len(ts_all)
    if n_total <= n_train:
        return None
    observed = ts_all.iloc[:n_train].reset_index(drop=True)
    t_c = observed.iloc[-1]
    span_h = (t_c - observed.iloc[0]).total_seconds() / 3600.0
    if span_h <= 0:
        return None
    horizon_end = t_c + pd.Timedelta(hours=horizon_h)
    truth_cum = int((ts_all <= horizon_end).sum())
    truth_inc = truth_cum - n_train

    # --- A: prod ---
    a_p = a_lo = a_hi = -1
    try:
        tl = build_timeline_from_timestamps([t.to_pydatetime() for t in observed])
        fc = forecast_responses(tl, horizon_until=pd.Timestamp(horizon_end))
        idx = min(
            range(len(fc.future_dates)),
            key=lambda i: abs((fc.future_dates[i] - horizon_end).total_seconds()),
        )
        a_p = int(round(float(fc.future_cum.iloc[idx])))
        a_lo = int(round(float(fc.ci_lower.iloc[idx])))
        a_hi = int(round(float(fc.ci_upper.iloc[idx])))
    except (ForecastError, ValueError, IndexError):
        pass

    # --- B: wave, detected (no oracle, test_skip=0) ---
    b_p = b_lo = b_hi = -1
    b_half = -1.0
    waves = detect_wave_starts(observed, form_type=ftype, test_skip=0)
    if waves:
        ws_det = waves[-1].timestamp
        res = _wave_total(observed, ws_det, n_train, horizon_end, ftype)
        if res:
            b_p, b_lo, b_hi, b_half = res

    # --- C: wave, oracle start + oracle test_skip ---
    c_p = c_lo = c_hi = -1
    c_half = -1.0
    past_starts = [s for s in oracle_starts if s <= t_c]
    if past_starts:
        ws_or = max(past_starts)
        res = _wave_total(observed, ws_or, n_train, horizon_end, ftype)
        if res:
            c_p, c_lo, c_hi, c_half = res

    # --- D: naive recent (rate last 1h) ---
    last_1h_start = t_c - pd.Timedelta(hours=1)
    n_last_1h = int(((observed > last_1h_start) & (observed <= t_c)).sum())
    d_p = int(round(n_train + n_last_1h * horizon_h))

    # --- E: naive overall ---
    rate_overall = n_train / span_h
    e_p = int(round(n_train + rate_overall * horizon_h))

    return Point(
        form_id="",
        ftype=ftype,
        n_train=n_train,
        horizon_h=horizon_h,
        truth_cum=truth_cum,
        truth_inc=truth_inc,
        a_prod=a_p,
        b_detected=b_p,
        c_oracle=c_p,
        d_naive_recent=d_p,
        e_naive_overall=e_p,
        a_lo=a_lo,
        a_hi=a_hi,
        b_lo=b_lo,
        b_hi=b_hi,
        c_lo=c_lo,
        c_hi=c_hi,
        b_half=b_half,
        c_half=c_half,
    )


# ---------- main ------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ann = pd.read_csv(REPO / "data" / "wave_annotations.csv")
    df = pd.read_csv(REPO / "data" / "Form Timestamp Collection.csv")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

    valid = ann[ann["waves_iso"].notna() & (ann["waves_iso"] != "")].copy()
    valid["ftype"] = valid["form_type"].fillna("unknown")
    train_ids, holdout_ids = _stratified_split(valid[["form_id", "ftype"]])
    print(f"Train: {len(train_ids)} forms, Holdout: {len(holdout_ids)} forms")

    def run_split(ids: set, label: str) -> list[Point]:
        rows: list[Point] = []
        subset = valid[valid["form_id"].isin(ids)]
        if args.limit:
            subset = subset.head(args.limit)
        for _, row in subset.iterrows():
            fid = row["form_id"]
            ftype = row["form_type"]
            test_skip = int(row["test_resp"] or 0)
            oracle_starts = [pd.Timestamp(w) for w in str(row["waves_iso"]).split("|")]
            ts_all = df[df["FORM_ID"] == fid]["TIMESTAMP"].sort_values().reset_index(drop=True)
            if len(ts_all) < 6:
                continue
            for n_train in N_TRAIN_VALUES:
                for h in HORIZON_HOURS:
                    pt = _backtest_point(ts_all, n_train, h, ftype, oracle_starts, test_skip)
                    if pt:
                        pt.form_id = fid
                        rows.append(pt)
        print(f"  {label}: {len(rows)} backtest points")
        return rows

    print("Computing conformal on TRAIN...")
    train_rows = run_split(train_ids, "train")
    print("Evaluating on HOLDOUT...")
    ho_rows = run_split(holdout_ids, "holdout")

    train_df = pd.DataFrame([{f.name: getattr(r, f.name) for f in fields(r)} for r in train_rows])
    holdout_df = pd.DataFrame([{f.name: getattr(r, f.name) for f in fields(r)} for r in ho_rows])

    # Conformal q from TRAIN (normalized residual on cumulative).
    def conformal_q(rowset: pd.DataFrame, col: str, half_col: str) -> tuple[float, dict]:
        ok = rowset[(rowset[col] >= 0) & (rowset[half_col] > 0)].copy()
        ok["nr"] = (ok[col] - ok["truth_cum"]).abs() / ok[half_col]
        q_g = float(ok["nr"].quantile(0.95)) if len(ok) else 3.0
        q_t = ok.groupby("ftype")["nr"].quantile(0.95).to_dict()
        return q_g, q_t

    q_b_g, q_b_t = conformal_q(train_df, "b_detected", "b_half")
    q_c_g, q_c_t = conformal_q(train_df, "c_oracle", "c_half")
    print(f"Conformal q (B detected): global={q_b_g:.2f}")
    print(f"Conformal q (C oracle):   global={q_c_g:.2f}")

    # ---------- metrics ----------
    out = []
    out.append(f"Train: {len(train_ids)} forms / {len(train_df)} points")
    out.append(f"Holdout: {len(holdout_ids)} forms / {len(holdout_df)} points")
    out.append(f"Conformal q_0.95: B(detected)={q_b_g:.2f}  C(oracle)={q_c_g:.2f}")
    out.append("")

    methods = {
        "A prod": ("a_prod", "a_lo", "a_hi", None, None),
        "B wave_detected": ("b_detected", "b_lo", "b_hi", "b_half", (q_b_g, q_b_t)),
        "C wave_oracle": ("c_oracle", "c_lo", "c_hi", "c_half", (q_c_g, q_c_t)),
        "D naive_recent": ("d_naive_recent", None, None, None, None),
        "E naive_overall": ("e_naive_overall", None, None, None, None),
    }

    hdr = f"{'Method':<18}{'n':>6}{'cumMAPE':>9}{'incSMAPE':>10}{'cov95':>7}{'cov+cf':>8}{'Winkler':>9}{'width':>7}{'bias':>8}"
    out.append("=== GLOBAL (holdout) ===")
    out.append(hdr)
    metric_store = {}
    for name, (col, lo_c, hi_c, half_c, q) in methods.items():
        ok = holdout_df[holdout_df[col] >= 0].copy()
        if not len(ok):
            continue
        ape = (ok[col] - ok["truth_cum"]).abs() / ok["truth_cum"].clip(lower=1)
        smape = ok.apply(
            lambda r, col=col: _smape(r[col] - r["n_train"], r["truth_inc"]),
            axis=1,
        )
        bias = (ok[col] - ok["truth_cum"]).median()
        cov = cov_cf = wink = width = np.nan
        if lo_c and hi_c:
            hit = (ok[lo_c] <= ok["truth_cum"]) & (ok["truth_cum"] <= ok[hi_c])
            cov = hit.mean() * 100
            wink = ok.apply(
                lambda r, lo_c=lo_c, hi_c=hi_c: _winkler(r["truth_cum"], r[lo_c], r[hi_c]),
                axis=1,
            ).median()
            width = (ok[hi_c] - ok[lo_c]).median()
        if half_c and q:
            q_g, q_t = q
            okq = ok[ok[half_c] > 0]
            hits = 0
            for _, r in okq.iterrows():
                qq = q_t.get(r["ftype"], q_g)
                lo = r[col] - qq * r[half_c]
                hi = r[col] + qq * r[half_c]
                hits += lo <= r["truth_cum"] <= hi
            cov_cf = hits / max(len(okq), 1) * 100
        metric_store[name] = {"ape": ape.values, "n": len(ok)}
        out.append(
            f"{name:<18}{len(ok):>6}{ape.median() * 100:>8.1f}%{smape.median():>10.3f}"
            f"{(f'{cov:.0f}%' if not np.isnan(cov) else '  -'):>7}"
            f"{(f'{cov_cf:.0f}%' if not np.isnan(cov_cf) else '  -'):>8}"
            f"{(f'{wink:.0f}' if not np.isnan(wink) else '-'):>9}"
            f"{(f'{width:.0f}' if not np.isnan(width) else '-'):>7}{bias:>8.0f}"
        )

    # Detection penalty
    both = holdout_df[(holdout_df["b_detected"] >= 0) & (holdout_df["c_oracle"] >= 0)]
    if len(both):
        b_ape = (
            (both["b_detected"] - both["truth_cum"]).abs() / both["truth_cum"].clip(lower=1)
        ).median()
        c_ape = (
            (both["c_oracle"] - both["truth_cum"]).abs() / both["truth_cum"].clip(lower=1)
        ).median()
        out.append("")
        out.append(
            f"DETECTION PENALTY (paired n={len(both)}): "
            f"detected={b_ape * 100:.1f}%  oracle={c_ape * 100:.1f}%  penalty=+{(b_ape - c_ape) * 100:.1f}pp"
        )

    # Wilcoxon significance
    out.append("")
    out.append("=== SIGNIFICANCE (Wilcoxon signed-rank, paired cum APE) ===")

    def paired_wilcoxon(c1: str, c2: str, label: str):
        sub = holdout_df[(holdout_df[c1] >= 0) & (holdout_df[c2] >= 0)]
        if len(sub) < 10:
            out.append(f"  {label}: n<10, skip")
            return
        a1 = (sub[c1] - sub["truth_cum"]).abs() / sub["truth_cum"].clip(lower=1)
        a2 = (sub[c2] - sub["truth_cum"]).abs() / sub["truth_cum"].clip(lower=1)
        diff = a1.values - a2.values
        if np.allclose(diff, 0):
            out.append(f"  {label}: identical")
            return
        try:
            stat, p = wilcoxon(a1.values, a2.values)
            winner = c1 if a1.median() < a2.median() else c2
            out.append(
                f"  {label}: n={len(sub)} med({c1})={a1.median() * 100:.1f}% "
                f"med({c2})={a2.median() * 100:.1f}% p={p:.4f} "
                f"{'**SIG**' if p < 0.05 else 'ns'} winner={winner}"
            )
        except ValueError as e:
            out.append(f"  {label}: {e}")

    paired_wilcoxon("b_detected", "d_naive_recent", "B detected vs D naive_recent")
    paired_wilcoxon("b_detected", "a_prod", "B detected vs A prod")
    paired_wilcoxon("c_oracle", "b_detected", "C oracle vs B detected (penalty)")
    paired_wilcoxon("d_naive_recent", "a_prod", "D naive_recent vs A prod")

    # Per-type (cum MAPE)
    out.append("")
    out.append("=== PER TYPE (cum MAPE, holdout) ===")
    out.append(f"{'ftype':<22}{'A prod':>9}{'B det':>9}{'C orac':>9}{'D naive':>9}{'n':>6}")
    for ft, g in holdout_df.groupby("ftype"):
        if len(g) < 5:
            continue

        def m(col, g=g):
            ok = g[g[col] >= 0]
            if not len(ok):
                return "  -"
            v = ((ok[col] - ok["truth_cum"]).abs() / ok["truth_cum"].clip(lower=1)).median()
            return f"{v * 100:.0f}%"

        out.append(
            f"{ft:<22}{m('a_prod'):>9}{m('b_detected'):>9}{m('c_oracle'):>9}{m('d_naive_recent'):>9}{len(g):>6}"
        )

    # Per horizon
    out.append("")
    out.append("=== PER HORIZON (cum MAPE, holdout) ===")
    out.append(f"{'horizon_h':<12}{'A prod':>9}{'B det':>9}{'C orac':>9}{'D naive':>9}{'n':>6}")
    for h, g in holdout_df.groupby("horizon_h"):

        def m(col, g=g):
            ok = g[g[col] >= 0]
            if not len(ok):
                return "  -"
            v = ((ok[col] - ok["truth_cum"]).abs() / ok["truth_cum"].clip(lower=1)).median()
            return f"{v * 100:.0f}%"

        out.append(
            f"{h:<12}{m('a_prod'):>9}{m('b_detected'):>9}{m('c_oracle'):>9}{m('d_naive_recent'):>9}{len(g):>6}"
        )

    report = "\n".join(out)
    print()
    print(report)

    fig_dir = REPO / "research" / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    holdout_df.to_csv(fig_dir / "18_endtoend_points.csv", index=False)
    (REPO / "research" / "reports" / "18_wave_endtoend.md").write_text(
        f"# 18 — Wave estimator end-to-end (no oracle)\n\n```\n{report}\n```\n",
        encoding="utf-8",
    )
    print("\nSaved: figures/18_endtoend_points.csv + 18_wave_endtoend.md")


if __name__ == "__main__":
    main()
