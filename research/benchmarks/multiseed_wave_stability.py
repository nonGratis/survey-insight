"""multiseed_wave_stability.py — чи стабільні числа P6/P8 чи везіння одного split.

Усі числа P6/P8 — з одного split (seed=42, 47 holdout форм). Тут повторюємо
оцінку wave(detected) vs naive(recent) на 5 сідах і дивимось mean±std
holdout MAPE + coverage. Малий std => число реальне, не артефакт split.

Без prod (повільний) — ключове питання: стабільна перевага wave над naive.

Запуск:
    .venv/Scripts/python.exe research/benchmarks/multiseed_wave_stability.py
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
from core.forecast.wave_detector import detect_wave_starts  # noqa: E402
from core.forecast.wave_estimator import estimate_wave  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
N_TRAIN_VALUES = (5, 8, 10, 15, 20)
HORIZON_HOURS = (0.5, 1.0, 2.0, 6.0)
SEEDS = (1, 7, 13, 42, 99)
TEST_SIZE = 0.32


def holdout_ids(valid: pd.DataFrame, seed: int) -> set:
    rng = np.random.RandomState(seed)
    counts = valid["ftype"].value_counts()
    strat = valid["ftype"].apply(lambda x: x if counts[x] >= 6 else "rare")
    ho = []
    for _, idx in valid.groupby(strat).groups.items():
        idx = list(idx)
        rng.shuffle(idx)
        ho.extend(idx[: max(1, int(len(idx) * TEST_SIZE))])
    return set(valid.loc[ho, "form_id"])


def eval_holdout(ids: set, valid: pd.DataFrame, df: pd.DataFrame) -> dict:
    wave_ape, naive_ape, wave_hit = [], [], []
    for _, row in valid[valid["form_id"].isin(ids)].iterrows():
        fid, ftype = row["form_id"], row["form_type"]
        ts = df[df["FORM_ID"] == fid]["TIMESTAMP"].sort_values().reset_index(drop=True)
        if len(ts) < 6:
            continue
        for n_train in N_TRAIN_VALUES:
            if len(ts) <= n_train:
                continue
            obs = ts.iloc[:n_train].reset_index(drop=True)
            t_c = obs.iloc[-1]
            if (t_c - obs.iloc[0]).total_seconds() <= 0:
                continue
            waves = detect_wave_starts(obs, form_type=ftype, test_skip=0)
            ws = waves[-1].timestamp if waves else obs.iloc[0]
            pre = int((obs < ws).sum())
            cur = obs[obs >= ws].reset_index(drop=True)
            n_last_1h = int(((obs > t_c - pd.Timedelta(hours=1)) & (obs <= t_c)).sum())
            for h in HORIZON_HOURS:
                he = t_c + pd.Timedelta(hours=h)
                truth = int((ts <= he).sum())
                # wave
                if len(cur) >= 5:
                    hfw = (he - ws).total_seconds() / 3600.0
                    if hfw > 0:
                        try:
                            wf = estimate_wave(cur.tolist(), horizon_h=hfw, form_type=ftype)
                            pred = pre + wf.point
                            wave_ape.append(abs(pred - truth) / max(truth, 1))
                            wave_hit.append((pre + wf.ci_lower) <= truth <= (pre + wf.ci_upper))
                        except ForecastError:
                            pass
                # naive recent
                naive = n_train + n_last_1h * h
                naive_ape.append(abs(naive - truth) / max(truth, 1))
    return {
        "wave_mape": float(np.median(wave_ape) * 100) if wave_ape else float("nan"),
        "naive_mape": float(np.median(naive_ape) * 100) if naive_ape else float("nan"),
        "wave_cov": float(np.mean(wave_hit) * 100) if wave_hit else float("nan"),
        "n": len(wave_ape),
    }


def main():
    ann = pd.read_csv(REPO / "data" / "wave_annotations.csv")
    df = pd.read_csv(REPO / "data" / "Form Timestamp Collection.csv")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    valid = ann[ann["waves_iso"].notna() & (ann["waves_iso"] != "")].copy()
    valid["ftype"] = valid["form_type"].fillna("unknown")

    rows = []
    print(f"{'seed':>6}{'wave_MAPE':>11}{'naive_MAPE':>12}{'wave_cov':>10}{'n':>7}")
    for s in SEEDS:
        r = eval_holdout(holdout_ids(valid, s), valid, df)
        rows.append(r)
        print(
            f"{s:>6}{r['wave_mape']:>10.1f}%{r['naive_mape']:>11.1f}%{r['wave_cov']:>9.1f}%{r['n']:>7}"
        )

    wm = np.array([r["wave_mape"] for r in rows])
    nm = np.array([r["naive_mape"] for r in rows])
    cv = np.array([r["wave_cov"] for r in rows])
    print()
    print(f"wave  MAPE: mean={wm.mean():.1f}%  std={wm.std():.1f}%  range=[{wm.min():.1f}, {wm.max():.1f}]")
    print(f"naive MAPE: mean={nm.mean():.1f}%  std={nm.std():.1f}%")
    print(f"wave  cov : mean={cv.mean():.1f}%  std={cv.std():.1f}%")
    print(f"wave beats naive in {int((wm < nm).sum())}/{len(SEEDS)} seeds")


if __name__ == "__main__":
    main()
