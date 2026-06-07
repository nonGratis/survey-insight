"""21_within_wave_diag.py — де стеля within-wave точності (G1 діагноз).

Мета сесії: ідеальний прогноз В МЕЖАХ хвилі. Перш ніж покращувати — міряємо
де саме погано: MAPE посадки хвилі за n_observed × form_type, на ОРАКУЛ-хвилях
(анотації = справжні старти, ізолюємо помилку ОЦІНЮВАЧА від помилки детектора).

Протокол: для кожної анотованої хвилі з >=8 відп., спостерігаємо перші n,
прогнозуємо посадку (estimate_wave horizon=3h від старту), truth = факт.
відп. у [wt, min(next_wave, wt+6h)]. APE = |pred_landing - truth| / truth.

Запуск:
    .venv/Scripts/python.exe research/benchmarks/21_within_wave_diag.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
from scipy.optimize import OptimizeWarning

warnings.simplefilter("ignore", OptimizeWarning)
warnings.simplefilter("ignore", RuntimeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.forecast import ForecastError  # noqa: E402
from core.forecast.wave_estimator import estimate_wave  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
N_OBS = (5, 8, 10, 15, 20)
WAVE_WINDOW_H = 6.0  # макс. вікно хвилі (хвиля <=3h, але буфер)


def main():
    ann = pd.read_csv(REPO / "data" / "wave_annotations.csv")
    df = pd.read_csv(REPO / "data" / "Form Timestamp Collection.csv")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

    rows = []
    for _, r in ann.iterrows():
        if not r["waves_iso"] or pd.isna(r["waves_iso"]):
            continue
        waves = [pd.Timestamp(w) for w in str(r["waves_iso"]).split("|")]
        ftype = r["form_type"]
        test_skip = int(r["test_resp"] or 0)
        grp = df[df["FORM_ID"] == r["form_id"]]["TIMESTAMP"].sort_values().iloc[test_skip:]
        grp = grp.reset_index(drop=True)
        for i, wt in enumerate(waves):
            nxt = waves[i + 1] if i + 1 < len(waves) else wt + pd.Timedelta(hours=24)
            wave_end = min(nxt, wt + pd.Timedelta(hours=WAVE_WINDOW_H))
            in_wave = grp[(grp >= wt) & (grp < wave_end)].reset_index(drop=True)
            truth = len(in_wave)
            if truth < 8:
                continue
            for n in N_OBS:
                if n >= truth:
                    continue
                obs = in_wave.iloc[:n]
                span_h = (obs.iloc[-1] - wt).total_seconds() / 3600.0
                if span_h <= 0:
                    continue
                try:
                    wf = estimate_wave(obs.tolist(), horizon_h=WAVE_WINDOW_H, form_type=ftype)
                except ForecastError:
                    continue
                ape = abs(wf.point - truth) / truth
                # naive within-wave baseline: n + recent-rate × remaining-window
                rate = n / span_h
                naive = n + rate * max(WAVE_WINDOW_H - span_h, 0)
                ape_naive = abs(naive - truth) / truth
                rows.append(
                    {
                        "ftype": ftype,
                        "n_obs": n,
                        "truth": truth,
                        "ape": ape,
                        "ape_naive": ape_naive,
                        "model": wf.model_name,
                    }
                )

    res = pd.DataFrame(rows)
    print(f"within-wave evals: {len(res)} (oracle waves)\n")

    print("=== MAPE by n_observed (wave estimator vs naive rate) ===")
    print(f"{'n_obs':>6}{'wave':>9}{'naive':>9}{'n':>7}")
    for n, g in res.groupby("n_obs"):
        print(f"{n:>6}{g.ape.median() * 100:>8.0f}%{g.ape_naive.median() * 100:>8.0f}%{len(g):>7}")

    print("\n=== MAPE by form_type (wave) ===")
    for ft, g in res.groupby("ftype"):
        if len(g) < 8:
            continue
        print(
            f"  {ft:<22} wave {g.ape.median() * 100:>3.0f}%  naive {g.ape_naive.median() * 100:>3.0f}%  (n={len(g)})"
        )

    print("\n=== model chosen (within-wave) ===")
    print(res["model"].value_counts().to_string())
    print(
        f"\nGLOBAL within-wave MAPE: wave={res.ape.median() * 100:.1f}%  naive={res.ape_naive.median() * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
