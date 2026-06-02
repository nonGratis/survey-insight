"""verify_current_wave.py — sanity proof for forecast_current_wave (prod path).

Доводить (не припускає): прод-вхід forecast_current_wave НІКОЛИ не дає
абсурдного CI — ні на wave-шляху, ні на fallback-шляху.

Секції:
  1. Anchor-форми зі скриншотів (47, 7433) на ранніх cutoff'ах.
  2. Fallback-форми: де поточна хвиля < 5 точок (спрацьовує forecast_responses) —
     перевірити що half-width ≤ 2× point (cap policy).
  3. Інваріант на всьому датасеті: для кожної форми × кілька cutoff'ів
     перевірити half ≤ 2× point. Будь-яке порушення друкується.

Запуск:
    .venv/Scripts/python.exe research/benchmarks/verify_current_wave.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.forecast import classify_form_type, forecast_current_wave  # noqa: E402
from core.forecast.wave_detector import detect_wave_starts  # noqa: E402
from core.timeline import build_timeline_from_timestamps  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CUTOFFS = (5, 8, 10, 15, 20, 30)


def _half_ok(fc) -> bool:
    """cap policy: half-width <= 2x point (max_relative=2.0)."""
    half = (fc.final_ci[1] - fc.final_ci[0]) / 2.0
    return half <= 2.0 * max(fc.final_estimate, 1) + 1


def main():
    df = pd.read_csv(REPO / "data" / "Form Timestamp Collection.csv")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    catalog = {}
    cat_path = REPO / "data" / "Form Catalog.tsv"
    if cat_path.exists():
        cat = pd.read_csv(cat_path, sep="\t", dtype=str).fillna("")
        catalog = dict(zip(cat["form_id"], cat["form_title"], strict=False))

    # --- 1. Anchors ---
    print("=== 1. ANCHOR forms (screenshots) ===")
    anchors = [
        ("47-form", "1p0ERtAe-_c4J_EL0H-f3Ykbbc6GBbWmqY4-SX1r9I3Y"),
        ("7433-form", "1GM-api8tg1DaVEE_NJ203b9K01TA4_uCW_3c2gkkh0c"),
    ]
    for label, fid in anchors:
        ts = df[df["FORM_ID"] == fid]["TIMESTAMP"].sort_values().reset_index(drop=True)
        ftype = classify_form_type(catalog.get(fid, ""))
        for n in (15, 30):
            tl = build_timeline_from_timestamps([t.to_pydatetime() for t in ts.iloc[:n]])
            fc, _ = forecast_current_wave(tl, form_type=ftype)
            half = (fc.final_ci[1] - fc.final_ci[0]) // 2
            flag = "OK" if _half_ok(fc) else "ABSURD"
            print(
                f"  [{flag}] {label} n={n}: {fc.final_estimate} +/-{half} CI={fc.final_ci} ({fc.model})"
            )

    # --- 2 & 3. Dataset-wide invariant + fallback census ---
    print("\n=== 2+3. Dataset-wide cap invariant (half <= 2x point) ===")
    form_ids = df["FORM_ID"].unique()
    n_checks = n_violations = n_fallback = n_wave = 0
    violations = []
    for fid in form_ids:
        ts = df[df["FORM_ID"] == fid]["TIMESTAMP"].sort_values().reset_index(drop=True)
        if len(ts) < 6:
            continue
        ftype = classify_form_type(catalog.get(fid, ""))
        for n in CUTOFFS:
            if len(ts) <= n:
                continue
            tl = build_timeline_from_timestamps([t.to_pydatetime() for t in ts.iloc[:n]])
            try:
                fc, _ = forecast_current_wave(tl, form_type=ftype)
            except Exception as exc:  # noqa: BLE001
                violations.append(f"{fid[:12]} n={n}: EXC {exc}")
                n_violations += 1
                continue
            n_checks += 1
            # fallback vs wave: detect current wave length
            waves = detect_wave_starts(ts.iloc[:n], form_type=ftype, test_skip=0)
            ws = waves[-1].timestamp if waves else ts.iloc[0]
            cur_len = int((ts.iloc[:n] >= ws).sum())
            if cur_len < 5:
                n_fallback += 1
            else:
                n_wave += 1
            if not _half_ok(fc):
                n_violations += 1
                violations.append(
                    f"{fid[:12]} n={n} ftype={ftype}: point={fc.final_estimate} CI={fc.final_ci} "
                    f"(fallback={cur_len < 5})"
                )

    print(f"checks={n_checks}  wave_path={n_wave}  fallback_path={n_fallback}")
    print(f"cap violations (half > 2x point): {n_violations}")
    if violations:
        print("VIOLATIONS (first 20):")
        for v in violations[:20]:
            print(f"  {v}")
    else:
        print("PROOF: 0 absurd intervals across entire dataset — wave AND fallback paths sane.")


if __name__ == "__main__":
    main()
