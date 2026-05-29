"""07_fit_priors.py — генерує core/forecast/priors_data.json з історії 177-форм.

Для кожної shape-категорії × моделі:
1. Зафітимо кожну форму у цій категорії всіма 3 моделями.
2. Зберемо параметри (K, r, t0 чи a, b, c).
3. Запишемо median і std кожного параметра.

Output: core/forecast/priors_data.json — використовується core.forecast.priors.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning

warnings.simplefilter("ignore", OptimizeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.forecast.models import DEFAULT_MODELS, fit_model  # noqa: E402
from core.forecast.priors import ShapePrior, save_priors  # noqa: E402
from core.forecast.shape_classifier import classify_timeline  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)

MIN_N_FOR_PRIOR_FIT = 20  # ігноруємо форми з малим N — нестабільні параметри


def main(input_path: Path) -> None:
    df = pd.read_csv(input_path)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df = df.drop_duplicates(subset=["FORM_ID", "TIMESTAMP"]).reset_index(drop=True)

    # Збираємо параметри per (shape, model).
    by_key: dict[tuple[str, str], list[tuple[float, ...]]] = {}

    eligible = [
        (fid, sorted(g["TIMESTAMP"].tolist()))
        for fid, g in df.groupby("FORM_ID")
        if len(g) >= MIN_N_FOR_PRIOR_FIT
    ]
    print(f"Forms with N>={MIN_N_FOR_PRIOR_FIT}: {len(eligible)}")

    for _fid, ts in eligible:
        shape = classify_timeline(pd.Series(ts))
        if shape == "insufficient":
            continue
        first = ts[0]
        t_train = np.array([(t - first).total_seconds() / 86400.0 for t in ts])
        y_train = np.arange(1, len(ts) + 1, dtype=float)
        for model in DEFAULT_MODELS:
            try:
                params, _pcov = fit_model(model, t_train, y_train, target=None)
                by_key.setdefault((model.name, shape), []).append(params)
            except Exception:  # noqa: BLE001
                continue

    # Median і std per key.
    priors = {}
    for key, fits in by_key.items():
        if len(fits) < 3:  # потребуємо ≥3 для оцінки std
            continue
        arr = np.array(fits)
        medians = np.median(arr, axis=0)
        stds = np.std(arr, axis=0, ddof=1)
        priors[key] = ShapePrior(
            model_name=key[0],
            shape=key[1],
            param_medians=tuple(float(x) for x in medians),
            param_stds=tuple(float(x) for x in stds),
            n_samples=len(fits),
        )

    save_priors(priors)
    print(f"\nSaved {len(priors)} priors to core/forecast/priors_data.json")
    print("\nSummary:")
    for p in sorted(priors.values(), key=lambda x: (x.model_name, x.shape)):
        print(
            f"  {p.model_name} × {p.shape}: n={p.n_samples}, "
            f"medians={tuple(round(x, 3) for x in p.param_medians)}, "
            f"stds={tuple(round(x, 3) for x in p.param_stds)}"
        )


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=repo_root / "data" / "Form Timestamp Collection.csv",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.input)
