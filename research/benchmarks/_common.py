"""_common.py — спільні хелпери для backtest-скриптів (08+).

Витягнуто з 08_/09_/10_ для уникнення копіпасти. Усе тут — pure helpers
без побічних ефектів окрім IO. Tests reuse them as-is.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- taxonomies (з 08_) ---------------------------------------------


def n_class(n_total: int) -> str:
    if n_total < 10:
        return "tiny"
    if n_total < 30:
        return "small"
    if n_total < 100:
        return "medium"
    if n_total < 1000:
        return "large"
    return "huge"


def tempo_class(timestamps: pd.Series) -> str:
    """burst / daily_flow / long_tail / sporadic (з 08_)."""
    ts = timestamps.sort_values().reset_index(drop=True)
    if len(ts) < 2:
        return "sporadic"
    deltas_sec = ts.diff().dropna().dt.total_seconds()
    deltas_sec = deltas_sec[deltas_sec > 0]
    if len(deltas_sec) == 0:
        return "burst"
    median_h = float(deltas_sec.median()) / 3600.0
    mean_s = float(deltas_sec.mean())
    std_s = float(deltas_sec.std()) if len(deltas_sec) > 1 else 0.0
    cv = std_s / mean_s if mean_s > 0 else 0.0
    if cv >= 2.0:
        return "sporadic"
    if median_h < 0.5:
        return "burst"
    if median_h < 12.0 and cv < 1.5:
        return "daily_flow"
    return "long_tail"


def duration_class(timestamps: pd.Series) -> str:
    ts = timestamps.sort_values()
    span_h = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 3600.0
    if span_h < 24:
        return "hours"
    if span_h < 24 * 7:
        return "days"
    if span_h < 24 * 30:
        return "weeks"
    return "months"


# ---------- loaders --------------------------------------------------------


def load_shapes(features_csv: Path) -> dict[str, str]:
    if not features_csv.exists():
        raise FileNotFoundError(f"Run 01_dataset_overview.py first to generate {features_csv}")
    df = pd.read_csv(features_csv)
    return dict(zip(df["form_id"], df["shape"], strict=True))


def load_form_types(form_types_csv: Path) -> dict[str, str]:
    """form_id -> form_type (event_registration/survey/.../other).

    Returns empty dict якщо файл відсутній (форма не була у каталозі).
    Caller може fallback на 'unknown' для missing форм.
    """
    if not form_types_csv.exists():
        return {}
    df = pd.read_csv(form_types_csv)
    return dict(zip(df["form_id"], df["form_type"], strict=True))


def build_eligible_forms(
    df: pd.DataFrame,
    shapes: dict[str, str],
    form_types: dict[str, str],
    min_n: int,
) -> tuple[list[dict], dict[str, int]]:
    """Створює список eligible форм з повним set таксономій + skip-stats.

    Returns:
        (eligible, skipped) де eligible — list[dict] з ключами
        form_id, timestamps, shape, n_class, tempo, duration_class, form_type.
    """
    eligible: list[dict] = []
    skipped = {"too_few": 0, "insufficient_shape": 0, "no_span": 0}
    for form_id, group in df.groupby("FORM_ID"):
        ts = group["TIMESTAMP"].sort_values().reset_index(drop=True)
        n = len(ts)
        if n < min_n:
            skipped["too_few"] += 1
            continue
        shape = shapes.get(form_id, "unknown")
        if shape == "insufficient":
            skipped["insufficient_shape"] += 1
            continue
        if (ts.iloc[-1] - ts.iloc[0]).total_seconds() <= 0:
            skipped["no_span"] += 1
            continue
        eligible.append(
            {
                "form_id": form_id,
                "timestamps": ts,
                "shape": shape,
                "n_class": n_class(n),
                "tempo": tempo_class(ts),
                "duration_class": duration_class(ts),
                "form_type": form_types.get(form_id, "unknown"),
            }
        )
    return eligible, skipped


def load_dataset(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    return df.drop_duplicates(subset=["FORM_ID", "TIMESTAMP"]).reset_index(drop=True)


# ---------- horizon utils --------------------------------------------------


def idx_for_horizon(future_dates: pd.DatetimeIndex, horizon_end: pd.Timestamp) -> int:
    future_dates = pd.DatetimeIndex(future_dates)
    target_date = pd.Timestamp(horizon_end.normalize())
    if target_date <= future_dates[0]:
        return 0
    if target_date >= future_dates[-1]:
        return len(future_dates) - 1
    idx = int(np.searchsorted(future_dates, target_date, side="left"))
    return min(idx, len(future_dates) - 1)


# ---------- metrics --------------------------------------------------------


def add_per_row_metrics(df: pd.DataFrame, point_col: str, lo_col: str, hi_col: str) -> pd.DataFrame:
    """Додає ape/hit/sharpness/signed_err/mode до df.

    df має мати truth + три названі колонки (point, lo, hi). truth > 0 фільтрується.
    """
    ok = df[df["truth"] > 0].copy()
    ok["ape"] = (ok["truth"] - ok[point_col]).abs() / ok["truth"]
    ok["hit"] = (ok[lo_col] <= ok["truth"]) & (ok["truth"] <= ok[hi_col])
    ok["sharpness"] = (ok[hi_col] - ok[lo_col]) / ok["truth"]
    ok["signed_err"] = (ok[point_col] - ok["truth"]) / ok["truth"]

    def _mode(row):
        if row["truth"] < row[lo_col]:
            return "overconfident_high"
        if row["truth"] > row[hi_col]:
            return "overconfident_low"
        return "in_ci"

    ok["mode"] = ok.apply(_mode, axis=1)
    return ok


def agg_by(
    metrics: pd.DataFrame, group_cols: str | list[str], hit_col: str = "hit"
) -> pd.DataFrame:
    """Per-group метрики. group_cols може бути str або list для cross-tab.

    Очікувані колонки в metrics: ape, <hit_col>, sharpness, signed_err.
    """
    agg = metrics.groupby(group_cols, observed=True).agg(
        n=("ape", "size"),
        mape_p50=("ape", "median"),
        mape_p90=("ape", lambda s: s.quantile(0.90)),
        coverage=(hit_col, "mean"),
        sharpness_p50=("sharpness", "median"),
        bias=("signed_err", "median"),
    )
    agg["n"] = agg["n"].astype(int)
    agg["mape_p50"] = (agg["mape_p50"] * 100).round(1)
    agg["mape_p90"] = (agg["mape_p90"] * 100).round(1)
    agg["coverage"] = (agg["coverage"] * 100).round(1)
    agg["sharpness_p50"] = agg["sharpness_p50"].round(2)
    agg["bias"] = (agg["bias"] * 100).round(1)
    return agg


# ---------- IO -------------------------------------------------------------


def file_sha256_short(path: Path, length: int = 12) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def df_to_md(df: pd.DataFrame) -> str:
    df = df.reset_index()
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in headers) + " |",
        "|" + "|".join("---:" if i > 0 else "---" for i in range(len(headers))) + "|",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def reindex_by(agg: pd.DataFrame, order: Iterable[str]) -> pd.DataFrame:
    """Безпечний reindex: лишає тільки order-категорії що реально присутні."""
    present = set(agg.index.tolist() if not isinstance(agg.index, pd.MultiIndex) else [])
    keep = [c for c in order if c in present]
    return agg.reindex(keep) if keep else agg


# ---------- common orderings ----------------------------------------------

SHAPE_ORDER = ["linear", "logarithmic", "logistic", "late_burst", "ill_fit", "unknown"]
N_CLASS_ORDER = ["tiny", "small", "medium", "large", "huge"]
TEMPO_ORDER = ["burst", "daily_flow", "long_tail", "sporadic"]
DURATION_ORDER = ["hours", "days", "weeks", "months"]
FORM_TYPE_ORDER = [
    "event_registration",
    "event_feedback",
    "survey",
    "recruitment",
    "service",
    "volunteer_donor",
    "political",
    "creative_submission",
    "holiday",
    "other",
    "unknown",
]
