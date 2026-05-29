"""09_early_blend_ab.py — A/B/C для ранніх передбачень (cutoffs 0.1, 0.2).

Hypothesis: при малих n_train (≤30) curve-fit нестійкий, бо коваріаційна
матриця параметрів роздута на 5–15 точках. Простий Poisson naive
(rate × horizon) має нижчу variance і кращу калібровку на коротких
горизонтах. Blend з sample-size-dependent weights повинен виграти у
ранній зоні без втрат у зрілій.

Методи (усі на тих самих cutoff-points що й 08_ для apples-to-apples):

- **A · model** — поточний `forecast_responses` (контроль).
- **B · naive** — Poisson MLE: λ̂ = n_train / cutoff_span_days,
  point = n_train + λ̂ · horizon_days; CI 95% — точний exact на
  Poisson-інкременті через γ-квантилі (Garwood 1936). Жодного curve-fit.
- **C · blend** — α · model + (1−α) · naive, де
  α = clip((n_train − 5) / 25, 0, 1).
  CI — лінійний blend bounds.

ЖОДНИХ змін у `core/forecast/`. Якщо C (або B) виграє — окремим коміттом
винесемо `naive.py` + flag у `service.py`. Поточний прогін — суто
research.

Запуск:
    .venv/Scripts/python.exe research/benchmarks/09_early_blend_ab.py
    .venv/Scripts/python.exe research/benchmarks/09_early_blend_ab.py --limit 5
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import OptimizeWarning
from scipy.stats import gamma

warnings.simplefilter("ignore", OptimizeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.forecast import ForecastError, forecast_responses  # noqa: E402
from core.timeline import build_timeline_from_timestamps  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)

# ---------- config ---------------------------------------------------------

MIN_N_FOR_BACKTEST = 10
CUTOFFS_DEFAULT = (0.1, 0.2, 0.3, 0.5, 0.7)
HORIZON_FRACTION = 0.25
MIN_TRAIN_POINTS = 5

BLEND_LOW_N = 5  # α=0 below this
BLEND_HIGH_N = 30  # α=1 above this

METHODS = ("model", "naive", "blend")


@dataclass(frozen=True)
class ABPoint:
    form_id: str
    shape: str
    n_total: int
    cutoff_frac: float
    n_train: int
    horizon_days: float
    truth: int
    # per-method
    model_point: int
    model_lo: int
    model_hi: int
    naive_point: int
    naive_lo: int
    naive_hi: int
    blend_point: int
    blend_lo: int
    blend_hi: int
    alpha: float
    error: str | None


# ---------- shape lookup ---------------------------------------------------


def _load_shapes(features_csv: Path) -> dict[str, str]:
    if not features_csv.exists():
        raise FileNotFoundError(f"Run 01_dataset_overview.py first to generate {features_csv}")
    df = pd.read_csv(features_csv)
    return dict(zip(df["form_id"], df["shape"], strict=True))


# ---------- naive Poisson --------------------------------------------------


def _naive_poisson(
    n_train: int, cutoff_span_seconds: float, horizon_seconds: float
) -> tuple[float, float, float]:
    """Naive Poisson rate forecast з exact 95% CI на інкременті.

    Рейт λ̂ = n_train / cutoff_span (за секунду). Очікувані надходження за
    горизонт = λ̂ · horizon_seconds. CI 95% на цілочисельному інкременті
    через γ-квантилі (Garwood exact Poisson CI).
    """
    rate = n_train / max(cutoff_span_seconds, 1.0)
    expected_inc = rate * horizon_seconds
    if expected_inc <= 0:
        return float(n_train), float(n_train), float(n_train)
    # Exact 95% Poisson CI for the increment:
    # If X ~ Poisson(λ), CI for λ given observed X uses gamma quantiles.
    # Here we want CI for X given λ (future increment is random).
    # Approach: treat expected_inc as mean of future Poisson, take 2.5/97.5
    # percentiles of Poisson(λ=expected_inc). γ.ppf на (X, λ) — стандартна
    # обертка через зв'язок Poisson↔gamma.
    lo_inc = float(gamma.ppf(0.025, expected_inc))
    hi_inc = float(gamma.ppf(0.975, expected_inc + 1))
    point = n_train + expected_inc
    lo = n_train + lo_inc
    hi = n_train + hi_inc
    return point, lo, hi


# ---------- blend weight ---------------------------------------------------


def _alpha(n_train: int) -> float:
    if n_train <= BLEND_LOW_N:
        return 0.0
    if n_train >= BLEND_HIGH_N:
        return 1.0
    return (n_train - BLEND_LOW_N) / (BLEND_HIGH_N - BLEND_LOW_N)


# ---------- A/B core -------------------------------------------------------


def _ab_one(
    form_id: str,
    timestamps: pd.Series,
    shape: str,
    cutoff_frac: float,
) -> ABPoint | None:
    n_total = len(timestamps)
    n_train = int(round(cutoff_frac * n_total))
    if n_train < MIN_TRAIN_POINTS:
        return None

    ts_sorted = timestamps.sort_values().reset_index(drop=True)
    first_ts = ts_sorted.iloc[0]
    last_ts = ts_sorted.iloc[-1]
    span_seconds = (last_ts - first_ts).total_seconds()
    if span_seconds <= 0:
        return None

    cutoff_ts = ts_sorted.iloc[n_train - 1]
    cutoff_span_seconds = (cutoff_ts - first_ts).total_seconds()
    if cutoff_span_seconds <= 0:
        return None

    horizon_seconds = max(cutoff_span_seconds * HORIZON_FRACTION, 86400.0)
    horizon_end = cutoff_ts + pd.Timedelta(seconds=horizon_seconds)
    truth = int((ts_sorted <= horizon_end).sum())

    # B · naive
    naive_p, naive_lo, naive_hi = _naive_poisson(n_train, cutoff_span_seconds, horizon_seconds)

    # A · model (current production)
    prefix_dt = [t.to_pydatetime() for t in ts_sorted.iloc[:n_train].tolist()]
    timeline = build_timeline_from_timestamps(prefix_dt)
    try:
        fc = forecast_responses(timeline)
        future_dates = pd.DatetimeIndex(fc.future_dates)
        target_date = pd.Timestamp(horizon_end.normalize())
        if target_date <= future_dates[0]:
            idx = 0
        elif target_date >= future_dates[-1]:
            idx = len(future_dates) - 1
        else:
            idx = int(np.searchsorted(future_dates, target_date, side="left"))
            idx = min(idx, len(future_dates) - 1)
        model_p = float(fc.future_cum.iloc[idx])
        model_lo = float(fc.ci_lower.iloc[idx])
        model_hi = float(fc.ci_upper.iloc[idx])
        err: str | None = None
    except ForecastError as e:
        # Якщо model падає, тестуємо B/C з naive як заміною model (degenerate
        # blend = pure naive). Це чесно: real prod-flow теж би деградував.
        model_p, model_lo, model_hi = naive_p, naive_lo, naive_hi
        err = f"model_fallback_to_naive: {e}"

    # C · blend
    a = _alpha(n_train)
    blend_p = a * model_p + (1 - a) * naive_p
    blend_lo = a * model_lo + (1 - a) * naive_lo
    blend_hi = a * model_hi + (1 - a) * naive_hi

    return ABPoint(
        form_id=form_id,
        shape=shape,
        n_total=n_total,
        cutoff_frac=cutoff_frac,
        n_train=n_train,
        horizon_days=horizon_seconds / 86400.0,
        truth=truth,
        model_point=int(round(model_p)),
        model_lo=int(round(model_lo)),
        model_hi=int(round(model_hi)),
        naive_point=int(round(naive_p)),
        naive_lo=int(round(naive_lo)),
        naive_hi=int(round(naive_hi)),
        blend_point=int(round(blend_p)),
        blend_lo=int(round(blend_lo)),
        blend_hi=int(round(blend_hi)),
        alpha=a,
        error=err,
    )


# ---------- metrics --------------------------------------------------------


def _per_method_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Перетворити wide → long: одна метрика на (point, method)."""
    rows = []
    for _, r in df.iterrows():
        if r["truth"] <= 0:
            continue
        for m in METHODS:
            p = r[f"{m}_point"]
            lo = r[f"{m}_lo"]
            hi = r[f"{m}_hi"]
            ape = abs(r["truth"] - p) / r["truth"]
            hit = lo <= r["truth"] <= hi
            sharp = (hi - lo) / r["truth"]
            signed = (p - r["truth"]) / r["truth"]
            rows.append(
                {
                    "form_id": r["form_id"],
                    "shape": r["shape"],
                    "n_train": r["n_train"],
                    "cutoff_frac": r["cutoff_frac"],
                    "method": m,
                    "truth": r["truth"],
                    "point": p,
                    "lo": lo,
                    "hi": hi,
                    "ape": ape,
                    "hit_95": hit,
                    "sharpness": sharp,
                    "signed_err": signed,
                    "alpha": r["alpha"],
                }
            )
    return pd.DataFrame(rows)


def _agg(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    agg = df.groupby(by, observed=True).agg(
        n=("ape", "size"),
        mape_p50=("ape", "median"),
        mape_p90=("ape", lambda s: s.quantile(0.90)),
        coverage=("hit_95", "mean"),
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


# ---------- figures --------------------------------------------------------


def _fig_per_cutoff(metrics: pd.DataFrame, value: str, ylabel: str, title: str) -> go.Figure:
    """Bar chart: x = cutoff, color = method."""
    pivot = (
        metrics.groupby(["cutoff_frac", "method"])
        .agg(
            v=(
                "hit_95" if value == "coverage" else "ape",
                "mean" if value == "coverage" else "median",
            )
        )
        .reset_index()
    )
    pivot["v"] = pivot["v"] * 100
    fig = go.Figure()
    colors = {"model": "#1f77b4", "naive": "#ff7f0e", "blend": "#2ca02c"}
    for m in METHODS:
        sub = pivot[pivot["method"] == m]
        fig.add_bar(name=m, x=sub["cutoff_frac"].astype(str), y=sub["v"], marker_color=colors[m])
    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title="Cutoff fraction",
        yaxis_title=ylabel,
    )
    if value == "coverage":
        fig.add_hline(y=95, line_dash="dash", line_color="gray", annotation_text="Nominal 95%")
    return fig


# ---------- main -----------------------------------------------------------


def _build_eligible_forms(
    df: pd.DataFrame, shapes: dict[str, str], min_n: int
) -> tuple[list[dict], dict[str, int]]:
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
        eligible.append({"form_id": form_id, "timestamps": ts, "shape": shape})
    return eligible, skipped


def main(
    input_path: Path,
    features_csv: Path,
    output_md: Path,
    figures_dir: Path,
    cutoffs: tuple[float, ...],
    min_n: int,
    limit: int | None,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df = df.drop_duplicates(subset=["FORM_ID", "TIMESTAMP"]).reset_index(drop=True)
    input_hash = _file_sha256_short(input_path)

    shapes = _load_shapes(features_csv)
    eligible, skipped = _build_eligible_forms(df, shapes, min_n)

    if limit:
        eligible = eligible[:limit]

    print(f"Eligible forms: {len(eligible)} (of {df['FORM_ID'].nunique()})")
    print(f"Skipped: {skipped}")
    print(f"Cutoffs: {cutoffs}")

    points: list[ABPoint] = []
    for i, form in enumerate(eligible, 1):
        for cutoff in cutoffs:
            p = _ab_one(form["form_id"], form["timestamps"], form["shape"], cutoff)
            if p is not None:
                points.append(p)
        if i % 25 == 0 or i == len(eligible):
            print(f"  processed {i}/{len(eligible)} forms...")

    points_df = pd.DataFrame([p.__dict__ for p in points])
    points_df.to_csv(figures_dir / "09_ab_points.csv", index=False)

    metrics = _per_method_metrics(points_df)
    metrics.to_csv(figures_dir / "09_ab_metrics.csv", index=False)

    # Global per method
    by_method = _agg(metrics, ["method"]).reindex(list(METHODS))

    # The decisive view: method × cutoff_frac
    by_method_cutoff = _agg(metrics, ["method", "cutoff_frac"])

    # Per-shape on early cutoffs (0.1, 0.2) only
    early = metrics[metrics["cutoff_frac"].isin([0.1, 0.2])]
    by_method_shape_early = _agg(early, ["method", "shape"])

    # Figures
    figs = {
        "mape_per_cutoff": _fig_per_cutoff(
            metrics, "ape", "MAPE p50 (%)", "MAPE p50 за cutoff × method"
        ),
        "coverage_per_cutoff": _fig_per_cutoff(
            metrics, "coverage", "Coverage (%)", "Coverage за cutoff × method"
        ),
    }
    fig_paths = {}
    for name, fig in figs.items():
        path = figures_dir / f"09_{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        fig_paths[name] = path.relative_to(output_md.parent)

    md = _render_markdown(
        n_forms_total=df["FORM_ID"].nunique(),
        n_forms_eligible=len(eligible),
        n_points=len(metrics) // len(METHODS),
        skipped=skipped,
        cutoffs=cutoffs,
        min_n=min_n,
        input_path=input_path,
        input_hash=input_hash,
        by_method=by_method,
        by_method_cutoff=by_method_cutoff,
        by_method_shape_early=by_method_shape_early,
        fig_paths=fig_paths,
    )
    output_md.write_text(md, encoding="utf-8")
    print(f"\nReport written to: {output_md}")


def _render_markdown(
    *,
    n_forms_total: int,
    n_forms_eligible: int,
    n_points: int,
    skipped: dict[str, int],
    cutoffs: tuple[float, ...],
    min_n: int,
    input_path: Path,
    input_hash: str,
    by_method: pd.DataFrame,
    by_method_cutoff: pd.DataFrame,
    by_method_shape_early: pd.DataFrame,
    fig_paths: dict[str, Path],
) -> str:
    return f"""# 09 — A/B/C для ранніх передбачень (cutoffs 0.1, 0.2)

**Generated:** {datetime.now().isoformat(timespec="seconds")}
**Source:** `{input_path}` (sha256:`{input_hash}`)
**Forms:** {n_forms_total} → {n_forms_eligible} eligible (N ≥ {min_n})
**Cutoffs:** {cutoffs} · **Horizon fraction:** {HORIZON_FRACTION}
**Backtest points / method:** {n_points}
**Skipped:** {skipped}

## Методи

- **A · model** — поточний `forecast_responses` (контроль, реплікує 08_).
- **B · naive** — Poisson MLE на (n_train, cutoff_span), exact γ-CI 95% на інкременті.
- **C · blend** — `α·model + (1−α)·naive`, де `α = clip((n_train − {BLEND_LOW_N}) / {BLEND_HIGH_N - BLEND_LOW_N}, 0, 1)`. На n_train={BLEND_LOW_N} → pure naive, на n_train≥{BLEND_HIGH_N} → pure model.

## Глобально (усі cutoffs)

{_df_to_md(by_method)}

## Decisive view: method × cutoff

{_df_to_md(by_method_cutoff)}

## Per-shape на ранніх cutoffs (0.1, 0.2)

{_df_to_md(by_method_shape_early)}

## Фігури

- [MAPE p50 за cutoff × method]({fig_paths["mape_per_cutoff"]})
- [Coverage за cutoff × method]({fig_paths["coverage_per_cutoff"]})

## Як читати

1. **Дивимось рядки cutoff=0.1, 0.2** у method × cutoff таблиці. Очікуємо
   що **C (blend)** має найнижчий MAPE_p50 і coverage найближче до 95%.
   Якщо B (pure naive) теж непогано — це сигнал що curve-fit на малих N
   взагалі не приносить value.
2. **Дивимось рядки cutoff=0.5, 0.7**: усі три методи мають збігатися
   близько до model (бо α→1). Якщо B сильно гірше — підтверджує що
   curve-fit вигідний на зрілих формах.
3. **Per-shape early-cutoff таблиця**: де A>>B (тобто model значно
   гірше за naive) — це shape-категорії, для яких curve-fit
   контрпродуктивний на ранніх етапах.

## Рішення

Promote → `core/forecast/naive.py` + flag у `service.py` ЯКЩО:

- C виграє A за MAPE_p50 на cutoff 0.1, 0.2 **і** не програє на 0.5, 0.7
  (не більше +2pp MAPE).
- C coverage на cutoff 0.1, 0.2 ≥ A coverage (тобто blend не звужує CI
  у поганий бік).

Інакше: задокументувати negative result, шукати інший підхід (можливо
Akaike model averaging замість selection).

## Артефакти

- `figures/09_ab_points.csv` — wide-format raw результати.
- `figures/09_ab_metrics.csv` — long-format per-method метрики.
"""


def _df_to_md(df: pd.DataFrame) -> str:
    df = df.reset_index()
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in headers) + " |",
        "|" + "|".join("---:" if i > 0 else "---" for i in range(len(headers))) + "|",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _file_sha256_short(path: Path, length: int = 12) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    p.add_argument(
        "--input", type=Path, default=repo_root / "data" / "Form Timestamp Collection.csv"
    )
    p.add_argument(
        "--features-csv",
        type=Path,
        default=repo_root / "research" / "reports" / "figures" / "01_per_form_features.csv",
    )
    p.add_argument(
        "--output", type=Path, default=repo_root / "research" / "reports" / "09_early_blend_ab.md"
    )
    p.add_argument(
        "--figures-dir", type=Path, default=repo_root / "research" / "reports" / "figures"
    )
    p.add_argument("--cutoffs", type=str, default=",".join(str(c) for c in CUTOFFS_DEFAULT))
    p.add_argument("--min-n", type=int, default=MIN_N_FOR_BACKTEST)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cutoffs = tuple(float(x) for x in args.cutoffs.split(","))
    main(
        input_path=args.input,
        features_csv=args.features_csv,
        output_md=args.output,
        figures_dir=args.figures_dir,
        cutoffs=cutoffs,
        min_n=args.min_n,
        limit=args.limit,
    )
