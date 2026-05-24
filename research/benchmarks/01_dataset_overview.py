"""01_dataset_overview.py — per-form features + shape classifier (fit-based).

Читає сирий timestamp-датасет, рахує per-form характеристики (включно з
auc_excess і fits трьох параметричних моделей), класифікує форму за
shape-категорією, будує гістограми. Output — markdown report у
research/reports/.

Класифікатор будується на двох рівнях:

1. **Descriptive features** (model-agnostic):
   - t50, t90  — фракції span'у до досягнення 50%/90% cumulative
   - auc_excess — середнє відхилення нормованого cumulative від
     лінійної інтерполяції ∈ [-0.5, +0.5]:
       +0.5 → миттєвий старт (concave, fast-then-slow)
       -0.5 → мить наприкінці (convex, slow-then-fast)
        0   → лінійний

2. **Best-fit family** (academic shape assignment):
   Фітимо три криві (linear, asymp_exp, logistic), порівнюємо R².
   Переможець за R² → форма «реально» описується цим класом.
   Якщо best_r2 < 0.85, форма "ill-fit" (потребує іншої моделі —
   мульти-хвильовий, Bass, тощо).

Категорії:
   insufficient   — N < 5
   linear         — best fit = linear (~ const rate)
   logarithmic    — best fit = asymp_exp (concave saturating)
   logistic       — best fit = logistic (S-shape)
   late_burst     — auc_excess < -0.20 (convex, агітація під дедлайн)
   ill_fit        — найкращий R² < 0.85 (multi-wave / нестандартна форма)

Запуск:
    .venv/Scripts/python.exe research/benchmarks/01_dataset_overview.py
"""

from __future__ import annotations

import argparse
import hashlib
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import OptimizeWarning, curve_fit

warnings.simplefilter("ignore", OptimizeWarning)


# ---------- параметричні моделі для shape-діагностики ---------------------


def _linear(t, a, b):
    return a * t + b


def _asymp_exp(t, a, b, c):
    return a * (1.0 - np.exp(-b * t)) + c


def _logistic(t, k, r, t0):
    return k / (1.0 + np.exp(-r * (t - t0)))


def _r2(y_actual, y_pred):
    ss_res = float(np.sum((y_actual - y_pred) ** 2))
    ss_tot = float(np.sum((y_actual - y_actual.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _fit_r2(model_fn, t, y, p0, bounds):
    try:
        popt, _ = curve_fit(model_fn, t, y, p0=p0, bounds=bounds, maxfev=500)
        return _r2(y, model_fn(t, *popt))
    except (RuntimeError, ValueError):
        return float("-inf")


def _fit_three(t, y):
    """Повертає (r2_linear, r2_asymp_exp, r2_logistic) для cumulative-кривої."""
    last = float(y[-1])
    duration = float(t[-1] - t[0]) if t[-1] > t[0] else 1.0

    r2_lin = _fit_r2(
        _linear,
        t,
        y,
        p0=(last / max(duration, 1e-9), 0.0),
        bounds=((-np.inf, -np.inf), (np.inf, np.inf)),
    )
    r2_aexp = _fit_r2(
        _asymp_exp,
        t,
        y,
        p0=(last, 1.0 / max(duration, 1e-9), 0.0),
        bounds=((0.0, 1e-6, 0.0), (np.inf, np.inf, np.inf)),
    )
    r2_log = _fit_r2(
        _logistic,
        t,
        y,
        p0=(last * 1.2, 1.0 / max(duration, 1e-9), duration / 2.0),
        bounds=((last * 0.5, 1e-6, -np.inf), (last * 20.0, np.inf, np.inf)),
    )
    return r2_lin, r2_aexp, r2_log


# ---------- per-form features ----------------------------------------------


@dataclass(frozen=True)
class FormFeatures:
    form_id: str
    n: int
    first_ts: pd.Timestamp
    last_ts: pd.Timestamp
    span_days: float
    rate_per_day: float
    t50_frac: float
    t90_frac: float
    auc_excess: float
    r2_linear: float
    r2_asymp_exp: float
    r2_logistic: float
    best_fit: str  # "linear" | "asymp_exp" | "logistic"
    best_r2: float
    shape: str


def _compute_features(form_id: str, timestamps: pd.Series) -> FormFeatures:
    n = len(timestamps)
    ts_sorted = timestamps.sort_values().reset_index(drop=True)
    first_ts = ts_sorted.iloc[0]
    last_ts = ts_sorted.iloc[-1]
    span_seconds = (last_ts - first_ts).total_seconds()
    span_days = span_seconds / 86400.0
    rate_per_day = n / max(span_days, 1.0 / 24.0)

    if n < 5 or span_seconds <= 0:
        return FormFeatures(
            form_id=form_id,
            n=n,
            first_ts=first_ts,
            last_ts=last_ts,
            span_days=span_days,
            rate_per_day=rate_per_day,
            t50_frac=float("nan"),
            t90_frac=float("nan"),
            auc_excess=float("nan"),
            r2_linear=float("nan"),
            r2_asymp_exp=float("nan"),
            r2_logistic=float("nan"),
            best_fit="—",
            best_r2=float("nan"),
            shape="insufficient",
        )

    t_frac = (ts_sorted - first_ts).dt.total_seconds().to_numpy() / span_seconds
    y_cum = np.arange(1, n + 1, dtype=float)
    y_norm = (y_cum - y_cum[0]) / (y_cum[-1] - y_cum[0])

    t50_frac = float(np.interp(0.5 * n, y_cum, t_frac))
    t90_frac = float(np.interp(0.9 * n, y_cum, t_frac))
    auc_excess = float(np.mean(y_norm - t_frac))  # signed area between curve and diag

    # Curve fits use t у днях (не нормованих), щоб параметри інтерпретувались
    t_days = (ts_sorted - first_ts).dt.total_seconds().to_numpy() / 86400.0
    r2_lin, r2_aexp, r2_log = _fit_three(t_days, y_cum)

    r2_map = {"linear": r2_lin, "asymp_exp": r2_aexp, "logistic": r2_log}
    best_fit = max(r2_map, key=r2_map.get)
    best_r2 = r2_map[best_fit]

    shape = _classify(best_r2, best_fit, auc_excess)

    return FormFeatures(
        form_id=form_id,
        n=n,
        first_ts=first_ts,
        last_ts=last_ts,
        span_days=span_days,
        rate_per_day=rate_per_day,
        t50_frac=t50_frac,
        t90_frac=t90_frac,
        auc_excess=auc_excess,
        r2_linear=r2_lin,
        r2_asymp_exp=r2_aexp,
        r2_logistic=r2_log,
        best_fit=best_fit,
        best_r2=best_r2,
        shape=shape,
    )


def _classify(best_r2: float, best_fit: str, auc_excess: float) -> str:
    """Class по best-fit + auc_excess override для late-burst.

    Late-burst (auc_excess < -0.20) — це convex форма, де навіть якщо
    logistic дає високий R², зміст інший: агітація під дедлайн.
    """
    if best_r2 < 0.85:
        return "ill_fit"
    if auc_excess < -0.20:
        return "late_burst"
    if best_fit == "linear":
        return "linear"
    if best_fit == "asymp_exp":
        return "logarithmic"
    if best_fit == "logistic":
        return "logistic"
    return "ill_fit"


# ---------- report generation ----------------------------------------------


_SHAPE_ORDER = ["insufficient", "linear", "logarithmic", "logistic", "late_burst", "ill_fit"]


def _md_table_summary(features: list[FormFeatures]) -> str:
    df = pd.DataFrame([f.__dict__ for f in features])
    grouped = df.groupby("shape").agg(
        forms=("form_id", "count"),
        n_min=("n", "min"),
        n_median=("n", "median"),
        n_max=("n", "max"),
        span_median=("span_days", "median"),
        rate_median=("rate_per_day", "median"),
        best_r2_median=("best_r2", "median"),
    )
    grouped = grouped.reindex(_SHAPE_ORDER, fill_value=0).dropna(how="all")
    grouped["n_median"] = grouped["n_median"].astype(int)
    grouped["span_median"] = grouped["span_median"].round(2)
    grouped["rate_median"] = grouped["rate_median"].round(1)
    grouped["best_r2_median"] = grouped["best_r2_median"].round(3)

    lines = [
        "| shape | forms | N min | N median | N max | span median (d) | rate median (/d) | best R² median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for shape, row in grouped.iterrows():
        lines.append(
            f"| **{shape}** | {int(row.forms)} | {int(row.n_min)} | {int(row.n_median)} | "
            f"{int(row.n_max)} | {row.span_median} | {row.rate_median} | {row.best_r2_median} |"
        )
    return "\n".join(lines)


def _figure_shape_counts(features: list[FormFeatures]) -> go.Figure:
    df = pd.DataFrame([{"shape": f.shape} for f in features])
    counts = df["shape"].value_counts().reindex(_SHAPE_ORDER, fill_value=0)
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        labels={"x": "Shape category", "y": "Number of forms"},
        title="Forms by shape category",
        text=counts.values,
    )
    fig.update_traces(textposition="outside")
    return fig


def _figure_n_histogram(features: list[FormFeatures]) -> go.Figure:
    df = pd.DataFrame([{"shape": f.shape, "n": f.n} for f in features])
    fig = px.histogram(
        df,
        x="n",
        color="shape",
        log_x=True,
        nbins=30,
        title="Distribution of responses-per-form (log scale)",
        labels={"n": "N responses (log)"},
    )
    fig.update_layout(barmode="overlay")
    fig.update_traces(opacity=0.7)
    return fig


def _figure_span_vs_rate(features: list[FormFeatures]) -> go.Figure:
    df = pd.DataFrame([f.__dict__ for f in features])
    df = df[df["shape"] != "insufficient"]
    fig = px.scatter(
        df,
        x="span_days",
        y="rate_per_day",
        color="shape",
        size="n",
        log_x=True,
        log_y=True,
        hover_data=["form_id", "n", "t50_frac", "t90_frac", "auc_excess", "best_r2"],
        title="span_days vs rate_per_day (size = N responses)",
    )
    return fig


def _figure_auc_vs_t50(features: list[FormFeatures]) -> go.Figure:
    df = pd.DataFrame([f.__dict__ for f in features])
    df = df[df["shape"] != "insufficient"]
    fig = px.scatter(
        df,
        x="t50_frac",
        y="auc_excess",
        color="shape",
        hover_data=["form_id", "n", "span_days", "t90_frac", "best_fit", "best_r2"],
        title="t50 vs auc_excess (виявляє форму кривої)",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_hline(y=-0.20, line_dash="dash", line_color="red",
                  annotation_text="late_burst threshold")
    return fig


def _figure_r2_comparison(features: list[FormFeatures]) -> go.Figure:
    df = pd.DataFrame([f.__dict__ for f in features])
    df = df[df["shape"] != "insufficient"]
    fig = go.Figure()
    for col, name in [("r2_linear", "linear"),
                       ("r2_asymp_exp", "asymp_exp"),
                       ("r2_logistic", "logistic")]:
        fig.add_trace(go.Box(y=df[col].clip(lower=0), name=name, boxmean=True))
    fig.update_layout(
        title="R² distribution per parametric family (clipped at 0)",
        yaxis_title="R²",
    )
    return fig


# ---------- main -----------------------------------------------------------


def main(input_path: Path, output_md: Path, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

    input_hash = _file_sha256_short(input_path)
    n_rows = len(df)
    n_forms = df["FORM_ID"].nunique()

    features: list[FormFeatures] = []
    for form_id, group in df.groupby("FORM_ID"):
        features.append(_compute_features(form_id, group["TIMESTAMP"]))

    features.sort(key=lambda f: f.n, reverse=True)

    figs = {
        "shape_counts": _figure_shape_counts(features),
        "n_histogram": _figure_n_histogram(features),
        "span_vs_rate": _figure_span_vs_rate(features),
        "auc_vs_t50": _figure_auc_vs_t50(features),
        "r2_comparison": _figure_r2_comparison(features),
    }
    fig_paths: dict[str, Path] = {}
    for name, fig in figs.items():
        path = figures_dir / f"01_{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        fig_paths[name] = path.relative_to(output_md.parent)

    per_form_csv = figures_dir / "01_per_form_features.csv"
    pd.DataFrame([f.__dict__ for f in features]).to_csv(per_form_csv, index=False)

    md = _render_markdown(
        n_rows=n_rows,
        n_forms=n_forms,
        input_path=input_path,
        input_hash=input_hash,
        features=features,
        fig_paths=fig_paths,
        per_form_csv=per_form_csv.relative_to(output_md.parent),
    )
    output_md.write_text(md, encoding="utf-8")
    print(f"Report written to: {output_md}")
    print(f"Figures dir: {figures_dir}")


def _render_markdown(
    *,
    n_rows: int,
    n_forms: int,
    input_path: Path,
    input_hash: str,
    features: list[FormFeatures],
    fig_paths: dict[str, Path],
    per_form_csv: Path,
) -> str:
    summary_table = _md_table_summary(features)
    n_dist = ", ".join(
        f"{shape}: {sum(1 for f in features if f.shape == shape)}"
        for shape in _SHAPE_ORDER
    )

    top10 = sorted(features, key=lambda f: f.n, reverse=True)[:10]
    top10_table = [
        "| form_id (short) | N | span (d) | rate/d | t50 | t90 | auc | best fit (R²) | shape |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for f in top10:
        short = f.form_id[:18] + "…"
        if f.shape == "insufficient":
            top10_table.append(
                f"| `{short}` | {f.n} | {f.span_days:.2f} | {f.rate_per_day:.1f} | "
                f"— | — | — | — | {f.shape} |"
            )
        else:
            top10_table.append(
                f"| `{short}` | {f.n} | {f.span_days:.2f} | {f.rate_per_day:.1f} | "
                f"{f.t50_frac:.2f} | {f.t90_frac:.2f} | {f.auc_excess:+.2f} | "
                f"{f.best_fit} ({f.best_r2:.3f}) | {f.shape} |"
            )

    return f"""# 01 — Dataset Overview

**Generated:** {datetime.now().isoformat(timespec="seconds")}
**Source:** `{input_path}` (sha256:`{input_hash}`)
**Rows:** {n_rows:,} · **Forms:** {n_forms}

## Shape classifier

Двошарова процедура:

1. **Descriptive features** (model-agnostic):
   - `t50`, `t90` — фракції span'у до 50%/90% cumulative
   - `auc_excess` ∈ [-0.5, +0.5] — середнє відхилення нормованого
     cumulative від діагоналі лінійної інтерполяції; знак показує
     concave (+) чи convex (−)

2. **Best-fit family**: фітимо linear, asymp_exp, logistic; обираємо
   модель з найвищим R².

Категорії:
```
N < 5                          → insufficient
best R² < 0.85                 → ill_fit (multi-wave / нестандарт)
auc_excess < -0.20             → late_burst (convex, агітація під дедлайн)
best_fit == "linear"           → linear
best_fit == "asymp_exp"        → logarithmic
best_fit == "logistic"         → logistic
```

## Розподіл по категоріях

{summary_table}

Розподіл: {n_dist}

## Топ-10 форм за N

{chr(10).join(top10_table)}

## Графіки

- [Форми за shape-категоріями]({fig_paths["shape_counts"]})
- [Гістограма N (log scale)]({fig_paths["n_histogram"]})
- [span vs rate scatter]({fig_paths["span_vs_rate"]})
- [t50 vs auc_excess (sanity-чек класифікатора)]({fig_paths["auc_vs_t50"]})
- [R² distribution per family (boxplot)]({fig_paths["r2_comparison"]})

## Per-form features

Повний CSV: [`{per_form_csv}`]({per_form_csv})

## Що означає для downstream

- **insufficient** (N<5): пропускаємо в backtest'і — нижче MIN_TRAIN_POINTS.
- **linear**: модельний пакет (logistic / Gompertz / asymp_exp) може
  систематично передбачати плато, якого немає. Кандидати на додавання
  `LinearModel` (1 параметр) у селектор.
- **logarithmic**: цільова аудиторія `AsymptoticExpModel` — має домінувати
  у empirical model selection (03_model_selection_empirical).
- **logistic**: `LogisticModel` має виграти AICc.
- **late_burst**: convex форма; жодна з трьох моделей не описує точно
  (вони всі concave/saturating). Кандидати на нову модель: power-law
  з positive curvature, або Bass diffusion. Очікуємо найгірший forecast
  на цьому класі.
- **ill_fit**: best R² < 0.85 — multi-wave або нестандарт. Може допомогти
  changepoint detection + per-segment fit.
"""


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
        "--input",
        type=Path,
        default=repo_root / "data" / "Form Timestamp Collection.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=repo_root / "research" / "reports" / "01_dataset_overview.md",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=repo_root / "research" / "reports" / "figures",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.input, args.output, args.figures_dir)
