"""17_hybrid_ci_decision.py — post-hoc hybrid CI decision rules.

Читає 16_ab_points.csv (A=prod, B=selector+delta) і обчислює гібридні
методи C/D без додаткового compute. Дає point-by-point рішення, який
з методів використати для практичного прода.

Методи:
- **A · prod** — поточний (NHPP+calibration).
- **B · selector_delta** — з 16_.
- **C · oracle** — теоретичний upper bound: per-point обираємо метод з
  кращим Winkler. Дає максимально-можливий outcome якщо б ми мали
  perfect rule.
- **D1 · rule_min_width** — use B if (b_width ≤ a_width). Парraint-free.
- **D2 · rule_capped** — use B if (b_width ≤ a_width AND b_width ≤ point × 5).
  Sanity cap проти ill-conditioned pcov explosions.
- **D3 · rule_horizon** — use B if horizon ≤ 12h AND model ∈ {asympexp, logistic}.
  Apriori signals тільки. Безпечне.

Compare Winkler / width / coverage на всіх 3710 backtest points.

Запуск:
    .venv/Scripts/python.exe research/benchmarks/17_hybrid_ci_decision.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px


def _winkler(truth: float, lower: float, upper: float, alpha: float = 0.05) -> float:
    width = max(0.0, upper - lower)
    return width + (2.0 / alpha) * max(lower - truth, 0.0) + (2.0 / alpha) * max(truth - upper, 0.0)


def _stable_model(name: str) -> bool:
    return name in {"asymptotic_exp", "logistic"}


def _is_short_horizon(h: float) -> bool:
    return h <= 12.0


def _build_hybrids(points_df: pd.DataFrame) -> pd.DataFrame:
    """Дописуємо C/D методи на основі A/B colonn."""
    df = points_df[
        (points_df["a_point"] >= 0) & (points_df["b_point"] >= 0) & (points_df["truth"] > 0)
    ].copy()

    # Per-point Winkler scores.
    df["a_winkler"] = [
        _winkler(t, lo, hi) for t, lo, hi in zip(df["truth"], df["a_lo"], df["a_hi"], strict=True)
    ]
    df["b_winkler"] = [
        _winkler(t, lo, hi) for t, lo, hi in zip(df["truth"], df["b_lo"], df["b_hi"], strict=True)
    ]
    df["a_width"] = df["a_hi"] - df["a_lo"]
    df["b_width"] = df["b_hi"] - df["b_lo"]
    df["a_cap"] = df["a_lo"] <= df["truth"]
    df["a_cap"] &= df["truth"] <= df["a_hi"]
    df["b_cap"] = df["b_lo"] <= df["truth"]
    df["b_cap"] &= df["truth"] <= df["b_hi"]

    # C: oracle (best Winkler per point).
    def _choose(row, rule):
        if rule == "c_oracle":
            return "b" if row["b_winkler"] < row["a_winkler"] else "a"
        if rule == "d1_min_width":
            return "b" if row["b_width"] <= row["a_width"] else "a"
        if rule == "d2_capped":
            point_est = row["b_point"]
            cap = max(20.0, 5.0 * point_est)
            if row["b_width"] <= row["a_width"] and row["b_width"] <= cap:
                return "b"
            return "a"
        if rule == "d3_horizon":
            if _is_short_horizon(row["horizon_hours"]) and _stable_model(row["selected_model"]):
                return "b"
            return "a"
        return "a"

    for rule in ["c_oracle", "d1_min_width", "d2_capped", "d3_horizon"]:
        choice = df.apply(lambda r, rule=rule: _choose(r, rule), axis=1)
        df[f"{rule}_choice"] = choice
        df[f"{rule}_lo"] = df.apply(
            lambda r, c=choice: (
                r[f"{r[c.name]}_lo"]
                if False
                else (r["b_lo"] if c.loc[r.name] == "b" else r["a_lo"])
            ),
            axis=1,
        )
        df[f"{rule}_hi"] = df.apply(
            lambda r, c=choice: r["b_hi"] if c.loc[r.name] == "b" else r["a_hi"], axis=1
        )
        df[f"{rule}_point"] = df.apply(
            lambda r, c=choice: r["b_point"] if c.loc[r.name] == "b" else r["a_point"], axis=1
        )
        df[f"{rule}_winkler"] = df.apply(
            lambda r, lo=f"{rule}_lo", hi=f"{rule}_hi": _winkler(r["truth"], r[lo], r[hi]), axis=1
        )
        df[f"{rule}_width"] = df[f"{rule}_hi"] - df[f"{rule}_lo"]
        df[f"{rule}_cap"] = (df[f"{rule}_lo"] <= df["truth"]) & (df["truth"] <= df[f"{rule}_hi"])
    return df


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = [
        ("a_prod", "a_winkler", "a_width", "a_cap"),
        ("b_selector_delta", "b_winkler", "b_width", "b_cap"),
        ("c_oracle", "c_oracle_winkler", "c_oracle_width", "c_oracle_cap"),
        ("d1_min_width", "d1_min_width_winkler", "d1_min_width_width", "d1_min_width_cap"),
        ("d2_capped", "d2_capped_winkler", "d2_capped_width", "d2_capped_cap"),
        ("d3_horizon", "d3_horizon_winkler", "d3_horizon_width", "d3_horizon_cap"),
    ]
    for name, w_col, wid_col, cap_col in methods:
        rows.append(
            {
                "method": name,
                "n": len(df),
                "winkler_p50": round(df[w_col].median(), 1),
                "winkler_p90": round(df[w_col].quantile(0.9), 1),
                "winkler_mean": round(df[w_col].mean(), 1),
                "width_p50": round(df[wid_col].median(), 1),
                "width_p90": round(df[wid_col].quantile(0.9), 1),
                "width_p99": round(df[wid_col].quantile(0.99), 1),
                "coverage_pct": round(df[cap_col].mean() * 100, 1),
            }
        )
    return pd.DataFrame(rows)


def _per_horizon(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in sorted(df["horizon_hours"].unique()):
        sub = df[df["horizon_hours"] == h]
        for name, w_col, wid_col, cap_col in [
            ("a_prod", "a_winkler", "a_width", "a_cap"),
            ("b_selector_delta", "b_winkler", "b_width", "b_cap"),
            ("c_oracle", "c_oracle_winkler", "c_oracle_width", "c_oracle_cap"),
            ("d1_min_width", "d1_min_width_winkler", "d1_min_width_width", "d1_min_width_cap"),
            ("d2_capped", "d2_capped_winkler", "d2_capped_width", "d2_capped_cap"),
            ("d3_horizon", "d3_horizon_winkler", "d3_horizon_width", "d3_horizon_cap"),
        ]:
            rows.append(
                {
                    "horizon_h": h,
                    "method": name,
                    "n": len(sub),
                    "winkler_p50": round(sub[w_col].median(), 1),
                    "width_p50": round(sub[wid_col].median(), 1),
                    "width_p90": round(sub[wid_col].quantile(0.9), 1),
                    "coverage_pct": round(sub[cap_col].mean() * 100, 1),
                }
            )
    return pd.DataFrame(rows)


def _per_type(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ft in sorted(df["form_type"].unique()):
        sub = df[df["form_type"] == ft]
        for name, w_col, wid_col, cap_col in [
            ("a_prod", "a_winkler", "a_width", "a_cap"),
            ("b_selector_delta", "b_winkler", "b_width", "b_cap"),
            ("d2_capped", "d2_capped_winkler", "d2_capped_width", "d2_capped_cap"),
        ]:
            rows.append(
                {
                    "form_type": ft,
                    "method": name,
                    "n": len(sub),
                    "winkler_p50": round(sub[w_col].median(), 1),
                    "width_p50": round(sub[wid_col].median(), 1),
                    "coverage_pct": round(sub[cap_col].mean() * 100, 1),
                }
            )
    return pd.DataFrame(rows)


def _choice_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-rule: яка частка вибрала B vs A?"""
    rows = []
    for rule in ["c_oracle", "d1_min_width", "d2_capped", "d3_horizon"]:
        chose_b = (df[f"{rule}_choice"] == "b").sum()
        rows.append(
            {
                "rule": rule,
                "chose_b_pct": round(chose_b / len(df) * 100, 1),
                "n": len(df),
            }
        )
    return pd.DataFrame(rows)


def main(points_csv: Path, output_md: Path, figures_dir: Path) -> None:
    df = pd.read_csv(points_csv)
    hybrid_df = _build_hybrids(df)
    summary = _summary(hybrid_df)
    per_horizon = _per_horizon(hybrid_df)
    per_type = _per_type(hybrid_df)
    choice_stats = _choice_stats(hybrid_df)

    figures_dir.mkdir(parents=True, exist_ok=True)
    hybrid_df.to_csv(figures_dir / "17_hybrid_points.csv", index=False)

    # Fig: winkler per method per horizon.
    fig = px.line(
        per_horizon,
        x="horizon_h",
        y="winkler_p50",
        color="method",
        markers=True,
        log_x=True,
        title="Median Winkler per horizon — все 6 методів",
        labels={"winkler_p50": "Winkler p50", "horizon_h": "Horizon (h)"},
    )
    path = figures_dir / "17_winkler_per_horizon.html"
    fig.write_html(path, include_plotlyjs="cdn")
    fig_rel = path.relative_to(output_md.parent)

    fig2 = px.line(
        per_horizon,
        x="horizon_h",
        y="width_p50",
        color="method",
        markers=True,
        log_x=True,
        title="Median CI width per horizon",
        labels={"width_p50": "Median width", "horizon_h": "Horizon (h)"},
    )
    path2 = figures_dir / "17_width_per_horizon.html"
    fig2.write_html(path2, include_plotlyjs="cdn")
    fig_rel2 = path2.relative_to(output_md.parent)

    md = f"""# 17 - Hybrid CI decision rules (post-hoc analysis)

**Generated:** {pd.Timestamp.now().isoformat(timespec="seconds")}
**Source:** 16_ab_points.csv ({len(df)} backtest points, {len(hybrid_df)} valid for hybrid)

## Методи

- **A · prod** — поточний (NHPP + P7 + P10 + P11).
- **B · selector_delta** — selector обирає модель, CI через delta-method.
- **C · oracle** — per-point вибір кращого Winkler (theoretical max).
- **D1 · rule_min_width** — use B if b_width <= a_width. Parameter-free.
- **D2 · rule_capped** — D1 + sanity cap: b_width <= max(20, 5 * point).
- **D3 · rule_horizon** — use B only if horizon <= 12h AND model in {{asympexp, logistic}}.

## Global summary

{summary.to_csv(index=False, sep="|")}

## Choice rates (яка частка пунктів обрала B)

{choice_stats.to_csv(index=False, sep="|")}

## Per horizon

{per_horizon.to_csv(index=False, sep="|")}

## Per form_type (тільки A vs B vs D2)

{per_type.to_csv(index=False, sep="|")}

## Figures

- [Winkler per horizon — все методи]({fig_rel})
- [Width per horizon]({fig_rel2})

## Інтерпретація

1. **C oracle** — upper bound. Якщо D-rules близькі — вони добре aproximate
   ідеальне рішення.
2. **D2 capped** — найбільш прод-realistic кандидат. Не вимагає знання
   winkler. Прозоре правило: "якщо delta-CI вузьке і не вибухає — використати".
3. **D3 horizon-based** — найкондервативніше, легко зрозуміти і пояснити.

Якщо D2 winkler_p50 ≤ A winkler_p50 І width значно менший → це winning
prod-кандидат. Промочуємо як P12.
"""
    output_md.write_text(md, encoding="utf-8")
    print(f"Report: {output_md}")
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nPer horizon:")
    print(per_horizon.to_string(index=False))
    print("\nChoice rates:")
    print(choice_stats.to_string(index=False))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    p.add_argument(
        "--points-csv",
        type=Path,
        default=repo_root / "research" / "reports" / "figures" / "16_ab_points.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=repo_root / "research" / "reports" / "17_hybrid_ci_decision.md",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=repo_root / "research" / "reports" / "figures",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.points_csv, args.output, args.figures_dir)
