# 01 — Dataset Overview

**Generated:** 2026-05-24T03:39:19
**Source:** `D:\GitHub\survey-insight\data\Form Timestamp Collection.csv` (sha256:`0dfa60a9a2cd`)
**Rows:** 27,627 · **Forms:** 177

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

| shape | forms | N min | N median | N max | span median (d) | rate median (/d) | best R² median |
|---|---:|---:|---:|---:|---:|---:|---:|
| **insufficient** | 16 | 1 | 1 | 4 | 0.41 | 13.8 | nan |
| **linear** | 2 | 31 | 56 | 81 | 6.53 | 8.8 | 0.968 |
| **logarithmic** | 83 | 5 | 46 | 7433 | 7.6 | 6.7 | 0.951 |
| **logistic** | 61 | 5 | 34 | 3139 | 7.17 | 5.1 | 0.949 |
| **late_burst** | 9 | 5 | 50 | 351 | 6.7 | 18.0 | 0.928 |
| **ill_fit** | 6 | 8 | 24 | 386 | 5.8 | 5.0 | 0.837 |

Розподіл: insufficient: 16, linear: 2, logarithmic: 83, logistic: 61, late_burst: 9, ill_fit: 6

## Топ-10 форм за N

| form_id (short) | N | span (d) | rate/d | t50 | t90 | auc | best fit (R²) | shape |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `1GM-api8tg1DaVEE_N…` | 7433 | 1.70 | 4383.2 | 0.02 | 0.45 | +0.39 | asymp_exp (0.970) | logarithmic |
| `15EZp01e49mqyyrZfn…` | 3139 | 1300.97 | 2.4 | 0.00 | 0.00 | +0.50 | logistic (0.931) | logistic |
| `13Lbv-OpFHnwMdkzlt…` | 1449 | 19.79 | 73.2 | 0.10 | 0.70 | +0.26 | asymp_exp (0.941) | logarithmic |
| `1qOYJsx8rtZcImp4OD…` | 1292 | 458.72 | 2.8 | 0.13 | 0.97 | +0.10 | logistic (0.866) | logistic |
| `1uBn4wyNG5LlIfdymz…` | 626 | 910.15 | 0.7 | 0.34 | 0.77 | +0.12 | asymp_exp (0.983) | logarithmic |
| `1_zS99FyaYl2eNPl9p…` | 602 | 8.15 | 73.8 | 0.13 | 0.39 | +0.33 | asymp_exp (0.964) | logarithmic |
| `1p0ERtAe-_c4J_EL0H…` | 594 | 4.58 | 129.8 | 0.17 | 0.41 | +0.29 | asymp_exp (0.952) | logarithmic |
| `1ci4V9v25Ifn2qojem…` | 545 | 88.35 | 6.2 | 0.08 | 0.14 | +0.41 | logistic (0.946) | logistic |
| `1IUps2ikeV37yMz9sa…` | 497 | 6.08 | 81.7 | 0.35 | 0.78 | +0.04 | logistic (0.866) | logistic |
| `1Tw_K22VdkSmUGpDY9…` | 447 | 3.20 | 139.7 | 0.71 | 0.75 | -0.14 | logistic (0.905) | logistic |

## Графіки

- [Форми за shape-категоріями](figures\01_shape_counts.html)
- [Гістограма N (log scale)](figures\01_n_histogram.html)
- [span vs rate scatter](figures\01_span_vs_rate.html)
- [t50 vs auc_excess (sanity-чек класифікатора)](figures\01_auc_vs_t50.html)
- [R² distribution per family (boxplot)](figures\01_r2_comparison.html)

## Per-form features

Повний CSV: [`figures\01_per_form_features.csv`](figures\01_per_form_features.csv)

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
