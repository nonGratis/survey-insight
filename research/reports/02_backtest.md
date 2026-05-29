# 02 — Rolling-Origin Backtest

**Generated:** 2026-05-29T13:22:53
**Source:** `D:\GitHub\survey-insight\data\Form Timestamp Collection.csv` (sha256:`0dfa60a9a2cd`)
**Forms:** 177 total → 96 eligible (N ≥ 30)
**Cutoffs:** (0.3, 0.5, 0.7) · **Horizon fraction:** 0.25
**Backtest points:** 288

## Глобальні метрики

| Метрика | Значення | Орієнтир |
|---|---:|---|
| **MAPE (median)** | 26.2% | < 15% — добре, < 25% — прийнятно |
| **Coverage 95% PI** | 30.9% | Має бути ≈ 95% |
| **Sharpness (median)** | 0.19 | width / truth |
| **Bias (median)** | +2.0% | 0% — unbiased |

## По shape-категоріях

| shape | n_points | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|
| ill_fit | 6 | 35.5 | 179.5 | 0.0 | 0.08 | -20.8 |
| late_burst | 18 | 23.7 | 78.4 | 27.8 | 0.31 | -18.0 |
| linear | 6 | 13.1 | 22.7 | 66.7 | 0.23 | -13.1 |
| logarithmic | 147 | 33.9 | 176.6 | 29.3 | 0.15 | 27.7 |
| logistic | 111 | 23.0 | 80.7 | 33.3 | 0.22 | -5.1 |

## По N-buckets

| n_bucket | n_points | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|
| <15 | 31 | 75.0 | 309.1 | 29.0 | 0.55 | 75.0 |
| 15-30 | 76 | 24.9 | 211.0 | 42.1 | 0.37 | 3.3 |
| 30-100 | 107 | 25.4 | 120.8 | 30.8 | 0.15 | -2.9 |
| 100-1k | 68 | 19.4 | 131.8 | 20.6 | 0.1 | 5.8 |
| 1k+ | 6 | 14.9 | 40.1 | 16.7 | 0.0 | -0.1 |

## Failure modes

| Mode | Опис |
|---|---|
| `in_ci` | Truth у [ci_lower, ci_upper] — успіх |
| `overconfident_low` | Truth > ci_upper (модель занизила) |
| `overconfident_high` | Truth < ci_lower (модель завищила) |

## Графіки

- [MAPE distribution by shape (boxplot)](figures\02_mape_box.html)
- [Empirical coverage vs nominal 95%](figures\02_coverage_bar.html)
- [Reliability diagram](figures\02_reliability.html)
- [Failure modes by shape](figures\02_failure_modes.html)
- [Bias vs N_train](figures\02_bias_by_n.html)

## Що з цього випливає

Дивись числа вище:

1. **Глобальний coverage**: якщо < 90% → CI занадто вузький
   (треба збільшити n_sims або врахувати додаткове джерело варіансу).
   Якщо > 98% → занадто широкий, sharpness страждає.

2. **Найгірший shape за MAPE**: кандидат на нову модель або
   shape-specific selector.

3. **Bias за N-bucket**: якщо bias систематично negative для малих N
   → модель консервативна на старті (засипана target-prior'ом?).
   Якщо positive — оптимістична. Це підказує напрямок calibration'у.

4. **Failure modes**: переважання `overconfident_low` означає, що CI
   треба зсунути вгору (або просто розширити). `overconfident_high` —
   зсунути вниз.

## Артефакти

- `figures/02_backtest_points.csv` — повний log усіх backtest-runs
- `figures/02_backtest_metrics.csv` — обчислені метрики
