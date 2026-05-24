# 02 — Rolling-Origin Backtest

**Generated:** 2026-05-24T08:02:53
**Source:** `D:\GitHub\survey-insight\data\Form Timestamp Collection.csv` (sha256:`0dfa60a9a2cd`)
**Forms:** 177 total → 96 eligible (N ≥ 30)
**Cutoffs:** (0.3, 0.5, 0.7) · **Horizon fraction:** 0.25
**Backtest points:** 288

## Глобальні метрики

| Метрика | Значення | Орієнтир |
|---|---:|---|
| **MAPE (median)** | 22.7% | < 15% — добре, < 25% — прийнятно |
| **Coverage 95% PI** | 30.6% | Має бути ≈ 95% |
| **Sharpness (median)** | 0.19 | width / truth |
| **Bias (median)** | -7.8% | 0% — unbiased |

## По shape-категоріях

| shape | n_points | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|
| ill_fit | 6 | 38.7 | 115.8 | 0.0 | 0.08 | -27.4 |
| late_burst | 18 | 30.7 | 68.6 | 27.8 | 0.31 | -24.8 |
| linear | 6 | 18.0 | 29.8 | 66.7 | 0.23 | -18.0 |
| logarithmic | 147 | 19.5 | 139.8 | 29.3 | 0.15 | 0.0 |
| logistic | 111 | 25.0 | 59.8 | 32.4 | 0.22 | -16.7 |

## По N-buckets

| n_bucket | n_points | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|
| <15 | 31 | 40.9 | 100.0 | 29.0 | 0.55 | -9.5 |
| 15-30 | 76 | 22.4 | 80.0 | 42.1 | 0.37 | -15.2 |
| 30-100 | 107 | 22.9 | 83.1 | 30.8 | 0.15 | -8.6 |
| 100-1k | 68 | 20.3 | 131.8 | 20.6 | 0.1 | 0.2 |
| 1k+ | 6 | 19.0 | 49.3 | 0.0 | 0.0 | -14.1 |

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
