# 02 — Rolling-Origin Backtest

**Generated:** 2026-05-24T04:15:37
**Source:** `D:\GitHub\survey-insight\data\Form Timestamp Collection.csv` (sha256:`0dfa60a9a2cd`)
**Forms:** 177 total → 96 eligible (N ≥ 30)
**Cutoffs:** (0.3, 0.5, 0.7) · **Horizon fraction:** 0.25
**Backtest points:** 288

## Глобальні метрики

| Метрика | Значення | Орієнтир |
|---|---:|---|
| **MAPE (median)** | 25.6% | < 15% — добре, < 25% — прийнятно |
| **Coverage 95% PI** | 47.6% | Має бути ≈ 95% |
| **Sharpness (median)** | 0.27 | width / truth |
| **Bias (median)** | -15.1% | 0% — unbiased |

## По shape-категоріях

| shape | n_points | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|
| ill_fit | 6 | 38.9 | 115.8 | 16.7 | 0.07 | -29.3 |
| late_burst | 18 | 31.0 | 69.8 | 33.3 | 0.2 | -24.8 |
| linear | 6 | 19.7 | 32.4 | 50.0 | 0.18 | -19.7 |
| logarithmic | 147 | 18.5 | 175.3 | 54.4 | 0.45 | -8.3 |
| logistic | 111 | 29.5 | 69.0 | 42.3 | 0.21 | -23.8 |

## По N-buckets

| n_bucket | n_points | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|
| <15 | 31 | 48.3 | 166.7 | 58.1 | 0.9 | -10.5 |
| 15-30 | 76 | 23.6 | 50.0 | 57.9 | 0.35 | -18.5 |
| 30-100 | 107 | 25.4 | 142.6 | 41.1 | 0.19 | -15.6 |
| 100-1k | 68 | 23.0 | 286.5 | 45.6 | 0.2 | -8.4 |
| 1k+ | 6 | 18.9 | 45.8 | 0.0 | 32.09 | -5.2 |

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
