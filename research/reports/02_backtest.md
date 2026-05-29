# 02 — Rolling-Origin Backtest

**Generated:** 2026-05-29T13:29:55
**Source:** `D:\GitHub\survey-insight\data\Form Timestamp Collection.csv` (sha256:`0dfa60a9a2cd`)
**Forms:** 177 total → 96 eligible (N ≥ 30)
**Cutoffs:** (0.3, 0.5, 0.7) · **Horizon fraction:** 0.25
**Backtest points:** 288

## Глобальні метрики

| Метрика | Значення | Орієнтир |
|---|---:|---|
| **MAPE (median)** | 25.4% | < 15% — добре, < 25% — прийнятно |
| **Coverage 95% PI** | 81.9% | Має бути ≈ 95% |
| **Sharpness (median)** | 1.43 | width / truth |
| **Bias (median)** | -2.1% | 0% — unbiased |

## По shape-категоріях

| shape | n_points | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|
| ill_fit | 6 | 33.9 | 179.5 | 83.3 | 0.72 | -22.0 |
| late_burst | 18 | 22.0 | 63.4 | 94.4 | 1.34 | -16.8 |
| linear | 6 | 13.1 | 20.0 | 100.0 | 1.69 | -13.1 |
| logarithmic | 147 | 33.9 | 203.8 | 76.9 | 1.33 | 15.4 |
| logistic | 111 | 20.0 | 59.8 | 85.6 | 1.5 | -5.9 |

## По N-buckets

| n_bucket | n_points | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|
| <15 | 31 | 59.1 | 316.7 | 90.3 | 2.17 | 0.0 |
| 15-30 | 76 | 22.7 | 204.8 | 86.8 | 2.18 | 4.8 |
| 30-100 | 107 | 22.0 | 105.7 | 79.4 | 1.26 | -5.1 |
| 100-1k | 68 | 19.6 | 142.2 | 79.4 | 1.04 | 5.0 |
| 1k+ | 6 | 19.0 | 47.4 | 50.0 | 0.35 | -4.0 |

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
