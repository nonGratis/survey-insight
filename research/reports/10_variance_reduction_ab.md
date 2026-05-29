# 10 — Variance-reduction A/B/C/D

**Generated:** 2026-05-29T19:17:37
**Source:** `D:\GitHub\survey-insight\data\Form Timestamp Collection.csv` (sha256:`0dfa60a9a2cd`)
**Forms:** 177 → 141 eligible (N ≥ 10)
**Cutoffs:** (0.1, 0.2, 0.3, 0.5, 0.7) · **Horizon fraction:** 0.25
**Backtest points / method:** 590
**Skipped:** {'too_few': 36, 'insufficient_shape': 0, 'no_span': 0}

## Методи

- **A · baseline** — поточний `forecast_responses` (контроль).
- **B · averaged** — Akaike model averaging (refit усі 3, weight `exp(-ΔAICc/2)`).
- **C · scaled** — A + CI scale `×1.5` при n_train≤15, лінійно до ×1.0 на n_train=30.
- **D · avg_scaled** — B + той самий CI scale.

## Глобально (усі cutoffs)

| method | n | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 590 | 29.4 | 141.2 | 87.3 | 3.2 | -10.5 |
| averaged | 590 | 29.2 | 139.6 | 87.3 | 3.18 | -11.1 |
| scaled | 590 | 29.4 | 141.2 | 89.2 | 3.59 | -10.5 |
| avg_scaled | 590 | 29.2 | 139.6 | 88.6 | 3.59 | -11.1 |

## Decisive view: method × cutoff

| method | cutoff_frac | n | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| averaged | 0.1 | 74 | 47.5 | 99.2 | 79.7 | 1.84 | -13.9 |
| averaged | 0.2 | 106 | 40.1 | 138.8 | 84.9 | 3.63 | -4.2 |
| averaged | 0.3 | 128 | 38.7 | 179.3 | 87.5 | 3.83 | -6.3 |
| averaged | 0.5 | 141 | 29.1 | 110.0 | 89.4 | 3.28 | -15.0 |
| averaged | 0.7 | 141 | 17.0 | 33.3 | 90.8 | 2.88 | -11.1 |
| avg_scaled | 0.1 | 74 | 47.5 | 99.2 | 82.4 | 2.03 | -13.9 |
| avg_scaled | 0.2 | 106 | 40.1 | 138.8 | 86.8 | 4.0 | -4.2 |
| avg_scaled | 0.3 | 128 | 38.7 | 179.3 | 89.1 | 4.89 | -6.3 |
| avg_scaled | 0.5 | 141 | 29.1 | 110.0 | 90.8 | 3.86 | -15.0 |
| avg_scaled | 0.7 | 141 | 17.0 | 33.3 | 90.8 | 3.22 | -11.1 |
| baseline | 0.1 | 74 | 46.9 | 99.2 | 85.1 | 1.92 | -7.7 |
| baseline | 0.2 | 106 | 40.1 | 149.7 | 84.9 | 3.63 | -4.2 |
| baseline | 0.3 | 128 | 39.0 | 180.1 | 87.5 | 3.83 | -6.3 |
| baseline | 0.5 | 141 | 29.1 | 118.8 | 87.2 | 3.29 | -15.0 |
| baseline | 0.7 | 141 | 17.6 | 33.3 | 90.1 | 2.79 | -11.1 |
| scaled | 0.1 | 74 | 46.9 | 99.2 | 87.8 | 2.03 | -7.7 |
| scaled | 0.2 | 106 | 40.1 | 149.7 | 87.7 | 4.0 | -4.2 |
| scaled | 0.3 | 128 | 39.0 | 180.1 | 89.1 | 4.71 | -6.3 |
| scaled | 0.5 | 141 | 29.1 | 118.8 | 90.1 | 3.77 | -15.0 |
| scaled | 0.7 | 141 | 17.6 | 33.3 | 90.1 | 3.22 | -11.1 |

## Per-shape на ранніх cutoffs (0.1, 0.2)

| method | shape | n | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| averaged | ill_fit | 4 | 111.7 | 297.9 | 75.0 | 7.27 | 111.7 |
| averaged | late_burst | 11 | 38.5 | 80.5 | 72.7 | 1.71 | -38.5 |
| averaged | linear | 3 | 45.5 | 57.1 | 66.7 | 4.45 | -32.0 |
| averaged | logarithmic | 96 | 43.2 | 100.0 | 88.5 | 2.21 | -1.1 |
| averaged | logistic | 66 | 41.7 | 130.8 | 77.3 | 3.83 | -9.5 |
| avg_scaled | ill_fit | 4 | 111.7 | 297.9 | 75.0 | 7.36 | 111.7 |
| avg_scaled | late_burst | 11 | 38.5 | 80.5 | 90.9 | 2.44 | -38.5 |
| avg_scaled | linear | 3 | 45.5 | 57.1 | 66.7 | 6.7 | -32.0 |
| avg_scaled | logarithmic | 96 | 43.2 | 100.0 | 89.6 | 2.59 | -1.1 |
| avg_scaled | logistic | 66 | 41.7 | 130.8 | 78.8 | 4.73 | -9.5 |
| baseline | ill_fit | 4 | 111.7 | 297.9 | 75.0 | 7.27 | 111.7 |
| baseline | late_burst | 11 | 38.5 | 80.5 | 72.7 | 1.63 | -38.5 |
| baseline | linear | 3 | 45.5 | 57.1 | 66.7 | 4.45 | -32.0 |
| baseline | logarithmic | 96 | 40.6 | 100.0 | 91.7 | 2.22 | 3.7 |
| baseline | logistic | 66 | 41.7 | 141.7 | 78.8 | 3.83 | -9.5 |
| scaled | ill_fit | 4 | 111.7 | 297.9 | 75.0 | 7.36 | 111.7 |
| scaled | late_burst | 11 | 38.5 | 80.5 | 90.9 | 2.44 | -38.5 |
| scaled | linear | 3 | 45.5 | 57.1 | 66.7 | 6.7 | -32.0 |
| scaled | logarithmic | 96 | 40.6 | 100.0 | 92.7 | 2.59 | 3.7 |
| scaled | logistic | 66 | 41.7 | 141.7 | 81.8 | 4.6 | -9.5 |

## Фігури

- [MAPE p50 за cutoff × method](figures\10_mape_per_cutoff.html)
- [Coverage за cutoff × method](figures\10_coverage_per_cutoff.html)

## Критерії promote

| Метрика | Поріг promote |
|---|---|
| MAPE@0.1, 0.2 | НЕ гірше за baseline (≤ +1pp) |
| Coverage@0.1, 0.2 | Ближче до 95% за baseline |
| MAPE@0.5, 0.7 | НЕ гірше за baseline (≤ +1pp) |
| Coverage@0.5, 0.7 | НЕ гірше за baseline (≥ −3pp) |

Якщо ≥1 з B/C/D відповідає — promote окремим коміттом. Інакше документуємо
як negative result.

## Артефакти

- `figures/10_ab_points.csv` — wide raw результати.
- `figures/10_ab_metrics.csv` — long per-method метрики.
