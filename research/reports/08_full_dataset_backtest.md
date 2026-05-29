# 08 — Full-dataset Diagnostic Backtest

**Generated:** 2026-05-29T19:24:20
**Source:** `D:\GitHub\survey-insight\data\Form Timestamp Collection.csv` (sha256:`0dfa60a9a2cd`)
**Forms:** 177 total → 141 eligible (N ≥ 10, shape ≠ insufficient)
**Cutoffs:** (0.1, 0.2, 0.3, 0.5, 0.7) · **Horizon fraction:** 0.25
**Backtest points:** 590
**Skipped:** {'too_few': 36, 'insufficient_shape': 0, 'no_span': 0}

## Глобальні метрики

| Метрика | Значення | Орієнтир |
|---|---:|---|
| **MAPE (median)** | 29.4% | < 15% — добре, < 25% — прийнятно |
| **Coverage 95% PI** | 89.2% | Має бути ≈ 95% |
| **Sharpness (median)** | 3.59 | width / truth |
| **Bias (median)** | -10.5% | 0% — unbiased |

## Sanity vs 02_ baseline (N≥30, cutoffs ∈ {0.3, 0.5, 0.7})

Очікуємо ≈ 87% coverage / 24.6% MAPE / -12% bias з [02_backtest.md](02_backtest.md).
Якщо суттєво різниться — bug у pipeline 08_, не в моделі.

| Метрика | 08_ на baseline-subset | 02_ baseline |
|---|---:|---:|
| n_points | 288 | 288 |
| MAPE_p50 | 24.6% | 24.6% |
| Coverage | 88.9% | 87% |
| Bias | -11.6% | -12% |

## Per-axis breakdowns

### Shape
| shape | n_points | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|
| ill_fit | 16 | 50.0 | 220.5 | 75.0 | 2.55 | 28.5 |
| late_burst | 29 | 32.7 | 79.8 | 93.1 | 2.45 | -28.4 |
| linear | 9 | 29.6 | 48.4 | 88.9 | 2.97 | -19.4 |
| logarithmic | 306 | 25.5 | 142.3 | 90.2 | 3.59 | -3.5 |
| logistic | 230 | 30.0 | 121.4 | 88.3 | 3.93 | -19.0 |

### N-class (нова таксономія)
| n_class | n_points | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|
| small | 132 | 29.7 | 128.9 | 90.9 | 6.77 | -11.1 |
| medium | 243 | 27.6 | 95.6 | 89.7 | 4.39 | -13.8 |
| large | 195 | 30.5 | 181.6 | 88.2 | 2.39 | -7.0 |
| huge | 20 | 32.7 | 122.2 | 80.0 | 1.17 | -5.7 |

### Tempo (нова таксономія: burst/daily/long-tail/sporadic)
| tempo | n_points | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|
| burst | 31 | 36.7 | 179.3 | 93.5 | 4.65 | -3.3 |
| daily_flow | 37 | 20.0 | 44.8 | 100.0 | 6.43 | -16.7 |
| long_tail | 92 | 20.0 | 55.5 | 96.7 | 5.71 | -14.6 |
| sporadic | 430 | 31.6 | 166.8 | 86.3 | 2.98 | -7.3 |

### Duration (нова таксономія: hours/days/weeks/months)
| duration_class | n_points | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|
| hours | 16 | 75.0 | 290.0 | 100.0 | 10.3 | 75.0 |
| days | 242 | 29.7 | 97.3 | 90.9 | 4.77 | -15.0 |
| weeks | 193 | 27.6 | 182.2 | 89.1 | 3.56 | -7.0 |
| months | 139 | 29.3 | 125.6 | 84.9 | 2.1 | -10.0 |

### Cutoff (як метод деградує з 10%→70% життя форми)
| cutoff_frac | n_points | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|
| 0.1 | 74.0 | 46.9 | 99.2 | 87.8 | 2.03 | -7.7 |
| 0.2 | 106.0 | 40.1 | 149.7 | 87.7 | 4.0 | -4.2 |
| 0.3 | 128.0 | 39.0 | 180.1 | 89.1 | 4.71 | -6.3 |
| 0.5 | 141.0 | 29.1 | 118.8 | 90.1 | 3.77 | -15.0 |
| 0.7 | 141.0 | 17.6 | 33.3 | 90.1 | 3.22 | -11.1 |

## Cross-tab heat-maps

- [MAPE p50 (%) — shape × n_class](figures\08_heat_mape_shape_n.html)
- [Coverage (%) — shape × n_class](figures\08_heat_cov_shape_n.html)
- [MAPE p50 (%) — tempo × cutoff](figures\08_heat_mape_tempo_cutoff.html)
- [Coverage (%) — tempo × cutoff](figures\08_heat_cov_tempo_cutoff.html)

### Shape × N-class (точні значення, MAPE/coverage)
| shape | n_class | n_points | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| ill_fit | large | 5 | 49.9 | 105.0 | 60.0 | 1.46 | 7.1 |
| ill_fit | medium | 4 | 115.1 | 315.5 | 100.0 | 3.47 | 76.4 |
| ill_fit | small | 7 | 50.0 | 214.5 | 71.4 | 3.33 | 50.0 |
| late_burst | large | 15 | 24.7 | 74.1 | 93.3 | 2.33 | -21.3 |
| late_burst | medium | 14 | 38.7 | 80.2 | 92.9 | 3.7 | -38.7 |
| linear | medium | 9 | 29.6 | 48.4 | 88.9 | 2.97 | -19.4 |
| logarithmic | huge | 10 | 27.4 | 61.5 | 80.0 | 1.11 | 10.1 |
| logarithmic | large | 110 | 31.9 | 190.2 | 90.0 | 2.44 | -4.5 |
| logarithmic | medium | 118 | 19.6 | 100.0 | 91.5 | 5.86 | -5.1 |
| logarithmic | small | 68 | 30.1 | 122.1 | 89.7 | 7.79 | 0.0 |
| logistic | huge | 10 | 62.5 | 319.6 | 80.0 | 1.53 | -13.6 |
| logistic | large | 65 | 30.3 | 147.1 | 86.2 | 2.36 | -10.0 |
| logistic | medium | 98 | 30.7 | 79.5 | 86.7 | 3.99 | -21.6 |
| logistic | small | 57 | 27.3 | 84.4 | 94.7 | 6.48 | -20.0 |

### Tempo × cutoff
| tempo | cutoff_frac | n_points | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| burst | 0.1 | 2 | 50.8 | 58.2 | 50.0 | 5.22 | 9.2 |
| burst | 0.2 | 6 | 44.7 | 155.6 | 83.3 | 4.11 | 6.5 |
| burst | 0.3 | 7 | 35.5 | 149.7 | 100.0 | 6.45 | 0.0 |
| burst | 0.5 | 8 | 38.9 | 140.0 | 100.0 | 13.23 | 8.3 |
| burst | 0.7 | 8 | 29.5 | 140.9 | 100.0 | 4.43 | -20.2 |
| daily_flow | 0.2 | 4 | 22.7 | 48.6 | 100.0 | 10.77 | -22.7 |
| daily_flow | 0.3 | 9 | 28.6 | 44.8 | 100.0 | 7.38 | -28.6 |
| daily_flow | 0.5 | 12 | 23.8 | 43.4 | 100.0 | 3.97 | -23.4 |
| daily_flow | 0.7 | 12 | 13.1 | 20.0 | 100.0 | 6.18 | -12.5 |
| long_tail | 0.1 | 5 | 13.8 | 60.5 | 100.0 | 13.55 | -13.8 |
| long_tail | 0.2 | 14 | 35.4 | 67.7 | 100.0 | 10.43 | -16.4 |
| long_tail | 0.3 | 21 | 14.3 | 54.5 | 100.0 | 9.0 | -6.7 |
| long_tail | 0.5 | 26 | 25.0 | 52.8 | 88.5 | 4.83 | -15.3 |
| long_tail | 0.7 | 26 | 19.8 | 29.0 | 100.0 | 3.61 | -18.1 |
| sporadic | 0.1 | 67 | 47.2 | 100.9 | 88.1 | 1.99 | -2.2 |
| sporadic | 0.2 | 82 | 40.1 | 156.6 | 85.4 | 3.6 | 1.3 |
| sporadic | 0.3 | 91 | 47.5 | 184.1 | 84.6 | 3.56 | -5.5 |
| sporadic | 0.5 | 95 | 30.3 | 204.8 | 88.4 | 3.41 | -12.2 |
| sporadic | 0.7 | 95 | 15.5 | 56.0 | 85.3 | 2.34 | -7.2 |

## Bar-chart фігури

- [Shape: MAPE & coverage](figures\08_shape_bars.html)
- [N-class: MAPE & coverage](figures\08_n_class_bars.html)
- [Tempo: MAPE & coverage](figures\08_tempo_bars.html)
- [Duration: MAPE & coverage](figures\08_duration_bars.html)
- [Cutoff: MAPE & coverage](figures\08_cutoff_bars.html)
- [Failure modes за shape](figures\08_failure_modes.html)

## Failure spotlight (cells з MAPE_p50 > 30% або coverage < 75%)

| shape | n_class | n_pts | MAPE_p50 | Coverage | Bias | Дамп |
|---|---|---:|---:|---:|---:|---|
| ill_fit | medium | 4 | 115.1% | 100.0% | +76.4% | [HTML](figures\08_failures\ill_fit_medium.html) |
| logistic | huge | 10 | 62.5% | 80.0% | -13.6% | [HTML](figures\08_failures\logistic_huge.html) |
| ill_fit | small | 7 | 50.0% | 71.4% | +50.0% | [HTML](figures\08_failures\ill_fit_small.html) |
| ill_fit | large | 5 | 49.9% | 60.0% | +7.1% | [HTML](figures\08_failures\ill_fit_large.html) |
| late_burst | medium | 14 | 38.7% | 92.9% | -38.7% | [HTML](figures\08_failures\late_burst_medium.html) |
| logarithmic | large | 110 | 31.9% | 90.0% | -4.5% | [HTML](figures\08_failures\logarithmic_large.html) |
| logistic | medium | 98 | 30.7% | 86.7% | -21.6% | [HTML](figures\08_failures\logistic_medium.html) |
| logistic | large | 65 | 30.3% | 86.2% | -10.0% | [HTML](figures\08_failures\logistic_large.html) |
| logarithmic | small | 68 | 30.1% | 89.7% | +0.0% | [HTML](figures\08_failures\logarithmic_small.html) |

## Висновки для prod (input для fix-сесії)

Перечитати таблиці вище і відповісти на питання:

1. **Де метод системно недопрогнозує** (bias більш ніж -15%)?
   Перевірити `by_shape`, `by_n_class`, `by_cutoff`. Малі N + ранні cutoffs —
   очікуваний негативний bias через K_min relaxation, що калібрований на N≥30.
2. **Де coverage сильно < 95%**? Це cells, що потребують локальної
   per-cell calibration multiplier'у (зараз глобальний ×10).
3. **Найгірші cells у failure spotlight** — це або (а) нова модель потрібна
   (наприклад LinearModel для shape=linear), (б) занижений MIN_TRAIN_POINTS
   для конкретного shape, (в) HORIZON_FRACTION надто агресивний для коротких
   форм. Обрати найбільш impactful напрямок.
4. **Тempo×cutoff heat-map**: чи burst-форми ламаються на ранніх 10–20%?
   Якщо так — це core prod-проблема (свіжа форма з потужним сплеском у
   перший день, користувач дивиться на прогноз і бачить unrealistic точку).

## Артефакти

- `figures/08_backtest_points.csv` — повний log усіх backtest-runs (з tempo, n_class, duration_class).
- `figures/08_backtest_metrics.csv` — обчислені метрики.
- `figures/08_failures/<shape>_<n_class>.html` — per-cell failure curves.
