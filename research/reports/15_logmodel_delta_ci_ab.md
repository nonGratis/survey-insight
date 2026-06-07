# 15 - LogModel + delta-method CI vs prod (Winkler A/B)

**Generated:** 2026-05-30T13:48:03
**Source:** `D:\GitHub\survey-insight\data\Form Timestamp Collection.csv` (sha256:`0dfa60a9a2cd`)
**Forms:** 177 -> 161 eligible (N >= 5)
**n_train:** (5, 10, 15, 20, 25, 30) - **horizons:** (2.0, 6.0, 24.0, 72.0, 168.0)
**Backtest points:** 3710
**Skipped:** {'too_few': 16, 'insufficient_shape': 0, 'no_span': 0}
**Fit failures:** A=0, B=0

## Контекст pivot-у

Користувач показав скриншот де прод-CI [62, 818] на формі з truth ~60.
Точка добре (point ~70), але CI безкорисний. R²=0.92, fit excellent —
але NHPP + CALIBRATION_MULTIPLIER + per-type widening роздуває CI без
залежності від якості fit.

Цей бенчмарк тестує **structural alternative**: bypass NHPP+multiplier,
використати класичну delta-method CI (curve_fit + numerical Jacobian +
Student t quantile) на простій log model.

## Методи

- **A · prod** — `forecast_responses(timeline, form_type=ft)` з усіма
  калібровками (P7+P10+P11). Контроль.
- **B · log_delta** — fit `y = a·ln(t+1) + b` через scipy curve_fit,
  delta-method CI без post-hoc widening.

Primary metric — **Winkler interval score** (proper scoring rule):
`W = (U − L) + (2/α)·max(L−y, 0) + (2/α)·max(y−U, 0)`. Lower = better.

## Win count (Winkler, per backtest point)

| | Кількість | % |
|---|---:|---:|
| **B (log_delta) виграє** | 1556 | 41.9% |
| A (prod) виграє | 2153 | 58.0% |
| Ties | 1 | 0.0% |

## Global per method

| method | n | mape_p50 | coverage | sharpness_p50 | width_p50 | width_p90 | winkler_p50 | winkler_mean | bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a_prod | 3710 | 37.2 | 81.6 | 4.39 | 158.0 | 656.0 | 234.0 | 2675.4 | -2.7 |
| b_log_delta | 3710 | 66.7 | 28.5 | 0.5 | 14.5 | 576.0 | 419.5 | 31899.4 | 30.8 |

## Per form_type (decisive)

| method | form_type | n | mape_p50 | coverage | sharpness_p50 | width_p50 | width_p90 | winkler_p50 | winkler_mean | bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a_prod | creative_submission | 210 | 26.8 | 92.9 | 5.15 | 99.0 | 438.0 | 116.0 | 188.5 | -5.5 |
| a_prod | event_feedback | 185 | 73.3 | 85.9 | 4.0 | 134.0 | 508.0 | 160.0 | 811.1 | 61.9 |
| a_prod | event_registration | 945 | 28.6 | 89.9 | 5.56 | 156.0 | 482.0 | 181.0 | 439.2 | -5.0 |
| a_prod | holiday | 205 | 44.2 | 77.6 | 8.85 | 256.0 | 1081.0 | 419.0 | 541.6 | 27.3 |
| a_prod | other | 155 | 37.5 | 96.1 | 4.41 | 105.0 | 327.0 | 105.0 | 153.0 | 0.0 |
| a_prod | political | 100 | 30.2 | 81.0 | 3.46 | 160.5 | 755.0 | 218.0 | 707.6 | -9.1 |
| a_prod | recruitment | 235 | 22.2 | 83.0 | 8.23 | 274.0 | 1045.0 | 320.0 | 440.0 | 0.0 |
| a_prod | service | 200 | 30.8 | 68.0 | 3.0 | 123.0 | 894.6 | 339.0 | 4648.8 | -16.7 |
| a_prod | survey | 550 | 54.0 | 57.8 | 1.11 | 209.0 | 1149.0 | 922.0 | 11037.7 | -36.0 |
| a_prod | unknown | 570 | 36.5 | 83.0 | 5.16 | 157.5 | 654.1 | 231.5 | 3296.7 | 0.0 |
| a_prod | volunteer_donor | 355 | 39.4 | 88.2 | 3.33 | 108.0 | 532.8 | 160.0 | 373.5 | 0.0 |
| b_log_delta | creative_submission | 210 | 20.0 | 46.7 | 0.2 | 4.0 | 25.1 | 63.0 | 260.2 | 0.0 |
| b_log_delta | event_feedback | 185 | 183.3 | 20.5 | 1.78 | 40.0 | 1854.8 | 2167.0 | 46035.4 | 183.3 |
| b_log_delta | event_registration | 945 | 37.5 | 30.8 | 0.3 | 8.0 | 236.0 | 200.0 | 12753.0 | 0.0 |
| b_log_delta | holiday | 205 | 273.7 | 17.6 | 1.82 | 83.0 | 501.8 | 2593.0 | 12321.1 | 273.7 |
| b_log_delta | other | 155 | 38.1 | 31.6 | 0.5 | 10.0 | 533.2 | 204.0 | 11024.2 | 9.1 |
| b_log_delta | political | 100 | 50.6 | 27.0 | 0.55 | 21.5 | 279.2 | 351.5 | 2099.3 | 38.6 |
| b_log_delta | recruitment | 235 | 82.1 | 33.2 | 0.67 | 19.0 | 169.8 | 322.0 | 2864.6 | 82.1 |
| b_log_delta | service | 200 | 48.3 | 38.5 | 0.39 | 10.5 | 723.8 | 280.0 | 30539.5 | 25.2 |
| b_log_delta | survey | 550 | 126.9 | 12.9 | 0.75 | 85.0 | 2861.4 | 10009.0 | 138313.1 | 126.9 |
| b_log_delta | unknown | 570 | 75.0 | 30.7 | 0.53 | 13.0 | 302.0 | 406.0 | 14249.8 | 27.9 |
| b_log_delta | volunteer_donor | 355 | 45.5 | 32.7 | 0.43 | 12.0 | 334.4 | 323.0 | 6489.8 | 20.6 |

## Per n_train

| method | n_train | n | mape_p50 | coverage | sharpness_p50 | width_p50 | width_p90 | winkler_p50 | winkler_mean | bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a_prod | 5 | 805 | 53.6 | 74.5 | 5.78 | 87.0 | 479.0 | 233.0 | 2506.1 | 0.0 |
| a_prod | 10 | 705 | 44.4 | 81.8 | 4.12 | 106.0 | 593.0 | 155.0 | 2577.7 | -6.9 |
| a_prod | 15 | 645 | 33.0 | 83.4 | 5.8 | 160.0 | 788.0 | 234.0 | 2492.3 | -6.2 |
| a_prod | 20 | 555 | 28.6 | 85.4 | 4.8 | 207.0 | 1045.0 | 220.0 | 2659.0 | 0.0 |
| a_prod | 25 | 520 | 26.7 | 82.7 | 3.31 | 220.0 | 939.1 | 254.0 | 2928.2 | -3.8 |
| a_prod | 30 | 480 | 27.0 | 85.2 | 2.83 | 245.0 | 733.4 | 262.0 | 3093.9 | -3.2 |
| b_log_delta | 5 | 805 | 148.6 | 44.6 | 2.88 | 51.0 | 1652.6 | 304.0 | 21346.0 | 148.6 |
| b_log_delta | 10 | 705 | 74.7 | 28.1 | 0.74 | 15.0 | 668.0 | 491.0 | 28410.7 | 43.8 |
| b_log_delta | 15 | 645 | 60.0 | 22.3 | 0.41 | 11.0 | 405.2 | 440.0 | 31149.3 | 22.2 |
| b_log_delta | 20 | 555 | 47.6 | 22.0 | 0.33 | 11.0 | 330.2 | 474.0 | 36945.8 | 16.7 |
| b_log_delta | 25 | 520 | 40.0 | 23.3 | 0.23 | 9.0 | 293.6 | 452.0 | 40299.8 | 5.6 |
| b_log_delta | 30 | 480 | 37.0 | 23.3 | 0.19 | 10.0 | 251.2 | 403.0 | 40794.9 | 3.2 |

## Per horizon_hours

| method | horizon_hours | n | mape_p50 | coverage | sharpness_p50 | width_p50 | width_p90 | winkler_p50 | winkler_mean | bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a_prod | 2.0 | 742 | 28.6 | 83.2 | 6.32 | 158.5 | 738.9 | 212.5 | 1337.6 | 5.9 |
| a_prod | 6.0 | 742 | 26.9 | 84.1 | 5.52 | 158.5 | 738.9 | 220.5 | 1890.5 | 0.0 |
| a_prod | 24.0 | 742 | 31.8 | 84.0 | 4.53 | 158.5 | 738.9 | 235.5 | 2550.2 | -11.8 |
| a_prod | 72.0 | 742 | 43.3 | 80.9 | 3.35 | 157.0 | 552.3 | 215.0 | 3545.3 | -9.1 |
| a_prod | 168.0 | 742 | 50.1 | 76.0 | 2.85 | 160.0 | 555.9 | 256.0 | 4053.5 | -16.7 |
| b_log_delta | 2.0 | 742 | 25.0 | 43.4 | 0.27 | 5.0 | 76.4 | 80.0 | 2149.5 | 7.8 |
| b_log_delta | 6.0 | 742 | 50.7 | 32.3 | 0.38 | 9.0 | 211.4 | 206.5 | 6937.5 | 19.3 |
| b_log_delta | 24.0 | 742 | 83.8 | 25.5 | 0.62 | 19.0 | 656.7 | 485.5 | 24013.4 | 54.5 |
| b_log_delta | 72.0 | 742 | 95.3 | 21.2 | 0.85 | 35.0 | 1311.6 | 1138.0 | 50124.2 | 75.0 |
| b_log_delta | 168.0 | 742 | 98.8 | 19.9 | 0.99 | 53.5 | 1967.3 | 1829.0 | 76272.5 | 95.9 |

## Figures

- [Winkler score per type](figures\15_winkler_per_type.html)
- [Absolute CI width per type](figures\15_width_per_type.html)
- [Coverage per type](figures\15_coverage_per_type.html)

## Як читати

- **width_p50, width_p90** — медіанна/p90 АБСОЛЮТНА ширина CI у units of
  responses. Це те, що бачить користувач на скріншоті. ↓ = краще.
- **winkler_p50** — proper score. ↓ = краще. Виграш на цій метриці означає
  CI одночасно вужче І capture truth.
- **coverage** — традиційна. Метрика КОЛИШНЬОЇ оптимізації.

Якщо B має нижче winkler І нижче width при coverage >= 80% → структурний
сигнал що delta-CI прод-практичніший. Тоді треба:
1. Revertити P10/P11 (multipliers).
2. Перенести LogModel і delta-CI у `core/forecast/` як основний механізм.
3. Додати інші моделі (Logistic, AsympExp) з delta-CI як альтернативи.

## Артефакти

- `figures/15_ab_points.csv` — wide-format raw результати.
- `figures/15_ab_metrics.csv` — long-format per-method метрики з Winkler.
