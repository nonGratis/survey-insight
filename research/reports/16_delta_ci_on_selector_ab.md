# 16 - Selector model + delta-CI vs prod (ізолюємо CI method)

**Generated:** 2026-05-30T14:05:30
**Source:** `D:\GitHub\survey-insight\data\Form Timestamp Collection.csv` (sha256:`0dfa60a9a2cd`)
**Forms:** 177 -> 161 eligible (N >= 5)
**n_train:** (5, 10, 15, 20, 25, 30) - **horizons:** (2.0, 6.0, 24.0, 72.0, 168.0)
**Backtest points:** 3710
**Skipped:** {'too_few': 16, 'insufficient_shape': 0, 'no_span': 0}
**Fit failures:** A=0, B=0

## Контекст

15_ показав що log+delta виграє Winkler на 2h horizon (-62%), але loses
на 168h (log diverges). Висновок: проблема НЕ у delta-method, а у виборі
моделі.

Цей бенчмарк ізолює "CI method" як змінну:
- Обидва методи використовують ТУ САМУ модель з AICc selector
  (asymp_exp / logistic / gompertz)
- Точка identical в обох
- Різниця: A використовує NHPP+P7+P10+P11 multipliers, B використовує
  classical delta-method CI на pcov моделі

Якщо B виграє Winkler глобально — це аргумент revert P7/P10/P11.

## Win rates по horizon

| horizon_h | n | b_wins | win_rate_pct |
|---|---:|---:|---:|
| 2.0 | 742.0 | 440.0 | 59.3 |
| 6.0 | 742.0 | 405.0 | 54.6 |
| 24.0 | 742.0 | 362.0 | 48.8 |
| 72.0 | 742.0 | 260.0 | 35.0 |
| 168.0 | 742.0 | 236.0 | 31.8 |

## Global per method

| method | n | mape_p50 | coverage | width_p50 | width_p90 | winkler_p50 | winkler_mean | bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| a_prod | 3710 | 37.2 | 81.6 | 158.0 | 656.0 | 234.0 | 2675.4 | -2.7 |
| b_selector_delta | 3710 | 37.2 | 79.3 | 69.0 | 963228.0 | 289.5 | 2528196.6 | -2.7 |

## Per horizon (decisive view)

| method | horizon_hours | n | mape_p50 | coverage | width_p50 | width_p90 | winkler_p50 | winkler_mean | bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a_prod | 2.0 | 742 | 28.6 | 83.2 | 158.5 | 738.9 | 212.5 | 1337.6 | 5.9 |
| a_prod | 6.0 | 742 | 26.9 | 84.1 | 158.5 | 738.9 | 220.5 | 1890.5 | 0.0 |
| a_prod | 24.0 | 742 | 31.8 | 84.0 | 158.5 | 738.9 | 235.5 | 2550.2 | -11.8 |
| a_prod | 72.0 | 742 | 43.3 | 80.9 | 157.0 | 552.3 | 215.0 | 3545.3 | -9.1 |
| a_prod | 168.0 | 742 | 50.1 | 76.0 | 160.0 | 555.9 | 256.0 | 4053.5 | -16.7 |
| b_selector_delta | 2.0 | 742 | 28.6 | 86.8 | 28.0 | 421035.4 | 81.0 | 1104727.2 | 5.9 |
| b_selector_delta | 6.0 | 742 | 26.9 | 82.9 | 28.0 | 421035.4 | 109.0 | 1104782.6 | 0.0 |
| b_selector_delta | 24.0 | 742 | 31.8 | 73.9 | 28.0 | 421035.4 | 205.5 | 1104862.8 | -11.8 |
| b_selector_delta | 72.0 | 742 | 43.3 | 77.1 | 150.0 | 2727281.8 | 720.0 | 4655213.9 | -9.1 |
| b_selector_delta | 168.0 | 742 | 50.1 | 75.7 | 282.5 | 2867748.3 | 979.5 | 4671396.4 | -16.7 |

## Per form_type

| method | form_type | n | mape_p50 | coverage | width_p50 | width_p90 | winkler_p50 | winkler_mean | bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a_prod | creative_submission | 210 | 26.8 | 92.9 | 99.0 | 438.0 | 116.0 | 188.5 | -5.5 |
| a_prod | event_feedback | 185 | 73.3 | 85.9 | 134.0 | 508.0 | 160.0 | 811.1 | 61.9 |
| a_prod | event_registration | 945 | 28.6 | 89.9 | 156.0 | 482.0 | 181.0 | 439.2 | -5.0 |
| a_prod | holiday | 205 | 44.2 | 77.6 | 256.0 | 1081.0 | 419.0 | 541.6 | 27.3 |
| a_prod | other | 155 | 37.5 | 96.1 | 105.0 | 327.0 | 105.0 | 153.0 | 0.0 |
| a_prod | political | 100 | 30.2 | 81.0 | 160.5 | 755.0 | 218.0 | 707.6 | -9.1 |
| a_prod | recruitment | 235 | 22.2 | 83.0 | 274.0 | 1045.0 | 320.0 | 440.0 | 0.0 |
| a_prod | service | 200 | 30.8 | 68.0 | 123.0 | 894.6 | 339.0 | 4648.8 | -16.7 |
| a_prod | survey | 550 | 54.0 | 57.8 | 209.0 | 1149.0 | 922.0 | 11037.7 | -36.0 |
| a_prod | unknown | 570 | 36.5 | 83.0 | 157.5 | 654.1 | 231.5 | 3296.7 | 0.0 |
| a_prod | volunteer_donor | 355 | 39.4 | 88.2 | 108.0 | 532.8 | 160.0 | 373.5 | 0.0 |
| b_selector_delta | creative_submission | 210 | 26.8 | 69.5 | 7.5 | 1158.8 | 81.5 | 633.3 | -5.5 |
| b_selector_delta | event_feedback | 185 | 73.3 | 87.6 | 1965.0 | 8494354.0 | 1965.0 | 2752954.4 | 61.9 |
| b_selector_delta | event_registration | 945 | 28.6 | 72.6 | 19.0 | 9431.4 | 122.0 | 371646.4 | -5.0 |
| b_selector_delta | holiday | 205 | 44.2 | 94.6 | 576.0 | 181221.0 | 725.0 | 87496.0 | 27.3 |
| b_selector_delta | other | 155 | 37.5 | 73.5 | 24.0 | 410573.2 | 125.0 | 375779.1 | 0.0 |
| b_selector_delta | political | 100 | 30.2 | 87.0 | 77.0 | 6553.7 | 172.0 | 1534.6 | -9.1 |
| b_selector_delta | recruitment | 235 | 22.2 | 71.5 | 21.0 | 1310.6 | 102.0 | 1976.7 | 0.0 |
| b_selector_delta | service | 200 | 30.8 | 78.0 | 11.0 | 3169378.0 | 203.0 | 970914.9 | -16.7 |
| b_selector_delta | survey | 550 | 54.0 | 90.4 | 68278.0 | 23457627.8 | 68278.0 | 14706522.0 | -36.0 |
| b_selector_delta | unknown | 570 | 36.5 | 81.9 | 96.0 | 265242.0 | 288.0 | 208556.5 | 0.0 |
| b_selector_delta | volunteer_donor | 355 | 39.4 | 74.4 | 23.0 | 20476.2 | 210.0 | 114157.9 | 0.0 |

## Per n_train

| method | n_train | n | mape_p50 | coverage | width_p50 | width_p90 | winkler_p50 | winkler_mean | bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a_prod | 5 | 805 | 53.6 | 74.5 | 87.0 | 479.0 | 233.0 | 2506.1 | 0.0 |
| a_prod | 10 | 705 | 44.4 | 81.8 | 106.0 | 593.0 | 155.0 | 2577.7 | -6.9 |
| a_prod | 15 | 645 | 33.0 | 83.4 | 160.0 | 788.0 | 234.0 | 2492.3 | -6.2 |
| a_prod | 20 | 555 | 28.6 | 85.4 | 207.0 | 1045.0 | 220.0 | 2659.0 | 0.0 |
| a_prod | 25 | 520 | 26.7 | 82.7 | 220.0 | 939.1 | 254.0 | 2928.2 | -3.8 |
| a_prod | 30 | 480 | 27.0 | 85.2 | 245.0 | 733.4 | 262.0 | 3093.9 | -3.2 |
| b_selector_delta | 5 | 805 | 53.6 | 97.1 | 1037.0 | 870438.0 | 1088.0 | 4536899.0 | 0.0 |
| b_selector_delta | 10 | 705 | 44.4 | 82.1 | 91.0 | 2230185.2 | 298.0 | 3400802.6 | -6.9 |
| b_selector_delta | 15 | 645 | 33.0 | 76.9 | 34.0 | 813241.6 | 182.0 | 2053892.9 | -6.2 |
| b_selector_delta | 20 | 555 | 28.6 | 78.0 | 39.0 | 1384007.0 | 168.0 | 1523238.6 | 0.0 |
| b_selector_delta | 25 | 520 | 26.7 | 65.2 | 20.0 | 935800.3 | 240.0 | 1251150.3 | -3.8 |
| b_selector_delta | 30 | 480 | 27.0 | 65.0 | 16.0 | 574958.6 | 284.0 | 1060590.0 | -3.2 |

## Per selected model

| method | selected_model | n | mape_p50 | coverage | width_p50 | width_p90 | winkler_p50 | winkler_mean | bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a_prod | asymptotic_exp | 2400 | 36.5 | 81.9 | 123.0 | 739.0 | 193.0 | 1337.8 | -3.2 |
| a_prod | gompertz | 980 | 38.9 | 78.9 | 207.0 | 622.0 | 286.5 | 5318.6 | 0.0 |
| a_prod | logistic | 330 | 33.3 | 87.9 | 196.0 | 425.1 | 217.0 | 4553.7 | -3.2 |
| b_selector_delta | asymptotic_exp | 2400 | 36.5 | 75.4 | 39.5 | 19115.1 | 240.0 | 1522725.4 | -3.2 |
| b_selector_delta | gompertz | 980 | 38.9 | 88.1 | 1073.0 | 11630677.0 | 1701.0 | 5781434.5 | 0.0 |
| b_selector_delta | logistic | 330 | 33.3 | 81.2 | 27.5 | 234244.0 | 94.0 | 179583.1 | -3.2 |

## Figures

- [Winkler per horizon (A vs B)](figures\16_winkler_per_horizon.html)
- [Width per horizon](figures\16_width_per_horizon.html)
- [Coverage per horizon](figures\16_coverage_per_horizon.html)

## Інтерпретація

- **B wins Winkler globally** + не падає на довгих горизонтах (бо
  saturating моделі не diverge) → revert P7/P10/P11 і впровадити
  delta-CI як стандарт. Це **революція** у calibration модулі.
- **B wins на 2h-6h, loses на >24h** → реалізувати hybrid: delta-CI для
  short horizon, NHPP для long. Менш агресивно.
- **B loses глобально** → 15_ результат був специфічний до log model,
  delta-CI само по собі не виграє. Документувати negative і повернутися
  до Tier 2 (LinearModel, Bass).

## Артефакти

- `figures/16_ab_points.csv` — wide raw.
- `figures/16_ab_metrics.csv` — long per-method метрики з Winkler.
