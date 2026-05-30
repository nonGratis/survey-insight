# 13 - Per-form-type calibration A/B (P11 candidate)

**Generated:** 2026-05-30T00:14:34
**Source:** `D:\GitHub\survey-insight\data\Form Timestamp Collection.csv` (sha256:`0dfa60a9a2cd`)
**Forms:** 177 -> 161 eligible (N >= 5)
**n_train:** (5, 10, 15, 20, 25, 30) - **horizon_hours:** (2.0, 6.0, 24.0, 72.0, 168.0)
**Backtest points:** 3710
**Skipped:** {'too_few': 16, 'insufficient_shape': 0, 'no_span': 0}

## Методи

- **A · baseline** — `forecast_responses(timeline)` з глобальним CALIBRATION_MULTIPLIER=10.0.
- **B · per_type** — `forecast_responses(timeline, form_type=ft)` з PER_TYPE_MULTIPLIER:
  - survey 28, service 20, holiday 16, recruitment 14, political 14,
  - volunteer 12, feedback 12, event_reg 11, creative 11, other 8, unknown 13.

## Global

| method | n | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|
| a_baseline | 3710 | 37.2 | 155.9 | 80.2 | 3.7 | -2.7 |
| b_per_type | 3710 | 37.2 | 155.9 | 81.6 | 4.39 | -2.7 |

## Per form_type (decisive view)

| method | form_type | n | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| a_baseline | creative_submission | 210 | 26.8 | 190.0 | 90.5 | 4.7 | -5.5 |
| a_baseline | event_feedback | 185 | 73.3 | 430.5 | 85.4 | 3.83 | 61.9 |
| a_baseline | event_registration | 945 | 28.6 | 115.0 | 89.6 | 5.09 | -5.0 |
| a_baseline | holiday | 205 | 44.2 | 211.0 | 77.6 | 5.94 | 27.3 |
| a_baseline | other | 155 | 37.5 | 231.4 | 96.1 | 4.77 | 0.0 |
| a_baseline | political | 100 | 30.2 | 87.0 | 80.0 | 2.91 | -9.1 |
| a_baseline | recruitment | 235 | 22.2 | 81.8 | 81.3 | 5.91 | 0.0 |
| a_baseline | service | 200 | 30.8 | 91.4 | 67.5 | 1.94 | -16.7 |
| a_baseline | survey | 550 | 54.0 | 97.8 | 52.5 | 0.91 | -36.0 |
| a_baseline | unknown | 570 | 36.5 | 216.8 | 81.9 | 4.5 | 0.0 |
| a_baseline | volunteer_donor | 355 | 39.4 | 216.7 | 87.9 | 3.29 | 0.0 |
| b_per_type | creative_submission | 210 | 26.8 | 190.0 | 92.9 | 5.15 | -5.5 |
| b_per_type | event_feedback | 185 | 73.3 | 430.5 | 85.9 | 4.0 | 61.9 |
| b_per_type | event_registration | 945 | 28.6 | 115.0 | 89.9 | 5.56 | -5.0 |
| b_per_type | holiday | 205 | 44.2 | 211.0 | 77.6 | 8.85 | 27.3 |
| b_per_type | other | 155 | 37.5 | 231.4 | 96.1 | 4.41 | 0.0 |
| b_per_type | political | 100 | 30.2 | 87.0 | 81.0 | 3.46 | -9.1 |
| b_per_type | recruitment | 235 | 22.2 | 81.8 | 83.0 | 8.23 | 0.0 |
| b_per_type | service | 200 | 30.8 | 91.4 | 68.0 | 3.0 | -16.7 |
| b_per_type | survey | 550 | 54.0 | 97.8 | 57.8 | 1.11 | -36.0 |
| b_per_type | unknown | 570 | 36.5 | 216.8 | 83.0 | 5.16 | 0.0 |
| b_per_type | volunteer_donor | 355 | 39.4 | 216.7 | 88.2 | 3.33 | 0.0 |

## Per n_train

| method | n_train | n | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| a_baseline | 5 | 805 | 53.6 | 200.0 | 72.8 | 4.85 | 0.0 |
| a_baseline | 10 | 705 | 44.4 | 168.7 | 81.0 | 3.5 | -6.9 |
| a_baseline | 15 | 645 | 33.0 | 153.9 | 82.3 | 4.85 | -6.2 |
| a_baseline | 20 | 555 | 28.6 | 144.4 | 84.0 | 4.12 | 0.0 |
| a_baseline | 25 | 520 | 26.7 | 120.6 | 81.9 | 3.0 | -3.8 |
| a_baseline | 30 | 480 | 27.0 | 102.8 | 82.7 | 2.49 | -3.2 |
| b_per_type | 5 | 805 | 53.6 | 200.0 | 74.5 | 5.78 | 0.0 |
| b_per_type | 10 | 705 | 44.4 | 168.7 | 81.8 | 4.12 | -6.9 |
| b_per_type | 15 | 645 | 33.0 | 153.9 | 83.4 | 5.8 | -6.2 |
| b_per_type | 20 | 555 | 28.6 | 144.4 | 85.4 | 4.8 | 0.0 |
| b_per_type | 25 | 520 | 26.7 | 120.6 | 82.7 | 3.31 | -3.8 |
| b_per_type | 30 | 480 | 27.0 | 102.8 | 85.2 | 2.83 | -3.2 |

## Per horizon

| method | horizon_hours | n | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| a_baseline | 2.0 | 742 | 28.6 | 199.1 | 81.8 | 5.1 | 5.9 |
| a_baseline | 6.0 | 742 | 26.9 | 137.5 | 82.5 | 4.44 | 0.0 |
| a_baseline | 24.0 | 742 | 31.8 | 98.5 | 82.2 | 3.67 | -11.8 |
| a_baseline | 72.0 | 742 | 43.3 | 188.7 | 79.5 | 3.13 | -9.1 |
| a_baseline | 168.0 | 742 | 50.1 | 185.3 | 75.2 | 2.53 | -16.7 |
| b_per_type | 2.0 | 742 | 28.6 | 199.1 | 83.2 | 6.32 | 5.9 |
| b_per_type | 6.0 | 742 | 26.9 | 137.5 | 84.1 | 5.52 | 0.0 |
| b_per_type | 24.0 | 742 | 31.8 | 98.5 | 84.0 | 4.53 | -11.8 |
| b_per_type | 72.0 | 742 | 43.3 | 188.7 | 80.9 | 3.35 | -9.1 |
| b_per_type | 168.0 | 742 | 50.1 | 185.3 | 76.0 | 2.85 | -16.7 |

## Per (form_type x horizon) (для перевірки чи покращення на ВСІХ горизонтах)

| method | form_type | horizon_hours | n | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| a_baseline | creative_submission | 2.0 | 42 | 11.5 | 58.4 | 92.9 | 7.65 | 0.0 |
| a_baseline | creative_submission | 6.0 | 42 | 16.7 | 59.8 | 92.9 | 6.12 | 0.0 |
| a_baseline | creative_submission | 24.0 | 42 | 25.9 | 63.8 | 88.1 | 4.99 | -17.4 |
| a_baseline | creative_submission | 72.0 | 42 | 44.0 | 222.5 | 90.5 | 3.87 | -8.7 |
| a_baseline | creative_submission | 168.0 | 42 | 67.9 | 266.2 | 88.1 | 3.69 | -20.6 |
| a_baseline | event_feedback | 2.0 | 37 | 134.5 | 431.0 | 89.2 | 4.47 | 134.5 |
| a_baseline | event_feedback | 6.0 | 37 | 76.2 | 317.0 | 89.2 | 3.72 | 72.0 |
| a_baseline | event_feedback | 24.0 | 37 | 76.3 | 280.7 | 91.9 | 3.1 | 50.0 |
| a_baseline | event_feedback | 72.0 | 37 | 68.3 | 447.3 | 83.8 | 3.5 | 30.0 |
| a_baseline | event_feedback | 168.0 | 37 | 61.9 | 475.3 | 73.0 | 3.5 | 40.0 |
| a_baseline | event_registration | 2.0 | 189 | 16.7 | 179.7 | 92.6 | 6.18 | 4.0 |
| a_baseline | event_registration | 6.0 | 189 | 20.0 | 108.7 | 91.5 | 5.77 | 0.0 |
| a_baseline | event_registration | 24.0 | 189 | 28.6 | 76.2 | 90.5 | 4.77 | -15.6 |
| a_baseline | event_registration | 72.0 | 189 | 38.7 | 120.9 | 88.4 | 4.58 | -16.1 |
| a_baseline | event_registration | 168.0 | 189 | 51.6 | 140.1 | 85.2 | 4.41 | -22.4 |
| a_baseline | holiday | 2.0 | 41 | 52.3 | 297.7 | 73.2 | 9.22 | 45.5 |
| a_baseline | holiday | 6.0 | 41 | 42.1 | 224.1 | 78.0 | 7.95 | 26.4 |
| a_baseline | holiday | 24.0 | 41 | 34.4 | 158.8 | 85.4 | 6.28 | 9.4 |
| a_baseline | holiday | 72.0 | 41 | 54.5 | 206.1 | 78.0 | 2.82 | 51.5 |
| a_baseline | holiday | 168.0 | 41 | 39.8 | 164.9 | 73.2 | 2.06 | 20.0 |
| a_baseline | other | 2.0 | 31 | 30.0 | 166.7 | 93.5 | 7.15 | 0.0 |
| a_baseline | other | 6.0 | 31 | 30.0 | 128.6 | 93.5 | 6.23 | 0.0 |
| a_baseline | other | 24.0 | 31 | 34.8 | 100.0 | 96.8 | 5.0 | -6.2 |
| a_baseline | other | 72.0 | 31 | 48.3 | 354.5 | 100.0 | 3.92 | 18.2 |
| a_baseline | other | 168.0 | 31 | 51.6 | 343.3 | 96.8 | 3.86 | 22.9 |
| a_baseline | political | 2.0 | 20 | 21.0 | 129.3 | 75.0 | 4.56 | 10.8 |
| a_baseline | political | 6.0 | 20 | 24.6 | 88.9 | 80.0 | 4.27 | -5.0 |
| a_baseline | political | 24.0 | 20 | 32.8 | 78.4 | 85.0 | 3.66 | -16.7 |
| a_baseline | political | 72.0 | 20 | 27.7 | 95.2 | 80.0 | 2.8 | -12.7 |
| a_baseline | political | 168.0 | 20 | 34.1 | 88.8 | 80.0 | 2.08 | -30.4 |
| a_baseline | recruitment | 2.0 | 47 | 22.2 | 104.4 | 72.3 | 9.26 | 22.2 |
| a_baseline | recruitment | 6.0 | 47 | 17.9 | 73.1 | 80.9 | 8.32 | 10.0 |
| a_baseline | recruitment | 24.0 | 47 | 14.3 | 50.5 | 89.4 | 6.89 | -6.2 |
| a_baseline | recruitment | 72.0 | 47 | 25.9 | 116.0 | 85.1 | 4.29 | -7.8 |
| a_baseline | recruitment | 168.0 | 47 | 37.7 | 101.7 | 78.7 | 3.27 | -21.9 |
| a_baseline | service | 2.0 | 40 | 20.7 | 87.6 | 67.5 | 2.05 | 0.0 |
| a_baseline | service | 6.0 | 40 | 20.0 | 91.0 | 70.0 | 1.78 | 0.0 |
| a_baseline | service | 24.0 | 40 | 25.7 | 87.6 | 65.0 | 1.55 | -22.1 |
| a_baseline | service | 72.0 | 40 | 37.5 | 88.2 | 70.0 | 2.05 | -30.2 |
| a_baseline | service | 168.0 | 40 | 46.4 | 94.8 | 65.0 | 2.28 | -39.6 |
| a_baseline | survey | 2.0 | 110 | 54.0 | 99.3 | 63.6 | 1.4 | -12.3 |
| a_baseline | survey | 6.0 | 110 | 50.0 | 98.1 | 59.1 | 1.06 | -26.7 |
| a_baseline | survey | 24.0 | 110 | 53.5 | 97.0 | 51.8 | 0.8 | -49.1 |
| a_baseline | survey | 72.0 | 110 | 56.5 | 97.5 | 46.4 | 0.74 | -37.4 |
| a_baseline | survey | 168.0 | 110 | 57.7 | 95.5 | 41.8 | 0.74 | -45.6 |
| a_baseline | unknown | 2.0 | 114 | 32.1 | 250.9 | 81.6 | 6.1 | 8.7 |
| a_baseline | unknown | 6.0 | 114 | 27.0 | 168.4 | 83.3 | 5.53 | 0.0 |
| a_baseline | unknown | 24.0 | 114 | 29.3 | 100.0 | 86.8 | 4.41 | -8.3 |
| a_baseline | unknown | 72.0 | 114 | 38.2 | 221.8 | 80.7 | 3.55 | -1.0 |
| a_baseline | unknown | 168.0 | 114 | 47.8 | 232.2 | 77.2 | 2.91 | -5.5 |
| a_baseline | volunteer_donor | 2.0 | 71 | 20.0 | 250.0 | 87.3 | 4.1 | 10.0 |
| a_baseline | volunteer_donor | 6.0 | 71 | 28.6 | 216.7 | 90.1 | 3.73 | 10.0 |
| a_baseline | volunteer_donor | 24.0 | 71 | 33.3 | 161.1 | 87.3 | 3.28 | 0.0 |
| a_baseline | volunteer_donor | 72.0 | 71 | 58.1 | 223.1 | 90.1 | 2.94 | 0.0 |
| a_baseline | volunteer_donor | 168.0 | 71 | 61.2 | 216.0 | 84.5 | 2.8 | -5.0 |
| b_per_type | creative_submission | 2.0 | 42 | 11.5 | 58.4 | 95.2 | 8.4 | 0.0 |
| b_per_type | creative_submission | 6.0 | 42 | 16.7 | 59.8 | 95.2 | 6.7 | 0.0 |
| b_per_type | creative_submission | 24.0 | 42 | 25.9 | 63.8 | 90.5 | 5.49 | -17.4 |
| b_per_type | creative_submission | 72.0 | 42 | 44.0 | 222.5 | 92.9 | 4.05 | -8.7 |
| b_per_type | creative_submission | 168.0 | 42 | 67.9 | 266.2 | 90.5 | 3.98 | -20.6 |
| b_per_type | event_feedback | 2.0 | 37 | 134.5 | 431.0 | 89.2 | 4.96 | 134.5 |
| b_per_type | event_feedback | 6.0 | 37 | 76.2 | 317.0 | 89.2 | 3.91 | 72.0 |
| b_per_type | event_feedback | 24.0 | 37 | 76.3 | 280.7 | 94.6 | 3.1 | 50.0 |
| b_per_type | event_feedback | 72.0 | 37 | 68.3 | 447.3 | 83.8 | 3.5 | 30.0 |
| b_per_type | event_feedback | 168.0 | 37 | 61.9 | 475.3 | 73.0 | 3.5 | 40.0 |
| b_per_type | event_registration | 2.0 | 189 | 16.7 | 179.7 | 92.6 | 6.7 | 4.0 |
| b_per_type | event_registration | 6.0 | 189 | 20.0 | 108.7 | 91.5 | 6.31 | 0.0 |
| b_per_type | event_registration | 24.0 | 189 | 28.6 | 76.2 | 91.5 | 5.27 | -15.6 |
| b_per_type | event_registration | 72.0 | 189 | 38.7 | 120.9 | 88.9 | 4.86 | -16.1 |
| b_per_type | event_registration | 168.0 | 189 | 51.6 | 140.1 | 85.2 | 4.74 | -22.4 |
| b_per_type | holiday | 2.0 | 41 | 52.3 | 297.7 | 73.2 | 14.59 | 45.5 |
| b_per_type | holiday | 6.0 | 41 | 42.1 | 224.1 | 78.0 | 12.58 | 26.4 |
| b_per_type | holiday | 24.0 | 41 | 34.4 | 158.8 | 85.4 | 10.01 | 9.4 |
| b_per_type | holiday | 72.0 | 41 | 54.5 | 206.1 | 78.0 | 2.82 | 51.5 |
| b_per_type | holiday | 168.0 | 41 | 39.8 | 164.9 | 73.2 | 2.06 | 20.0 |
| b_per_type | other | 2.0 | 31 | 30.0 | 166.7 | 93.5 | 5.74 | 0.0 |
| b_per_type | other | 6.0 | 31 | 30.0 | 128.6 | 93.5 | 5.0 | 0.0 |
| b_per_type | other | 24.0 | 31 | 34.8 | 100.0 | 96.8 | 4.03 | -6.2 |
| b_per_type | other | 72.0 | 31 | 48.3 | 354.5 | 100.0 | 3.29 | 18.2 |
| b_per_type | other | 168.0 | 31 | 51.6 | 343.3 | 96.8 | 3.31 | 22.9 |
| b_per_type | political | 2.0 | 20 | 21.0 | 129.3 | 75.0 | 6.34 | 10.8 |
| b_per_type | political | 6.0 | 20 | 24.6 | 88.9 | 85.0 | 5.93 | -5.0 |
| b_per_type | political | 24.0 | 20 | 32.8 | 78.4 | 85.0 | 5.08 | -16.7 |
| b_per_type | political | 72.0 | 20 | 27.7 | 95.2 | 80.0 | 3.14 | -12.7 |
| b_per_type | political | 168.0 | 20 | 34.1 | 88.8 | 80.0 | 2.48 | -30.4 |
| b_per_type | recruitment | 2.0 | 47 | 22.2 | 104.4 | 76.6 | 12.7 | 22.2 |
| b_per_type | recruitment | 6.0 | 47 | 17.9 | 73.1 | 83.0 | 11.43 | 10.0 |
| b_per_type | recruitment | 24.0 | 47 | 14.3 | 50.5 | 89.4 | 9.61 | -6.2 |
| b_per_type | recruitment | 72.0 | 47 | 25.9 | 116.0 | 85.1 | 5.78 | -7.8 |
| b_per_type | recruitment | 168.0 | 47 | 37.7 | 101.7 | 80.9 | 4.31 | -21.9 |
| b_per_type | service | 2.0 | 40 | 20.7 | 87.6 | 67.5 | 3.03 | 0.0 |
| b_per_type | service | 6.0 | 40 | 20.0 | 91.0 | 70.0 | 2.82 | 0.0 |
| b_per_type | service | 24.0 | 40 | 25.7 | 87.6 | 67.5 | 2.44 | -22.1 |
| b_per_type | service | 72.0 | 40 | 37.5 | 88.2 | 70.0 | 3.21 | -30.2 |
| b_per_type | service | 168.0 | 40 | 46.4 | 94.8 | 65.0 | 4.58 | -39.6 |
| b_per_type | survey | 2.0 | 110 | 54.0 | 99.3 | 69.1 | 1.88 | -12.3 |
| b_per_type | survey | 6.0 | 110 | 50.0 | 98.1 | 65.5 | 1.41 | -26.7 |
| b_per_type | survey | 24.0 | 110 | 53.5 | 97.0 | 58.2 | 1.06 | -49.1 |
| b_per_type | survey | 72.0 | 110 | 56.5 | 97.5 | 52.7 | 0.93 | -37.4 |
| b_per_type | survey | 168.0 | 110 | 57.7 | 95.5 | 43.6 | 0.78 | -45.6 |
| b_per_type | unknown | 2.0 | 114 | 32.1 | 250.9 | 82.5 | 7.8 | 8.7 |
| b_per_type | unknown | 6.0 | 114 | 27.0 | 168.4 | 85.1 | 6.73 | 0.0 |
| b_per_type | unknown | 24.0 | 114 | 29.3 | 100.0 | 86.8 | 5.38 | -8.3 |
| b_per_type | unknown | 72.0 | 114 | 38.2 | 221.8 | 81.6 | 3.65 | -1.0 |
| b_per_type | unknown | 168.0 | 114 | 47.8 | 232.2 | 78.9 | 3.12 | -5.5 |
| b_per_type | volunteer_donor | 2.0 | 71 | 20.0 | 250.0 | 87.3 | 4.62 | 10.0 |
| b_per_type | volunteer_donor | 6.0 | 71 | 28.6 | 216.7 | 90.1 | 4.43 | 10.0 |
| b_per_type | volunteer_donor | 24.0 | 71 | 33.3 | 161.1 | 88.7 | 3.3 | 0.0 |
| b_per_type | volunteer_donor | 72.0 | 71 | 58.1 | 223.1 | 90.1 | 3.24 | 0.0 |
| b_per_type | volunteer_donor | 168.0 | 71 | 61.2 | 216.0 | 84.5 | 2.91 | -5.0 |

## Figures

- [Per-type coverage A vs B](figures\13_per_type_coverage.html)
- [Per-type sharpness A vs B](figures\13_per_type_sharpness.html)

## Критерії promote

| Критерій | Поріг |
|---|---|
| Per-type cov: survey, service, holiday | > 80% (з 53/67/78 → ≥80) |
| Per-type cov: усі типи | gap до 95% ≤ 10pp |
| Sharpness | НЕ зростає більш ніж +50% global |
| MAPE | незмінне (point estimate не торкаємо) |
| Global cov | gain ≥ 0 |

## Артефакти

- `figures/13_ab_points.csv` — wide-format raw.
- `figures/13_ab_metrics.csv` — long-format per-method метрики.
