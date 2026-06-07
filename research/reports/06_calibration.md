# 06 — CI Calibration Sweep

**Target coverage:** 95%
**Backtest points:** 288

## Sweep

| Multiplier | Coverage | Median Width |
|---:|---:|---:|
| 1.0 | 0.309 | 13 |
| 2.0 | 0.538 | 26 |
| 3.0 | 0.590 | 39 |
| 5.0 | 0.670 | 65 |
| 7.0 | 0.698 | 91 |
| 10.0 | 0.726 | 130 |
| 15.0 | 0.757 | 195 |
| 20.0 | 0.764 | 260 |
| 30.0 | 0.781 | 390 |
| 50.0 | 0.788 | 650 |
| 100.0 | 0.792 | 1300 |

## Recommended

**CALIBRATION_MULTIPLIER = 100.0** → emp.coverage = 0.792

Записати у `core/forecast/calibration.py`.
