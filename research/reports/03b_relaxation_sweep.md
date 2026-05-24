# 03b — K_min Relaxation Sweep

Запуск на 96 формах × 3 cutoff'и × 7 factors.

## Sweep results

| factor | n | bias % | |bias| | MAPE % | coverage % | score |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 288 | -6.25 | 6.25 | 27.67 | 50.00 | 51.25 |
| 1.05 | 288 | -2.62 | 2.62 | 27.87 | 51.04 | 46.58 |
| 1.1 | 288 | +0.00 | 0.00 | 26.96 | 53.47 | 41.53 |
| 1.15 | 288 | +2.74 | 2.74 | 25.02 | 55.21 | 42.54 |
| 1.2 | 288 | +5.41 | 5.41 | 24.32 | 57.29 | 43.12 |
| 1.3 | 288 | +9.48 | 9.48 | 22.42 | 61.46 | 43.02 |
| 1.5 | 288 | +16.46 | 16.46 | 27.55 | 67.01 | 44.44 |

## Найкращий factor: **1.1**

За composite score = |bias| + |coverage - 95|.

## Графік

[K_min sweep: bias / coverage / MAPE](figures\03b_relaxation_sweep.html)

## Recommendation

Змінити `_capacity_bounds` у `core/forecast/models.py`:

```python
def _capacity_bounds(y, target):
    last = float(y[-1])
    if target is not None and target > 0:
        return max(last * 1.1, 0.3 * target), max(last * 1.05, 3.0 * target)
    return max(last * 1.1, 1.0), max(last * 10.0, 10.0)
```

Очікувані ефекти на production:
- Bias: -6.2% → +0.0%
- Coverage: 50.0% → 53.5%
- MAPE: 27.7% → 27.0%
