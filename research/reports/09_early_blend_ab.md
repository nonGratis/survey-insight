# 09 — A/B/C для ранніх передбачень (cutoffs 0.1, 0.2)

**Generated:** 2026-05-29T18:58:29
**Source:** `D:\GitHub\survey-insight\data\Form Timestamp Collection.csv` (sha256:`0dfa60a9a2cd`)
**Forms:** 177 → 141 eligible (N ≥ 10)
**Cutoffs:** (0.1, 0.2, 0.3, 0.5, 0.7) · **Horizon fraction:** 0.25
**Backtest points / method:** 590
**Skipped:** {'too_few': 36, 'insufficient_shape': 0, 'no_span': 0}

## Методи

- **A · model** — поточний `forecast_responses` (контроль, реплікує 08_).
- **B · naive** — Poisson MLE на (n_train, cutoff_span), exact γ-CI 95% на інкременті.
- **C · blend** — `α·model + (1−α)·naive`, де `α = clip((n_train − 5) / 25, 0, 1)`. На n_train=5 → pure naive, на n_train≥30 → pure model.

## Глобально (усі cutoffs)

| method | n | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|
| model | 590 | 29.4 | 141.2 | 87.3 | 3.2 | -10.5 |
| naive | 590 | 59.8 | 1701.4 | 25.6 | 0.62 | 44.6 |
| blend | 590 | 39.2 | 969.6 | 56.4 | 2.44 | 12.9 |

## Decisive view: method × cutoff

| method | cutoff_frac | n | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| blend | 0.1 | 74 | 174.8 | 2090.2 | 29.7 | 1.99 | 174.8 |
| blend | 0.2 | 106 | 80.3 | 1211.8 | 41.5 | 2.67 | 80.3 |
| blend | 0.3 | 128 | 62.9 | 955.5 | 50.0 | 2.8 | 50.8 |
| blend | 0.5 | 141 | 30.3 | 533.3 | 68.1 | 2.67 | -6.2 |
| blend | 0.7 | 141 | 17.0 | 144.0 | 75.9 | 2.16 | -4.6 |
| model | 0.1 | 74 | 46.9 | 99.2 | 85.1 | 1.92 | -7.7 |
| model | 0.2 | 106 | 40.1 | 149.7 | 84.9 | 3.63 | -4.2 |
| model | 0.3 | 128 | 39.0 | 180.1 | 87.5 | 3.83 | -6.3 |
| model | 0.5 | 141 | 29.1 | 118.8 | 87.2 | 3.29 | -15.0 |
| model | 0.7 | 141 | 17.6 | 33.3 | 90.1 | 2.79 | -11.1 |
| naive | 0.1 | 74 | 567.4 | 3369.1 | 10.8 | 1.3 | 567.4 |
| naive | 0.2 | 106 | 265.8 | 2487.3 | 9.4 | 1.08 | 265.8 |
| naive | 0.3 | 128 | 113.6 | 2239.7 | 21.9 | 0.95 | 113.6 |
| naive | 0.5 | 141 | 35.0 | 1002.0 | 29.8 | 0.5 | 21.7 |
| naive | 0.7 | 141 | 14.3 | 360.0 | 44.7 | 0.37 | 12.9 |

## Per-shape на ранніх cutoffs (0.1, 0.2)

| method | shape | n | mape_p50 | mape_p90 | coverage | sharpness_p50 | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| blend | ill_fit | 4 | 911.7 | 2800.9 | 25.0 | 7.15 | 911.7 |
| blend | late_burst | 11 | 60.0 | 83.7 | 54.5 | 1.31 | -24.7 |
| blend | linear | 3 | 9.1 | 417.8 | 66.7 | 1.48 | 9.1 |
| blend | logarithmic | 96 | 207.1 | 2087.3 | 28.1 | 2.35 | 207.1 |
| blend | logistic | 66 | 89.4 | 1108.2 | 45.5 | 2.47 | 88.0 |
| model | ill_fit | 4 | 111.7 | 297.9 | 75.0 | 7.27 | 111.7 |
| model | late_burst | 11 | 38.5 | 80.5 | 72.7 | 1.63 | -38.5 |
| model | linear | 3 | 45.5 | 57.1 | 66.7 | 4.45 | -32.0 |
| model | logarithmic | 96 | 40.6 | 100.0 | 91.7 | 2.22 | 3.7 |
| model | logistic | 66 | 41.7 | 141.7 | 78.8 | 3.83 | -9.5 |
| naive | ill_fit | 4 | 2143.9 | 3150.0 | 0.0 | 4.48 | 2143.9 |
| naive | late_burst | 11 | 55.6 | 83.7 | 36.4 | 0.22 | -28.3 |
| naive | linear | 3 | 36.0 | 475.2 | 33.3 | 1.09 | 36.0 |
| naive | logarithmic | 96 | 815.7 | 3611.5 | 2.1 | 1.61 | 815.7 |
| naive | logistic | 66 | 72.5 | 1263.0 | 16.7 | 0.76 | 31.2 |

## Фігури

- [MAPE p50 за cutoff × method](figures\09_mape_per_cutoff.html)
- [Coverage за cutoff × method](figures\09_coverage_per_cutoff.html)

## Як читати

1. **Дивимось рядки cutoff=0.1, 0.2** у method × cutoff таблиці. Очікуємо
   що **C (blend)** має найнижчий MAPE_p50 і coverage найближче до 95%.
   Якщо B (pure naive) теж непогано — це сигнал що curve-fit на малих N
   взагалі не приносить value.
2. **Дивимось рядки cutoff=0.5, 0.7**: усі три методи мають збігатися
   близько до model (бо α→1). Якщо B сильно гірше — підтверджує що
   curve-fit вигідний на зрілих формах.
3. **Per-shape early-cutoff таблиця**: де A>>B (тобто model значно
   гірше за naive) — це shape-категорії, для яких curve-fit
   контрпродуктивний на ранніх етапах.

## Verdict — **NEGATIVE RESULT**

Гіпотеза «blending з naive Poisson покращить ранні cutoffs» **спростована
на 590-point full-dataset**. Жоден критерій promote не виконано:

| Критерій | Очікувалось | Реально | Результат |
|---|---|---|---|
| C MAPE@0.1 < A MAPE@0.1 | < 46.9% | **174.8%** | ✗ FAIL (+128pp) |
| C MAPE@0.2 < A MAPE@0.2 | < 40.1% | **80.3%** | ✗ FAIL (+40pp) |
| C MAPE@0.7 ≤ A + 2pp | ≤ 19.6% | 17.0% | ✓ pass |
| C coverage@0.1 ≥ A | ≥ 85.1% | **29.7%** | ✗ FAIL (-55pp) |

Не промоутимо. `core/forecast/` лишається без змін.

### Чому naive так сильно програв

Naive Poisson припускає **homogeneous rate λ** на весь горизонт. Реальні
форми мають **rate decay**: перші години після шарингу — пік потоку,
далі рейт згасає. На cutoff=0.1 ми бачимо саме peak-rate, naive
екстраполює його forward → bias +567%.

Curve-моделі (Logistic/Gompertz/AsympExp) явно параметризують саме цей
decay — тому навіть нестабільний fit на n_train=5..15 **краще за
homogeneous-rate assumption** для цього домену.

### Єдиний виняток

На **shape=linear** (n=3 у early-cutoff bucket, занадто мало для
рішення) blend MAPE=9.1% vs model 45.5%. Логічно: лінійні форми = саме
ті, де homogeneous rate й справді тримається. Якщо буде окремий
classifier-tier для shape=linear, можна повернутися до цієї ідеї там
точково. Зараз — лишити як ткритий tail-insight.

### Пропоновані наступні експерименти

1. **Akaike weighted model averaging** — замість selection-by-AICc усереднити
   всі 3 моделі за вагами `wi = exp(-ΔAICᵢ/2) / Σ exp(-ΔAICⱼ/2)`. Не змінює
   inductive bias (decay-capturing), але прибирає selection-variance — головне
   джерело шуму на малих N.
2. **Stronger priors на малих N** — у `priors.py:narrow_bounds_with_prior`
   зменшити `PRIOR_N_SIGMA` з 2.0 до 1.0 для `n_train<15`. Сильніше тягне
   prediction до population-median, але робастніше.
3. **Per-cutoff calibration multiplier** — замість глобального ×10 у
   `calibration.py`, scale CI ширше на ранніх cutoffs: ×15 для n_train<15,
   ×10 для n_train≥15. Не покращить MAPE, але закриє coverage gap
   (85% → ~95%).

## Артефакти

- `figures/09_ab_points.csv` — wide-format raw результати.
- `figures/09_ab_metrics.csv` — long-format per-method метрики.
