# 05 — Segmented vs Default Forecast (A/B Comparison)

**Generated:** 2026-05-29T02:25:54
**Source:** `D:\GitHub\survey-insight\data\Form Timestamp Collection.csv` (sha256:`0dfa60a9a2cd`)
**Eligible forms:** 96 · **Points per mode:** 288 / 288

## TL;DR

| Метрика | Default | Segmented | Δ |
|---|---:|---:|---:|
| **MAPE (median)** | 22.7% | 54.0% | +31.3pp |
| **Coverage 95%** | 30.6% | 21.5% | -9.0pp |
| **CP total** | — | 371 | — |
| **CP per backtest (avg)** | — | 1.29 | — |

**Paired test:** Wilcoxon W=938, p=8.267e-18, n_paired=288

## Per-shape ΔMAPE

| shape | n | mape_default | mape_segmented | delta_mape | cov_default | cov_segmented | delta_cov | bias_default | bias_segmented |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ill_fit | 6 | 38.7 | 67.7 | 29.0 | 0.0 | 0.0 | 0.0 | -27.4 | -55.8 |
| late_burst | 18 | 30.7 | 69.4 | 38.7 | 27.8 | 16.7 | -11.100000000000001 | -24.8 | -67.1 |
| linear | 6 | 18.0 | 27.1 | 9.100000000000001 | 66.7 | 50.0 | -16.700000000000003 | -18.0 | -27.1 |
| logarithmic | 147 | 19.5 | 59.4 | 39.9 | 29.3 | 21.8 | -7.5 | 0.0 | -12.2 |
| logistic | 111 | 25.0 | 48.8 | 23.799999999999997 | 32.4 | 21.6 | -10.799999999999997 | -16.7 | -36.8 |

Інтерпретація:
- `delta_mape < 0` → segmented краще (модель захоплює структуру через CP).
- `delta_mape > 0` → segmented гірше (false-positive CP розрізали корисну
  криву на дрібні шматки).
- `delta_cov > 0` → segmented дає кращий PI calibration.

## Графіки

- [Δ MAPE per shape (bar)](figures\05_delta_mape.html)
- [Average CP count per shape](figures\05_cp_freq.html)

## Висновок для production: чесний негативний результат

**Гіпотеза з research/04 (87% Ljung-Box rejection → потрібна сегментація)
НЕ ПІДТВЕРДИЛАСЬ для дефолтних параметрів PELT.**

Спостерігаємо погіршення СКРІЗЬ:
- `late_burst` (очікувана target-категорія): MAPE 30.7% → 69.4% (+38.7pp)
- `ill_fit` (очікувана target-категорія): MAPE 38.7% → 67.7% (+29.0pp)
- `logarithmic` (51% точок): MAPE 19.5% → 59.4% (+39.9pp)
- `logistic` (39% точок): MAPE 25.0% → 48.8% (+23.8pp)

Wilcoxon W=938, p=8.3e-18 — статистично значуща погана різниця.

### Чому так

PELT з `penalty=10` знаходить **~1.3 CP в середньому per backtest**, навіть
на чистих logarithmic-кривих, де реальних хвиль немає (false positives).
Це призводить до того, що training subset = post-last-CP сегмент стає
закоротким (часто 5-15 точок замість 30-90), і параметрична модель
overfit'ить локальний нахил, не бачачи глобального тренду насичення.

Корінна проблема: PELT детектує зміни **середнього значення rate**, але
для cumulative-сурваїв rate ЗАКОНОМІРНО спадає з часом (бо modeled
saturation). PELT приймає це за хвилі-CP.

### Production-наслідки

**`auto_segment=False` як default у `forecast_with_segmentation`.** Модуль
залишається тільки для:
1. **Візуалізації** — CP-маркери на графіку як інформативний overlay
   (користувач бачить, де PELT думає, що були хвилі).
2. **Експертний opt-in** — `auto_segment=True` доступний як kwarg для
   advanced UI або A/B-experiments на тих формах, де користувач знає
   про справжню хвилю агітації.

### Що варто спробувати у майбутньому (поза цим скоупом)

1. **Cost model 'rbf' замість 'l2'** — детектує зміни дисперсії і шейпу,
   не лише середнього. Може бути робастніше до природного спаду rate.
2. **Per-shape penalty tuning** — sweep penalty ∈ {30, 50, 100} на форми
   шейпів `late_burst` / `ill_fit` окремо.
3. **Detrending rate перед PELT** — спершу відняти AsymptoticExp-фіт,
   потім шукати CP у residuals (а не у raw rate).
4. **CUSUM замість PELT** — більш sensitive до monotonic зростання rate.

Ці експерименти заслуговують окремого диплом-розділу. Поточний звіт
фіксує negative result — academic-grade falsification гіпотези.
