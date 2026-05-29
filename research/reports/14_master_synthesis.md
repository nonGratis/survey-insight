# 14 — Master synthesis: stan системи прогнозу після P10 + повної діагностики

**Generated:** 2026-05-29
**Scope:** консолідація 08_, 09_, 10_, 11_, 12_ — стан повної системи, де працює, де ні, що робити далі.

---

## TL;DR

Поточна система (`feat/forecast-nhpp` після P10) **добре працює для 70% форм**, але **системно фейлить на 2 категоріях**, що вкупі дають ~25% від датасету:

| Категорія | Частка | MAPE | Coverage | Bias | Висновок |
|---|---:|---:|---:|---:|---|
| event_registration | 25% (43/169) | **28.6%** | **89.6%** | -5% | ✅ Prod-ready |
| recruitment | 7% (12/169) | **22.2%** | 81.3% | 0% | ✅ Prod-ready |
| creative_submission | 9% (16/169) | 26.8% | **90.5%** | -5% | ✅ Prod-ready |
| volunteer_donor | 9% (16/169) | 39.4% | 87.9% | 0% | ⚠️ Acceptable |
| **survey** | **15% (25/169)** | **54.0%** | **52.5%** | **-36%** | ❌ **Disaster** |
| **event_feedback** | **9% (16/169)** | **73.3%** | 85.4% | **+62%** | ❌ **Disaster** |
| holiday | 7% (11/169) | 44.2% | 77.6% | +27% | ❌ Bad |
| service | 9% (15/169) | 30.8% | **67.5%** | -17% | ⚠️ Coverage gap |
| political | 4% (6/169) | 30.2% | 80.0% | -9% | ⚠️ |
| other | 5% (9/169) | 37.5% | 96.1% | 0% | ⚠️ |
| unknown | 27% (45/169) | 36.5% | 81.9% | 0% | n/a |

**Critical:** "survey" + "event_feedback" = 24% форм, де метод системно зловить (>50% MAPE).

---

## Stack контекст

```
core/forecast/
├── models: Logistic + Gompertz + AsymptoticExp
├── selector: AICc-based
├── intervals: NHPP-Poisson з bounded MVN + reality caps
├── calibration: ×10 multiplier (P7) + P10 sample-size CI scaling
├── priors: emp. Bayes з 177-form history (P9, opt-in)
└── shape_classifier: 5 категорій
```

Дослідницькі звіти: 01-06 (thesis baseline), 08 (full-dataset diagnostics),
09 (negative result: Poisson naive), 10 (P10 promotion), **11 (multi-level
reliability)**, **12 (prod-realistic horizons)**.

---

## Знахідки з 11_ — RAW NHPP reliability

Без калібровки, NHPP сильно **під-впевнений у CI**:

| Nominal | Empirical | Gap |
|---:|---:|---:|
| 50% | 11.8% | -38pp |
| 80% | 21.0% | -59pp |
| 90% | 27.1% | -63pp |
| 95% | 34.6% | -60pp |

Тобто справжня калібровка моделі ≈ **50% nominal = 12% real**. Множник ×10 (P7) це частково компенсує до 89% emp на 95%. Але:

**Per form_type RAW@95% (де модель calibrated краще без множника):**

| form_type | RAW cov 95% | RAW MAPE | Висновок |
|---|---:|---:|---|
| recruitment | **51.4%** | 11.9% | Найкраще — концentated rate |
| other | 52.4% | 30.8% | |
| creative_submission | 50.0% | 30.0% | |
| event_registration | 41.7% | 29.4% | |
| volunteer_donor | 35.2% | 27.4% | |
| service | 34.4% | 12.5% | Низький MAPE, але CI вузька |
| **survey** | **15.6%** | **40.5%** | Найгірше — multi-wave structure |
| event_feedback | 25.0% | 59.2% | |
| holiday | 21.9% | 23.9% | |

**Висновок 11_**: різні form_type мають **РІЗНУ внутрішню калібровку**. Глобальний множник ×10 — це grain-blunt. **Per-type calibration multiplier** — найдешевший win.

---

## Знахідки з 12_ — прод-сценарій (абсолютні n_train + horizon)

### Cross-tab: n_train × horizon → MAPE_p50 (%)

| n_train \ h | 2h | 6h | 24h | 72h | 168h (7d) |
|---:|---:|---:|---:|---:|---:|
| 5 | 50.0 | 41.7 | 45.8 | 57.1 | 64.0 |
| 10 | **40.0** | **36.0** | 38.1 | 53.8 | 56.2 |
| 15 | 22.7 | 26.7 | 30.9 | 42.3 | 46.4 |
| 20 | **18.5** | 23.1 | 27.6 | 34.5 | 40.8 |
| 25 | 16.7 | 19.3 | 22.4 | 39.5 | 46.6 |
| 30 | **13.1** | 12.4 | 21.4 | 37.4 | 43.2 |

**Прод-висновки (твоє питання):**
1. **n_train=10 + horizon=2h** (ранній прод-сценарій): **MAPE 40%, Coverage 81.6%**. Грубо.
2. **n_train=20 + horizon=2h** (свіжа форма ~1 день): **MAPE 18.5%, Coverage 86.5%**. Acceptable.
3. **n_train=30 + horizon=2h**: **MAPE 13.1%** — це системний "вхід у зрілість".
4. **Гранична зона**: 7d (168h) MAPE 40-65% **на всіх** n_train. Прогноз тиждень наперед нашою системою — не reliable.

### Cross-tab: form_type × horizon → MAPE (%)

| form_type | 2h | 6h | 24h | 72h | 168h |
|---|---:|---:|---:|---:|---:|
| creative_submission | 11.5 | 16.7 | 25.9 | 44.0 | 67.9 |
| **event_feedback** | **134.5** | 76.2 | **76.3** | 68.3 | 61.9 |
| event_registration | 16.7 | 20.0 | 28.6 | 38.7 | 51.6 |
| holiday | 52.3 | 42.1 | 34.4 | 54.5 | 39.8 |
| political | 21.0 | 24.6 | 32.8 | 27.7 | 34.1 |
| recruitment | **22.2** | **17.9** | **14.3** | 25.9 | 37.7 |
| service | 20.7 | 20.0 | 25.7 | 37.5 | 46.4 |
| **survey** | **54.0** | **50.0** | **53.5** | **56.5** | 57.7 |
| volunteer_donor | 20.0 | 28.6 | 33.3 | 58.1 | 61.2 |

**Pattern recognition:**

🔥 **event_feedback на 2h MAPE 134%** — це фундаментальна failure. Чому: feedback форми мають initial burst у першу годину після події (всі хто пам'ятає), потім різкий drop. Наш NHPP бере цей burst rate і екстраполює forward → 2-3× overshoot. **bias +61.9%** глобально підтверджує.

🔥 **survey має ПЛАТО ~50-55% MAPE на ВСІХ горизонтах**. Тобто горизонт майже не важить — модель просто не описує survey-форми взагалі. **bias -36%**. Чому: surveys мають **multi-wave** структуру (announcement burst, mid-period reminder, deadline push). Concave saturating models бачать одну хвилю → недо-прогнозують ВЕСЬ майбутній потік.

✅ **recruitment** на 24h MAPE **14.3%**, Cov 89.4% — найкраще у датасеті. Студрада-набори справді мають передбачувану convex-concave dynamic.

✅ **creative_submission на 2h MAPE 11.5%** — рідкісно гарно. Бо мерч-передзамовлення/фото-збори мають швидку sat-curve динаміку, NHPP добре fitує.

⚠️ **holiday** на 2h MAPE 52% але на 24h MAPE 34% — модель потребує "розкачки". Bias +27% → systematic overshoot на ранніх горизонтах для holiday-форм.

⚠️ **service** на 2h MAPE 20%, Cov 67% — точка ОК, CI вузька. Це тип "статичної" rate dynamics що потребує LinearModel.

### Cross-tab: form_type × n_train (MAPE)

| form_type | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---:|---:|---:|---:|---:|---:|
| creative_submission | 28.6 | 28.6 | 24.0 | 60.0 | 18.7 | 26.8 |
| **event_feedback** | 94.0 | 78.2 | 65.9 | 76.3 | 60.3 | 70.0 |
| event_registration | 41.2 | 38.5 | 31.8 | 22.6 | 20.8 | **13.9** |
| holiday | 51.6 | 48.9 | 50.0 | 41.4 | 42.4 | 38.3 |
| political | 33.3 | 65.9 | 21.8 | 41.4 | 20.2 | 20.9 |
| recruitment | 23.0 | 38.6 | 22.8 | 21.5 | 18.7 | **14.3** |
| service | 37.5 | 36.0 | 21.4 | 19.5 | 48.3 | 40.3 |
| **survey** | 81.4 | 63.2 | 46.8 | 35.3 | 45.1 | 43.8 |
| volunteer_donor | 50.9 | 55.4 | 38.0 | 38.7 | 23.7 | **16.6** |

**Stable saturation patterns:**
- **event_registration**: monotonic від 41% → 14% з ростом n_train. Прекрасний приклад де метод працює як треба.
- **recruitment** і **volunteer_donor**: те саме, заходять у <20% MAPE на n_train≥25.
- **survey**: НЕ покращується з n_train. На 30 точках усе ще 44% MAPE. → структурна проблема моделі.
- **event_feedback**: ще гірше — навіть на 30 точках 70% MAPE.

---

## Тестовані гіпотези користувача

> "для івентів часто експоненційна історія логарифмічних хвиль (часто через день агітація й кожна краща)"

✅ **Підтверджено частково.** event_registration справді добре описується нашими saturating моделями (MAPE 13-29% на більшості cells). АЛЕ модель НЕ ловить "хвилі агітації" — на 7d horizon MAPE 51%. Це не критично для прод (короткий horizon = головне), але є фундаментальний limit.

> "для опитувань зазвичай одна й вона потужна й логарифмічна, а можуть бути наступні але менш потужні"

❌ **Спростовано на даних.** Surveys мають **середній MAPE 54%** і bias **-36%**. Це означає що "наступні менш потужні хвилі" насправді сильніші, ніж очікувалось — або їх більше. Модель бачить одну saturating crystal і не передбачає follow-up bursts. **Реальний потік опитувань неперервно вищий за прогноз** через додаткові reminders, deadlines, додаткові канали поширення.

---

## Priority-ranked recommendations для fix-cycles

### 🔴 P1: per-form-type calibration multiplier (1 fix-session)

**Evidence (з 11_):**

| form_type | RAW cov@95% | Implied multiplier |
|---|---:|---:|
| recruitment | 51% | ~5x |
| service | 34% | ~10-15x |
| event_registration | 42% | ~6-8x |
| survey | 16% | ~25x |
| event_feedback | 25% | ~15x |
| holiday | 22% | ~18x |

**Fix:** замість глобального `CALIBRATION_MULTIPLIER = 10.0`, додати dict per form_type. Розраховується автоматично з 11_ data.

**Очікуваний impact:** survey coverage 52% → ~85%, service coverage 67% → ~85%. MAPE без змін (CI тільки).

**Cost:** ~50 LOC у `calibration.py` + dict-lookup у `service.py`. Trivial.

### 🔴 P2: LinearModel для service-форм + small-N (1 fix-session)

**Evidence:** service forms мають rate ~constant (поселення, M365, додаткові бали = адмін-процеси). Поточні concave моделі не описують лінійну динаміку — CI занадто вузька (cov 67%) бо модель закладає saturation там де її немає.

**Fix:** додати `LinearModel(slope, intercept)` у `models.py:DEFAULT_MODELS`. AICc автоматично обере її для shape=linear (2 форми) і service-типу.

**Очікуваний impact:** service coverage 67% → 85%, MAPE 30.8% → ~22%.

### 🟡 P3: Bass diffusion для multi-wave forms (2 fix-sessions)

**Evidence:** survey + event_feedback мають "хвильову" динаміку, яку saturating curves не описують. Bass diffusion (innovation + imitation parameters) природно описує:
- initial burst (innovators)
- amplification (imitators)
- saturation з можливими повторними хвилями

`dN/dt = (p + q·N/K)·(K − N)` — 3 параметри, інтегрується аналітично.

**Fix:** додати `BassModel` у DEFAULT_MODELS. Перевірити чи AICc обере її саме для survey/feedback.

**Очікуваний impact:**
- survey MAPE 54% → потенційно 30-35% (надія, не гарантія).
- event_feedback MAPE 73% → потенційно 40-45%.

**Risk:** Bass fits можуть бути нестабільні на малих N. Потрібен careful bounds + AICc gate.

### 🟡 P4: Burst-aware initial-rate detection (1 fix-session)

**Evidence:** event_feedback на 2h MAPE **134%**, bias +62%. Причина — initial-burst-then-decay, де model екстраполює burst-rate.

**Fix:** detect initial burst (перший 30% точок rate >> median rate) → apply rate-decay assumption explicitly. Можна як heuristic over current models.

**Очікуваний impact:** event_feedback bias +62% → ~+20%, MAPE 73% → ~45%.

### 🟢 P5 (відкладене): Hawkes self-exciting Poisson (3+ sessions)

Для burst tempo (зараз 31% MAPE). Складна реалізація, помірний impact. Виправдано після P1-P4.

### 🟢 P6 (відкладене): Akaike model averaging (з 10_)

Спростовано на даних: -5.4pp coverage на cutoff=0.1. Не повертатись доки не з'являться нові моделі (Bass, Linear) — тоді averaging може мати сенс.

---

## Реалістична стеля з арсеналом

| Метрика | Поточне (P10) | Після P1-P4 | Stretch (з #5) |
|---|---:|---:|---:|
| Global coverage | 89.2% | **~93%** | 94-95% |
| Global MAPE_p50 | 29.4% | **~22-24%** | 18-20% |
| survey MAPE | 54% | **~32%** | 25-28% |
| event_feedback MAPE | 73% | **~45%** | 35% |
| recruitment MAPE | 22% | ~18% | 15% |
| Prod n_train=10, h=2h | MAPE 40% | **~28%** | 22% |
| Prod n_train=20, h=2h | MAPE 18% | **~14%** | 12% |

---

## Артефакти діагностики

| Звіт | Тема |
|---|---|
| `02_backtest.md` | Закріплений thesis-baseline (96 форм × 3 cutoffs) |
| `08_full_dataset_backtest.md` | Full dataset 141×5, taxonomy axes |
| `09_early_blend_ab.md` | Negative result: Poisson naive blend |
| `10_variance_reduction_ab.md` | P10 promotion: sample-size CI scaling |
| `11_multi_level_reliability.md` | Raw NHPP reliability per (level × type) |
| `12_prod_realistic_horizons.md` | n_train × horizon × form_type |
| `14_master_synthesis.md` | **(цей)** |

## Артефакти даних

- `data/Form Catalog.tsv` — католог 169 форм з типами
- `figures/07_form_types.csv` — form_id → form_type
- `figures/08_*` — full dataset diagnostics
- `figures/11_*` — multi-level reliability
- `figures/12_prod_*` — prod scenarios + heat-maps

---

## Що далі

Якщо буде запит на наступний fix-cycle — рекомендую **P1 (per-type calibration)** першим: дешево, безпечно, точно дає coverage gain, не торкає point estimate. P2 (LinearModel) — наступний.

Питання користувача "які рези через пару годин якщо маємо перші 10-30 відповідей" має **прямий числовий answer** з 12_:
- 10 відповідей + 2h: **40% MAPE, 82% Cov** (acceptable не дуже)
- 20 відповідей + 2h: **18% MAPE, 86% Cov** (acceptable)
- 30 відповідей + 2h: **13% MAPE, 82% Cov** (very good)

Це числа які можна показувати у UI як "точність прогнозу залежно від кількості відповідей" → користувачі знатимуть коли довіряти.
