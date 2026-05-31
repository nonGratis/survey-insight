# Variant γ refactor — Roadmap

**Статус: ЗАВЕРШЕНО (P12 + P14 + P16).** P13 відкладено (optional safety net), P15 — negative result (reverted).

**Прийнято:** 2026-05-30. Користувач погодив повну архітектурну переробку CI subsystem замість косметичних multiplier-тюнів.

**Мотивація:** Поточний CI flow (9 шарів post-hoc корекцій) на формі з R²=0.92 видавав CI [62, 818] width=756, тоді як справжня pcov-uncertainty width ~12. Multipliers працюють незалежно від fit-quality → структурно неправильно.

## Цільова архітектура

```
forecast_responses(timeline)
  → curve_fit (model parameters + pcov)
  → primary CI = delta_method_ci(pcov, t_future, Student-t)
  → cap_width (explosion guard на ill-conditioned pcov)
  → conformal_adjustment (calibrated на empirical residuals)
  → cumulative_floor
```

Жодних magic multipliers. CI відображає (i) справжню параметричну uncertainty, (ii) historical residual distribution.

---

## Кроки

### ✅ P12 (DONE) — Delta-method CI як primary

**Файли:** `core/forecast/delta_ci.py` (новий), `core/forecast/service.py`.

**Зміни:**
- Новий модуль `delta_ci.py` з `delta_method_ci()` + `cap_width()`.
- `forecast_responses(ci_method="delta")` тепер default. `"nhpp"` — legacy fallback.
- На R²=0.92 form: width 1097 → 12 (91× narrower). Smoke-test пройдено.

**Замок:** Coverage просяде. Це очікувано — наступні кроки відновлять.

### ⏳ P13 — Bootstrap residual fallback

**Файли:** `core/forecast/bootstrap_ci.py` (новий).

**Зміни:**
- `bootstrap_residual_ci(t_train, y_train, t_future, model, params, n_samples=500)`.
- Resample fit residuals → percentile of cumulative trajectories.
- Викликається коли `cap_width` спрацював (delta-CI explosion) АБО коли `pcov` degenerate.

**Очікуваний impact:** safety net на Gompertz + multi-wave forms. Coverage не падає катастрофічно навіть на survey/feedback.

**Estimate:** 1 сесія, ~150 LOC + tests.

### ⏳ P14 — Conformal calibration

**Файли:** `core/forecast/conformal.py` (новий), `data/conformal_quantiles.json` (артефакт).

**Зміни:**
- Cross-form calibration set: для кожної форми × cutoff_fraction обчислити normalized residual = (truth − point) / σ̂ де σ̂ = delta-CI half-width.
- Per-(shape, n_class, horizon-bucket) quantile = q_{1−α}(|normalized residual|).
- `forecast_responses` після delta-CI масштабує width × q для досягнення target coverage.

**Гарантія:** empirical coverage → 1−α (95%) asymptotically (Vovk 2005, Romano 2019).

**Risk:** calibration set leakage. Required: rigorous train/test split за form_id, не за cutoff.

**Estimate:** 1-2 сесії, ~250 LOC + offline calibration script + tests.

### ⏳ P15 — Model arsenal cleanup + additions

**Файли:** `core/forecast/models.py`.

**Зміни:**
1. Drop `GompertzModel` з `DEFAULT_MODELS` (16_ показав: pcov ill-conditioned у 26% випадків, delta-CI explodes).
2. Add `LinearModel(a*t + b)` — для service форм з constant rate.
3. Add `BassModel((p + q·N/K)(K − N))` — для multi-wave (survey, feedback).

**Очікуваний impact:**
- service MAPE 31% → ~22%, coverage 68% → 85%.
- survey MAPE 54% → ~35%, coverage 53% → 80%.
- event_feedback MAPE 73% → ~50%.

**Estimate:** 2 сесії (Linear=1, Bass=1 з ретельним bounds-tuning).

### ⏳ P16 — Cleanup + tests + docs

**Файли:** усе core/forecast/, тести, CLAUDE.md.

**Зміни:**
- Видалити dead-code paths що не використовуються.
- Оновити docstring + module headers.
- Comprehensive test suite для P12-P15.
- Final integrated benchmark 20_post_gamma.py — порівняння пре-/пост-γ метрик.

**Estimate:** 1 сесія.

---

## Total

- 5 кроків × ~1.5 сесії = **6-8 сесій**.
- Очікувані фінальні числа на 3710 backtest points:
  - Width median: 158 → ~15-25 (8-10× ↓)
  - Coverage: 82% → 90-93% (через conformal)
  - MAPE: 37% → ~25% (через нові моделі)
  - Winkler median: 234 → ~80 (3× ↓)
- На screenshot-сценарії: width 756 → 15-25.

---

## Atomic commits

Кожен крок = окремий комміт. Імена:
- `refactor(forecast): delta-method CI primary (P12)`
- `feat(forecast): bootstrap residual fallback (P13)`
- `feat(forecast): conformal calibration (P14)`
- `feat(forecast): linear + bass models, drop gompertz (P15)`
- `chore(forecast): cleanup legacy + comprehensive tests (P16)`

---

## Status

- [x] P12 — Delta-method CI primary (commit `0a1bf99`)
- [ ] P13 — Bootstrap fallback (optional safety net; tractable після P15)
- [x] P14 — Conformal calibration (commit `29a16d0`) — **coverage 91.8% global, 95% @ 2h**
- [ ] P15 — Model arsenal: **NEGATIVE result, reverted.** See "P15 attempt" нижче.
- [x] P16 — Cleanup + tests (commit `4fadb32`) — service.py spрощено, 37 нових тестів, 109/109 pass.

## Post-P14 prod numbers (3710 backtest points)

| | Pre-γ | After P14 | Δ |
|---|---:|---:|---:|
| MAPE | 37.2% | 37.2% | 0 |
| Coverage | 81.6% | **91.8%** | **+10pp** |
| width_p50 | 158 | 98 | -38% |
| Winkler_p50 | 234 | 135 | **-42%** |

**Per horizon at user's prod-scenario (2h):**
- Coverage: 83% → **95.0%** (nominal!)
- width_p50: 158 → 54 (3× narrower)
- Winkler: 212 → 62 (-71%)

**Screenshot scenario (R²=0.92):** CI width 1097 → 27 (40× narrower, properly calibrated).

## P15 attempt — NEGATIVE result

Спроба: drop Gompertz, add LinearModel + BassDiffusionModel.

**Результат на full benchmark (3710 точок):**

| | P14 (current) | P15 attempt | Δ |
|---|---:|---:|---:|
| MAPE | 37.2% | **60.6%** | **+23pp** ❌ |
| Coverage | 91.8% | **78.8%** | **−13pp** ❌ |
| Winkler | 135 | 156 | +16 |

**Що зламалося:**
- **LinearModel extrapolates без saturation cap.** AICc обирає Linear у ранній фазі (parsimony bonus від 2-х параметрів vs 3), але на long horizon Linear розходиться. MAPE@72h = 98%, MAPE@168h = 200%.
- **BassDiffusionModel identifiability issues** на ~28 точках. Параметри (p, q, m) часто нестабільні, pcov ill-conditioned. Не покращив survey/event_feedback (як гіпотеза очікувала); навпаки погіршив (survey MAPE 54%→165%).

**Лекції:**
- AICc — це in-sample fit criterion. Для **extrapolation** треба cross-validation або out-of-sample selection.
- Linear without saturation cap небезпечний у DEFAULT_MODELS.
- Bass потребує **довшої історії** (≥50 точок) для стабільної ідентифікації, тоді як наш use-case — short-prefix forecasting на 5-30 точках.

**Можливі варіанти P15-v2 для майбутнього (НЕ робив зараз):**

1. **LinearModel з K_max cap у predict** — `predict = min(a*t + b, K_max)`, де K_max з `_capacity_bounds`. Розв'язує extrapolation issue.
2. **Bass тільки при n ≥ 50** — додати threshold у `models_for_n_points`.
3. **Cross-validation замість AICc** для selector — дорого compute-wise але теоретично правильно для extrapolation.
4. **Тільки drop Gompertz** (без додавання). Per 16_, Gompertz pcov ill-conditioned у 26% випадків. Можливо AsympExp + Logistic покриють без втрат. Окремий test.

**Поточне рішення:** залишити P14 як final γ-кадр. Huge n_class (54% cov) і survey (82% cov, 1121 width) — fundamental limits, які потребують **окремих research-сесій з ретельними safety guards**.

## Поточний фінал γ-refactor (P12 + P14)

| | Pre-γ | **Final γ (P14)** | Δ |
|---|---:|---:|---:|
| MAPE | 37.2% | 37.2% | 0 (point intact) |
| Coverage | 81.6% | **91.8%** | **+10pp** |
| width_p50 | 158 | 98 | -38% |
| Winkler_p50 | 234 | 135 | -42% |
| **Cov@2h** | 83% | **95.0%** | nominal! |
| **Screenshot CI width** | 1097 | 27 | **40× narrower** |

Це **головна перемога γ-refactor**. Решта (huge cov, survey) — fundamental model limits, не cosmetic fix-able.
