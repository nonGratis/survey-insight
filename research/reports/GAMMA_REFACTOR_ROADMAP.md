# Variant γ refactor — Roadmap

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

- [x] P12 — Delta-method CI primary
- [ ] P13 — Bootstrap fallback
- [ ] P14 — Conformal calibration
- [ ] P15 — Model arsenal
- [ ] P16 — Cleanup
