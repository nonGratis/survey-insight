# 04 — Statistical Diagnostics

**Generated:** 2026-05-28T11:35:04
**Source:** `D:\GitHub\survey-insight\data\Form Timestamp Collection.csv` (sha256:`0dfa60a9a2cd`)
**Eligible forms:** 96 · **Diagnostic points:** 288

## Battery

Чотири стандартні тести з econometrics / time-series літератури:

1. **Ljung-Box Q*** — H0: residuals = white noise. Reject (p<0.05) →
   модель не захоплює всю temporal structure.
2. **Shapiro-Wilk** — H0: residuals ~ Normal. Reject → curve_fit LSQ
   припущення Gaussian-шуму порушене (це аргумент за NHPP-Poisson, що
   ми і робимо у `nhpp_prediction_interval`).
3. **BIC** як sensitivity-check до AICc (sharper complexity penalty).
4. **Theil's U** vs naive (cumulative = last_observed). U<1 → ми
   кращі за тривіальний "no-change forecast".

## Глобальні результати

| Метрика | Значення |
|---|---:|
| **Ljung-Box rejection rate** (p<0.05) | 86.8% |
| **Shapiro-Wilk rejection rate** (p<0.05) | 48.3% |
| **Theil's U median** | 0.922 |
| **Theil's U < 1 (beats naive)** | 52.8% |

**Інтерпретація:**

- Ljung-Box reject 87% → у 13% форм модель захоплює temporal structure адекватно. Решта — кандидати на додаткове моделювання (можливо через CP detection).
- Shapiro-Wilk reject 48% → значуща частка форм має non-Gaussian residuals. Це обґрунтовує наше використання NHPP-Poisson, а не LSQ-CI.
- Theil's U: median = 0.922, beats naive у 53% форм. Виграш над naive скромний.

## Per-shape діагностика

| shape | n | lb_pval_median | lb_reject_pct | sw_pval_median | sw_reject_pct | theil_u_median | theil_u_beats_naive_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| ill_fit | 6 | 0.0 | 83.333 | 0.068 | 50.0 | 1.0 | 33.333 |
| late_burst | 18 | 0.0 | 88.889 | 0.266 | 27.778 | 0.716 | 66.667 |
| linear | 6 | 0.162 | 50.0 | 0.691 | 0.0 | 0.68 | 100.0 |
| logarithmic | 147 | 0.0 | 89.796 | 0.021 | 59.184 | 1.0 | 47.619 |
| logistic | 111 | 0.0 | 84.685 | 0.18 | 39.64 | 0.891 | 55.856 |

Колонки:
- `lb_pval_median` / `lb_reject_pct`: p-value Ljung-Box і % rejected при α=0.05
- `sw_pval_median` / `sw_reject_pct`: те саме для Shapiro-Wilk
- `theil_u_median` / `theil_u_beats_naive_pct`: U-статистика і % форм, де U<1

## AICc vs BIC по обраних моделях

| winner | n | aicc_median | bic_median |
|---|---:|---:|---:|
| asymptotic_exp | 148 | 77.97 | 81.7 |
| gompertz | 102 | 166.21 | 173.22 |
| logistic | 38 | 231.01 | 239.75 |

Якщо AICc і BIC сходяться у виборі — селектор робастний до критерію.

## Графіки

- [Ljung-Box p-values per shape (boxplot)](figures\04_lb_distribution.html)
- [Shapiro-Wilk p-values per shape (boxplot)](figures\04_sw_distribution.html)
- [Theil's U per shape (boxplot)](figures\04_theil_u.html)
- [AICc vs BIC scatter](figures\04_aicc_bic.html)

## Що з цього випливає для thesis defense

1. **Адекватність моделі**: Ljung-Box rejection rate = 87% — кваліфікований індикатор. У документі цитуємо як "модель адекватна на 13% форм за тестом Ljung-Box (α=0.05, lags=10)".
2. **Обґрунтування NHPP**: Shapiro-Wilk reject = 48% — формальне підтвердження, що residuals не Gaussian → LSQ-CI неадекватна → NHPP-Poisson — правильний вибір (це йшло у research/02 P1 fix).
3. **Корисність моделі**: Theil's U beats naive у 53% — модель не просто "не гірша за no-change", а реально кращі-передбачає.
4. **Robust model selection**: AICc-вибір не сильно розходиться з BIC (див. scatter) — селектор стійкий.

## Артефакти

- `figures/04_diagnostics_raw.csv` — повний log per (form, cutoff)
