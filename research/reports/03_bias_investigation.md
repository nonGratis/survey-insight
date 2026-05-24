# 03 — Bias Investigation (P4)

**Generated:** 2026-05-24T04:23:01
**Source:** `D:\GitHub\survey-insight\data\Form Timestamp Collection.csv` (sha256:`0dfa60a9a2cd`)
**Eligible forms:** 96 · **Backtest points (default):** 288

## TL;DR

- **Default (production)**: bias_p50 = -6.2%, MAPE = 27.7%, coverage = 50.0%
- **Relaxed-K (E1: K_min = last·1.5)**: bias_p50 = +16.5%, MAPE = 27.5%, coverage = 67.0%
- **Linear baseline (E2)**: bias_p50 = +25.2%, MAPE = 44.7%, coverage = 11.8%

**H1 (K_min floor): ✅ CONFIRMED** — релаксація K_min → bias change of +22.7pp. Wilcoxon W=350, p=4.893e-30 (default vs relaxed-K, paired)

**H7 (K hits floor): ⚠️ PARTIAL** — в середньому 12.3% фітів закінчуються з K = K_min (плато forced).

**H9 (model family limit): ✅ CONFIRMED** — linear baseline bias = +25.2% vs default -6.2% (Δ = +31.5pp). Linear краще → saturating-моделі за-saturate.

## Гіпотези й тести

### H3/H6: Bias per winning model

| winner | n | bias_p50 | bias_p25 | bias_p75 | mape_p50 | coverage |
|---|---:|---:|---:|---:|---:|---:|
| asymptotic_exp | 116 | -7.2 | -24.7 | 66.1 | 31.0 | 49.1 |
| gompertz | 132 | -9.0 | -25.4 | 53.0 | 26.6 | 43.2 |
| logistic | 40 | 7.6 | -11.8 | 47.0 | 23.5 | 75.0 |

### Per-shape bias

| shape | n | bias_p50 | bias_p25 | bias_p75 | mape_p50 | coverage |
|---|---:|---:|---:|---:|---:|---:|
| ill_fit | 6 | -29.3 | -43.7 | -20.6 | 38.9 | 16.7 |
| late_burst | 18 | -6.4 | -38.7 | 77.8 | 50.1 | 44.4 |
| linear | 6 | -16.4 | -26.4 | -15.3 | 16.4 | 66.7 |
| logarithmic | 147 | -3.1 | -15.4 | 104.5 | 22.8 | 55.8 |
| logistic | 111 | -11.8 | -31.2 | 23.7 | 30.0 | 44.1 |

### H4: Bias per cutoff_frac

| cutoff_frac | n | bias_p50 | bias_p25 | bias_p75 | mape_p50 | coverage |
|---|---:|---:|---:|---:|---:|---:|
| 0.3 | 96.0 | 5.1 | -25.1 | 181.2 | 50.6 | 60.4 |
| 0.5 | 96.0 | -10.3 | -31.0 | 26.8 | 30.2 | 44.8 |
| 0.7 | 96.0 | -7.9 | -19.8 | 17.1 | 19.4 | 44.8 |

### H5: Bias per horizon-bucket

| horizon_bucket | n | bias_p50 | bias_p25 | bias_p75 | mape_p50 | coverage |
|---|---:|---:|---:|---:|---:|---:|
| <1d | 199 | -1.3 | -19.8 | 110.7 | 30.9 | 55.8 |
| 1-3d | 54 | -7.5 | -17.3 | 15.5 | 17.0 | 48.1 |
| 3-7d | 8 | -28.5 | -38.5 | -23.0 | 28.5 | 0.0 |
| 7-30d | 9 | -19.1 | -40.0 | 3.9 | 19.1 | 44.4 |
| >30d | 18 | -22.7 | -37.9 | -13.0 | 26.0 | 16.7 |

### Bias per N-bucket

| n_bucket | n | bias_p50 | bias_p25 | bias_p75 | mape_p50 | coverage |
|---|---:|---:|---:|---:|---:|---:|
| <15 | 31 | 14.3 | -46.4 | 109.4 | 58.6 | 58.1 |
| 15-30 | 76 | -10.7 | -24.3 | 33.0 | 25.8 | 57.9 |
| 30-100 | 107 | -7.0 | -25.0 | 30.7 | 25.8 | 43.0 |
| 100-1k | 68 | -3.5 | -17.3 | 89.5 | 23.3 | 51.5 |
| 1k+ | 6 | 25.3 | -7.6 | 98.1 | 42.9 | 16.7 |

### H7: Frequency of K stuck at lower bound

| winner | % K-at-floor |
|---|---:|
| asymptotic_exp | 0.0% |
| gompertz | 29.5% |
| logistic | 7.5% |

## Контрольні experiments

### E1 — Relaxed K_min (last → last·1.5)

| Метрика | Default | Relaxed-K | Δ |
|---|---:|---:|---:|
| Bias (median) | -6.2% | +16.5% | +22.7pp |
| MAPE (median) | 27.7% | 27.5% | -0.1pp |
| Coverage 95% | 50.0% | 67.0% | +17.0pp |

### E2 — Linear baseline (просте y = a·t + b)

| Метрика | Default | Linear baseline | Δ |
|---|---:|---:|---:|
| Bias (median) | -6.2% | +25.2% | +31.5pp |
| MAPE (median) | 27.7% | 44.7% | +17.0pp |
| Coverage 95% | 50.0% | 11.8% | -38.2pp |

## Графіки

- [Bias per winning model × shape (boxplot)](figures\03_bias_per_winner.html)
- [Bias vs horizon_days (scatter)](figures\03_bias_vs_horizon.html)
- [% K-at-floor per winner (bar)](figures\03_k_at_floor.html)
- [K/truth ratio vs bias (scatter)](figures\03_k_ratio_vs_bias.html)
- [Experiment comparison (bar)](figures\03_experiment_comparison.html)

## Recommendation для P4 фіксу в core/forecast/

На основі чисел вище — конкретний фікс випливає з:
- Якщо H1 ✅ і Δbias істотний → **relax K_min в `_capacity_bounds`** (production fix).
- Якщо H9 ✅ → треба **додати LinearModel у селектор** як 4-у модель.
- Якщо H7 ✅ і >30% фітів сидять на floor → це самостійна причина.

Якщо всі три ❌ — bias має іншу природу (initial guess, AICc-bias, multi-wave forms). Тоді потрібен наступний рівень investigation.

## Артефакти

- `figures/03_default_metrics.csv` — default backtest results
- `figures/03_relaxed_metrics.csv` — з релаксованим K_min
- `figures/03_linear_metrics.csv` — linear baseline
