"""Post-stratification weighting & representativeness.

Призначення: оцінити, наскільки зібрана вибірка відповідей репрезентує
генеральну сукупність, і дати ваги, що коригують перекоси за відомими
вимірами (підрозділ, курс, стать, спеціальність — будь-які).

Один вимір (`Dimension`) = одне питання форми, що несе значення страти, +
таблиця популяції {страта: N_h} в АБСОЛЮТНИХ числах. Вимірів довільна
кількість; композитна вага — добуток вимірних ваг (мультиплікативна, тобто
перша ітерація raking без подальшого вирівнювання).

Математика (звірена 1:1 з еталонним прототипом, data/*.csv):

  n_target = ceil( SRS-обсяг з FPC для заданого MoE )           (ціль вибірки)
  w_h      = (N_h/N · n_target) / n_h                           (вага страти)
  w_i      = Π_dim w_{dim,h(i)}                                 (композит, добуток)
  DEFF     = n·Σw² / (Σw)²   = 1 + CV²(w)   (Kish, інваріант до масштабу)
  n_eff    = n / DEFF
  MoE      = z·√(p(1−p)/n)                  (БЕЗ FPC: вибірка ≪ сукупності)
  MoE_DEFF = MoE · √DEFF
  need     = n_target · DEFF                (скільки треба при цьому DEFF)

Чисті функції без I/O. Джерело даних (Forms API / CSV-імпорт) — окремо;
сюди приходять уже розпарсені популяції та відповіді в порядку R_ID
(порядок повернення API — submission order, без ресорту за createTime).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

# --- іменовані константи (методологія вибіркових обстежень) -----------------
Z_95 = 1.96  # z-квантиль для 95% довіри (двостороння); 1.96² = 3.8416
DEFAULT_MOE = 0.05  # цільова гранична похибка 5% для розрахунку n_target
MAX_VAR_P = 0.5  # частка максимальної дисперсії p(1−p): консервативна оцінка
RID_COLUMN = "R_ID"  # наскрізний ідентифікатор респондента (порядок submission)


def required_sample_size(
    population: int,
    *,
    moe: float = DEFAULT_MOE,
    z: float = Z_95,
    p: float = MAX_VAR_P,
) -> int:
    """Цільовий обсяг вибірки n_target для заданого MoE (SRS + FPC, ceil).

    n0 = z²·p(1−p)/e² ;  n = n0 / (1 + (n0−1)/N).  Округлення вгору, бо
    обсяг не може бути меншим за розрахунковий (інакше MoE > цілі).

    Приклад: N=2111, MoE=5% → n0=384.16 → n=325.14 → 326 (= eталон).
    """
    if population <= 0:
        return 0
    n0 = z * z * p * (1.0 - p) / (moe * moe)
    n_fpc = n0 / (1.0 + (n0 - 1.0) / population)
    return math.ceil(n_fpc)


def margin_of_error(n: int, *, z: float = Z_95, p: float = MAX_VAR_P) -> float:
    """Гранична похибка частки для досягнутого n (SRS, без FPC, частка).

    FPC свідомо не застосовуємо: вибірка значно менша за сукупність, а
    еталон рахує саме так (n=497, N=2111 → MoE=4.40%).
    """
    if n <= 0:
        return 0.0
    return z * math.sqrt(p * (1.0 - p) / n)


def design_effect(weights: Sequence[float]) -> float:
    """DEFF за Kish: n·Σw²/(Σw)² = 1 + CV²(w). Інваріант до масштабу ваг.

    Порожні / нульова сума → 1.0 (немає дизайн-ефекту).
    """
    ws = [w for w in weights if w is not None and math.isfinite(w)]
    n = len(ws)
    s = sum(ws)
    if n == 0 or s <= 0:
        return 1.0
    return n * sum(w * w for w in ws) / (s * s)


def cumulative_design_effect(weights: Sequence[float]) -> list[float]:
    """DEFF наростаючим підсумком: DEFF_k за першими k вагами (порядок R_ID).

    Показує, як дизайн-ефект еволюціонував у міру надходження відповідей.
    Рахуємо інкрементально по running Σw та Σw² → O(n). NaN-ваги ігноруємо
    (k у формулі = к-сть валідних ваг до позиції включно).
    """
    out: list[float] = []
    s = s2 = 0.0
    k = 0
    for w in weights:
        if w is not None and math.isfinite(w):
            s += w
            s2 += w * w
            k += 1
        out.append(k * s2 / (s * s) if k > 0 and s > 0 else 1.0)
    return out


@dataclass(frozen=True)
class Dimension:
    """Один вимір стратифікації: питання + популяція страт."""

    name: str  # людська назва ("Підрозділ", "Курс", "Стать"…)
    column: str  # колонка відповіді, що несе значення страти
    population: Mapping[str, int]  # страта → N_h (абсолютні числа)

    @property
    def total_population(self) -> int:
        return int(sum(self.population.values()))


@dataclass(frozen=True)
class StratumStat:
    """Підсумок по одній страті одного виміру (рядок таблиці ваг)."""

    dimension: str
    stratum: str
    population: int  # N_h
    sample: int  # n_h (з живих відповідей)
    req_sample: float  # P_h · n_target — скільки треба для пропорції
    weight: float  # w_h = req_sample / n_h
    sampling_fraction: float  # n_h / N_h
    coverage: float  # n_h / req_sample (1.0 = рівно за планом)
    lack: float  # max(req_sample − n_h, 0) — «ще треба» відповідей


@dataclass
class WeightingResult:
    """Повний результат зважування: метрики + страти + per-respondent кадр."""

    n: int  # досягнутий обсяг
    population: int  # N (за головним виміром)
    n_target: int  # цільовий обсяг для MoE
    deff: float
    n_eff: float
    moe: float  # частка (0..1)
    moe_deff: float  # частка (0..1)
    sample_need: float  # n_target · DEFF
    lack_overall: float  # sample_need − n (>0 = бракує)
    coverage_raw: float  # n / n_target (репрезентативність БЕЗ DEFF)
    coverage_eff: float  # n_eff / n_target (з урахуванням DEFF)
    strata: list[StratumStat] = field(default_factory=list)
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)

    def strata_frame(self) -> pd.DataFrame:
        """Таблиця ваг (для UI; сортуй за `lack` desc для under-representation)."""
        return pd.DataFrame(
            [
                {
                    "Вимір": s.dimension,
                    "Страта": s.stratum,
                    "N_h": s.population,
                    "n_h": s.sample,
                    "Треба (P_h·n_target)": round(s.req_sample, 1),
                    "Вага w_h": round(s.weight, 4),
                    "Частка вибірки": round(s.sampling_fraction, 4),
                    "Покриття": round(s.coverage, 3),
                    "Ще треба": round(s.lack, 1),
                }
                for s in self.strata
            ]
        )


def _apply_cap(weight: float, cap: float | None) -> float:
    """Обрізати вагу зверху до cap (якщо заданий). Default off → None."""
    if cap is not None and math.isfinite(weight):
        return min(weight, cap)
    return weight


def stratum_weights(
    population: Mapping[str, int],
    counts: Mapping[str, int],
    n_target: int,
    *,
    caps: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Ваги страт одного виміру: w_h = (N_h/N · n_target)/n_h.

    n_h береться з ЖИВИХ відповідей (`counts`), популяція N_h — з таблиці.
    n_h = 0 → вага NaN (страта присутня в популяції, але без відповідей;
    у вибірці її ваги немає кого нести). `caps[stratum]` обрізає вагу зверху.
    """
    total = sum(population.values())
    out: dict[str, float] = {}
    for stratum, n_h_pop in population.items():
        n_h = counts.get(stratum, 0)
        if n_h <= 0 or total <= 0:
            out[stratum] = math.nan
            continue
        w = (n_h_pop / total * n_target) / n_h
        cap = caps.get(stratum) if caps else None
        out[stratum] = _apply_cap(w, cap)
    return out


def _running_normalized_timeline(
    values: Sequence[str],
    population: Mapping[str, int],
    n_target: int,
) -> list[float]:
    """Нормована таймлайн-вага виміру на момент кожної відповіді.

    Для респондента i (в порядку R_ID) беремо стан вибірки на [t_1, t_i]:
    running n_h(t_i) = к-сть страти серед перших i. Сира миттєва вага —
    та сама формула, що й статична: (N_h/N · n_target)/n_h(t_i). Далі
    нормуємо на середню вагу присутніх страт у цей момент, тож ряд
    центрований навколо 1 (перший респондент = 1.0), а DEFF не зачеплено
    (інваріант до масштабу). Це «яка була вага в той момент» — еволюція
    репрезентативності, придатна для графіка.
    """
    total = sum(population.values())
    run: Counter[str] = Counter()
    present_pop = 0  # Σ N_h присутніх страт (для середньої ваги)
    out: list[float] = []
    for i, v in enumerate(values, start=1):
        if run[v] == 0:  # нова страта зʼявилась — додаємо її N_h
            present_pop += population.get(v, 0)
        run[v] += 1
        n_h = run[v]
        n_h_pop = population.get(v, 0)
        if total <= 0 or n_h_pop <= 0 or present_pop <= 0:
            out.append(math.nan)
            continue
        raw = (n_h_pop / total * n_target) / n_h
        mean_w = n_target * (present_pop / total) / i  # середня вага на t_i
        out.append(raw / mean_w if mean_w > 0 else math.nan)
    return out


def compute_weighting(
    responses: pd.DataFrame,
    dimensions: Sequence[Dimension],
    *,
    moe: float = DEFAULT_MOE,
    caps: Mapping[str, Mapping[str, float]] | None = None,
    primary_dimension: str | None = None,
) -> WeightingResult:
    """Порахувати повне зважування за довільним набором вимірів.

    Args:
        responses: рядки-респонденти В ПОРЯДКУ R_ID (порядок API). Має містити
            колонки кожного `Dimension.column`; R_ID додається, якщо немає.
        dimensions: виміри стратифікації (≥1).
        moe: цільова гранична похибка для n_target.
        caps: {назва_виміру: {страта: cap}} — обрізання ваг зверху (default off).
        primary_dimension: чий N береться за головну сукупність N і n_target
            (default — перший вимір).

    Returns:
        WeightingResult з метриками, per-stratum таблицею та per-respondent
        кадром (R_ID, w_{dim}, w_{dim}_timeline, w, w_timeline).
    """
    if not dimensions:
        raise ValueError("потрібен принаймні один вимір стратифікації")

    frame = responses.reset_index(drop=True).copy()
    if RID_COLUMN not in frame.columns:
        frame.insert(0, RID_COLUMN, range(1, len(frame) + 1))

    primary = next((d for d in dimensions if d.name == primary_dimension), dimensions[0])
    population = primary.total_population
    n = len(frame)
    n_target = required_sample_size(population, moe=moe)

    # --- per-dimension: статичні ваги + таймлайн ----------------------------
    strata: list[StratumStat] = []
    static_cols: list[str] = []
    for dim in dimensions:
        values = frame[dim.column].astype(str).tolist()
        counts = Counter(values)
        dim_caps = caps.get(dim.name) if caps else None
        w_h = stratum_weights(dim.population, counts, n_target, caps=dim_caps)

        col = f"w_{dim.name}"
        tl_col = f"w_{dim.name}_timeline"
        frame[col] = [w_h.get(v, math.nan) for v in values]
        frame[tl_col] = _running_normalized_timeline(values, dim.population, n_target)
        static_cols.append(col)

        dim_total = dim.total_population
        for stratum, n_h_pop in dim.population.items():
            n_h = counts.get(stratum, 0)
            req = n_h_pop / dim_total * n_target if dim_total else 0.0
            strata.append(
                StratumStat(
                    dimension=dim.name,
                    stratum=stratum,
                    population=n_h_pop,
                    sample=n_h,
                    req_sample=req,
                    weight=w_h.get(stratum, math.nan),
                    sampling_fraction=(n_h / n_h_pop if n_h_pop else math.nan),
                    coverage=(n_h / req if req > 0 else math.nan),
                    lack=max(req - n_h, 0.0),
                )
            )

    # --- композитні ваги: добуток вимірних (мультиплікативний raking) --------
    tl_cols = [f"w_{d.name}_timeline" for d in dimensions]
    frame["w"] = frame[static_cols].prod(axis=1, skipna=False)
    frame["w_timeline"] = frame[tl_cols].prod(axis=1, skipna=False)

    composite = [w for w in frame["w"].tolist() if w is not None and math.isfinite(w)]
    deff = design_effect(composite)
    n_eff = len(composite) / deff if deff > 0 else float(len(composite))
    moe_val = margin_of_error(n)
    moe_deff = moe_val * math.sqrt(deff)
    sample_need = n_target * deff

    return WeightingResult(
        n=n,
        population=population,
        n_target=n_target,
        deff=deff,
        n_eff=n_eff,
        moe=moe_val,
        moe_deff=moe_deff,
        sample_need=sample_need,
        lack_overall=sample_need - n,
        coverage_raw=(n / n_target if n_target else math.nan),
        coverage_eff=(n_eff / n_target if n_target else math.nan),
        strata=strata,
        frame=frame,
    )
