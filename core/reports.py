"""Білдери доменних звітів: результат аналізу → блоки `Report` (модель рендеру).

Шар композиції між доменом (forms_quality / weighting / crosstab / …) та
універсальним рендером (core.report). Структура — за SRP:

  • кожна СЕКЦІЯ — окрема чиста функція `*_section(...) -> list[block]`, що знає
    лише ЩО показати у своєму розділі, але не ЯК малювати (це core.report) і не
    як компонувати документ;
  • `full_report(...)` лише КОМПОНУЄ передані секції в один документ (розриви
    між секціями), нічого не обчислюючи;
  • тонкі обгортки `representativeness_report` / `questions_report` збирають
    типовий документ із кількох секцій (зворотна сумісність із локальними
    кнопками).

Глобальна сторінка «Звіт» викликає ті самі секційні білдери, що й локальні
кнопки, — нуль дублювання (DRY). Числа форматуються по-українськи (кома).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from core.forms_quality import (
    SORT_BY_COUNT,
    analyze_form_design,
    analyze_responses,
    anonymize_distribution,
    question_options,
    sort_distribution,
)
from core.report import (
    BarChart,
    Heading,
    Metric,
    Metrics,
    PageBreak,
    Para,
    Report,
    TableBlock,
)
from core.weighting import WeightingResult

_STRATA_HEADERS = ("Вимір", "Страта", "N_h", "n_h", "Треба", "Вага", "Покриття", "Ще треба")
_STRATA_WIDTHS = (0.14, 0.26, 0.09, 0.09, 0.10, 0.10, 0.10, 0.12)
_ASSOC_HEADERS = ("Питання 1", "Питання 2", "Міра", "Ефект", "Сила", "p (FDR)")
_ASSOC_WIDTHS = (0.27, 0.27, 0.14, 0.10, 0.10, 0.12)


def _uk(value: float, spec: str) -> str:
    """Відформатувати число за spec із комою як десятковим роздільником."""
    return format(value, spec).replace(".", ",")


def _pct(fraction: float, digits: int = 1) -> str:
    return _uk(fraction * 100, f".{digits}f") + " %"


# --- конфіг дескриптивної секції (дзеркало екранних налаштувань) -------------
@dataclass(frozen=True)
class DescriptiveConfig:
    """Налаштування дескриптивної секції — ті самі, що на екрані «Відповіді»."""

    anonymize: bool = False
    other_label: str = "Інше*"
    keep_other_last: bool = True
    hide_only_other: bool = False
    sort_mode: str = SORT_BY_COUNT
    render_mode: str = "chart"  # "table" | "chart" | "both"
    top_n: int = 30


# ===================== СЕКЦІЇ (кожна — своя відповідальність) ================


def overview_section(structure: dict, responses: list[dict]) -> list[object]:
    """Титульна секція: загальні характеристики опитування."""
    stats = analyze_responses(structure, responses)
    n_open = sum(1 for s in stats.values() if s.is_text)
    return [
        Heading("Загальні характеристики", level=2),
        Metrics(
            columns=4,
            items=[
                Metric("Відповідей", str(len(responses))),
                Metric("Питань", str(len(stats))),
                Metric("Відкритих", str(n_open)),
            ],
        ),
    ]


def descriptive_section(
    structure: dict, responses: list[dict], config: DescriptiveConfig | None = None
) -> list[object]:
    """Результат кожного питання — окремою сторінкою (таблиця та/або діаграма).

    Поважає ті самі налаштування, що й екран: анонімізація, приховування
    «лише Інше», сортування, ліміт, формат (таблиця/діаграма).
    """
    cfg = config or DescriptiveConfig()
    designs = {d.question_id: d for d in analyze_form_design(structure)}
    stats = analyze_responses(structure, responses)
    options = question_options(structure)

    blocks: list[object] = []
    first = True
    for qid, s in stats.items():
        dist = s.distribution
        if not s.is_text and cfg.anonymize:
            dist = anonymize_distribution(dist, options.get(qid, []), cfg.other_label)
        if not s.is_text and cfg.hide_only_other and set(dist) == {cfg.other_label}:
            continue

        if not first:
            blocks.append(PageBreak())
        first = False
        design = designs.get(qid)
        blocks.append(Heading(design.title if design else qid, level=2))
        blocks.append(
            Para(
                f"Відповіли {s.n_answered}/{s.n_total} · пропуск {_uk(s.non_response_pct, '.1f')} %"
            )
        )

        if s.is_text:
            if s.text_median_len:
                blocks.append(
                    Para(f"Медіана довжини відповіді — {int(s.text_median_len)} символів.")
                )
            continue
        if not dist:
            blocks.append(Para("Немає відповідей."))
            continue

        items = sort_distribution(
            dist, cfg.sort_mode, options.get(qid, []), cfg.keep_other_last, cfg.other_label
        )[: cfg.top_n]
        total = max(s.n_answered, 1)
        if cfg.render_mode in ("table", "both"):
            rows = [[value, str(count), _pct(count / total)] for value, count in items]
            blocks.append(
                TableBlock(headers=("Варіант", "К-сть", "%"), rows=rows, col_widths=(0.6, 0.2, 0.2))
            )
        if cfg.render_mode in ("chart", "both"):
            blocks.append(BarChart(labels=[v for v, _ in items], values=[c for _, c in items]))
    return blocks


def representativeness_section(res: WeightingResult) -> list[object]:
    """Секція репрезентативності: показники + вердикт + таблиця ваг."""
    metrics = Metrics(
        columns=4,
        items=[
            Metric("Репрезентативність", _pct(res.coverage_eff, 0)),
            Metric("Без DEFF", _pct(res.coverage_raw, 0)),
            Metric("DEFF (Кіш)", _uk(res.deff, ".2f")),
            Metric("Ефективний n", _uk(res.n_eff, ".0f")),
            Metric("Відповідей (n)", str(res.n)),
            Metric("Ціль n_target", str(res.n_target)),
            Metric("MoE", _pct(res.moe)),
            Metric("MoE з DEFF", _pct(res.moe_deff)),
        ],
    )
    lack = max(res.sample_need - res.n, 0.0)
    if lack > 0:
        verdict = (
            f"Бракує ще ~<b>{_uk(lack, '.0f')}</b> відповідей до цілі з урахуванням "
            f"дизайну вибірки (потрібно n_target·DEFF = {_uk(res.sample_need, '.0f')})."
        )
    else:
        verdict = (
            f"Ціль досягнуто: зібрано {res.n} ≥ потрібних {_uk(res.sample_need, '.0f')} "
            f"(n_target·DEFF)."
        )
    ordered = sorted(res.strata, key=lambda s: s.lack, reverse=True)
    rows = [
        [
            s.dimension,
            s.stratum,
            str(s.population),
            str(s.sample),
            _uk(s.req_sample, ".1f"),
            _uk(s.weight, ".3f"),
            _uk(s.coverage, ".2f"),
            _uk(s.lack, ".1f"),
        ]
        for s in ordered
    ]
    return [
        Heading("Репрезентативність вибірки", level=2),
        metrics,
        Para(verdict),
        Heading("Таблиця ваг (за недопредставленістю)", level=3),
        Para("Покриття < 1 — недобір, > 1 — надлишок страти у вибірці."),
        TableBlock(headers=_STRATA_HEADERS, rows=rows, col_widths=_STRATA_WIDTHS),
    ]


def associations_section(rows: Sequence[Sequence[str]]) -> list[object]:
    """Секція зв'язків: таблиця найсильніших попарних асоціацій (вже відформатована)."""
    if not rows:
        return [
            Heading("Зв'язки між питаннями", level=2),
            Para("Недостатньо придатних питань для аналізу зв'язків."),
        ]
    return [
        Heading("Зв'язки між питаннями", level=2),
        Para("Найсильніші попарні зв'язки за розміром ефекту (FDR-скориговано)."),
        TableBlock(headers=_ASSOC_HEADERS, rows=rows, col_widths=_ASSOC_WIDTHS),
    ]


def dynamics_section(items: Sequence[Metric], note: str = "") -> list[object]:
    """Секція динаміки надходження (показники прогнозу — числами)."""
    blocks: list[object] = [
        Heading("Динаміка надходження відповідей", level=2),
        Metrics(columns=3, items=list(items)),
    ]
    if note:
        blocks.append(Para(note))
    return blocks


# ===================== КОМПОЗИЦІЯ + ОБГОРТКИ ================================


def full_report(
    title: str,
    subtitle: str,
    sections: Sequence[Sequence[object]],
    footer: str = "Survey Insight",
) -> Report:
    """Скомпонувати документ із секцій: кожна наступна — з нової сторінки."""
    blocks: list[object] = []
    for index, section in enumerate(sections):
        if index > 0:
            blocks.append(PageBreak())
        blocks.extend(section)
    return Report(title=title, subtitle=subtitle, footer=footer, blocks=blocks)


def representativeness_report(res: WeightingResult, form_title: str = "") -> Report:
    """Звіт про репрезентативність вибірки (локальна кнопка «Зважування»)."""
    return full_report(
        "Звіт про репрезентативність вибірки",
        f"Форма: {form_title}" if form_title else "",
        [representativeness_section(res)],
        footer="Survey Insight · Зважування",
    )


def questions_report(
    structure: dict,
    responses: list[dict],
    form_title: str = "",
    config: DescriptiveConfig | None = None,
) -> Report:
    """Повний звіт за результатами опитування (титул + сторінка на питання)."""
    return full_report(
        "Звіт за результатами опитування",
        f"Форма: {form_title}" if form_title else "",
        [overview_section(structure, responses), descriptive_section(structure, responses, config)],
        footer="Survey Insight · Звіт",
    )
