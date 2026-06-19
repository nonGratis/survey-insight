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
from typing import Literal

from core.form_flow import parse_form_flow
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
    FlowChart,
    FlowChartEdge,
    FlowChartNode,
    Heading,
    Metric,
    Metrics,
    PageBreak,
    Para,
    Report,
    TableBlock,
)
from core.response_weights import (
    compute_configured_response_weights,
    weighted_response_distribution,
)
from core.weighting import Dimension, WeightingResult

_STRATA_HEADERS = ("Вимір", "Страта", "N_h", "n_h", "Треба", "Вага", "Покриття", "Ще треба")
_STRATA_WIDTHS = (0.14, 0.26, 0.09, 0.09, 0.10, 0.10, 0.10, 0.12)
_ASSOC_HEADERS = ("Питання 1", "Питання 2", "Міра", "Ефект", "Сила", "p (FDR)")
_ASSOC_WIDTHS = (0.27, 0.27, 0.14, 0.10, 0.10, 0.12)
QuestionTableMode = Literal["none", "flagged", "all"]


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
    weighting_dimensions: Sequence[Dimension] = ()
    weighting_cap_value: float = 0.0
    weighting_moe_pct: float = 5.0


@dataclass(frozen=True)
class OverviewConfig:
    """Налаштування секції огляду форми."""

    include_response_metrics: bool = True
    include_question_audit: bool = True
    question_table_mode: QuestionTableMode = "flagged"
    include_flow_map: bool = True


# ===================== СЕКЦІЇ (кожна — своя відповідальність) ================


def _overview_question_table(designs, mode: QuestionTableMode) -> list[object]:
    if mode == "none":
        return []
    selected = designs if mode == "all" else [design for design in designs if design.flags]
    if not selected:
        return [Para("Питань із прапорами якості не виявлено.")]

    title = "Усі питання форми" if mode == "all" else "Питання, що потребують уваги"
    return [
        Heading(title, level=3),
        TableBlock(
            headers=("Запитання", "Тип", "Опцій", "Обов'язк.", "Прапори"),
            rows=[
                [
                    design.title,
                    design.qtype_label,
                    str(design.n_options) if design.n_options is not None else "—",
                    "так" if design.required else "ні",
                    ", ".join(design.flags) if design.flags else "—",
                ]
                for design in selected
            ],
            col_widths=(0.42, 0.17, 0.10, 0.11, 0.20),
        ),
    ]


def _overview_flow_map(structure: dict) -> list[object]:
    flow = parse_form_flow(structure)
    has_interesting_flow = (
        flow.section_count > 1
        or flow.conditional_edge_count > 0
        or bool(flow.unreachable_section_ids)
        or flow.has_cycles
    )
    if not has_interesting_flow:
        return []

    title_by_node = {node.id: node.title for node in flow.nodes}
    blocks: list[object] = [
        Heading("Карта переходів", level=3),
        FlowChart(
            nodes=[
                FlowChartNode(
                    id=node.id,
                    label="\n".join((node.title, *node.detail_lines[:1])),
                    kind=node.kind,
                    flagged=node.id in flow.unreachable_section_ids,
                )
                for node in flow.nodes
            ],
            edges=[
                FlowChartEdge(
                    source=edge.source,
                    target=edge.target,
                    label=edge.label,
                    kind=edge.kind,
                )
                for edge in flow.edges
            ],
        ),
    ]
    if flow.unreachable_section_ids:
        unreachable = [
            title_by_node.get(section_id, section_id) for section_id in flow.unreachable_section_ids
        ]
        blocks.append(
            Para(
                "Недосяжні секції: "
                + ", ".join(unreachable)
                + ". Перевірте умови переходів і завершення форми."
            )
        )
    return blocks


def overview_section(
    structure: dict, responses: list[dict], config: OverviewConfig | None = None
) -> list[object]:
    """Титульна секція: загальні характеристики опитування."""
    cfg = config or OverviewConfig()
    stats = analyze_responses(structure, responses)
    designs = analyze_form_design(structure)
    n_open = sum(1 for s in stats.values() if s.is_text)
    n_flagged = sum(1 for d in designs if d.flags)
    sections_count = sum(1 for item in structure.get("items", []) if "pageBreakItem" in item) + 1

    blocks: list[object] = [Heading("Огляд форми", level=2)]
    metric_items: list[Metric] = []
    if cfg.include_response_metrics:
        metric_items.extend(
            [
                Metric("Відповідей", str(len(responses))),
                Metric("Питань", str(len(stats))),
                Metric("Відкритих", str(n_open)),
            ]
        )
    if cfg.include_question_audit:
        metric_items.extend(
            [
                Metric("Секцій", str(sections_count)),
                Metric("Питань з прапорами", str(n_flagged)),
            ]
        )
    if metric_items:
        blocks.append(
            Metrics(
                columns=4,
                items=metric_items,
            )
        )
    if cfg.include_question_audit:
        blocks.extend(_overview_question_table(designs, cfg.question_table_mode))
    if cfg.include_flow_map:
        blocks.extend(_overview_flow_map(structure))
    return blocks


def _format_count(value: float, *, weighted: bool) -> str:
    if weighted:
        return _uk(float(value), ".1f")
    return str(int(value))


def _format_bar_label(count: float, denominator: float, *, weighted: bool) -> str:
    return f"{_pct(count / denominator)} · {_format_count(count, weighted=weighted)}"


def _descriptive_distribution(
    responses: list[dict],
    qid: str,
    raw_distribution: dict[str, int],
    raw_denominator: int,
    options: list[str],
    cfg: DescriptiveConfig,
) -> tuple[dict[str, float | int], float, bool]:
    """Підготувати розподіл для PDF: raw або weighted з leave-one-dimension-out."""
    if cfg.weighting_dimensions:
        try:
            weights = compute_configured_response_weights(
                responses,
                cfg.weighting_dimensions,
                exclude_column=qid,
                cap_value=cfg.weighting_cap_value,
                moe_pct=cfg.weighting_moe_pct,
            )
        except (KeyError, ValueError):
            weights = None
        if weights is not None:
            weighted = weighted_response_distribution(
                responses,
                qid,
                weights,
                allowed_options=options,
                anonymize=cfg.anonymize,
                anonymized_label=cfg.other_label,
            )
            if weighted is not None:
                return weighted.distribution, weighted.denominator, True

    dist: dict[str, float | int] = raw_distribution
    if cfg.anonymize:
        dist = anonymize_distribution(dist, options, cfg.other_label)
    return dist, float(max(raw_denominator, 1)), False


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
        dist: dict[str, float | int] = s.distribution
        total = float(max(s.n_answered, 1))
        weighted = False
        if not s.is_text and dist:
            dist, total, weighted = _descriptive_distribution(
                responses,
                qid,
                s.distribution,
                s.n_answered,
                options.get(qid, []),
                cfg,
            )
            if cfg.hide_only_other and set(dist) == {cfg.other_label}:
                continue

        if not first:
            blocks.append(PageBreak())
        first = False
        design = designs.get(qid)
        blocks.append(Heading(design.title if design else qid, level=2))
        meta = f"Відповіли {s.n_answered}/{s.n_total} · пропуск {_uk(s.non_response_pct, '.1f')} %"
        if weighted:
            meta += f" · зважено, нормовано до n={_uk(total, '.0f')}"
        blocks.append(Para(meta))

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
        if cfg.render_mode in ("table", "both"):
            rows = [
                [value, _format_count(float(count), weighted=weighted), _pct(float(count) / total)]
                for value, count in items
            ]
            blocks.append(
                TableBlock(headers=("Варіант", "К-сть", "%"), rows=rows, col_widths=(0.6, 0.2, 0.2))
            )
        if cfg.render_mode in ("chart", "both"):
            blocks.append(
                BarChart(
                    labels=[v for v, _ in items],
                    values=[float(c) for _, c in items],
                    value_labels=[
                        _format_bar_label(float(count), total, weighted=weighted)
                        for _, count in items
                    ],
                )
            )
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
