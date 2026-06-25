"""Збір вхідних даних для секцій глобального звіту (інтеграційний шар).

SRP: цей модуль ЛИШЕ дістає та готує дані з зовнішніх сервісів (Sheets,
Forms API, прогноз) для секційних білдерів `core.reports`. Він не малює PDF
(це core.report) і не вирішує компонування (це core.reports.full_report).

Тут — I/O і доменна оркестрація, тому модуль у шарі UI, а не в core.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from core.context_tables import ContextTable, assign_tables_to_questions
from core.crosstab import PairAssociation, association_scan, crosstab
from core.crosstab_frame import answer_values
from core.forecast import ForecastError, classify_form_type, forecast_current_wave
from core.forms_api import get_linked_sheet_id, parse_question_types
from core.report import Metric
from core.sheets_api import SheetsApiError
from core.timeline import build_timeline_from_timestamps
from core.weighting import Dimension, WeightingResult, compute_weighting
from ui.google_data import list_response_timestamps, scan_population_tables

_SHORT = 40  # обрізання довгих формулювань у таблиці зв'язків


def _first_answers(responses: list[dict], qid: str) -> list[str]:
    """Перша відповідь кожного респондента на питання (порожній рядок, якщо немає)."""
    out: list[str] = []
    for r in responses:
        vals = answer_values(r, qid)
        out.append(vals[0].strip() if vals else "")
    return out


def _single_choice(structure: dict) -> dict[str, str]:
    """{question_id: title} для питань з одиночним вибором."""
    return {q.id: q.title for q in parse_question_types(structure) if q.type == "MULTIPLE_CHOICE"}


def auto_weighting(structure: dict, responses: list[dict]) -> WeightingResult | None:
    """Порахувати репрезентативність авто-детектом таблиць популяції у Sheet.

    Повертає None, якщо немає привʼязаного Sheet або придатних таблиць —
    тоді секція репрезентативності у звіт не включається.
    """
    sheet_id = get_linked_sheet_id(structure)
    if not sheet_id:
        return None
    try:
        tables = scan_population_tables(sheet_id)
    except (RuntimeError, SheetsApiError):
        return None
    return weighting_from_tables(structure, responses, tables)


def weighting_from_tables(
    structure: dict, responses: list[dict], tables: Sequence[ContextTable]
) -> WeightingResult | None:
    """Порахувати ваги з уже отриманих таблиць популяції (чиста, без I/O).

    Спільне ядро авто-зважування для глобального звіту й вкладки «Крос-таби»
    (DRY): будує кадр одиночних виборів, ексклюзивно зіставляє таблиці з
    питаннями й рахує пост-стратифікаційні ваги. None, якщо немає придатних
    таблиць або жодного виміру не зіставлено.
    """
    if not tables:
        return None
    single = _single_choice(structure)
    frame = pd.DataFrame({qid: _first_answers(responses, qid) for qid in single})
    option_sets = {qid: list(dict.fromkeys(v for v in frame[qid] if v)) for qid in single}
    option_sets = {qid: opts for qid, opts in option_sets.items() if opts}
    assigned = assign_tables_to_questions(option_sets, tables)
    dims = [Dimension(single[qid], qid, m.population) for qid, m in assigned.items()]
    if not dims:
        return None
    frame["R_ID"] = range(1, len(frame) + 1)
    return compute_weighting(frame, dims)


def top_association_rows(
    structure: dict, responses: list[dict], top_n: int = 15
) -> list[list[str]]:
    """Найсильніші попарні зв'язки питань з одиночним вибором (для секції зв'язків).

    Уже відформатовані рядки таблиці (χ²/Cramér's V, FDR-скориговано). Свідома
    спрощена версія для звіту — лише номінальні пари одиночного вибору.
    """
    single = _single_choice(structure)
    cols = {qid: _first_answers(responses, qid) for qid in single}
    eligible = [qid for qid in single if len({v for v in cols[qid] if v}) >= 2]

    pairs: list[PairAssociation] = []
    for i, q1 in enumerate(eligible):
        for q2 in eligible[i + 1 :]:
            try:
                ct = crosstab(cols[q1], cols[q2])
            except ValueError:
                continue
            pairs.append(
                PairAssociation(
                    q1,
                    q2,
                    "cramers_v",
                    ct.cramers_v,
                    0.0,
                    ct.n,
                    ct.p_value_design,
                    low_expected=ct.low_expected,
                )
            )
    if not pairs:
        return []

    def _short(text: str) -> str:
        return (text[:_SHORT] + "…") if len(text) > _SHORT else text

    rows: list[list[str]] = []
    for pr in association_scan(pairs)[:top_n]:
        p_fdr = "<0,001" if pr.p_fdr < 0.001 else f"{pr.p_fdr:.3f}".replace(".", ",")
        rows.append(
            [
                _short(single[pr.q1]),
                _short(single[pr.q2]),
                "Cramér's V",
                f"{pr.effect:.2f}".replace(".", ","),
                pr.effect_label,
                p_fdr,
            ]
        )
    return rows


def dynamics_metrics(form_id: str, form_title: str) -> tuple[list[Metric], str]:
    """Показники динаміки надходження: «Зараз» + прогноз посадки поточної хвилі.

    Прогноз — числами (без графіка). За будь-якого збою повертає лише поточний
    лічильник, не валячи звіт.
    """
    try:
        timestamps = list_response_timestamps(form_id)
    except Exception:  # noqa: BLE001 — звіт не має падати через прогноз
        return [], ""

    items = [Metric("Зараз", str(len(timestamps)))]
    if len(timestamps) < 2:
        return items, ""
    try:
        timeline = build_timeline_from_timestamps(timestamps)
        forecast, _ = forecast_current_wave(timeline, form_type=classify_form_type(form_title))
    except (ForecastError, ValueError):
        return items, ""
    half = (forecast.final_ci[1] - forecast.final_ci[0]) // 2
    items.append(Metric("Прогноз (ця хвиля)", f"{forecast.final_estimate} ±{half}"))
    items.append(Metric("Модель", forecast.model))
    return items, "Прогноз посадки поточної хвилі без нової агітації (95% ДІ)."


def report_subtitle(form_title: str) -> str:
    stamp = datetime.now().strftime("%d.%m.%Y")
    base = f"Форма: {form_title}" if form_title else ""
    return f"{base} · {stamp}" if base else stamp
