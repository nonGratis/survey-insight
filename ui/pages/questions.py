"""Сторінка «Запитання» — розподіли відповідей + крос-аналіз зв'язків.

Дві вкладки після збору:
- 📊 Відповіді  — розподіли + % пропуску.
- 🔀 Крос-таби  — таблиці спряженості та міри зв'язку між парами питань
                  (χ²/Cramér's V, Spearman, Odds Ratio, Pearson) з поправкою на
                  пост-стратифікаційні ваги (Rao-Scott) та BH-FDR.

Дані — через Forms API (`answers`); Sheet потрібен лише для авто-зважування
крос-табів. Форма обирається глобально (спільний sidebar-пікер).
"""

from __future__ import annotations

import math
from textwrap import fill

import altair as alt
import pandas as pd
import streamlit as st

from core.auth import credentials_from_dict
from core.context_tables import scan_sheets_for_tables
from core.crosstab import (
    ASSOCIATION_FILTER_MODES,
    IMPORTANT_EFFECT_THRESHOLD,
    PairAssociation,
    association_scan,
    classify_association,
    crosstab,
    filter_associations,
    numeric_correlation,
    odds_ratio_2x2,
    ordinal_correlation,
)
from core.crosstab_frame import (
    Var,
    answer_values,
    build_analysis_frame,
    pair_association,
    short_label,
    to_float,
)
from core.forms_api import (
    FormsApiError,
    get_form_structure,
    get_linked_sheet_id,
    list_form_responses,
    parse_question_types,
)
from core.forms_quality import (
    SORT_MODES,
    analyze_responses,
    anonymize_distribution,
    canonicalize_distribution,
    sort_distribution,
)
from core.logger import get_logger
from core.report import render_pdf
from core.reports import DescriptiveConfig, questions_report
from core.sheets_api import SheetsApiError, fetch_all_grids
from core.weighting import RID_COLUMN, compute_weighting
from ui.components.action_bar import ActionBarStatus, render_action_bar, render_action_status
from ui.components.auth_widget import ensure_api_access
from ui.components.form_picker import clear_forms_cache
from ui.components.metric_bar import MetricItem, render_metric_bar
from ui.components.mode_switch import render_mode_switch
from ui.components.page_shell import (
    render_empty_state,
    render_error_state,
    render_page_header,
)
from ui.report_data import weighting_from_tables

log = get_logger(__name__)

RESPONSE_AXIS_LABEL_LIMIT_PX = 320
RESPONSE_AXIS_LABEL_LINE_HEIGHT_PX = 13


def _wrap_width_for_labels(labels: pd.Series) -> int:
    """Підібрати ширину переносу, щоб ліва шкала не забирала більшість графіка."""
    longest = max((len(str(value)) for value in labels), default=0)
    if longest > 80:
        return 28
    if longest > 50:
        return 30
    if longest > 30:
        return 32
    return 36


def _wrap_axis_label(value: object, width: int) -> str:
    """Wrap one categorical axis label; Vega turns line breaks into multiline text."""
    return fill(str(value), width=width, break_long_words=False, break_on_hyphens=False)


def _response_axis() -> alt.Axis:
    """Left answer axis capped so the plot area keeps at least half of the container."""
    return alt.Axis(
        labelLimit=RESPONSE_AXIS_LABEL_LIMIT_PX,
        labelLineHeight=RESPONSE_AXIS_LABEL_LINE_HEIGHT_PX,
        labelOverlap=False,
        labelPadding=8,
        labelExpr="split(datum.label, '\\n')",
    )


# Логіка анонімізації/сортування розподілу — спільна з PDF-звітом
# (core.forms_quality.anonymize_distribution / sort_distribution), щоб екран і
# звіт давали ІДЕНТИЧНИЙ результат (DRY).


render_page_header("Запитання")

if not ensure_api_access():
    st.stop()

creds = credentials_from_dict(st.session_state["credentials"])
action = render_action_bar(
    creds,
    refresh_scope="questions",
    show_status=False,
)
if not action.selected_form:
    st.stop()
form_id = action.selected_form["id"]


@st.cache_data(ttl=120, show_spinner="Завантажую структуру форми…")
def _cached_structure(form_id_: str, _creds_token: str) -> dict:
    return get_form_structure(creds, form_id_)


@st.cache_data(ttl=300, show_spinner="Завантажую відповіді для аналізу…")
def _cached_responses(form_id_: str, _creds_token: str) -> list[dict]:
    return list_form_responses(creds, form_id_)


@st.cache_data(ttl=300, show_spinner="Шукаю таблиці популяції у Sheet…")
def _cached_grids(sheet_id_: str, _creds_token: str) -> dict[str, list[list[str]]]:
    return fetch_all_grids(creds, sheet_id_)


if action.refresh_clicked:
    clear_forms_cache()
    _cached_structure.clear()
    _cached_responses.clear()
    _cached_grids.clear()
    st.rerun()


try:
    structure = _cached_structure(form_id, creds.token or "")
except FormsApiError as exc:
    log.exception("ui_questions_get_structure_failed", extra={"form_id": form_id})
    render_error_state("Не вдалося завантажити форму.", details=str(exc))
    st.stop()

form_questions = parse_question_types(structure)

QUESTION_MODES = ["Відповіді", "Крос-таби"]
mode = render_mode_switch("Режим аналізу", QUESTION_MODES, key="questions_mode")

if mode == "Відповіді":
    # ПІСЛЯ збору: розподіли + якість даних по питаннях.
    responses = _cached_responses(form_id, creds.token or "")
    render_action_status(ActionBarStatus(responses=len(responses), note="розподіли відповідей"))
    if not responses:
        render_empty_state("Аналіз зʼявиться після перших відповідей форми.")
    else:
        render_metric_bar([MetricItem("Відповідей", len(responses))], columns=3)
        stats = analyze_responses(structure, responses)
        options_by_id = {q.id: q.options for q in form_questions}
        with st.expander("Фільтр анонімізації", expanded=False):
            anonymize_open_values = st.checkbox(
                "Згорнути відкриті відповіді, яких немає серед кодованих варіантів",
                value=False,
            )
            anonymized_label = st.text_input("Мітка для інших", value="Інше*")
            keep_other_last = st.checkbox(
                'Тримати "Інше*" в кінці графіка',
                value=True,
            )
            hide_only_other_questions = st.checkbox(
                'Прибирати питання, де лишилось лише "Інше*"',
                value=False,
            )
        sort_mode = st.selectbox("Сортування", options=list(SORT_MODES), index=0)

        # PDF успадковує поточні екранні налаштування (анонімізація, сортування)
        # — той самий core.reports, що й глобальний «Звіт» (DRY).
        _pdf_config = DescriptiveConfig(
            anonymize=anonymize_open_values,
            other_label=anonymized_label,
            keep_other_last=keep_other_last,
            hide_only_other=hide_only_other_questions,
            sort_mode=sort_mode,
            render_mode="chart",
        )
        st.download_button(
            ":material/picture_as_pdf: Завантажити звіт за результатами (PDF)",
            data=render_pdf(
                questions_report(
                    structure, responses, structure.get("info", {}).get("title", ""), _pdf_config
                )
            ),
            file_name=f"report_{form_id}.pdf",
            mime="application/pdf",
            help="З поточними налаштуваннями екрана; результат кожного питання — окремою сторінкою.",
        )
        title_by_id = {q.id: q.title for q in form_questions}
        for qid, s in stats.items():
            st.markdown(f"**{title_by_id.get(qid, qid)}**")
            meta = f"відповіли {s.n_answered}/{s.n_total} · пропуск {s.non_response_pct}%"
            if s.is_text:
                if s.text_median_len:
                    meta += f" · медіана довжини {int(s.text_median_len)} симв."
                st.caption(meta)
            else:
                st.caption(meta)
                if s.distribution:
                    chart_distribution = s.distribution
                    if anonymize_open_values:
                        chart_distribution = anonymize_distribution(
                            chart_distribution,
                            options_by_id.get(qid, []),
                            anonymized_label,
                        )
                    chart_distribution = canonicalize_distribution(
                        chart_distribution,
                        options_by_id.get(qid, []),
                    )
                    if hide_only_other_questions and set(chart_distribution) == {anonymized_label}:
                        continue
                    sorted_items = sort_distribution(
                        chart_distribution,
                        sort_mode,
                        options_by_id.get(qid, []),
                        keep_other_last,
                        anonymized_label,
                    )[:30]
                    top_items = sorted_items
                    chart_df = pd.DataFrame(top_items, columns=["Відповідь", "Кількість"])
                    chart_df["%"] = chart_df["Кількість"] / max(s.n_answered, 1) * 100
                    wrap_width = _wrap_width_for_labels(chart_df["Відповідь"])
                    chart_df["Відповідь_перенесена"] = chart_df["Відповідь"].map(
                        lambda value, width=wrap_width: _wrap_axis_label(value, width)
                    )
                    max_label_lines = max(
                        (str(value).count("\n") + 1 for value in chart_df["Відповідь_перенесена"]),
                        default=1,
                    )
                    chart_df["Підпис"] = chart_df.apply(
                        lambda row: f"{row['%']:.1f}% · {int(row['Кількість'])}",
                        axis=1,
                    )
                    row_height = max(30, RESPONSE_AXIS_LABEL_LINE_HEIGHT_PX * max_label_lines + 8)
                    chart_height = max(420, min(1200, row_height * len(chart_df)))
                    y_order = chart_df["Відповідь_перенесена"].tolist()
                    base = (
                        alt.Chart(chart_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("Кількість:Q", title="Відповідей"),
                            y=alt.Y(
                                "Відповідь_перенесена:N",
                                sort=y_order,
                                title=None,
                                axis=_response_axis(),
                            ),
                            tooltip=[
                                alt.Tooltip("Відповідь:N", title="Відповідь"),
                                alt.Tooltip("Кількість:Q", title="Відповідей"),
                                alt.Tooltip("%:Q", title="%", format=".1f"),
                            ],
                        )
                    )
                    labels = (
                        alt.Chart(chart_df)
                        .mark_text(align="left", baseline="middle", dx=4)
                        .encode(
                            x=alt.X("Кількість:Q"),
                            y=alt.Y("Відповідь_перенесена:N", sort=y_order),
                            text="Підпис:N",
                        )
                    )
                    st.altair_chart(
                        (base + labels).properties(height=chart_height), width="stretch"
                    )
            st.divider()
        st.caption("Аналіз зв'язків між питаннями — на вкладці «🔀 Крос-таби».")


# ======================= Крос-таби =========================================


def _auto_weights(form: dict, responses: list[dict]) -> list[float] | None:
    """Per-respondent ваги через авто-детект таблиць популяції у Sheet, або None.

    Тонка I/O-обгортка: дістає таблиці популяції з привʼязаного Sheet (через
    кеш Streamlit) і делегує розрахунок спільному ядру `weighting_from_tables`
    (те саме, що й глобальний «Звіт» — DRY), повертаючи лише колонку ваг.
    """
    sheet_id = get_linked_sheet_id(form)
    if not sheet_id:
        return None
    try:
        tables = scan_sheets_for_tables(_cached_grids(sheet_id, creds.token or ""))
    except SheetsApiError:
        return None
    result = weighting_from_tables(form, responses, tables)
    if result is None:
        return None
    return [float(w) if w is not None and math.isfinite(w) else 1.0 for w in result.frame["w"]]


def _configured_weights(form_id_: str, responses: list[dict]) -> tuple[list[float] | None, str]:
    """Per-respondent ваги з активної конфігурації сторінки «Зважування»."""
    config = st.session_state.get(f"weighting_config_{form_id_}")
    if not config:
        return None, ""
    dimensions = config.get("dimensions") or []
    if not dimensions:
        return None, ""

    qids = [dim.column for dim in dimensions]
    rows: list[dict[str, object]] = []
    for i, resp in enumerate(responses, start=1):
        row: dict[str, object] = {RID_COLUMN: i}
        for qid in qids:
            values = answer_values(resp, qid)
            row[qid] = values[0].strip() if values else ""
        rows.append(row)

    cap_value = float(config.get("cap_value", 0.0) or 0.0)
    caps = None
    if cap_value > 0:
        caps = {dim.name: {stratum: cap_value for stratum in dim.population} for dim in dimensions}

    try:
        result = compute_weighting(
            pd.DataFrame(rows),
            dimensions,
            moe=float(config.get("moe_pct", 5.0)) / 100.0,
            caps=caps,
        )
    except (KeyError, ValueError):
        return None, ""
    weights = [float(w) if w is not None and math.isfinite(w) else 1.0 for w in result.frame["w"]]
    return weights, f"конфігурація «Зважування» · {len(dimensions)} вим."


def _verdict(effect: float, label: str, p: float, significant: bool, weighted: bool) -> str:
    sig = "статистично значущий" if significant else "статистично НЕзначущий"
    note = " (з урахуванням ваг)" if weighted else ""
    p_txt = "p<0,001" if p < 0.001 else f"p={p:.3f}"
    return f"Зв'язок **{label}** (ефект {effect:.2f}), {sig} ({p_txt}, FDR){note}."


def _render_pair(frame: pd.DataFrame, meta: dict[str, Var], k1: str, k2: str, w) -> None:
    v1, v2 = meta[k1], meta[k2]
    both_metric = v1.kind in ("ordinal", "numeric") and v2.kind in ("ordinal", "numeric")

    if both_metric:
        is_numeric = v1.kind == "numeric" and v2.kind == "numeric"
        cr = (numeric_correlation if is_numeric else ordinal_correlation)(
            frame[k1].map(to_float), frame[k2].map(to_float), w
        )
        name = "Pearson r" if cr.method == "pearson" else "Spearman ρ"
        mc = st.columns(3)
        mc[0].metric(name, f"{cr.coef:+.2f}")
        mc[1].metric("Сила", cr.effect_label)
        mc[2].metric("n", cr.n)
        direction = "додатний" if cr.coef >= 0 else "відʼємний"
        p_txt = "p<0,001" if cr.p_value < 0.001 else f"p={cr.p_value:.3f}"
        st.caption(f"{name} = {cr.coef:+.3f} ({direction} напрям), {p_txt}.")
        return

    ct = crosstab(frame[k1], frame[k2], w)
    mc = st.columns(4)
    mc[0].metric("Cramér's V", f"{ct.cramers_v:.2f}", help="Сила зв'язку 0..1.")
    mc[1].metric("Сила", ct.effect_label)
    mc[2].metric("χ²", f"{ct.chi2:.1f}", help=f"df={ct.dof}")
    p_show = ct.p_value_design
    mc[3].metric(
        "p (з вагами)" if w is not None else "p", "<0,001" if p_show < 0.001 else f"{p_show:.3f}"
    )

    st.markdown(_verdict(ct.cramers_v, ct.effect_label, p_show, ct.significant, w is not None))
    if ct.low_expected:
        warn = "Таблиця розріджена (очікувані частоти <5): χ² ненадійний."
        if ct.fisher_p is not None:
            warn += f" Точний тест Фішера: p={ct.fisher_p:.3f}."
        st.warning(warn)
    orr = odds_ratio_2x2(ct)
    if orr is not None:
        st.caption(
            f"Відношення шансів (OR) = {orr.odds_ratio:.2f} "
            f"(95% ДІ {orr.ci_low:.2f}–{orr.ci_high:.2f})."
        )

    st.markdown("**Таблиця спряженості** (частка по рядку, %)")
    row_pct = ct.table.div(ct.table.sum(axis=1), axis=0).mul(100).round(1)
    st.dataframe(row_pct, width="stretch")

    index_name = ct.table.index.name or "index"
    long = ct.table.reset_index().melt(id_vars=index_name, var_name="col", value_name="freq")
    long.columns = ["row", "col", "freq"]
    heat = (
        alt.Chart(long)
        .mark_rect()
        .encode(
            x=alt.X("col:N", title=short_label(v2.label)),
            y=alt.Y("row:N", title=short_label(v1.label)),
            color=alt.Color("freq:Q", title="Частота", scale=alt.Scale(scheme="blues")),
            tooltip=["row", "col", alt.Tooltip("freq:Q", format=".1f")],
        )
    )
    st.altair_chart(heat, width="stretch")


def _render_overview(frame: pd.DataFrame, meta: dict[str, Var], w, var_keys: list[str]) -> None:
    pairs: list[PairAssociation] = []
    for i, k1 in enumerate(var_keys):
        for k2 in var_keys[i + 1 :]:
            try:
                pairs.append(pair_association(frame, meta, k1, k2, w))
            except ValueError:
                continue
    if not pairs:
        st.info("Недостатньо даних для матриці зв'язків.")
        return
    scanned = association_scan(pairs)

    st.markdown("**Матриця сили зв'язків** (Cramér's V / |ρ|)")
    rows = []
    for pr in scanned:
        rows.append({"q1": meta[pr.q1].label, "q2": meta[pr.q2].label, "effect": pr.effect})
        rows.append({"q1": meta[pr.q2].label, "q2": meta[pr.q1].label, "effect": pr.effect})
    mat = pd.DataFrame(rows)
    heat = (
        alt.Chart(mat)
        .mark_rect()
        .encode(
            x=alt.X("q2:N", title=None),
            y=alt.Y("q1:N", title=None),
            color=alt.Color(
                "effect:Q", title="Сила", scale=alt.Scale(scheme="oranges", domain=[0, 1])
            ),
            tooltip=["q1", "q2", alt.Tooltip("effect:Q", format=".2f")],
        )
        .properties(height=max(220, 22 * len(var_keys)))
    )
    st.altair_chart(heat, width="stretch")

    st.markdown("**Найсильніші зв'язки** (за спаданням ефекту, FDR-скориговано)")
    measure_label = {"cramers_v": "Cramér's V", "spearman": "Spearman ρ", "pearson": "Pearson r"}
    measure_options = list(measure_label)
    filter_mode = render_mode_switch(
        "Фільтр зв'язків",
        ASSOCIATION_FILTER_MODES,
        key="association_overview_filter_mode",
    )
    f1, f2, f3 = st.columns([1.15, 1.0, 1.85])
    min_effect = f1.slider(
        "Мінімальна сила ефекту",
        min_value=0.0,
        max_value=1.0,
        value=IMPORTANT_EFFECT_THRESHOLD,
        step=0.01,
        disabled=filter_mode != "Важливі",
        help="Поріг практичної важливості для режиму «Важливі».",
    )
    hide_sparse = f2.checkbox(
        "Приховати розріджені",
        value=filter_mode == "Важливі",
        key=f"association_hide_sparse_{filter_mode}",
        help="Ховає пари з малими очікуваними частотами, де статистика менш надійна.",
    )
    selected_measures = f3.multiselect(
        "Тип міри",
        options=measure_options,
        default=measure_options,
        format_func=lambda name: measure_label[name],
    )

    filtered = filter_associations(
        scanned,
        filter_mode,
        min_effect=min_effect,
        hide_sparse=hide_sparse,
        measures=selected_measures,
    )
    important_count = sum(classify_association(pr) in {"ключовий", "важливий"} for pr in scanned)
    unreliable_count = sum(pr.low_expected for pr in scanned)
    st.caption(
        f"Показано {len(filtered)} з {len(scanned)} зв'язків · "
        f"важливих {important_count} · ненадійних {unreliable_count}"
    )
    if not filtered:
        st.info(
            "За поточними фільтрами зв'язків не знайдено. "
            "Зменшіть поріг сили ефекту, увімкніть розріджені таблиці або перейдіть у режим «Усі»."
        )
        return

    table = pd.DataFrame(
        [
            {
                "Запитання 1": meta[pr.q1].label,
                "Запитання 2": meta[pr.q2].label,
                "Статус": classify_association(pr),
                "Міра": measure_label[pr.measure],
                "Ефект": round(pr.effect, 3),
                "Сила": pr.effect_label,
                "p (FDR)": round(pr.p_fdr, 4),
                "Значущий": "так" if pr.significant else "ні",
                "Розріджена": "так" if pr.low_expected else "",
            }
            for pr in filtered
        ]
    )
    st.dataframe(table.head(40), width="stretch", hide_index=True)
    st.caption(
        "Ефект — розмір зв'язку (0–0,1 немає, 0,1–0,3 слабкий, 0,3–0,5 помірний, "
        ">0,5 сильний). «Значущий» — за FDR-скоригованим p<0,05. При великому n "
        "орієнтуйтесь на ефект, а не на p."
    )


if mode == "Крос-таби":
    responses = _cached_responses(form_id, creds.token or "")
    render_action_status(ActionBarStatus(responses=len(responses), note="крос-аналіз"))
    if not responses:
        st.info("Крос-аналіз зʼявиться після збору відповідей.")
    else:
        ct_frame, variables = build_analysis_frame(structure, responses)
        meta = {v.key: v for v in variables}
        if len(variables) < 2:
            st.info("Замало придатних питань для крос-аналізу (потрібно ≥2).")
        else:
            use_w = st.toggle(
                "Враховувати ваги (постстратифікація)",
                value=False,
                help="Зважені частки + поправка Rao-Scott (χ²/DEFF). Потрібні таблиці популяції у Sheet.",
            )
            weights = None
            if use_w:
                weight_source = ""
                weights, weight_source = _configured_weights(form_id, responses)
                if weights is None:
                    weights = _auto_weights(structure, responses)
                    weight_source = "авто-детект таблиць популяції у Sheet" if weights else ""
                if weights is None:
                    st.info(
                        "Ваги недоступні: налаштуйте «Зважування» або додайте таблиці популяції у Sheet. "
                        "Показано незважено."
                    )
                else:
                    st.caption(f"Ваги: {weight_source}.")
            mode = st.radio("Режим", ["Пара питань", "Огляд зв'язків"], horizontal=True)

            if mode == "Пара питань":
                keys = [v.key for v in variables]
                c1, c2 = st.columns(2)
                k1 = c1.selectbox("Запитання 1 (рядки)", keys, format_func=lambda k: meta[k].label)
                k2 = c2.selectbox(
                    "Запитання 2 (стовпці)",
                    keys,
                    index=min(1, len(keys) - 1),
                    format_func=lambda k: meta[k].label,
                )
                if k1 == k2:
                    st.info("Оберіть два різні питання.")
                else:
                    _render_pair(ct_frame, meta, k1, k2, weights)
            else:
                chosen = st.multiselect(
                    "Запитання для матриці",
                    options=[v.key for v in variables],
                    default=[v.key for v in variables][:12],
                    format_func=lambda k: meta[k].label,
                )
                if len(chosen) < 2:
                    st.info("Оберіть принаймні два питання.")
                else:
                    _render_overview(ct_frame, meta, weights, chosen)
