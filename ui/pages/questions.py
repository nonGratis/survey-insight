"""Сторінка «Питання» — якість анкети + крос-аналіз зв'язків між питаннями.

Три вкладки життєвого циклу:
- 🔧 Дизайн     — лінтер формулювань (до публікації, без відповідей).
- 📊 Відповіді  — розподіли + % пропуску (після збору).
- 🔀 Крос-таби  — таблиці спряженості та міри зв'язку між парами питань
                  (χ²/Cramér's V, Spearman, Odds Ratio, Pearson) з поправкою на
                  пост-стратифікаційні ваги (Rao-Scott) та BH-FDR.

Дані — через Forms API (`answers`); Sheet потрібен лише для авто-зважування
крос-табів. Форма обирається глобально (спільний sidebar-пікер).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from textwrap import fill

import altair as alt
import pandas as pd
import streamlit as st

from core.auth import credentials_from_dict
from core.context_tables import assign_tables_to_questions, scan_sheets_for_tables
from core.crosstab import (
    PairAssociation,
    association_scan,
    crosstab,
    numeric_correlation,
    odds_ratio_2x2,
    ordinal_correlation,
)
from core.forms_api import (
    FormsApiError,
    get_form_structure,
    get_linked_sheet_id,
    list_form_responses,
    parse_question_types,
)
from core.forms_quality import analyze_form_design, analyze_responses
from core.logger import get_logger
from core.sheets_api import SheetsApiError, fetch_all_grids
from core.weighting import Dimension, compute_weighting
from ui.components.auth_widget import ensure_api_access
from ui.components.form_picker import render_form_picker

log = get_logger(__name__)

NUMERIC_FRACTION = 0.8  # частка числових відповідей, аби питання вважати числовим
MAX_LABEL = 45  # обрізання довгих формулювань у віджетах вибору крос-табів


def _wrap_width_for_labels(labels: pd.Series) -> int:
    """Підібрати ширину переносу для підписів у межах 14..22 символів."""
    longest = max((len(str(value)) for value in labels), default=0)
    if longest > 80:
        return 14
    if longest > 50:
        return 16
    if longest > 30:
        return 18
    return 22


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def _anonymize_distribution(
    distribution: dict[str, int],
    allowed_options: list[str],
    anonymized_label: str,
) -> dict[str, int]:
    """Згорнути значення поза кодованими options у спільну анонімну мітку."""
    allowed = {_normalize_text(option): option for option in allowed_options}
    output: dict[str, int] = {}
    for raw_value, count in distribution.items():
        canonical = allowed.get(_normalize_text(raw_value))
        key = canonical if canonical is not None else anonymized_label
        output[key] = output.get(key, 0) + count
    return output


def _sort_distribution_items(
    distribution: dict[str, int],
    sort_mode: str,
    form_options: list[str],
    keep_label_last: bool,
    label_last_value: str,
) -> list[tuple[str, int]]:
    """Сортувати розподіл за величиною, алфавітом або порядком у формі."""
    items = list(distribution.items())

    def _label_last_flag(value: str) -> int:
        return 1 if keep_label_last and value == label_last_value else 0

    if sort_mode == "Алфавіт":
        return sorted(
            items,
            key=lambda kv: (
                _label_last_flag(kv[0]),
                _normalize_text(kv[0]),
            ),
        )
    if sort_mode == "Порядок у формі":
        order_map = {_normalize_text(option): index for index, option in enumerate(form_options)}
        return sorted(
            items,
            key=lambda kv: (
                _label_last_flag(kv[0]),
                order_map.get(_normalize_text(kv[0]), len(order_map)),
                _normalize_text(kv[0]),
            ),
        )
    return sorted(
        items,
        key=lambda kv: (
            _label_last_flag(kv[0]),
            -kv[1],
            _normalize_text(kv[0]),
        ),
    )


st.title("Запитання")

if not ensure_api_access():
    st.stop()

creds = credentials_from_dict(st.session_state["credentials"])
choice = render_form_picker(creds)
if not choice:
    st.stop()
form_id = choice["id"]


@st.cache_data(ttl=120, show_spinner="Завантажую структуру форми…")
def _cached_structure(form_id_: str, _creds_token: str) -> dict:
    return get_form_structure(creds, form_id_)


@st.cache_data(ttl=300, show_spinner="Завантажую відповіді для аналізу…")
def _cached_responses(form_id_: str, _creds_token: str) -> list[dict]:
    return list_form_responses(creds, form_id_)


@st.cache_data(ttl=300, show_spinner="Шукаю таблиці популяції у Sheet…")
def _cached_grids(sheet_id_: str, _creds_token: str) -> dict[str, list[list[str]]]:
    return fetch_all_grids(creds, sheet_id_)


try:
    structure = _cached_structure(form_id, creds.token or "")
except FormsApiError as exc:
    log.exception("ui_questions_get_structure_failed", extra={"form_id": form_id})
    st.error(f"Не вдалося завантажити форму: {exc}")
    st.stop()

st.caption(f"Форма: **{structure.get('info', {}).get('title', '—')}**")
form_questions = parse_question_types(structure)

tab_design, tab_responses, tab_crosstab = st.tabs(["🔧 Дизайн", "📊 Відповіді", "🔀 Крос-таби"])

with tab_design:
    # ДО публікації / без відповідей: лінтер якості формулювання питань.
    designs = analyze_form_design(structure)
    if not designs:
        st.info("У формі немає питань для аналізу.")
    else:
        n_flagged = sum(1 for d in designs if d.flags)
        n_open = sum(1 for d in designs if d.qtype in ("text", "paragraph"))
        dcols = st.columns(3)
        dcols[0].metric("Питань", len(designs))
        dcols[1].metric("З прапорами", n_flagged)
        dcols[2].metric("Відкритих", n_open)
        st.dataframe(
            [
                {
                    "Запитання": (d.title[:70] + "…") if len(d.title) > 70 else d.title,
                    "Тип": d.qtype_label,
                    "Опцій": d.n_options if d.n_options is not None else "—",
                    "Обовʼязк.": "так" if d.required else "ні",
                    "Прапори": ", ".join(d.flags) if d.flags else "—",
                }
                for d in designs
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Прапори — евристичні підказки якості формулювання (довжина, "
            "можливе подвійне «та/або», к-сть опцій), не вирок."
        )

with tab_responses:
    # ПІСЛЯ збору: розподіли + якість даних по питаннях.
    responses = _cached_responses(form_id, creds.token or "")
    if not responses:
        st.info("Аналіз зʼявиться після перших відповідей форми.")
    else:
        st.metric("Відповідей", len(responses))
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
        sort_mode = st.selectbox(
            "Сортування",
            options=["За величиною", "Алфавіт", "Порядок у формі"],
            index=0,
        )
        by_id = {d.question_id: d for d in analyze_form_design(structure)}
        for qid, s in stats.items():
            d = by_id.get(qid)
            st.markdown(f"**{d.title if d else qid}**")
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
                        chart_distribution = _anonymize_distribution(
                            chart_distribution,
                            options_by_id.get(qid, []),
                            anonymized_label,
                        )
                    if hide_only_other_questions and set(chart_distribution) == {anonymized_label}:
                        continue
                    sorted_items = _sort_distribution_items(
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
                        lambda value, width=wrap_width: fill(str(value), width=width)
                    )
                    chart_df["Підпис"] = chart_df.apply(
                        lambda row: f"{row['%']:.1f}% · {int(row['Кількість'])}",
                        axis=1,
                    )
                    chart_height = max(420, min(900, 30 * len(chart_df)))
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
                                axis=alt.Axis(labelLimit=0, labelLineHeight=14, labelPadding=6),
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


@dataclass
class _Var:
    """Аналітична змінна крос-аналізу (питання або бінарна опція checkbox)."""

    key: str  # унікальний ключ колонки у frame
    label: str  # людська назва
    kind: str  # "nominal" | "ordinal" | "numeric"


def _short(text: str) -> str:
    return (text[:MAX_LABEL] + "…") if len(text) > MAX_LABEL else text


def _to_float(value: str) -> float:
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        return math.nan


def _answer_list(resp: dict, qid: str) -> list[str]:
    ans = resp.get("answers", {}).get(qid, {})
    return [a.get("value", "") for a in ans.get("textAnswers", {}).get("answers", [])]


def _iter_form_questions(form: dict):
    """(qid, title, type, options) для кожного питання форми."""
    for item in form.get("items", []):
        q = item.get("questionItem", {}).get("question")
        if not q:
            continue
        qid = q.get("questionId", "")
        if not qid:
            continue
        title = item.get("title", qid)
        if "choiceQuestion" in q:
            ch = q["choiceQuestion"]
            opts = [o.get("value", "") for o in ch.get("options", [])]
            yield qid, title, ch.get("type", "RADIO"), opts
        elif "scaleQuestion" in q:
            yield qid, title, "SCALE", []
        elif "textQuestion" in q:
            yield qid, title, "TEXT", []


def _build_analysis_frame(form: dict, responses: list[dict]) -> tuple[pd.DataFrame, list[_Var]]:
    """Кадр «респондент × змінна» + типізація для крос-аналізу.

    Числові choice/шкали → порядкові; інші choice → номінальні; CHECKBOX →
    бінарні індикатори по кожній опції; текст із числами → числове. Вільний
    текст і дати — поза аналізом.
    """
    cols: dict[str, list[str]] = {}
    variables: list[_Var] = []
    for qid, title, qtype, options in _iter_form_questions(form):
        per_resp = [_answer_list(r, qid) for r in responses]
        answered = [bool(v) for v in per_resp]
        if not any(answered):
            continue

        if qtype == "CHECKBOX":
            for opt in options:
                if not opt:
                    continue
                key = f"{qid}::{opt}"
                cols[key] = [
                    ("так" if opt in vals else "ні") if ans else ""
                    for vals, ans in zip(per_resp, answered, strict=True)
                ]
                variables.append(_Var(key, f"{_short(title)} → {opt}", "nominal"))
            continue

        first = [vals[0] if vals else "" for vals in per_resp]
        non_empty = [v for v in first if v.strip()]
        if not non_empty:
            continue
        numeric_share = sum(not math.isnan(_to_float(v)) for v in non_empty) / len(non_empty)

        if qtype == "SCALE" or (
            qtype in ("RADIO", "DROP_DOWN") and numeric_share >= NUMERIC_FRACTION
        ):
            kind = "ordinal"
        elif qtype == "TEXT":
            if numeric_share < NUMERIC_FRACTION:
                continue
            kind = "numeric"
        else:
            kind = "nominal"
        cols[qid] = first
        variables.append(_Var(qid, _short(title), kind))

    frame = pd.DataFrame(cols)
    variables = [v for v in variables if frame[v.key].replace("", pd.NA).nunique() >= 2]
    return frame, variables


def _auto_weights(form: dict, responses: list[dict]) -> list[float] | None:
    """Композитні ваги через авто-детект таблиць популяції у Sheet, або None."""
    sheet_id = get_linked_sheet_id(form)
    if not sheet_id:
        return None
    try:
        tables = scan_sheets_for_tables(_cached_grids(sheet_id, creds.token or ""))
    except SheetsApiError:
        return None
    if not tables:
        return None

    single = {
        qid: title for qid, title, t, _ in _iter_form_questions(form) if t in ("RADIO", "DROP_DOWN")
    }
    wframe = pd.DataFrame(
        {qid: [(_answer_list(r, qid) or [""])[0].strip() for r in responses] for qid in single}
    )
    option_sets = {qid: list(dict.fromkeys(v for v in wframe[qid].tolist() if v)) for qid in single}
    option_sets = {q: o for q, o in option_sets.items() if o}
    assigned = assign_tables_to_questions(option_sets, tables)
    dims = [Dimension(single[qid], qid, m.population) for qid, m in assigned.items()]
    if not dims:
        return None
    res = compute_weighting(wframe.assign(R_ID=range(1, len(wframe) + 1)), dims)
    return [float(w) if w is not None and math.isfinite(w) else 1.0 for w in res.frame["w"]]


def _verdict(effect: float, label: str, p: float, significant: bool, weighted: bool) -> str:
    sig = "статистично значущий" if significant else "статистично НЕзначущий"
    note = " (з урахуванням ваг)" if weighted else ""
    p_txt = "p<0,001" if p < 0.001 else f"p={p:.3f}"
    return f"Зв'язок **{label}** (ефект {effect:.2f}), {sig} ({p_txt}, FDR){note}."


def _pair_assoc(frame: pd.DataFrame, meta: dict[str, _Var], k1: str, k2: str, w) -> PairAssociation:
    t1, t2 = meta[k1].kind, meta[k2].kind
    if t1 == "numeric" and t2 == "numeric":
        cr = numeric_correlation(frame[k1].map(_to_float), frame[k2].map(_to_float), w)
        return PairAssociation(
            k1, k2, "pearson", abs(cr.coef), math.copysign(1, cr.coef), cr.n, cr.p_value
        )
    if t1 in ("ordinal", "numeric") and t2 in ("ordinal", "numeric"):
        cr = ordinal_correlation(frame[k1].map(_to_float), frame[k2].map(_to_float), w)
        return PairAssociation(
            k1, k2, "spearman", abs(cr.coef), math.copysign(1, cr.coef), cr.n, cr.p_value
        )
    ct = crosstab(frame[k1], frame[k2], w)
    return PairAssociation(
        k1,
        k2,
        "cramers_v",
        ct.cramers_v,
        0.0,
        ct.n,
        ct.p_value_design,
        low_expected=ct.low_expected,
    )


def _render_pair(frame: pd.DataFrame, meta: dict[str, _Var], k1: str, k2: str, w) -> None:
    v1, v2 = meta[k1], meta[k2]
    both_metric = v1.kind in ("ordinal", "numeric") and v2.kind in ("ordinal", "numeric")

    if both_metric:
        is_numeric = v1.kind == "numeric" and v2.kind == "numeric"
        cr = (numeric_correlation if is_numeric else ordinal_correlation)(
            frame[k1].map(_to_float), frame[k2].map(_to_float), w
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
            x=alt.X("col:N", title=_short(v2.label)),
            y=alt.Y("row:N", title=_short(v1.label)),
            color=alt.Color("freq:Q", title="Частота", scale=alt.Scale(scheme="blues")),
            tooltip=["row", "col", alt.Tooltip("freq:Q", format=".1f")],
        )
    )
    st.altair_chart(heat, width="stretch")


def _render_overview(frame: pd.DataFrame, meta: dict[str, _Var], w, var_keys: list[str]) -> None:
    pairs: list[PairAssociation] = []
    for i, k1 in enumerate(var_keys):
        for k2 in var_keys[i + 1 :]:
            try:
                pairs.append(_pair_assoc(frame, meta, k1, k2, w))
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
    table = pd.DataFrame(
        [
            {
                "Запитання 1": meta[pr.q1].label,
                "Запитання 2": meta[pr.q2].label,
                "Міра": measure_label[pr.measure],
                "Ефект": round(pr.effect, 3),
                "Сила": pr.effect_label,
                "p (FDR)": round(pr.p_fdr, 4),
                "Значущий": "так" if pr.significant else "ні",
                "Розріджена": "⚠️" if pr.low_expected else "",
            }
            for pr in scanned
        ]
    )
    st.dataframe(table.head(40), width="stretch", hide_index=True)
    st.caption(
        "Ефект — розмір зв'язку (0–0,1 немає, 0,1–0,3 слабкий, 0,3–0,5 помірний, "
        ">0,5 сильний). «Значущий» — за FDR-скоригованим p<0,05. При великому n "
        "орієнтуйтесь на ефект, а не на p."
    )


with tab_crosstab:
    responses = _cached_responses(form_id, creds.token or "")
    if not responses:
        st.info("Крос-аналіз зʼявиться після збору відповідей.")
    else:
        ct_frame, variables = _build_analysis_frame(structure, responses)
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
                weights = _auto_weights(structure, responses)
                if weights is None:
                    st.info(
                        "Ваги недоступні (немає таблиць популяції у Sheet) — показано незважено."
                    )
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
