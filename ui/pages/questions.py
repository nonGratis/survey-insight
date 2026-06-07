"""Сторінка «Питання» — якість анкети по питаннях (дизайн + відповіді).

Дві сутності життєвого циклу як вкладки:
- 🔧 Дизайн — лінтер формулювань (до публікації, без відповідей).
- 📊 Відповіді — розподіли + % пропуску (після збору).

Замінює стару сторінку «Огляд форми» (яка дублювала цю логіку через Sheet).
Тут — через Forms API (`answers`), Sheet не потрібен. Форма обирається
глобально (спільний sidebar-пікер).
"""

from __future__ import annotations

from textwrap import fill

import altair as alt
import pandas as pd
import streamlit as st

from core.auth import credentials_from_dict
from core.forms_api import (
    FormsApiError,
    get_form_structure,
    list_form_responses,
    parse_question_types,
)
from core.forms_quality import analyze_form_design, analyze_responses
from core.logger import get_logger
from ui.components.auth_widget import ensure_api_access
from ui.components.form_picker import render_form_picker

log = get_logger(__name__)


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
) -> list[tuple[str, int]]:
    """Сортувати розподіл за величиною, алфавітом або порядком у формі."""
    items = list(distribution.items())
    if sort_mode == "Алфавіт":
        return sorted(items, key=lambda kv: _normalize_text(kv[0]))
    if sort_mode == "Порядок у формі":
        order_map = {_normalize_text(option): index for index, option in enumerate(form_options)}
        return sorted(
            items,
            key=lambda kv: (
                order_map.get(_normalize_text(kv[0]), len(order_map)),
                _normalize_text(kv[0]),
            ),
        )
    return sorted(items, key=lambda kv: (-kv[1], _normalize_text(kv[0])))

st.title("Питання")

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


try:
    structure = _cached_structure(form_id, creds.token or "")
except FormsApiError as exc:
    log.exception("ui_questions_get_structure_failed", extra={"form_id": form_id})
    st.error(f"Не вдалося завантажити форму: {exc}")
    st.stop()

st.caption(f"Форма: **{structure.get('info', {}).get('title', '—')}**")
form_questions = parse_question_types(structure)

tab_design, tab_responses = st.tabs(["🔧 Дизайн", "📊 Відповіді"])

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
                    "Питання": (d.title[:70] + "…") if len(d.title) > 70 else d.title,
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
                    sorted_items = _sort_distribution_items(
                        chart_distribution,
                        sort_mode,
                        options_by_id.get(qid, []),
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
                    st.altair_chart((base + labels).properties(height=chart_height), width="stretch")
            st.divider()
        st.caption("Крос-табуляції питання × питання + χ²-тест — наступний крок.")
