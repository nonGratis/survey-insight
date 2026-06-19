"""Сторінка «Дизайн форми» — pre-data лінтер структури анкети."""

from __future__ import annotations

import streamlit as st

from core.auth import credentials_from_dict
from core.form_flow import flow_to_dot, parse_form_flow
from core.forms_api import FormsApiError, get_form_structure
from core.forms_quality import analyze_form_design
from core.logger import get_logger
from ui.components.action_bar import ActionBarStatus, render_action_bar, render_action_status
from ui.components.auth_widget import ensure_api_access
from ui.components.form_picker import clear_forms_cache
from ui.components.metric_bar import MetricItem, render_metric_bar
from ui.components.page_shell import render_empty_state, render_error_state, render_page_header

log = get_logger(__name__)

render_page_header("Дизайн форми")

if not ensure_api_access():
    st.stop()

creds = credentials_from_dict(st.session_state["credentials"])
action = render_action_bar(
    creds,
    refresh_scope="form_design",
    show_status=False,
)
if not action.selected_form:
    st.stop()
form_id = action.selected_form["id"]


@st.cache_data(ttl=120, show_spinner="Завантажую структуру форми…")
def _cached_structure(form_id_: str, _creds_token: str) -> dict:
    return get_form_structure(creds, form_id_)


if action.refresh_clicked:
    clear_forms_cache()
    _cached_structure.clear()
    st.rerun()


try:
    structure = _cached_structure(form_id, creds.token or "")
except FormsApiError as exc:
    log.exception("ui_form_design_get_structure_failed", extra={"form_id": form_id})
    render_error_state("Не вдалося завантажити форму.", details=str(exc))
    st.stop()


render_action_status(ActionBarStatus(note="аналіз структури форми"))
designs = analyze_form_design(structure)
if not designs:
    render_empty_state("У формі немає питань для аналізу.")
else:
    n_flagged = sum(1 for d in designs if d.flags)
    n_open = sum(1 for d in designs if d.qtype in ("text", "paragraph"))
    render_metric_bar(
        [
            MetricItem("Питань", len(designs)),
            MetricItem("З прапорами", n_flagged),
            MetricItem("Відкритих", n_open),
        ],
        columns=3,
    )
    flow = parse_form_flow(structure)
    with st.container(border=True):
        st.subheader("Карта переходів")
        render_metric_bar(
            [
                MetricItem("Секцій", flow.section_count),
                MetricItem("Умовних переходів", flow.conditional_edge_count),
                MetricItem("Недосяжних", len(flow.unreachable_section_ids)),
                MetricItem("Цикли", "є" if flow.has_cycles else "немає"),
            ],
            columns=4,
        )
        flow_dot = flow_to_dot(flow)
        has_interesting_flow = (
            flow.section_count > 1
            or flow.conditional_edge_count > 0
            or bool(flow.unreachable_section_ids)
            or flow.has_cycles
        )
        if has_interesting_flow:
            st.graphviz_chart(flow_dot, width="stretch")
        else:
            st.caption("Переходів між секціями немає, тому граф не показується.")
        if flow.unreachable_section_ids:
            title_by_node = {node.id: node.title for node in flow.nodes}
            unreachable_titles = [
                title_by_node.get(section_id, section_id)
                for section_id in flow.unreachable_section_ids[:5]
            ]
            st.caption(
                "Недосяжні секції: "
                + ", ".join(unreachable_titles)
                + ". Перевірте умови переходів і завершення форми."
            )
        elif has_interesting_flow:
            st.caption("Суцільні стрілки — умовні переходи, пунктир — звичайний перехід далі.")

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
        "Прапори — евристичні підказки якості формулювання: довжина, можливе подвійне "
        "«та/або», кількість опцій. Це не вирок, а швидкий pre-data аудит анкети."
    )
