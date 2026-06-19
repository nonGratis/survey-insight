"""Сторінка «Дизайн форми» — pre-data лінтер структури анкети."""

from __future__ import annotations

import streamlit as st

from core.auth import credentials_from_dict
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
