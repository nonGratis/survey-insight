"""Сторінка «Звіт» — глобальний аналітичний PDF-звіт за обраною формою.

Це справді ГЛОБАЛЬНА функція (не дубль локальних кнопок): один документ зводить
кілька розділів аналізу — дескриптивну статистику, репрезентативність, зв'язки
між питаннями та динаміку надходження. Розділи й формат налаштовуються панеллю.

Композиція — спільними секційними білдерами `core.reports` (ті самі, що й
локальні кнопки) → нуль дублювання. Важкі обчислення (Sheets, прогноз)
виконуються лише за натиском «Сформувати», а не на кожному перемальовуванні.
"""

from __future__ import annotations

import streamlit as st

from core.auth import credentials_from_dict
from core.forms_api import FormsApiError, get_form_structure, list_form_responses
from core.forms_quality import SORT_MODES
from core.logger import get_logger
from core.report import render_pdf
from core.reports import (
    DescriptiveConfig,
    associations_section,
    descriptive_section,
    dynamics_section,
    full_report,
    overview_section,
    representativeness_section,
)
from ui.components.auth_widget import ensure_api_access
from ui.components.form_picker import render_form_picker
from ui.components.page_shell import (
    render_empty_state,
    render_error_state,
    render_form_caption,
    render_page_header,
)
from ui.report_data import auto_weighting, dynamics_metrics, report_subtitle, top_association_rows

log = get_logger(__name__)

_RENDER_LABELS = {"Діаграми": "chart", "Таблиці": "table", "Діаграми + таблиці": "both"}

render_page_header("Звіт")

if not ensure_api_access():
    st.stop()

creds = credentials_from_dict(st.session_state["credentials"])
choice = render_form_picker(creds)
if not choice:
    st.stop()
form_id = choice["id"]


@st.cache_data(ttl=300, show_spinner="Завантажую дані форми…")
def _load(form_id_: str, _creds_token: str) -> tuple[dict, list[dict]]:
    return get_form_structure(creds, form_id_), list_form_responses(creds, form_id_)


try:
    structure, responses = _load(form_id, creds.token or "")
except FormsApiError as exc:
    log.exception("ui_report_load_failed", extra={"form_id": form_id})
    render_error_state("Не вдалося завантажити форму.", details=str(exc))
    st.stop()

form_title = structure.get("info", {}).get("title", "")
render_form_caption(form_title)

if not responses:
    render_empty_state("Звіт зʼявиться після перших відповідей форми.")
    st.stop()

# --- розділи звіту ----------------------------------------------------------
st.subheader("Розділи")
cols = st.columns(4)
inc_descriptive = cols[0].checkbox("Дескриптив", value=True)
inc_representativeness = cols[1].checkbox("Репрезентативність", value=True)
inc_associations = cols[2].checkbox("Зв'язки", value=True)
inc_dynamics = cols[3].checkbox("Динаміка", value=True)

with st.expander("Налаштування дескриптиву"):
    render_mode_label = st.radio("Формат", options=list(_RENDER_LABELS), index=0, horizontal=True)
    sort_mode = st.selectbox("Сортування", options=list(SORT_MODES), index=0)
    anonymize = st.checkbox("Анонімізувати відкриті відповіді", value=False)
    other_label = st.text_input("Мітка для інших", value="Інше*")
    keep_other_last = st.checkbox("Тримати «Інше*» в кінці", value=True)
    hide_only_other = st.checkbox("Прибирати питання лише з «Інше*»", value=False)
    top_n = st.number_input("Максимум варіантів на питання", min_value=5, max_value=100, value=30)

config = DescriptiveConfig(
    anonymize=anonymize,
    other_label=other_label,
    keep_other_last=keep_other_last,
    hide_only_other=hide_only_other,
    sort_mode=sort_mode,
    render_mode=_RENDER_LABELS[render_mode_label],
    top_n=int(top_n),
)

_pdf_key = f"report_pdf_{form_id}"

if st.button(":material/picture_as_pdf: Сформувати звіт", type="primary"):
    sections: list[list[object]] = []
    if inc_descriptive:
        sections.append(overview_section(structure, responses))
        sections.append(descriptive_section(structure, responses, config))
    if inc_representativeness:
        with st.spinner("Рахую репрезентативність…"):
            weighting = auto_weighting(creds, structure, responses)
        if weighting is not None:
            sections.append(representativeness_section(weighting))
        else:
            st.caption(
                "Репрезентативність пропущено: у привʼязаному Sheet немає таблиць популяції."
            )
    if inc_associations:
        with st.spinner("Шукаю зв'язки…"):
            sections.append(associations_section(top_association_rows(structure, responses)))
    if inc_dynamics:
        with st.spinner("Будую прогноз динаміки…"):
            items, note = dynamics_metrics(creds, form_id, form_title)
        if items:
            sections.append(dynamics_section(items, note))

    if not sections:
        st.warning("Оберіть принаймні один розділ.")
        st.session_state.pop(_pdf_key, None)
    else:
        st.session_state[_pdf_key] = render_pdf(
            full_report(
                "Аналітичний звіт за результатами опитування",
                report_subtitle(form_title),
                sections,
                footer="Survey Insight · Звіт",
            )
        )

if st.session_state.get(_pdf_key):
    st.download_button(
        ":material/download: Завантажити повний звіт (PDF)",
        data=st.session_state[_pdf_key],
        file_name=f"report_{form_id}.pdf",
        mime="application/pdf",
    )
