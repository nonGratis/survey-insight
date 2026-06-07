"""Сторінка «Питання» — якість анкети по питаннях (дизайн + відповіді).

Дві сутності життєвого циклу як вкладки:
- 🔧 Дизайн — лінтер формулювань (до публікації, без відповідей).
- 📊 Відповіді — розподіли + % пропуску (після збору).

Замінює стару сторінку «Огляд форми» (яка дублювала цю логіку через Sheet).
Тут — через Forms API (`answers`), Sheet не потрібен. Форма обирається
глобально (спільний sidebar-пікер).
"""

from __future__ import annotations

import streamlit as st

from core.auth import credentials_from_dict
from core.forms_api import FormsApiError, get_form_structure, list_form_responses
from core.forms_quality import analyze_form_design, analyze_responses
from core.logger import get_logger
from ui.components.auth_widget import ensure_api_access
from ui.components.form_picker import render_form_picker

log = get_logger(__name__)

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
                    top = dict(sorted(s.distribution.items(), key=lambda kv: -kv[1])[:12])
                    st.bar_chart(top, horizontal=True)
            st.divider()
        st.caption("Крос-табуляції питання × питання + χ²-тест — наступний крок.")
