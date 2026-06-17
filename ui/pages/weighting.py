"""Сторінка «Зважування» — постстратифікація та репрезентативність (standalone).

Самодостатня сторінка поза рештою вкладок: рахує ваги, що коригують перекоси
вибірки відповідей за відомими вимірами (підрозділ, курс, стать, спеціальність —
будь-якими), та метрики репрезентативності (DEFF Кіша, n_eff, MoE, MoE_DEFF).

Потік:
  1. Обрати форму (глобальний пікер) → структура + відповіді (Forms API).
  2. R_ID = наскрізний номер у порядку надходження (порядок повернення API).
  3. Популяція страт: авто-детект таблиць у привʼязаному Sheet АБО ручний CSV.
     Один файл/таблиця = один вимір. Завжди абсолютні N_h.
  4. compute_weighting → ваги, композит (добуток), таймлайни.
  5. BAN-метрики, таблиця ваг (сортовна за недопредставленістю), таймлайн ваг
     і DEFF, експорт повного кадру з R_ID.

Математика — у core.weighting (звірена 1:1 з еталоном). Детекція таблиць —
core.context_tables. Тут лише оркестрація та UI.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from core.auth import credentials_from_dict
from core.context_tables import (
    ContextTable,
    assign_tables_to_questions,
    match_population,
    parse_population_csv,
    scan_sheets_for_tables,
)
from core.forms_api import (
    FormsApiError,
    get_form_structure,
    get_linked_sheet_id,
    list_form_responses,
    parse_question_types,
)
from core.logger import get_logger
from core.report import render_pdf
from core.reports import representativeness_report
from core.sheets_api import SheetsApiError, fetch_all_grids
from core.weighting import (
    RID_COLUMN,
    Dimension,
    compute_weighting,
    cumulative_design_effect,
)
from ui.components.auth_widget import ensure_api_access
from ui.components.form_picker import render_form_picker

log = get_logger(__name__)

# Питання-кандидати у виміри — лише одиночний вибір (страта = один варіант).
SINGLE_CHOICE_TYPES = {"MULTIPLE_CHOICE"}

st.title("Зважування")
st.caption(
    "Постстратифікація: коригуємо перекоси вибірки за відомими вимірами та "
    "оцінюємо репрезентативність. R_ID — наскрізний номер відповіді (порядок "
    "надходження), присутній у кожному експорті."
)

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


@st.cache_data(ttl=300, show_spinner="Завантажую відповіді…")
def _cached_responses(form_id_: str, _creds_token: str) -> list[dict]:
    return list_form_responses(creds, form_id_)


@st.cache_data(ttl=300, show_spinner="Шукаю таблиці популяції у Sheet…")
def _cached_grids(sheet_id_: str, _creds_token: str) -> dict[str, list[list[str]]]:
    return fetch_all_grids(creds, sheet_id_)


try:
    structure = _cached_structure(form_id, creds.token or "")
    responses = _cached_responses(form_id, creds.token or "")
except FormsApiError as exc:
    log.exception("ui_weighting_load_failed", extra={"form_id": form_id})
    st.error(f"Не вдалося завантажити форму: {exc}")
    st.stop()

st.caption(f"Форма: **{structure.get('info', {}).get('title', '—')}**")

# placeholder for BAN metrics (rendered after weighting is computed)
metrics_placeholder = st.container()

if not responses:
    st.info("Зважування зʼявиться після перших відповідей форми.")
    st.stop()


def _build_frame(responses_: list[dict], qids: list[str]) -> pd.DataFrame:
    """Кадр респондентів у порядку API: R_ID + значення обраних питань.

    R_ID присвоюється за порядком повернення API (submission order) — БЕЗ
    ресорту за createTime (за високого навантаження timestamp'и збігаються).
    """
    rows: list[dict] = []
    for i, r in enumerate(responses_, start=1):
        row: dict[str, object] = {RID_COLUMN: i, "createTime": r.get("createTime", "")}
        answers = r.get("answers", {})
        for qid in qids:
            vals = answers.get(qid, {}).get("textAnswers", {}).get("answers", [])
            row[qid] = vals[0].get("value", "") if vals else ""
        rows.append(row)
    return pd.DataFrame(rows)


# --- питання-кандидати (одиночний вибір) ------------------------------------
questions = [q for q in parse_question_types(structure) if q.type in SINGLE_CHOICE_TYPES]
if not questions:
    st.info(
        "У формі немає питань з одиночним вибором — нема за чим стратифікувати. "
        "Додайте питання типу «один варіант» (підрозділ, курс, стать тощо)."
    )
    st.stop()

frame = _build_frame(responses, [q.id for q in questions])

# --- авто-детект таблиць популяції у привʼязаному Sheet ----------------------
sheet_id = get_linked_sheet_id(structure)
auto_tables: list[ContextTable] = []
if sheet_id:
    try:
        grids = _cached_grids(sheet_id, creds.token or "")
        auto_tables = scan_sheets_for_tables(grids)
    except SheetsApiError as exc:
        log.warning("ui_weighting_sheet_scan_failed", extra={"sheet_id": sheet_id})
        st.warning(f"Не вдалося просканувати Sheet (зважування лише з CSV): {exc}")

tab_settings, tab_table, tab_timeline, tab_deff, tab_export = st.tabs(
    [
        "⚙️ Налаштування",
        "📋 Таблиця ваг",
        "📈 Ваги",
        "📈 Дизайн ефект",
        "⬇️ Експорт",
    ]
)

with tab_settings:
    st.subheader("1. Виміри стратифікації та популяція")
    st.caption(
        "Для кожного виміру потрібна таблиця популяції (страта → абсолютна "
        "кількість). Знаходимо її автоматично у Sheet або імпортуйте CSV. "
        "Один файл = один вимір."
    )

    # Значення-страти кожного питання (як у відповідях) — для зіставлення.
    option_sets: dict[str, list[str]] = {
        q.id: list(dict.fromkeys(v for v in frame[q.id].astype(str).tolist() if v.strip()))
        for q in questions
    }
    option_sets = {qid: opts for qid, opts in option_sets.items() if opts}
    # Ексклюзивне зіставлення: одна таблиця → одне питання (без хибних збігів
    # спільних lookup-аркушів з купою малопотужних питань).
    assigned = assign_tables_to_questions(option_sets, auto_tables) if auto_tables else {}

    # --- конфігурація вимірів ----------------------------------------------------
    dimensions: list[Dimension] = []
    for q in questions:
        option_values = option_sets.get(q.id)
        if not option_values:
            continue

        auto = assigned.get(q.id)
        with st.expander(
            f"{q.title or q.id}  ·  {len(option_values)} страт"
            + (f"  ·  авто: {auto.table.source}" if auto else "  ·  потрібен CSV"),
            expanded=auto is not None,
        ):
            population: dict[str, int] | None = None
            source_label = ""

            if auto is not None:
                st.success(
                    f"Знайдено у «{auto.table.source}»: покрито "
                    f"{auto.matched}/{len(option_values)} страт "
                    f"({auto.coverage * 100:.0f}%)."
                )
                if auto.unmatched_options:
                    st.caption("Без популяції: " + ", ".join(auto.unmatched_options[:10]))
                population = dict(auto.population)
                source_label = auto.table.source

            upload = st.file_uploader(
                "Імпорт CSV популяції (страта, кількість) — замінює авто",
                type=["csv", "tsv", "txt"],
                key=f"pop_csv_{q.id}",
            )
            if upload is not None:
                try:
                    text = upload.getvalue().decode("utf-8-sig")
                    table = parse_population_csv(text, source=f"CSV: {upload.name}")
                    # Користувач свідомо обрав цей CSV для цього питання → беремо
                    # будь-яке перекриття без порогів (на відміну від авто-детекту).
                    matched = match_population(option_values, table)
                    if matched.matched == 0:
                        st.error(
                            "Жодна страта CSV не збіглася з варіантами питання — "
                            "перевірте, що перший стовпець містить назви варіантів."
                        )
                    else:
                        population = dict(matched.population)
                        source_label = table.source
                        st.success(f"CSV: покрито {matched.matched}/{len(option_values)} страт.")
                except (ValueError, UnicodeDecodeError) as exc:
                    st.error(f"Не вдалося прочитати CSV: {exc}")

            include = st.checkbox(
                "Включити цей вимір у зважування",
                value=population is not None,
                key=f"incl_{q.id}",
                disabled=population is None,
            )
            if include and population:
                dimensions.append(
                    Dimension(name=q.title or q.id, column=q.id, population=population)
                )
                st.caption(
                    f"N = {sum(population.values())} ({len(population)} страт) · {source_label}"
                )

    if not dimensions:
        st.info(
            "Додайте хоча б один вимір: увімкніть авто-знайдений або імпортуйте CSV "
            "популяції. Без популяції ваги порахувати неможливо."
        )
        st.stop()

    # --- опційний cap ваг (default off) -----------------------------------------
    with st.expander("Параметри (необовʼязково)"):
        cap_value = st.number_input(
            "Обрізати ваги зверху (cap), 0 = вимкнено",
            min_value=0.0,
            value=0.0,
            step=0.5,
            help=(
                "Великі ваги (рідкісні страти) роздувають DEFF. Cap обмежує вагу "
                "кожної страти зверху. За замовчуванням вимкнено."
            ),
        )
        moe_pct = st.number_input(
            "Цільова гранична похибка, %",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
        )

caps = None
if cap_value > 0:
    caps = {d.name: {s: cap_value for s in d.population} for d in dimensions}

res = compute_weighting(frame, dimensions, moe=moe_pct / 100.0, caps=caps)

with metrics_placeholder:
    row1 = st.columns(4)
    row1[0].metric(
        "Репрезентативність",
        f"{res.coverage_eff * 100:.0f}%",
        help=(
            "Частка цільового обсягу з урахуванням дизайну вибірки (DEFF). "
            "≥100% = ефективна вибірка покриває ціль."
        ),
    )
    row1[1].metric(
        "Без DEFF",
        f"{res.coverage_raw * 100:.0f}%",
        help="Сире покриття цілі без урахування нерівних ваг.",
    )
    row1[2].metric("DEFF (Кіш)", f"{res.deff:.2f}", help="1 + CV²(ваг). Втрата ефективності.")
    # Ефективний обсяг вибірки (n_eff)
    row1[3].metric(
        "Ефективний n",
        f"{res.n_eff:.0f}",
        help=f"Ефективний обсяг = відповіді / DEFF = {res.n} / {res.deff:.2f}",
    )

    row2 = st.columns(4)
    row2[0].metric("Відповідей (n)", res.n)
    row2[1].metric("Ціль", res.n_target, help=f"Розрахунок SRS+FPC для MoE, N={res.population}")
    row2[2].metric("MoE", f"{res.moe * 100:.1f}%", help="Гранична похибка частки (p=0.5, без FPC).")
    row2[3].metric(
        "MoE з DEFF",
        f"{res.moe_deff * 100:.1f}%",
        delta=f"+{(res.moe_deff - res.moe) * 100:.1f}%",
        delta_color="inverse",
        help="MoE · √DEFF — реальна похибка з урахуванням дизайну.",
    )

    lack = max(res.sample_need - res.n, 0.0)
    if lack > 0:
        st.warning(
            f"Бракує ще ~**{lack:.0f}** відповідей до цілі з урахуванням DEFF "
            f"(потрібно n_target·DEFF = {res.sample_need:.0f})."
        )
    else:
        st.success(
            f"Ціль досягнуто: зібрано {res.n} ≥ потрібних {res.sample_need:.0f} (n_target·DEFF)."
        )

with tab_table:
    st.caption(
        "Сортовано за «Ще треба» (недопредставленість) спадання: зверху страти, "
        "яким найбільше бракує відповідей. Покриття <1 = недобір, >1 = надлишок."
    )
    strata_df = res.strata_frame().sort_values("Ще треба", ascending=False, ignore_index=True)
    # Rename numeric weight column to use subscript h (Unicode) for visual subscripts
    if "Вага w_h" in strata_df.columns:
        strata_df = strata_df.rename(columns={"Вага w_h": "Вага wₕ"})

    st.dataframe(
        strata_df,
        hide_index=True,
        width="stretch",
        column_config={
            "Покриття": st.column_config.NumberColumn(format="%.2f"),
            "Вага wₕ": st.column_config.NumberColumn(format="%.3f"),
        },
    )

with tab_timeline:
    tl_weight_cols = ["w_timeline"] + [f"w_{d.name}_timeline" for d in dimensions]
    tl_labels = {"w_timeline": "Композит (w)"} | {
        f"w_{d.name}_timeline": d.name for d in dimensions
    }
    chosen = st.multiselect(
        "Які ваги показати на таймлайні",
        options=tl_weight_cols,
        default=["w_timeline"],
        format_func=lambda c: tl_labels.get(c, c),
    )
    if chosen:
        tl = res.frame[[RID_COLUMN, *chosen]].rename(columns=tl_labels).set_index(RID_COLUMN)
        st.caption("Вага кожного респондента на момент його надходження (центровано навколо 1).")
        st.line_chart(tl)

    # (DEFF chart moved to its own tab)

with tab_deff:
    deff_data = {"Композит": cumulative_design_effect(res.frame["w"].tolist())}
    for d in dimensions:
        deff_data[d.name] = cumulative_design_effect(res.frame[f"w_{d.name}"].tolist())
    deff_df = pd.DataFrame(deff_data)
    deff_df.index = res.frame[RID_COLUMN]
    st.caption("DEFF наростаючим підсумком — як дизайн-ефект змінювався з надходженням відповідей.")

    # Інтерактивний фільтр: дозволяє показувати тільки вибрані серії DEFF
    series = list(deff_df.columns)
    chosen = st.multiselect("Показати серії (фільтр)", options=series, default=series)
    if not chosen:
        st.info("Виберіть принаймні одну серію для відображення.")
    else:
        fold_cols = chosen
        deff_reset = deff_df.reset_index()[[RID_COLUMN] + chosen]
        # визначаємо мін/макс для домену та додаємо невеликий паддінг
        ymin = float(deff_reset[chosen].min().min())
        ymax = float(deff_reset[chosen].max().max())
        if ymin == ymax:
            ymin -= 0.5
            ymax += 0.5
        else:
            pad = (ymax - ymin) * 0.02
            ymin -= pad
            ymax += pad

        chart = (
            alt.Chart(deff_reset)
            .transform_fold(fold_cols, as_=["metric", "value"])
            .mark_line()
            .encode(
                x=alt.X(RID_COLUMN, title="R_ID"),
                y=alt.Y("value:Q", title="DEFF", scale=alt.Scale(domain=[ymin, ymax], zero=False)),
                color=alt.Color(
                    "metric:N",
                    title="Метрика",
                    legend=alt.Legend(orient="bottom", direction="horizontal"),
                ),
                tooltip=[RID_COLUMN, "metric:N", alt.Tooltip("value:Q", format=".3f")],
            )
        )
        st.altair_chart(chart, width="stretch")

with tab_export:
    csv_bytes = res.frame.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        ":material/download: Завантажити зважений кадр (CSV)",
        data=csv_bytes,
        file_name=f"weighting_{form_id}.csv",
        mime="text/csv",
        help=(
            "Повний кадр: R_ID, значення страт, ваги кожного виміру (статичні та "
            "таймлайн), композитна вага w і w_timeline. R_ID — у кожному рядку."
        ),
    )

    form_title = structure.get("info", {}).get("title", "")
    pdf_bytes = render_pdf(representativeness_report(res, form_title))
    st.download_button(
        ":material/picture_as_pdf: Завантажити звіт про репрезентативність (PDF)",
        data=pdf_bytes,
        file_name=f"representativeness_{form_id}.pdf",
        mime="application/pdf",
        help="PDF-звіт: показники репрезентативності та таблиця ваг (за недопредставленістю).",
    )
