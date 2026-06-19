from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_navigation_order_matches_analysis_workflow() -> None:
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    titles = ["Каталог", "Дизайн форми", "Динаміка", "Зважування", "Запитання", "Звіт"]
    positions = [source.index(f'title="{title}"') for title in titles]
    assert positions == sorted(positions)


def test_heavy_analysis_pages_do_not_use_streamlit_tabs() -> None:
    form_design = (ROOT / "ui/pages/form_design.py").read_text(encoding="utf-8")
    questions = (ROOT / "ui/pages/questions.py").read_text(encoding="utf-8")
    weighting = (ROOT / "ui/pages/weighting.py").read_text(encoding="utf-8")
    mode_switch = (ROOT / "ui/components/mode_switch.py").read_text(encoding="utf-8")
    assert "st.tabs(" not in form_design
    assert "st.tabs(" not in questions
    assert "st.tabs(" not in weighting
    assert "st.pills(" not in form_design
    assert "st.pills(" not in questions
    assert "st.pills(" not in weighting
    assert "render_mode_switch(" in questions
    assert "render_mode_switch(" in weighting
    assert "st.segmented_control(" in mode_switch


def test_form_design_is_separate_page_from_questions() -> None:
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    form_design = (ROOT / "ui/pages/form_design.py").read_text(encoding="utf-8")
    questions = (ROOT / "ui/pages/questions.py").read_text(encoding="utf-8")
    assert '"ui/pages/form_design.py"' in app
    assert 'title="Дизайн форми"' in app
    assert 'render_page_header("Дизайн форми")' in form_design
    assert "analyze_form_design" in form_design
    assert "Дизайн форми" not in questions
    assert 'QUESTION_MODES = ["Відповіді", "Крос-таби"]' in questions


def test_form_design_page_renders_flow_map() -> None:
    form_design = (ROOT / "ui/pages/form_design.py").read_text(encoding="utf-8")
    assert "parse_form_flow" in form_design
    assert "flow_to_dot" in form_design
    assert 'st.subheader("Карта переходів")' in form_design
    assert "st.graphviz_chart(" in form_design
    assert 'st.expander("Debug карти переходів (тимчасово)"' in form_design
    assert "raw structure.items" in form_design
    assert 'MetricItem("Секцій"' in form_design
    assert 'MetricItem("Умовних переходів"' in form_design
    assert 'MetricItem("Недосяжних"' in form_design


def test_weighting_settings_are_lazy_and_persisted() -> None:
    weighting = (ROOT / "ui/pages/weighting.py").read_text(encoding="utf-8")
    assert 'if mode == "Налаштування":' in weighting
    assert "weighting_config_" in weighting
    assert "stored_config = st.session_state.get(_config_key)" in weighting


def test_catalog_table_selects_global_form_without_action_columns() -> None:
    catalog = (ROOT / "ui/pages/catalog.py").read_text(encoding="utf-8")
    assert "st.column_config.LinkColumn" not in catalog
    assert '"FormID": f.id' in catalog
    assert 'on_select="rerun"' in catalog
    assert 'selection_mode="single-row"' in catalog
    assert "st.session_state[FORM_KEY] = selected_form_id" in catalog


def test_catalog_exposes_publication_status_metrics_and_filter() -> None:
    catalog = (ROOT / "ui/pages/catalog.py").read_text(encoding="utf-8")
    assert 'STATUS_OPEN = "Відкриті"' in catalog
    assert 'STATUS_CLOSED = "Закриті"' in catalog
    assert 'STATUS_UNPUBLISHED = "Неопубліковані"' in catalog
    assert 'STATUS_UNKNOWN = "Невідомо"' in catalog
    assert 'key="catalog_publication_status"' in catalog
    assert 'MetricItem("Відкритих"' in catalog
    assert 'MetricItem("Закритих"' in catalog
    assert 'MetricItem("Неопублікованих"' in catalog
    assert "ActionBarStatus(note=" not in catalog


def test_catalog_table_uses_dynamic_min_max_height() -> None:
    catalog = (ROOT / "ui/pages/catalog.py").read_text(encoding="utf-8")
    assert "TABLE_MIN_HEIGHT_PX = 360" in catalog
    assert "TABLE_MAX_HEIGHT_PX = 680" in catalog
    assert "def _table_height(row_count: int) -> int:" in catalog
    assert "height=_table_height(len(display))" in catalog


def test_form_label_lives_in_action_bar_not_page_caption() -> None:
    action_bar = (ROOT / "ui/components/action_bar.py").read_text(encoding="utf-8")
    assert "si-action-label" in action_bar
    assert "white-space: nowrap" in action_bar
    assert "[0.8, 8.25, 0.55, 0.55]" in action_bar
    assert "action_bar_refresh_" in action_bar
    assert "action_bar_open_" in action_bar
    assert "min-width: 2.5rem" in action_bar
    assert "margin-left: 0.25rem" in action_bar
    assert "actions_col.columns" not in action_bar

    for page in ["form_design.py", "dynamics.py", "questions.py", "weighting.py", "export.py"]:
        source = (ROOT / f"ui/pages/{page}").read_text(encoding="utf-8")
        assert "render_form_caption" not in source


def test_form_picker_keeps_global_selection_separate_from_widget_key() -> None:
    form_picker = (ROOT / "ui/components/form_picker.py").read_text(encoding="utf-8")
    action_bar = (ROOT / "ui/components/action_bar.py").read_text(encoding="utf-8")
    assert 'FORM_KEY = "global_form_id"' in form_picker
    assert 'FORM_WIDGET_PREFIX = "global_form_select"' in form_picker
    assert "def prepare_form_widget(" in form_picker
    assert "def sync_form_widget(" in form_picker
    assert "prepare_form_widget(refresh_scope, by_id)" in action_bar
    assert "on_change=sync_form_widget" in action_bar
    assert "key=widget_key" in action_bar
    assert "key=FORM_KEY" not in action_bar


def test_questions_association_overview_has_priority_filters() -> None:
    questions = (ROOT / "ui/pages/questions.py").read_text(encoding="utf-8")
    crosstab = (ROOT / "core/crosstab.py").read_text(encoding="utf-8")
    assert "ASSOCIATION_FILTER_MODES" in questions
    for mode in ["Важливі", "Значущі", "Сильні", "Ненадійні", "Усі"]:
        assert mode in crosstab
    assert 'key="association_overview_filter_mode"' in questions
    assert '"Статус": classify_association(pr)' in questions
    assert "filter_associations(" in questions
    assert "За поточними фільтрами зв'язків не знайдено" in questions


def test_question_response_charts_cap_left_axis_labels() -> None:
    questions = (ROOT / "ui/pages/questions.py").read_text(encoding="utf-8")
    assert "RESPONSE_AXIS_LABEL_LIMIT_PX = 320" in questions
    assert "def _wrap_axis_label(" in questions
    assert "def _response_axis(" in questions
    assert "labelExpr=\"split(datum.label, '\\\\n')\"" in questions
    assert "labelOverlap=False" in questions
    assert "axis=_response_axis()" in questions
    assert "labelLimit=0" not in questions


def test_report_page_uses_builder_layout() -> None:
    export = (ROOT / "ui/pages/export.py").read_text(encoding="utf-8")
    assert "REPORT_SECTION_DEFS" in export
    assert "st.container(border=True)" in export
    assert 'st.subheader("Секції звіту")' in export
    assert '"Preview"' not in export
    assert "preview_col" not in export
    assert 'if key == "descriptive" and section_state[key]:' in export
    assert "if inc_descriptive:" in export
    assert 'type="primary"' in export
    assert 'width="stretch"' in export
    assert "st.dataframe(" not in export
