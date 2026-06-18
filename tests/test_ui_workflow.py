from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_navigation_order_matches_analysis_workflow() -> None:
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    titles = ["Каталог", "Запитання", "Динаміка", "Зважування", "Звіт"]
    positions = [source.index(f'title="{title}"') for title in titles]
    assert positions == sorted(positions)


def test_heavy_analysis_pages_do_not_use_streamlit_tabs() -> None:
    questions = (ROOT / "ui/pages/questions.py").read_text(encoding="utf-8")
    weighting = (ROOT / "ui/pages/weighting.py").read_text(encoding="utf-8")
    mode_switch = (ROOT / "ui/components/mode_switch.py").read_text(encoding="utf-8")
    assert "st.tabs(" not in questions
    assert "st.tabs(" not in weighting
    assert "st.pills(" not in questions
    assert "st.pills(" not in weighting
    assert "render_mode_switch(" in questions
    assert "render_mode_switch(" in weighting
    assert "st.segmented_control(" in mode_switch


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
    assert 'st.markdown("**Форма:**")' in action_bar
    assert "[0.62, 9.38, 1.15]" in action_bar

    for page in ["dynamics.py", "questions.py", "weighting.py", "export.py"]:
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
