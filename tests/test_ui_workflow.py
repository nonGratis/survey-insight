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
    assert "st.tabs(" not in questions
    assert "st.tabs(" not in weighting


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
