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
