"""Survey Insight — точка входу Streamlit-додатку.

Використовує API st.navigation + st.Page (Streamlit >=1.36) для явного
оголошення сторінок. Бізнес-логіка живе у core/, UI — у ui/pages/.
Доступ до сторінок гейтується OAuth-логіном (Google identity).
"""
import streamlit as st

from ui.components.auth_widget import render_login

st.set_page_config(page_title="Survey Insight", layout="wide")

logged_in = render_login()

if not logged_in:
    st.title("Survey Insight")
    st.info(
        "Увійди через Google у бічній панелі ліворуч, щоб отримати доступ "
        "до аналізу, зважування і експорту."
    )
    st.stop()

pages = [
    st.Page("ui/pages/analyze.py", title="Аналіз", icon=":material/analytics:"),
    st.Page("ui/pages/weighting.py", title="Зважування", icon=":material/balance:"),
    st.Page("ui/pages/export.py", title="Експорт", icon=":material/download:"),
]

nav = st.navigation(pages)
nav.run()
