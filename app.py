"""Survey Insight — точка входу Streamlit-додатку.

Використовує API st.navigation + st.Page (Streamlit >=1.36) для явного
оголошення сторінок. Бізнес-логіка живе у core/, UI — у ui/pages/.
"""
import streamlit as st

st.set_page_config(page_title="Survey Insight", layout="wide")

pages = [
    st.Page("ui/pages/analyze.py", title="Аналіз", icon=":material/analytics:"),
    st.Page("ui/pages/weighting.py", title="Зважування", icon=":material/balance:"),
    st.Page("ui/pages/export.py", title="Експорт", icon=":material/download:"),
]

nav = st.navigation(pages)
nav.run()
