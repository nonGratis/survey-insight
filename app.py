"""Survey Insight — точка входу Streamlit-додатку.

Використовує API st.navigation + st.Page (Streamlit >=1.36) для явного
оголошення сторінок. Бізнес-логіка живе у core/, UI — у ui/pages/.
Доступ до сторінок гейтується OAuth-логіном (Google identity).
"""

import streamlit as st

# Налаштовуємо logging першим, до будь-яких інших імпортів core/, щоб
# модулі одразу отримали сконфігурований root logger.
from core.logger import setup_logging

setup_logging()

from ui.components.auth_widget import (  # noqa: E402
    ensure_login_state,
    render_login_button,
    render_profile,
)

st.set_page_config(page_title="Survey Insight", layout="wide")

logged_in = ensure_login_state()

if not logged_in:
    hero_cols = st.columns([1, 2, 1], gap="large")
    with hero_cols[1]:
        st.subheader("Вхід")
        st.write("Увійди через Google, щоб отримати доступ до сторінок з аналізом форм.")
        render_login_button(location="main")

    st.divider()

    info_cols = st.columns(3, gap="large")
    with info_cols[0]:
        st.subheader("Про продукт")
        st.write(
            "Survey Insight об'єднує відповіді з Google Forms,"
            " робить зрозумілі зрізи та готує дані для експорту."
        )
    with info_cols[1]:
        st.subheader("FAQ")
        with st.expander("Як отримати доступ?"):
            st.write(
                "Увійди через Google. Після входу сторінки та функції системи стануть доступні."
            )
        with st.expander("Чому потрібні дозволи?"):
            st.write("Доступ потрібен, щоб читати та редагувати форми та таблиці, які ти обираєш.")
        with st.expander("Чи зберігаються дані?"):
            st.write("Дані використовуються лише для побудови звітів у цій сесії.")
    with info_cols[2]:
        st.subheader("Підтримка")
        st.write("Потрібна допомога або демо? Напиши на пошту: shapovalov.andrii@edu.kpi.ua")

    st.stop()

render_profile(location="sidebar")

pages = [
    st.Page(
        "ui/pages/catalog.py",
        title="Каталог",
        icon=":material/table_view:",
        url_path="catalog",
    ),
    st.Page(
        "ui/pages/dynamics.py",
        title="Динаміка",
        icon=":material/show_chart:",
        url_path="dynamics",
    ),
    st.Page(
        "ui/pages/questions.py",
        title="Запитання",
        icon=":material/quiz:",
        url_path="questions",
    ),
    st.Page(
        "ui/pages/weighting.py",
        title="Зважування",
        icon=":material/balance:",
        url_path="weighting",
    ),
    st.Page(
        "ui/pages/export.py",
        title="Експорт",
        icon=":material/download:",
        url_path="export",
    ),
]

nav = st.navigation(pages)
nav.run()
