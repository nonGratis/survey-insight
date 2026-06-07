# Survey Insight

[![CI](https://github.com/nonGratis/survey-insight/actions/workflows/ci.yml/badge.svg)](https://github.com/nonGratis/survey-insight/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

Хмарна інформаційно-аналітична система для обробки, статистично коректного аналізу та візуалізації результатів соціологічних опитувань в освітньому середовищі. Збирає дані безпосередньо з Google Forms, застосовує методи вибіркового обстеження (постстратифікаційне зважування, ефект дизайну, аналіз зв'язків між питаннями) та прогнозує динаміку надходження відповідей.

Бакалаврський дипломний проєкт, **КПІ ім. Ігоря Сікорського, ФІОТ**, спеціальність **123 «Комп'ютерна інженерія»**. Автор — Андрій Шаповалов (ІО-23), `shapovalov.andrii@edu.kpi.ua`.

> ✅ 195 модульних тестів · `ruff` + `mypy` clean · CI (ruff + mypy + pytest + docker build).

## Стек

Python 3.11 · Streamlit · pandas · NumPy · SciPy · statsmodels · ruptures · Plotly · Altair · Google Forms / Drive / Sheets API · Docker · Google Cloud Run (цільове середовище розгортання).

## Можливості

Система організована як багатосторінковий вебзастосунок:

- **Каталог** — перелік усіх Google Forms організації з метаданими (власник, кількість відповідей, статус збору) і прогресивним підвантаженням деталей у фоні.
- **Динаміка** — кумулятивний графік надходження відповідей + прогноз насичення поточної хвилі активності: детекція хвиль агітації алгоритмом **CUSUM** → апроксимація кривими насичення з вибором за критерієм **AICc** → довірчі інтервали (дельта-метод + конформне калібрування). Валідовано ≈15 % MAPE / ≈87 % покриття на реальних формах.
- **Питання** — лінтер якості формулювань (до збору), розподіли відповідей із сортуванням та анонімізацією відкритих варіантів, і **крос-таби**: таблиці спряженості та міри зв'язку між парами питань (**χ² + Cramér's V**, Spearman, Odds Ratio, Pearson) з поправкою на ваги (**Rao-Scott**) та на множинні порівняння (**Бенджаміні-Хохберг / FDR**).
- **Зважування** — постстратифікаційне зважування за довільними вимірами (підрозділ, курс, стать…): ваги страт, **ефект дизайну Кіша (DEFF)**, ефективний обсяг вибірки, гранична похибка (MoE та MoE·√DEFF), таблиця ваг за недопредставленістю та наскрізний ідентифікатор респондента **R_ID**. Автодетекція таблиць генеральної сукупності у прив'язаному Sheet + ручний CSV-імпорт.
- **Експорт** — вивантаження результатів і зважених масивів у **CSV** та **PDF-звіт**.

Дані надходять напряму через Forms API (прив'язаний Sheet потрібен лише для таблиць генеральної сукупності); архітектура без збереження стану (stateless), без локальних баз даних.

## Локальний запуск

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Відкриється на `http://localhost:8501`.

### Налаштування доступу (Google OAuth)

Для роботи з Google Forms/Sheets потрібні OAuth-облікові дані (client ID та secret) проєкту Google Cloud з увімкненими Forms, Drive і Sheets API, задані через змінні середовища. Порядок отримання — у [документації Google OAuth 2.0](https://developers.google.com/identity/protocols/oauth2). Демо працює в Testing-режимі (лише для доданих test users).

## Запуск у Docker

```powershell
docker build -t survey-insight:dev .
docker run --rm -p 8501:8501 survey-insight:dev
```

Цільове середовище промислового розгортання — Google Cloud Run (порт через `$PORT`, stateless-контейнер; запланований деплой).

## Структура

```
app.py              точка входу; реєстрація сторінок через st.navigation()
core/               бізнес-логіка та інтеграції (детерміновані функції без I/O)
  weighting.py        постстратифікація + DEFF Кіша
  crosstab.py         таблиці спряженості + міри зв'язку
  context_tables.py   авто-детект таблиць популяції + CSV-імпорт
  forms_quality.py    лінтер анкети + розподіли відповідей
  forecast/           детекція хвиль (CUSUM), моделі насичення, довірчі інтервали
  forms_api.py · sheets_api.py · auth.py · google_throttle.py   доступ до Google API
ui/                 Streamlit-шар
  pages/              catalog · analysis · questions · weighting · export
  components/         auth_widget · form_picker
tests/              pytest (195 тестів)
research/           дослідницькі бенчмарки та звіти
data/               робочі та синтетичні дані (великі файли .gitignore-нуті)
```

## Якість

```powershell
ruff check . ; ruff format --check . ; mypy core/ ; pytest -q
```

## Ліцензія

MIT — див. [LICENSE](LICENSE).
