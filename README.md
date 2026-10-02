# Survey Insight

[![CI](https://github.com/nonGratis/survey-insight/actions/workflows/ci.yml/badge.svg)](https://github.com/nonGratis/survey-insight/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

Хмарна інформаційно-аналітична система для обробки, статистично коректного аналізу та візуалізації результатів соціологічних опитувань в освітньому середовищі. Збирає дані безпосередньо з Google Forms, застосовує методи вибіркового обстеження (постстратифікаційне зважування, ефект дизайну, аналіз зв'язків між питаннями) та прогнозує динаміку надходження відповідей.

Бакалаврський дипломний проєкт, **КПІ ім. Ігоря Сікорського, ФІОТ**, спеціальність **123 «Комп'ютерна інженерія»**. Автор — Андрій Шаповалов (ІО-23), `shapovalov.andrii@edu.kpi.ua`.

> ✅ Модульні тести (pytest), `ruff` і `mypy` запускаються в CI на кожен PR разом зі збіркою Docker-образу.

## Стек

Python 3.11 · Streamlit (web) · FastAPI (API, worker) · pandas · NumPy · SciPy · ruptures · Plotly · Altair · ReportLab · Google Forms / Drive / Sheets API · Firestore · Cloud KMS · Cloud Tasks · Cloud Storage · Docker · Google Cloud Run.

## Можливості

Вебзастосунок із шести сторінок:

- **Каталог** — перелік усіх Google Forms організації з метаданими (власник, кількість відповідей, статус збору) і прогресивним підвантаженням деталей у фоні.
- **Дизайн форми** — перевірка анкети ще до збору відповідей: таблиця питань із прапорами якості та карта переходів між секціями.
- **Динаміка** — кумулятивний графік надходження відповідей + прогноз насичення поточної хвилі активності: детекція хвиль агітації алгоритмом **CUSUM** → апроксимація кривими насичення з вибором за критерієм **AICc** → довірчі інтервали (дельта-метод + конформне калібрування). Валідовано ≈15 % MAPE / ≈87 % покриття на реальних формах.
- **Зважування** — постстратифікаційне зважування за довільними вимірами (підрозділ, курс, стать…): ваги страт, **ефект дизайну Кіша (DEFF)**, ефективний обсяг вибірки, гранична похибка (MoE та MoE·√DEFF), таблиця ваг за недопредставленістю та наскрізний ідентифікатор респондента **R_ID**. Автодетекція таблиць генеральної сукупності у прив'язаному Sheet + ручний CSV-імпорт.
- **Запитання** — розподіли відповідей із сортуванням та анонімізацією відкритих варіантів і **крос-таби**: таблиці спряженості та міри зв'язку між парами питань (**χ² + Cramér's V**, Spearman, Odds Ratio, Pearson) з поправкою на ваги (**Rao-Scott**) та на множинні порівняння (**Бенджаміні-Хохберг / FDR**).
- **Звіт** — зведений **PDF-звіт** за обраною формою: огляд анкети, дескриптивна статистика, репрезентативність, зв'язки між питаннями, динаміка. Розділи й формат налаштовуються.

Зважений масив і прогноз вивантажуються в **CSV** зі сторінок «Зважування» та «Динаміка».

## Архітектура

Один Docker-образ запускається як три сервіси Cloud Run; роль задає змінна `SERVICE`:

- **web** — інтерфейс на Streamlit. У розгорнутому застосунку до Google не звертається і токенів не бачить: усі дані отримує через API.
- **api** — FastAPI: вхід через Google OAuth, сесії, читання Google Forms / Drive / Sheets від імені користувача.
- **worker** — обробник фонових задач звітів через Cloud Tasks. Поки що каркас: PDF формує web.

Дані й безпека:

- токени Google зберігаються лише на боці API, у Firestore, зашифровані ключем Cloud KMS;
- сирі відповіді респондентів не пишуться ні в базу, ні у сховище, ні в логи: вони живуть лише в пам'яті сервісу (прив'язаний Sheet потрібен тільки для таблиць генеральної сукупності);
- у логах користувача позначає лише хеш ідентифікатора.

Інструкція розгортання — у [deploy/cloud-run/README.md](deploy/cloud-run/README.md).

## Локальний запуск

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
streamlit run app.py
```

Відкриється на `http://localhost:8501`.

Без `APP_ENV=production` застосунок працює в локальному demo-режимі: Streamlit сам входить у Google і читає форми напряму, без сервісу API. Для цього потрібен OAuth-клієнт проєкту Google Cloud з увімкненими Forms, Drive і Sheets API та redirect URI `http://localhost:8501`; його JSON кладуть у `config/credentials.json` (тека в git не потрапляє). Порядок отримання — у [документації Google OAuth 2.0](https://developers.google.com/identity/protocols/oauth2). Demo працює в Testing-режимі, лише для доданих test users.

API і worker локально запускаються так само, як у контейнері, і без `APP_ENV=production` тримають стан у пам'яті:

```powershell
python -m uvicorn api.main:app --port 8080
```

Щоб увійти через такий локальний API, йому потрібен OAuth-клієнт у змінній `GOOGLE_OAUTH_CLIENT_CONFIG_JSON`; повний перелік змінних — у [deploy/cloud-run/env.example](deploy/cloud-run/env.example).

Точні версії всіх залежностей зафіксовано в `constraints.txt`: pip застосовує його автоматично (рядок `-c constraints.txt` у `requirements.txt`), тож CI, Docker і локальне середовище збирають однаковий набір. `requirements.txt` — лише те, що потрібно для роботи застосунку; `requirements-dev.txt` додає pytest, ruff і mypy. Оновлення версій щотижня пропонує Dependabot окремими PR, які мають пройти CI.

## Запуск у Docker

```powershell
docker build -t survey-insight:dev .
docker run --rm -p 8501:8080 survey-insight:dev
```

Контейнер слухає порт `$PORT` (типово 8080); команда вище відкриває web на `http://localhost:8501`. `-e SERVICE=api` або `-e SERVICE=worker` запускає відповідний сервіс замість web. Для входу в demo-режимі змонтуйте теку з OAuth-клієнтом: `-v ${PWD}\config:/app/config`.

## Структура

```
app.py              точка входу web: реєстрація сторінок через st.navigation()
core/               бізнес-логіка і статистика (без імпорту streamlit)
  weighting.py        постстратифікація + DEFF Кіша
  crosstab.py         таблиці спряженості + міри зв'язку
  context_tables.py   авто-детект таблиць популяції + CSV-імпорт
  forms_quality.py    лінтер анкети + розподіли відповідей
  form_flow.py        граф переходів між секціями форми
  forecast/           детекція хвиль (CUSUM), моделі насичення, довірчі інтервали
  report.py · reports.py   PDF-звіт: рендер і секції
  forms_api.py · forms_catalog.py · sheets_api.py   читання Google Forms / Drive / Sheets
  auth.py             вхід у Google для локального demo-режиму
  saas/               сесії, OAuth, зашифровані токени; адаптери Firestore, KMS, GCS, Cloud Tasks
api/                FastAPI: OAuth, сесії, доступ до Google Forms / Drive / Sheets
worker/             фонові задачі звітів (Cloud Tasks)
ui/                 Streamlit-шар; до API ходить лише через saas_api.py
  pages/              catalog · form_design · dynamics · weighting · questions · export
  components/         auth_widget · form_picker · action_bar · metric_bar · mode_switch · page_shell
tests/              pytest
deploy/cloud-run/   інструкція розгортання
assets/fonts/       шрифти Liberation Sans для PDF
```

## Якість

```powershell
ruff check . ; ruff format --check . ; mypy core/ ; pytest tests/ -q
```

## Ліцензія

MIT — див. [LICENSE](LICENSE).
