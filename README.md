# Survey Insight

Хмарна інформаційно-аналітична система для обробки та візуалізації результатів соціологічних досліджень в освітньому середовищі.

Бакалаврська робота, **КПІ ФІОТ**, спеціальність **123 Комп'ютерна інженерія**.

## Стек

Streamlit · Python 3.11 · pandas · Plotly · Google Sheets API · Docker · Google Cloud Run

## Ключові фічі

- Автодетекція типів питань.
- Постстратифікаційне зважування з DEFF Кіша.
- Метрика репрезентативності в реалтаймі.
- Прогноз кількості відповідей (логарифмічна модель + changepoint detection).
- Крос-табуляції, експорт у long format та PDF.
- Опціональна генерація синтетичних даних.

## Локальний запуск

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Відкриється на `http://localhost:8501`.

## Запуск у Docker

```powershell
docker build -t survey-insight:dev .
docker run --rm -p 8501:8501 survey-insight:dev
```

## Структура

```
core/    — бізнес-логіка (weighting, metrics, detection); чисті функції + класи стану
ui/      — Streamlit-шар (pages/)
tests/   — pytest
config/  — конфіг (yaml), пізніше
data/    — синтетичні та робочі дані; великі файли .gitignore-нуті
```

## Ліцензія

MIT — див. [LICENSE](LICENSE).
