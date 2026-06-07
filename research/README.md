# `research/` — академічна еволюція моделі прогнозу

Каталог для роботи, що **не входить** у production-pipeline, але обґрунтовує
архітектурні рішення в `core/forecast/`. Сюди йдуть:

- **`gas/`** — Google Apps Script для збору сирих timestamp'ів з усіх форм
  у master-spreadsheet. Джерело даних для бенчмарків.
- **`data/`** — сирі CSV з GAS-збору. **У git не комітимо** (PII-чутливі form_id).
  Структура: `data/raw/Form_Timestamp_Collection_<YYYYMMDD>.csv`.
- **`benchmarks/`** *(планується)* — rolling-origin CV, CI calibration,
  empirical model selection. Кожен бенчмарк — окремий Python-скрипт + markdown report.
- **`reports/`** *(планується)* — фінальні markdown-репорти з графіками,
  які потім цитуються у тексті диплому.

## Workflow

1. **Збір даних**: GAS у [`gas/master_collector.gs`](gas/master_collector.gs)
   запускається трігером 4h або вручну → master Google Sheet з аркушами
   `Timestamps`, `Forms`, `Errors`.
2. **Експорт**: завантажити `Timestamps` як CSV у `data/raw/`.
3. **Бенчмарки** (TBD): скрипти у `benchmarks/` читають CSV → продукують
   markdown-репорти у `reports/`.

## Принципи

- Production-код у `core/`, `ui/` залишається чистим; жодного research-імпорту
  з main pipeline.
- Кожен скрипт reproducible: hardcoded seed, документована версія датасету.
- Кожен report цитує: метрику, дату, hash датасету, обраний commit `core/forecast/`.

## Не комітимо

- `data/raw/*.csv` — містить ідентифікатори чужих Google Forms.
- Згенеровані графіки/notebook outputs.
