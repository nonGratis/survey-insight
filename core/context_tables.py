"""Контекстні таблиці популяції: авто-детект у Sheet + ручний CSV-імпорт.

Контекстна таблиця = два стовпці: {назва страти → абсолютна кількість N_h}.
Один файл/таблиця = один вимір стратифікації (підрозділ, курс, стать…).

Два джерела, однаковий результат (`ContextTable`):
- **авто-детект**: скан значень аркушів привʼязаного Google Sheet — шукаємо
  суміжну пару стовпців, де лівий = текстові мітки, правий = цілі ≥ 0;
- **ручний імпорт**: CSV/TSV (мітка, кількість) — той самий парсер.

Знайдені таблиці зіставляються з варіантами питання форми (`match_population`):
обираємо таблицю, чиї мітки найкраще покривають опції питання. Зіставлення
нечутливе до регістру/пробілів, але популяція повертається з ключами рівно
як у відповідях (щоб коректно зʼєднатись по значенню).

Чисті функції без I/O — приймають уже прочитані матриці значень / текст.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# --- іменовані пороги детекції (не «магічні») -------------------------------
MIN_TABLE_ROWS = 2  # мінімум рядків, щоб вважати пару стовпців таблицею
MIN_OPTION_COVERAGE = 0.5  # частка опцій питання, яку має покрити таблиця
MIN_MATCHED_STRATA = 2  # мінімум спільних страт для впевненого зіставлення
# Таблиця популяції має ОДИН рядок на страту. Якщо у відрізку багато
# повторів міток (унікальних ≪ рядків) — це не популяція, а стовпець даних
# (напр. аркуш відповідей: 7433 рядки → 23 унікальних факультети). Відкидаємо.
MIN_UNIQUE_RATIO = 0.9


@dataclass(frozen=True)
class ContextTable:
    """Розпізнана таблиця популяції одного виміру."""

    source: str  # де знайдено: назва аркуша або "CSV: <файл>"
    label_header: str  # заголовок стовпця міток (або "")
    count_header: str  # заголовок стовпця кількостей (або "")
    population: dict[str, int]  # страта → N_h (порядок як у джерелі)

    @property
    def total(self) -> int:
        return sum(self.population.values())

    @property
    def n_strata(self) -> int:
        return len(self.population)


def _norm(s: str) -> str:
    """Нормалізувати мітку для зіставлення: trim + casefold + стиснення пробілів."""
    return " ".join(str(s).split()).casefold()


def _as_count(value: str) -> int | None:
    """Спробувати прочитати клітинку як невідʼємне ціле (N_h).

    Приймає "46", "1 234", "1 234", "46,0"/"46.0" (ціле зі сміттям-дробом).
    Відхиляє відсотки, відʼємні, текст. Повертає None, якщо не кількість.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or "%" in raw:
        return None
    cleaned = raw.replace(" ", "").replace(" ", "").replace(",", ".")
    try:
        num = float(cleaned)
    except ValueError:
        return None
    if num < 0 or not math.isfinite(num):
        return None
    if abs(num - round(num)) > 1e-9:  # популяція — ЦІЛІ числа осіб
        return None
    return int(round(num))


def _column(grid: Sequence[Sequence[str]], col: int) -> list[str]:
    return [(row[col] if col < len(row) else "") for row in grid]


def scan_grid_for_tables(grid: Sequence[Sequence[str]], source: str) -> list[ContextTable]:
    """Знайти всі 2-стовпцеві таблиці популяції у матриці значень аркуша.

    Алгоритм: для кожної суміжної пари стовпців (c, c+1) йдемо рядками і
    збираємо максимальні відрізки, де ліва клітинка — непорожній текст, а
    права — ціле ≥ 0 (`_as_count`). Відрізок довжиною ≥ MIN_TABLE_ROWS стає
    кандидатом; заголовки беремо з рядка над відрізком (якщо права клітинка
    там НЕ число — це шапка). Дублікати міток у межах таблиці відкидаємо
    (лишаємо перше входження).

    Returns:
        Список ContextTable, відсортований за к-стю страт (спадання) — більші
        таблиці зазвичай інформативніші.
    """
    if not grid:
        return []
    width = max((len(r) for r in grid), default=0)
    tables: list[ContextTable] = []

    for c in range(width - 1):
        labels = _column(grid, c)
        counts = _column(grid, c + 1)
        run_start: int | None = None
        for r in range(len(grid) + 1):
            is_row = (
                r < len(grid) and str(labels[r]).strip() != "" and _as_count(counts[r]) is not None
            )
            if is_row and run_start is None:
                run_start = r
            elif not is_row and run_start is not None:
                table = _build_table(labels, counts, run_start, r, source)
                if table is not None:
                    tables.append(table)
                run_start = None

    tables.sort(key=lambda t: t.n_strata, reverse=True)
    return tables


def _build_table(
    labels: Sequence[str],
    counts: Sequence[str],
    start: int,
    end: int,
    source: str,
) -> ContextTable | None:
    """Зібрати ContextTable з відрізка [start, end) пари стовпців.

    Відрізок з масовими повторами міток (унікальних/рядків < MIN_UNIQUE_RATIO)
    відкидаємо — це стовпець даних (аркуш відповідей), а не таблиця популяції.
    """
    n_rows = end - start
    if n_rows < MIN_TABLE_ROWS:
        return None
    population: dict[str, int] = {}
    for r in range(start, end):
        key = str(labels[r]).strip()
        val = _as_count(counts[r])
        if key and val is not None and key not in population:
            population[key] = val
    if len(population) < MIN_TABLE_ROWS:
        return None
    if len(population) / n_rows < MIN_UNIQUE_RATIO:
        return None  # забагато дублікатів страт → не популяція
    # Заголовок — рядок над відрізком, якщо його права клітинка не число.
    label_header = count_header = ""
    if start > 0 and _as_count(counts[start - 1]) is None:
        label_header = str(labels[start - 1]).strip()
        count_header = str(counts[start - 1]).strip()
    return ContextTable(
        source=source,
        label_header=label_header,
        count_header=count_header,
        population=population,
    )


def scan_sheets_for_tables(grids: dict[str, Sequence[Sequence[str]]]) -> list[ContextTable]:
    """Сканувати кілька аркушів (назва → матриця значень) на таблиці популяції."""
    found: list[ContextTable] = []
    for tab_name, grid in grids.items():
        found.extend(scan_grid_for_tables(grid, source=tab_name))
    return found


def parse_population_csv(text: str, *, source: str = "CSV") -> ContextTable:
    """Розпарсити ручний CSV/TSV у ContextTable (мітка, кількість).

    Авто-визначення роздільника (`,`/`;`/`\\t`). Перший рядок вважаємо шапкою,
    якщо його друга клітинка не число. Рядки без валідної кількості — пропуск.

    Raises:
        ValueError: якщо не знайдено жодної валідної пари (мітка, кількість).
    """
    sample = text[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ";" if sample.count(";") >= sample.count(",") else ","

    rows = [
        r for r in csv.reader(io.StringIO(text), delimiter=delimiter) if any(c.strip() for c in r)
    ]
    if not rows:
        raise ValueError("CSV порожній — немає рядків.")

    label_header = count_header = ""
    start = 0
    if len(rows[0]) >= 2 and _as_count(rows[0][1]) is None:
        label_header, count_header = rows[0][0].strip(), rows[0][1].strip()
        start = 1

    population: dict[str, int] = {}
    for row in rows[start:]:
        if len(row) < 2:
            continue
        key = row[0].strip()
        val = _as_count(row[1])
        if key and val is not None and key not in population:
            population[key] = val
    if len(population) < 1:
        raise ValueError("Не знайдено жодної пари (страта, кількість) у CSV.")
    return ContextTable(
        source=source,
        label_header=label_header,
        count_header=count_header,
        population=population,
    )


@dataclass(frozen=True)
class PopulationMatch:
    """Результат зіставлення таблиці з опціями питання."""

    table: ContextTable
    population: dict[str, int]  # ключі = значення опцій питання (як у відповідях)
    matched: int  # к-сть спільних страт
    coverage: float  # matched / к-сть опцій питання
    unmatched_options: list[str]  # опції без популяції
    extra_strata: list[str]  # страти таблиці без опції


def match_population(option_values: Iterable[str], table: ContextTable) -> PopulationMatch:
    """Зіставити одну таблицю з варіантами питання (нечутливо до регістру).

    Популяцію перекладаємо на ключі = значення опцій питання (щоб join по
    відповіді працював точно). Рахуємо покриття опцій.
    """
    options = list(dict.fromkeys(str(o) for o in option_values))
    norm_to_count = {_norm(k): v for k, v in table.population.items()}
    population: dict[str, int] = {}
    unmatched: list[str] = []
    for opt in options:
        count = norm_to_count.get(_norm(opt))
        if count is not None:
            population[opt] = count
        else:
            unmatched.append(opt)
    matched_norms = {_norm(o) for o in population}
    extra = [k for k in table.population if _norm(k) not in matched_norms]
    coverage = len(population) / len(options) if options else 0.0
    return PopulationMatch(
        table=table,
        population=population,
        matched=len(population),
        coverage=coverage,
        unmatched_options=unmatched,
        extra_strata=extra,
    )


def best_match(
    option_values: Iterable[str], tables: Sequence[ContextTable]
) -> PopulationMatch | None:
    """Обрати таблицю, що найкраще покриває опції питання.

    Кандидат проходить, якщо покрив ≥ MIN_MATCHED_STRATA страт і ≥
    MIN_OPTION_COVERAGE опцій. Серед прохідних — максимум matched, далі
    coverage. None, якщо жодна не підходить (тоді — ручний імпорт).
    """
    options = list(option_values)
    best: PopulationMatch | None = None
    for table in tables:
        m = match_population(options, table)
        if m.matched < MIN_MATCHED_STRATA or m.coverage < MIN_OPTION_COVERAGE:
            continue
        if best is None or (m.matched, m.coverage) > (best.matched, best.coverage):
            best = m
    return best


def assign_tables_to_questions(
    option_sets: dict[str, Sequence[str]], tables: Sequence[ContextTable]
) -> dict[str, PopulationMatch]:
    """Ексклюзивно зіставити таблиці з питаннями: одна таблиця → одне питання.

    Без ексклюзивності спільний lookup-аркуш (напр. «scales») збігається з
    купою малопотужних питань (так/ні, Likert) — хибні виміри. Тут кожна
    таблиця дістається ЛИШЕ найсильнішому претенденту (greedy за matched,
    потім coverage), а кожне питання отримує не більше однієї таблиці.

    Args:
        option_sets: {question_id: значення-опції як у відповідях}.
        tables: знайдені контекстні таблиці.

    Returns:
        {question_id: PopulationMatch} лише для впевнено зіставлених питань.
    """
    candidates: list[tuple[int, float, str, int, PopulationMatch]] = []
    for qid, options in option_sets.items():
        for ti, table in enumerate(tables):
            m = match_population(options, table)
            if m.matched < MIN_MATCHED_STRATA or m.coverage < MIN_OPTION_COVERAGE:
                continue
            candidates.append((m.matched, m.coverage, qid, ti, m))

    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    assigned: dict[str, PopulationMatch] = {}
    used_tables: set[int] = set()
    for _matched, _cov, qid, ti, m in candidates:
        if qid in assigned or ti in used_tables:
            continue
        assigned[qid] = m
        used_tables.add(ti)
    return assigned
