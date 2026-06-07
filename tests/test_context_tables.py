"""Tests for core.context_tables — auto-detect + CSV import + matching."""

from __future__ import annotations

import pytest

from core.context_tables import (
    ContextTable,
    assign_tables_to_questions,
    best_match,
    match_population,
    parse_population_csv,
    scan_grid_for_tables,
    scan_sheets_for_tables,
)

# Реалістична матриця аркуша: ліворуч таблиця популяції, праворуч сміття.
GRID = [
    ["Підрозділ", "Кількість", "", "примітка"],
    ["ФІОТ", "443", "", "x"],
    ["ФЕА", "104", "", ""],
    ["ПБФ", "201", "", "коментар"],
    ["", "", "", ""],
    ["Разом", "748", "", ""],  # "Разом" теж пара (текст, число) — окремий run
]


def test_scan_finds_population_table():
    tables = scan_grid_for_tables(GRID, source="Аркуш1")
    assert tables, "має знайти принаймні одну таблицю"
    top = tables[0]
    assert top.population == {"ФІОТ": 443, "ФЕА": 104, "ПБФ": 201}
    assert top.label_header == "Підрозділ"
    assert top.count_header == "Кількість"
    assert top.source == "Аркуш1"
    assert top.total == 748


def test_scan_ignores_percent_and_text_counts():
    grid = [
        ["A", "50%"],  # відсоток — не кількість
        ["B", "abc"],  # текст — не кількість
        ["C", "10"],
        ["D", "20"],
    ]
    tables = scan_grid_for_tables(grid, source="s")
    assert tables[0].population == {"C": 10, "D": 20}


def test_scan_rejects_too_short_run():
    grid = [["only", "5"]]  # 1 рядок < MIN_TABLE_ROWS
    assert scan_grid_for_tables(grid, source="s") == []


def test_scan_rejects_response_grid_with_duplicate_labels():
    # Імітація аркуша відповідей: 100 рядків, лише 2 унікальні факультети +
    # суміжний числовий стовпець. Це НЕ популяція (масові дублікати міток).
    grid = [["ФІОТ" if i % 2 else "ФЕА", str(i)] for i in range(100)]
    assert scan_grid_for_tables(grid, source="responses") == []


def test_scan_keeps_one_row_per_stratum_table():
    grid = [["ФІОТ", "443"], ["ФЕА", "104"], ["ПБФ", "201"]]
    tables = scan_grid_for_tables(grid, source="dept")
    assert tables and tables[0].population == {"ФІОТ": 443, "ФЕА": 104, "ПБФ": 201}


def test_scan_empty_grid():
    assert scan_grid_for_tables([], source="s") == []


def test_scan_sheets_multiple_tabs():
    grids = {
        "dept": [["ФІОТ", "443"], ["ФЕА", "104"]],
        "course": [["1 курс", "73"], ["2 курс", "2038"]],
    }
    tables = scan_sheets_for_tables(grids)
    assert {t.source for t in tables} == {"dept", "course"}


def test_parse_csv_comma():
    t = parse_population_csv("strata,population\nФІОТ,443\nФЕА,104\n", source="CSV: dept")
    assert t.population == {"ФІОТ": 443, "ФЕА": 104}
    assert t.label_header == "strata"
    assert t.source == "CSV: dept"


def test_parse_csv_semicolon_no_header():
    t = parse_population_csv("ФІОТ;443\nФЕА;104")
    assert t.population == {"ФІОТ": 443, "ФЕА": 104}
    assert t.label_header == ""  # без шапки


def test_parse_csv_tab_and_spaces_in_numbers():
    t = parse_population_csv("ФІОТ\t1 443\nФЕА\t104")
    assert t.population == {"ФІОТ": 1443, "ФЕА": 104}


def test_parse_csv_empty_raises():
    with pytest.raises(ValueError):
        parse_population_csv("\n\n")


def test_match_population_keys_to_option_values():
    from core.context_tables import ContextTable

    table = ContextTable("s", "", "", {"фіот": 443, "ФЕА ": 104, "ПБФ": 201})
    # опції питання у «канонічному» написанні відповідей
    m = match_population(["ФІОТ", "ФЕА", "ХІМ"], table)
    assert m.population == {"ФІОТ": 443, "ФЕА": 104}  # case/space-insensitive
    assert m.matched == 2
    assert m.unmatched_options == ["ХІМ"]
    assert "ПБФ" in m.extra_strata


def test_best_match_picks_highest_coverage():
    from core.context_tables import ContextTable

    good = ContextTable("good", "", "", {"A": 10, "B": 20, "C": 30})
    poor = ContextTable("poor", "", "", {"A": 10, "X": 5})
    m = best_match(["A", "B", "C"], [poor, good])
    assert m is not None and m.table.source == "good"


def test_best_match_none_when_below_threshold():
    table = ContextTable("s", "", "", {"X": 1, "Y": 2})  # 0 спільних з опціями
    assert best_match(["A", "B", "C"], [table]) is None


def test_assign_exclusive_one_table_per_question():
    dept = ContextTable("DEPARTMENT", "", "", {"ФІОТ": 443, "ФЕА": 104, "ПБФ": 201})
    course = ContextTable("COURSE", "", "", {"1 курс": 73, "2 курс": 2038})
    option_sets = {
        "q_dept": ["ФІОТ", "ФЕА", "ПБФ"],
        "q_course": ["1 курс", "2 курс"],
    }
    assigned = assign_tables_to_questions(option_sets, [dept, course])
    assert assigned["q_dept"].table.source == "DEPARTMENT"
    assert assigned["q_course"].table.source == "COURSE"


def test_assign_shared_lookup_goes_to_single_best_question():
    # «scales» збігається з двома питаннями; має дістатись лише сильнішому
    # (3 спільні страти > 2), друге лишається без таблиці.
    scales = ContextTable("scales", "", "", {"так": 1, "ні": 2, "не знаю": 3})
    option_sets = {
        "q_yesno": ["так", "ні"],  # 2 страти
        "q_three": ["так", "ні", "не знаю"],  # 3 страти — сильніший
    }
    assigned = assign_tables_to_questions(option_sets, [scales])
    assert "q_three" in assigned
    assert "q_yesno" not in assigned  # таблиця вже зайнята
