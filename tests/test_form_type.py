"""Tests for core.forecast.form_type (keyword classifier)."""

from __future__ import annotations

import pytest

from core.forecast.form_type import classify_form_type


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Обираємо формат навчання", "survey"),
        ("ДА Стан організації процесу вибору", "survey"),
        ("Реєстрація на лекцію It-Jim", "event_registration"),
        ("Кіновечір Атлантида", "event_registration"),
        ("Набір до Студентської ради КПІ", "recruitment"),
        ("Реєстрація КПІшників-донорів", "volunteer_donor"),
        ("Донація 24.07", "volunteer_donor"),
        ("Таємний Санта", "holiday"),
        ("Форма для отримання Microsoft365", "service"),
        ("Реєстрація у чергу на поселення 2024", "service"),
        ("Ubisoft Фідбек 2025", "event_feedback"),
        ("Збір фото для Фотосушки 2024", "creative_submission"),
    ],
)
def test_classify_known_titles(title, expected):
    assert classify_form_type(title) == expected


def test_unmatched_title_is_other():
    assert classify_form_type("zzz qqq xyz 123") == "other"


def test_empty_title_is_other():
    assert classify_form_type("") == "other"


def test_optional_fields_contribute():
    # Keyword only in description → still classified.
    assert classify_form_type("Форма", description="збір крові донорів") == "volunteer_donor"


def test_priority_specific_over_survey():
    # "опитування" (survey) + "донор" (volunteer) → volunteer wins (higher priority).
    assert classify_form_type("Опитування донорів крові") == "volunteer_donor"
