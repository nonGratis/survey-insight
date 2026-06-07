"""End-to-end тести оркестратора `forecast_responses`."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from core.forecast import ForecastError, forecast_responses
from core.timeline import build_timeline_from_timestamps


def _make_timeline(daily_counts: list[int], start: datetime = datetime(2025, 5, 1)):
    """Допоміжне: список денних приходів → TimelineSeries."""
    ts: list[datetime] = []
    for day_idx, count in enumerate(daily_counts):
        for _ in range(count):
            ts.append(start + timedelta(days=day_idx, hours=10))
    return build_timeline_from_timestamps(ts)


def test_raises_on_empty_timeline():
    tl = build_timeline_from_timestamps([])
    with pytest.raises(ForecastError, match="порожня"):
        forecast_responses(tl)


def test_raises_on_too_few_points():
    # Лише 4 точки — нижче MIN_TRAIN_POINTS=5
    tl = _make_timeline([1, 1, 1, 1])
    with pytest.raises(ForecastError, match="Замало точок"):
        forecast_responses(tl)


def test_works_with_minimum_points():
    """5 точок будь-якого span'у мають давати валідний прогноз (A1+B1)."""
    tl = _make_timeline([5])  # 5 відповідей одного дня
    fc = forecast_responses(tl, target=20)
    assert fc.final_estimate >= 5
    assert fc.final_ci[0] >= 5


def test_works_with_sub_daily_span():
    """47 відповідей за 15 хвилин — раніше падало, тепер працює."""
    base = datetime(2025, 5, 1, 12, 0, 0)
    ts = [base + timedelta(seconds=20 * i) for i in range(47)]
    tl = build_timeline_from_timestamps(ts)
    fc = forecast_responses(tl, target=100)
    assert fc.final_estimate >= 47


def test_ci_lower_never_below_last_observed():
    # 14 днів, sum=47, схоже на твій кейс
    daily = [3, 5, 4, 3, 4, 3, 2, 4, 3, 4, 3, 3, 3, 3]
    tl = _make_timeline(daily)
    fc = forecast_responses(tl, target=100)
    last_observed = int(tl.cumulative.iloc[-1])
    assert fc.final_ci[0] >= last_observed
    assert fc.ci_lower.min() >= last_observed
    assert fc.final_estimate >= last_observed


def test_ci_bounds_are_monotonic():
    daily = [3, 5, 4, 3, 4, 3, 2, 4, 3, 4, 3, 3, 3, 3]
    tl = _make_timeline(daily)
    fc = forecast_responses(tl, target=100)
    assert (fc.ci_lower.diff().dropna() >= 0).all()
    assert (fc.ci_upper.diff().dropna() >= 0).all()
    assert (fc.future_cum.diff().dropna() >= 0).all()


def test_target_changes_forecast():
    """Зміна target має впливати на результат (soft prior на K).

    Використовуємо ненасичену лінійну траєкторію — там bounds кусаються
    сильніше, ніж на чітко-S-кривих, де всі моделі сходяться до одного K.
    """
    # Потрібно ≥ SMALL_SAMPLE_THRESHOLD точок (10), щоб селектор пробував
    # усі три моделі — Logistic особливо чутливий до bounds на K.
    daily = [1] * 12
    tl = _make_timeline(daily)
    fc_small = forecast_responses(tl, target=15, random_seed=1)
    fc_large = forecast_responses(tl, target=1500, random_seed=1)
    assert (
        fc_small.final_estimate != fc_large.final_estimate or fc_small.final_ci != fc_large.final_ci
    )


def test_returns_valid_dates_index():
    daily = [3, 5, 4, 3, 4, 3, 2]
    tl = _make_timeline(daily)
    fc = forecast_responses(tl, target=50)
    assert isinstance(fc.future_dates, pd.DatetimeIndex)
    assert len(fc.future_dates) >= 1
    # Перша майбутня дата — наступний день після останнього факту
    last_known = tl.daily_counts.index[-1]
    assert fc.future_dates[0] == last_known + pd.Timedelta(days=1)


def test_model_name_is_one_of_three():
    daily = [3, 5, 4, 3, 4, 3, 2, 4, 3, 4, 3, 3, 3, 3]
    tl = _make_timeline(daily)
    fc = forecast_responses(tl, target=100)
    assert fc.model in {"logistic", "gompertz", "asymptotic_exp"}
    assert fc.aicc < float("inf")


def test_reproducible_with_seed():
    daily = [3, 5, 4, 3, 4, 3, 2, 4, 3, 4, 3, 3, 3, 3]
    tl = _make_timeline(daily)
    fc1 = forecast_responses(tl, target=100, random_seed=7)
    fc2 = forecast_responses(tl, target=100, random_seed=7)
    assert fc1.final_estimate == fc2.final_estimate
    assert fc1.final_ci == fc2.final_ci


def test_works_without_target():
    daily = [3, 5, 4, 3, 4, 3, 2, 4, 3, 4, 3, 3, 3, 3]
    tl = _make_timeline(daily)
    fc = forecast_responses(tl, target=None)
    assert fc.final_estimate >= int(tl.cumulative.iloc[-1])


def test_three_days_minimum_works():
    tl = _make_timeline([2, 3, 2])  # рівно 3 дні
    fc = forecast_responses(tl, target=20)
    assert fc.final_estimate >= 7
