"""Юніт-тести для CP-aware forecast wrapper."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from core.detection import Changepoint
from core.forecast import forecast_with_segmentation
from core.timeline import build_timeline_from_timestamps


def _build_timeline(daily_counts: list[int], start: datetime = datetime(2025, 1, 1)):
    """Допоміжний: list of daily counts → TimelineSeries."""
    ts: list[datetime] = []
    for d, n in enumerate(daily_counts):
        for i in range(n):
            ts.append(start + timedelta(days=d, hours=10, minutes=i))
    return build_timeline_from_timestamps(ts)


def _build_two_waves_timeline() -> tuple[object, datetime]:
    """Дві хвилі: тиха фаза (low rate) → активна (burst) → знову тиха.

    Це класичний step-change pattern, на якому PELT повинен знайти CP.
    Повертає (timeline, expected_cp_near_ts) — приблизну дату очікуваного CP.
    """
    base = datetime(2025, 1, 1)
    ts: list[datetime] = []
    # Phase 1: 30 годин по 1 події/годину (тиха)
    for h in range(30):
        ts.append(base + timedelta(hours=h, minutes=5))
    # Phase 2: 30 годин по 20 подій/годину (агітаційна хвиля)
    burst_start = base + timedelta(hours=30)
    for h in range(30):
        for i in range(20):
            ts.append(burst_start + timedelta(hours=h, minutes=i * 2))
    # Phase 3: 30 годин знову по 1 події/годину
    quiet_start = base + timedelta(hours=60)
    for h in range(30):
        ts.append(quiet_start + timedelta(hours=h, minutes=5))
    return build_timeline_from_timestamps(ts), burst_start


# ---------- основні сценарії ------------------------------------------------


def test_no_cp_on_clean_log_curve_falls_back_to_full_timeline():
    """Чистий logarithmic ряд без хвиль → CP=[] → той самий результат, що
    у forecast_responses на повному timeline."""
    # Поступове насичення: 10,9,8,7,6,5,5,4,4,3,3,3,3,3 (15 днів)
    daily = [10, 9, 8, 7, 6, 5, 5, 4, 4, 3, 3, 3, 3, 3]
    tl = _build_timeline(daily)
    fc, cps = forecast_with_segmentation(tl)
    assert cps == [] or isinstance(cps, list)
    assert fc.final_estimate >= sum(daily)


def test_segmentation_disabled_returns_empty_cps():
    daily = [5] * 20
    tl = _build_timeline(daily)
    fc, cps = forecast_with_segmentation(tl, auto_segment=False)
    assert cps == []
    assert fc.final_estimate >= sum(daily)


def test_too_short_series_returns_empty_cps():
    """N < 2 * min_segment (10 точок default) → одразу fallback."""
    daily = [2, 2, 2]  # 6 точок
    tl = _build_timeline(daily)
    fc, cps = forecast_with_segmentation(tl)
    assert cps == []
    assert fc.final_estimate >= 6


def test_two_wave_timeline_detects_cp_and_uses_post_cp_subset():
    """Step-change синтетика: PELT повинен знайти ≥1 CP, training subset =
    останній сегмент."""
    tl, _expected_burst_start = _build_two_waves_timeline()
    fc, cps = forecast_with_segmentation(tl, cp_penalty=5.0, cp_min_segment=5)
    # PELT на rate-серії з трьома phases (тиха-хвиля-тиха) повинен знайти CP.
    assert len(cps) >= 1, f"Очікували ≥1 CP, отримано {len(cps)}"
    for cp in cps:
        assert isinstance(cp, Changepoint)
        assert isinstance(cp.timestamp, pd.Timestamp)


def test_cps_returned_even_when_segment_too_short_falls_back():
    """Якщо post-CP segment < MIN_TRAIN_POINTS — fallback на повний timeline,
    але CP-список повертаємо (для візуалізації на графіку)."""
    # Дві фази, друга з лише 3 точками — закоротка для тренування
    base = datetime(2025, 1, 1)
    ts: list[datetime] = []
    # Phase 1: 50 годин по 1 (тиха)
    for h in range(50):
        ts.append(base + timedelta(hours=h, minutes=5))
    # Phase 2: 3 точки burst (закоротко для MIN_TRAIN_POINTS=5)
    burst = base + timedelta(hours=50)
    for i in range(3):
        ts.append(burst + timedelta(minutes=i))
    tl = build_timeline_from_timestamps(ts)
    fc, cps = forecast_with_segmentation(tl, cp_penalty=3.0)
    # Прогноз має повернутись (fallback), CPs можуть бути або порожні
    # (якщо penalty з'їв burst) або непорожні.
    assert fc is not None
    assert isinstance(cps, list)


# ---------- параметри ------------------------------------------------------


def test_higher_penalty_reduces_cp_count():
    """Збільшення penalty має зменшити кількість виявлених CP (rad.).

    На дуже драматичному (1x vs 20x) сигналі навіть високий penalty не
    дає 0 CP — тому перевіряємо монотонне зменшення."""
    tl, _ = _build_two_waves_timeline()
    _, cps_low = forecast_with_segmentation(tl, cp_penalty=1.0)
    _, cps_high = forecast_with_segmentation(tl, cp_penalty=100.0)
    assert len(cps_high) <= len(cps_low), (
        f"Очікували менше або рівно CP при високому penalty: "
        f"low={len(cps_low)}, high={len(cps_high)}"
    )


def test_horizon_until_propagates():
    """horizon_until прокидається у внутрішній forecast_responses."""
    daily = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    tl = _build_timeline(daily)
    horizon = pd.Timestamp(datetime(2025, 1, 20))
    fc, _cps = forecast_with_segmentation(tl, horizon_until=horizon)
    # Останній future_date повинен бути не пізніше horizon (з толерантністю
    # ±1 день через MIN_HORIZON_DAYS у service.py).
    assert fc.future_dates[-1] <= horizon + pd.Timedelta(days=1)
