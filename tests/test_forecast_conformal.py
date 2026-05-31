"""Tests for core.forecast.conformal (P14)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.forecast import conformal
from core.forecast.conformal import (
    apply_conformal_adjustment,
    horizon_bucket_from_days,
    lookup_quantile,
    n_class_from_n,
    reset_cache,
)


@pytest.fixture(autouse=True)
def _reset_cache_each_test():
    """Кожен тест починає з чистого cache (бо deeply caches JSON)."""
    reset_cache()
    yield
    reset_cache()


class TestNClassFromN:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (5, "tiny"),
            (9, "tiny"),
            (10, "small"),
            (29, "small"),
            (30, "medium"),
            (99, "medium"),
            (100, "large"),
            (999, "large"),
            (1000, "huge"),
            (10000, "huge"),
        ],
    )
    def test_thresholds(self, n: int, expected: str):
        assert n_class_from_n(n) == expected


class TestHorizonBucketFromDays:
    @pytest.mark.parametrize(
        ("days", "expected"),
        [
            (0.05, "short"),  # 1.2h
            (0.25, "short"),  # 6h
            (1.0, "mid"),  # 24h
            (3.0, "mid"),  # 72h
            (3.001, "long"),  # 72.024h
            (7.0, "long"),  # 168h
        ],
    )
    def test_bucket_boundaries(self, days: float, expected: str):
        assert horizon_bucket_from_days(days) == expected


class TestLookupQuantile:
    def test_exact_cell_match(self):
        """Cell present у JSON → exact lookup."""
        q = lookup_quantile("medium", "short")
        assert q > 0
        assert np.isfinite(q)

    def test_bucket_fallback_when_cell_missing(self):
        """Cell missing → fallback на bucket."""
        q_unknown = lookup_quantile("nonexistent_class", "short")
        q_bucket = lookup_quantile("_", "short")
        assert q_unknown == q_bucket

    def test_global_fallback_when_bucket_missing(self):
        """Bucket missing → fallback на global."""
        q_unknown = lookup_quantile("nonexistent", "nonexistent_bucket")
        q_global = lookup_quantile("_", "_")
        assert q_unknown == q_global

    def test_no_json_uses_default(self, monkeypatch, tmp_path):
        """Якщо JSON відсутній → default 1.0 (no adjustment)."""
        fake_path = tmp_path / "nonexistent.json"
        monkeypatch.setattr(conformal, "_QUANTILES_PATH", fake_path)
        reset_cache()
        q = lookup_quantile("medium", "short")
        assert q == 1.0

    def test_json_loaded_once(self, tmp_path, monkeypatch):
        """JSON cache — повторні lookup-и не re-парсять."""
        fake_json = tmp_path / "q.json"
        fake_json.write_text(json.dumps({"quantiles": {"_|_": 7.0}}), encoding="utf-8")
        monkeypatch.setattr(conformal, "_QUANTILES_PATH", fake_json)
        reset_cache()
        q1 = lookup_quantile("medium", "short")
        # Modify JSON; should NOT reload (cache hit).
        fake_json.write_text(json.dumps({"quantiles": {"_|_": 99.0}}), encoding="utf-8")
        q2 = lookup_quantile("medium", "short")
        assert q1 == q2 == 7.0
        # After reset_cache, re-reads.
        reset_cache()
        q3 = lookup_quantile("medium", "short")
        assert q3 == 99.0


class TestApplyConformalAdjustment:
    def test_widens_ci_with_high_q(self, monkeypatch, tmp_path):
        """High q → wider CI."""
        fake = tmp_path / "q.json"
        fake.write_text(json.dumps({"quantiles": {"_|_": 10.0}}), encoding="utf-8")
        monkeypatch.setattr(conformal, "_QUANTILES_PATH", fake)
        reset_cache()

        point = np.array([100.0])
        lower = np.array([95.0])
        upper = np.array([105.0])
        horizon_arr = np.array([1.0])
        new_lo, new_hi = apply_conformal_adjustment(
            point, lower, upper, n_train=20, horizon_days_arr=horizon_arr
        )
        # half = 5, new_half = 5*10 = 50. CI = [50, 150].
        assert new_lo[0] == pytest.approx(50.0)
        assert new_hi[0] == pytest.approx(150.0)

    def test_narrows_ci_with_low_q(self, monkeypatch, tmp_path):
        """q < 1 → narrower CI."""
        fake = tmp_path / "q.json"
        fake.write_text(json.dumps({"quantiles": {"_|_": 0.5}}), encoding="utf-8")
        monkeypatch.setattr(conformal, "_QUANTILES_PATH", fake)
        reset_cache()

        point = np.array([100.0])
        lower = np.array([90.0])
        upper = np.array([110.0])
        new_lo, new_hi = apply_conformal_adjustment(
            point, lower, upper, n_train=20, horizon_days_arr=np.array([1.0])
        )
        # half = 10, new = 10*0.5 = 5. CI = [95, 105].
        assert new_lo[0] == pytest.approx(95.0)
        assert new_hi[0] == pytest.approx(105.0)

    def test_preserves_point(self, monkeypatch, tmp_path):
        """Apply ніколи не змінює point estimate."""
        fake = tmp_path / "q.json"
        fake.write_text(json.dumps({"quantiles": {"_|_": 5.0}}), encoding="utf-8")
        monkeypatch.setattr(conformal, "_QUANTILES_PATH", fake)
        reset_cache()

        point = np.array([42.0, 50.0])
        new_lo, new_hi = apply_conformal_adjustment(
            point,
            np.array([40.0, 48.0]),
            np.array([44.0, 52.0]),
            n_train=10,
            horizon_days_arr=np.array([0.1, 2.0]),
        )
        # Conformal не змінює point. (Перевіряємо через симетричність CI навколо point.)
        assert new_hi[0] - point[0] == pytest.approx(point[0] - new_lo[0])
        assert new_hi[1] - point[1] == pytest.approx(point[1] - new_lo[1])

    def test_horizon_dependent_q(self, monkeypatch, tmp_path):
        """Різні horizon → різні q (per-bucket lookup)."""
        fake = tmp_path / "q.json"
        fake.write_text(
            json.dumps({"quantiles": {"_|short": 2.0, "_|mid": 5.0, "_|long": 10.0, "_|_": 7.0}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(conformal, "_QUANTILES_PATH", fake)
        reset_cache()

        point = np.array([100.0, 100.0, 100.0])
        lower = np.array([95.0, 95.0, 95.0])  # half=5
        upper = np.array([105.0, 105.0, 105.0])
        horizons = np.array([0.1, 2.0, 5.0])  # short, mid, long
        new_lo, new_hi = apply_conformal_adjustment(
            point, lower, upper, n_train=20, horizon_days_arr=horizons
        )
        # halfs: 5*2=10, 5*5=25, 5*10=50
        assert new_hi[0] - new_lo[0] == pytest.approx(20.0)
        assert new_hi[1] - new_lo[1] == pytest.approx(50.0)
        assert new_hi[2] - new_lo[2] == pytest.approx(100.0)
