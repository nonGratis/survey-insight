from __future__ import annotations

from datetime import date

import pytest

from core.version import DEV_VERSION, AppVersion, current_version


def test_reads_commit_and_build_date_from_the_environment() -> None:
    version = current_version({"APP_VERSION": "cd3b096", "APP_BUILD_DATE": "2026-10-03"})

    assert version == AppVersion("cd3b096", date(2026, 10, 3))
    assert version.label == "cd3b096 · 03.10.2026"


def test_without_build_values_the_version_is_dev() -> None:
    version = current_version({})

    assert version == AppVersion(DEV_VERSION, None)
    assert version.label == "dev"


def test_accepts_a_release_tag_as_well_as_a_commit_hash() -> None:
    assert current_version({"APP_VERSION": "v1.2.0"}).version == "v1.2.0"


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "**bold**", "<b>x</b>", "a b", "-leading-dash", "x" * 41],
)
def test_a_value_that_is_not_a_code_label_falls_back_to_dev(raw: str) -> None:
    assert current_version({"APP_VERSION": raw}).version == DEV_VERSION


@pytest.mark.parametrize("raw", ["", "03.10.2026", "2026-13-01", "soon"])
def test_an_unparseable_build_date_is_left_out(raw: str) -> None:
    version = current_version({"APP_VERSION": "cd3b096", "APP_BUILD_DATE": raw})

    assert version.build_date is None
    assert version.label == "cd3b096"


def test_reads_the_process_environment_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_VERSION", "abc1234")
    monkeypatch.delenv("APP_BUILD_DATE", raising=False)

    assert current_version().label == "abc1234"
