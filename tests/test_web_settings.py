"""Web service settings: in production a missing URL is an error, not the local demo sign-in."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.saas.settings import load_web_settings
from ui import startup

PRODUCTION = {
    "APP_ENV": "production",
    "APP_BASE_URL": "https://app.example.com/",
    "API_BASE_URL": "https://api.example.com",
}


def test_production_signs_in_through_the_api_with_both_urls() -> None:
    settings = load_web_settings(PRODUCTION)

    assert settings.signs_in_through_api
    assert settings.app_base_url == "https://app.example.com"
    assert settings.api_base_url == "https://api.example.com"


@pytest.mark.parametrize(
    "missing", [["API_BASE_URL"], ["APP_BASE_URL"], ["APP_BASE_URL", "API_BASE_URL"]]
)
def test_production_without_a_url_is_an_error_not_the_demo_sign_in(missing: list[str]) -> None:
    env = {name: value for name, value in PRODUCTION.items() if name not in missing}

    with pytest.raises(ValueError, match=f"web service: {', '.join(missing)}$"):
        load_web_settings(env)


def test_an_empty_url_counts_as_missing() -> None:
    with pytest.raises(ValueError, match="API_BASE_URL"):
        load_web_settings({**PRODUCTION, "API_BASE_URL": ""})


@pytest.mark.parametrize("name", ["APP_BASE_URL", "API_BASE_URL"])
def test_production_urls_must_use_https(name: str) -> None:
    with pytest.raises(ValueError, match=f"{name} must use HTTPS"):
        load_web_settings({**PRODUCTION, name: "http://example.com"})


def test_outside_production_the_local_demo_sign_in_keeps_its_defaults() -> None:
    settings = load_web_settings({})  # APP_ENV unset means development

    assert not settings.signs_in_through_api
    assert settings.app_base_url == "http://localhost:8501"
    assert settings.api_base_url == "http://localhost:8000"


def test_an_unknown_app_env_is_an_error() -> None:
    with pytest.raises(ValueError, match="APP_ENV must be one of"):
        load_web_settings({"APP_ENV": "prod"})


def test_the_data_facade_no_longer_falls_back_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ui.google_data import is_saas_mode

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("APP_BASE_URL", "https://app.example.com")
    monkeypatch.delenv("API_BASE_URL", raising=False)

    # before: False, i.e. Streamlit itself signed in to Google and held the tokens
    with pytest.raises(ValueError, match="API_BASE_URL"):
        is_saas_mode()


def test_startup_check_refuses_a_misconfigured_production_web(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("APP_BASE_URL", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)

    assert startup.main() == 1
    assert "refusing to start" in capsys.readouterr().err


def test_startup_check_passes_a_complete_production_web(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in PRODUCTION.items():
        monkeypatch.setenv(name, value)

    assert startup.main() == 0


def test_the_image_runs_the_startup_check_before_streamlit() -> None:
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")

    assert "web|*) python -m ui.startup && exec streamlit run app.py" in dockerfile
