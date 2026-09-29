"""safe_next_url decides where the OAuth callback may send the browser, login ticket included."""

from __future__ import annotations

import pytest

from api.urls import safe_next_url

APP = "https://app.example.com"


@pytest.mark.parametrize(
    "next_url",
    [
        pytest.param("/", id="root"),
        pytest.param("/catalog", id="local-path"),
        pytest.param("/catalog/forms/42", id="nested-local-path"),
        pytest.param("/catalog?form_id=abc&tab=stats#top", id="query-and-fragment"),
        pytest.param("https://app.example.com/", id="app-root"),
        pytest.param("https://app.example.com/catalog?form_id=abc", id="app-url-with-query"),
    ],
)
def test_targets_inside_the_app_are_kept(next_url: str) -> None:
    assert safe_next_url(next_url, APP) == next_url


@pytest.mark.parametrize(
    "next_url",
    [
        pytest.param("", id="empty"),
        pytest.param("catalog", id="path-without-leading-slash"),
        pytest.param("evil.example/x", id="bare-host"),
        pytest.param("https://evil.example/", id="foreign-url"),
        pytest.param("http://evil.example/", id="foreign-http-url"),
        pytest.param("javascript:alert(1)", id="javascript-scheme"),
        pytest.param("data:text/html,x", id="data-scheme"),
        pytest.param("https://app.example.com@evil.example/", id="app-host-as-userinfo"),
        pytest.param("https://app.example.com.evil.example/", id="app-host-as-subdomain"),
        pytest.param("\\\\evil.example", id="leading-backslashes"),
        pytest.param(" /catalog", id="leading-space"),
    ],
)
def test_everything_else_falls_back_to_the_root(next_url: str) -> None:
    assert safe_next_url(next_url, APP) == "/"


def test_the_app_base_url_may_carry_a_path_prefix() -> None:
    base = "https://example.com/survey"

    assert safe_next_url("https://example.com/survey/catalog", base) == (
        "https://example.com/survey/catalog"
    )
    assert safe_next_url("https://example.com/surveyor", base) == "/"
    assert safe_next_url("https://example.com/other", base) == "/"


def test_a_trailing_slash_on_the_app_base_url_is_ignored() -> None:
    assert safe_next_url("https://app.example.com/catalog", APP + "/") == (
        "https://app.example.com/catalog"
    )


def test_absolute_urls_are_refused_when_no_app_base_url_is_configured() -> None:
    assert safe_next_url("https://app.example.com/catalog", "") == "/"
    assert safe_next_url("/catalog", "") == "/catalog"
