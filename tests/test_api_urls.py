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
        pytest.param("/?form_id=abc", id="root-with-query"),
        pytest.param("/#top", id="root-with-fragment"),
        # Only the invisible and the ambiguous are refused, not escapes or plain text.
        pytest.param("/catalog?return=%2Fweighting", id="encoded-slash-in-query"),
        pytest.param("/catalog?q=опитування", id="non-ascii-text"),
        pytest.param("https://app.example.com", id="app-root-without-slash"),
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
        # Scheme-relative: a browser reads "//host/path" as https://host/path.
        pytest.param("//evil.example/x", id="scheme-relative"),
        pytest.param("//", id="bare-double-slash"),
        pytest.param("///evil.example/x", id="three-slashes"),
        pytest.param("////evil.example/x", id="four-slashes"),
        # A browser reads "\" as "/", so these are "//evil.example" to it.
        pytest.param("/\\evil.example", id="backslash-after-slash"),
        pytest.param("//\\evil.example", id="double-slash-then-backslash"),
        pytest.param("/catalog\\..\\x", id="backslash-deeper-in-path"),
        pytest.param("https://app.example.com/\\evil.example", id="backslash-in-app-url"),
        # A browser drops tab/CR/LF anywhere in a URL, so "/<TAB>/host" is "//host" to it.
        pytest.param("/\t/evil.example", id="tab-between-slashes"),
        pytest.param("/\n/evil.example", id="newline-between-slashes"),
        pytest.param("/\r\n/evil.example", id="crlf-between-slashes"),
        pytest.param("/catalog\t", id="trailing-tab"),
        pytest.param("https://app.example.com/\t/evil.example", id="tab-in-app-url"),
        # Other whitespace and invisible characters: no legitimate link contains them.
        pytest.param("/ /evil.example", id="space-between-slashes"),
        pytest.param("/catalog ", id="trailing-space"),
        pytest.param("/\x00/evil.example", id="nul"),
        pytest.param("/\x7f/evil.example", id="delete"),
        pytest.param("/\x85/evil.example", id="c1-control"),
        pytest.param("/ /evil.example", id="no-break-space"),
        pytest.param("/ /evil.example", id="line-separator"),
        pytest.param("/​/evil.example", id="zero-width-space"),
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


@pytest.mark.parametrize("app_base_url", [APP, APP + "/"])
def test_the_app_root_is_kept_with_or_without_a_trailing_slash(app_base_url: str) -> None:
    assert safe_next_url("https://app.example.com", app_base_url) == "https://app.example.com"
    assert safe_next_url("https://app.example.com/", app_base_url) == "https://app.example.com/"


def test_absolute_urls_are_refused_when_no_app_base_url_is_configured() -> None:
    assert safe_next_url("https://app.example.com/catalog", "") == "/"
    assert safe_next_url("/catalog", "") == "/catalog"


def test_scheme_relative_urls_are_refused_when_no_app_base_url_is_configured() -> None:
    # An empty base must not become the prefix "/", which every local path starts with.
    assert safe_next_url("//evil.example/x", "") == "/"
