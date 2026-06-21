from __future__ import annotations

import os

from google.oauth2.credentials import Credentials

from core.saas.adapters.google_oauth import GoogleOAuthWebClient


class _FakeFlow:
    def __init__(self) -> None:
        self.credentials = Credentials(token="access-token")
        self.received_code: str | None = None

    def fetch_token(self, *, code: str) -> None:
        self.received_code = code
        assert os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] == "1"


class _FakeAuthorizationFlow:
    def __init__(self) -> None:
        self.authorization_kwargs: dict[str, object] | None = None

    def authorization_url(self, **kwargs):
        self.authorization_kwargs = kwargs
        return "https://accounts.example/auth", kwargs["state"]


def test_google_oauth_identity_start_does_not_reuse_prior_grants(monkeypatch) -> None:
    flow = _FakeAuthorizationFlow()
    client = GoogleOAuthWebClient(
        client_config_json='{"web": {"token_uri": "https://oauth.example/token"}}',
        api_base_url="https://api.example.com",
    )
    monkeypatch.setattr(client, "_flow", lambda **_kwargs: flow)

    url = client.authorization_url(
        state="state",
        code_verifier="verifier",
        scopes=["openid"],
    )

    assert url == "https://accounts.example/auth"
    assert flow.authorization_kwargs is not None
    assert flow.authorization_kwargs["include_granted_scopes"] == "false"


def test_google_oauth_exchange_code_relaxes_scope_mismatch_temporarily(monkeypatch) -> None:
    flow = _FakeFlow()
    client = GoogleOAuthWebClient(
        client_config_json='{"web": {"token_uri": "https://oauth.example/token"}}',
        api_base_url="https://api.example.com",
    )
    monkeypatch.delenv("OAUTHLIB_RELAX_TOKEN_SCOPE", raising=False)
    monkeypatch.setattr(client, "_flow", lambda **_kwargs: flow)

    credentials = client.exchange_code(
        code="oauth-code",
        state="state",
        code_verifier="verifier",
        scopes=["openid"],
    )

    assert flow.received_code == "oauth-code"
    assert credentials.token == "access-token"
    assert "OAUTHLIB_RELAX_TOKEN_SCOPE" not in os.environ
