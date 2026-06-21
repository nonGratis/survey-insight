"""Google OAuth web-flow adapter for the SaaS API."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Protocol, cast

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build


class GoogleOAuthClient(Protocol):
    def authorization_url(
        self,
        *,
        state: str,
        code_verifier: str,
        scopes: Sequence[str],
        include_granted_scopes: bool = False,
    ) -> str: ...

    def exchange_code(
        self,
        *,
        code: str,
        state: str,
        code_verifier: str,
        scopes: Sequence[str],
    ) -> Credentials: ...

    def user_info(self, credentials: Credentials) -> dict[str, Any]: ...


class GoogleOAuthWebClient:
    def __init__(self, *, client_config_json: str, api_base_url: str) -> None:
        if not api_base_url:
            raise ValueError("API base URL is required.")
        self.client_config = json.loads(client_config_json) if client_config_json else None
        self.redirect_uri = f"{api_base_url.rstrip('/')}/v1/auth/google/callback"
        self.relax_token_scope = True

    def authorization_url(
        self,
        *,
        state: str,
        code_verifier: str,
        scopes: Sequence[str],
        include_granted_scopes: bool = False,
    ) -> str:
        flow = self._flow(
            scopes=scopes,
            code_verifier=code_verifier,
            state=state,
        )
        url, _state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true" if include_granted_scopes else "false",
            prompt="consent",
            state=state,
        )
        return url

    def exchange_code(
        self,
        *,
        code: str,
        state: str,
        code_verifier: str,
        scopes: Sequence[str],
    ) -> Credentials:
        flow = self._flow(
            scopes=scopes,
            code_verifier=code_verifier,
            state=state,
        )
        with _oauthlib_scope_policy(relax=self.relax_token_scope):
            flow.fetch_token(code=code)
        return cast(Credentials, flow.credentials)

    def user_info(self, credentials: Credentials) -> dict[str, Any]:
        service = build("oauth2", "v2", credentials=credentials, cache_discovery=False)
        return dict(service.userinfo().get().execute())

    def _flow(
        self,
        *,
        scopes: Sequence[str],
        code_verifier: str,
        state: str,
    ) -> Flow:
        return Flow.from_client_config(
            self._client_config(),
            scopes=list(scopes),
            redirect_uri=self.redirect_uri,
            code_verifier=code_verifier,
            autogenerate_code_verifier=False,
            state=state,
        )

    def _client_config(self) -> dict[str, Any]:
        if self.client_config is None:
            raise ValueError("Google OAuth client config JSON is required.")
        return self.client_config


@contextmanager
def _oauthlib_scope_policy(*, relax: bool):
    # Google may return a broader scope set when include_granted_scopes=true
    # reuses previously granted Forms/Drive scopes. That is valid for
    # incremental auth, but oauthlib raises unless relaxed explicitly.
    previous = os.environ.get("OAUTHLIB_RELAX_TOKEN_SCOPE")
    if relax:
        os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OAUTHLIB_RELAX_TOKEN_SCOPE", None)
        else:
            os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = previous
