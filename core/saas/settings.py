"""Environment-backed SaaS settings with production safety checks."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

APP_ENVS = frozenset({"development", "production", "test"})


@dataclass(frozen=True)
class SaaSSettings:
    app_env: str
    app_base_url: str
    api_base_url: str
    gcp_project_id: str
    firestore_database: str
    kms_key_name: str
    gcs_bucket: str
    cloud_tasks_location: str
    tasks_queue_name: str
    worker_base_url: str
    cloud_tasks_service_account_email: str
    google_oauth_client_config_json: str
    session_pepper: str

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    def validate(self) -> None:
        _require_known_env(self.app_env)
        if self.is_production:
            _require_https("APP_BASE_URL", self.app_base_url)
            _require_https("API_BASE_URL", self.api_base_url)
            required = {
                "GCP_PROJECT_ID": self.gcp_project_id,
                "FIRESTORE_DATABASE": self.firestore_database,
                "KMS_KEY_NAME": self.kms_key_name,
                "GCS_BUCKET": self.gcs_bucket,
                "CLOUD_TASKS_LOCATION": self.cloud_tasks_location,
                "TASKS_QUEUE_NAME": self.tasks_queue_name,
                "WORKER_BASE_URL": self.worker_base_url,
                "CLOUD_TASKS_SERVICE_ACCOUNT_EMAIL": self.cloud_tasks_service_account_email,
                "GOOGLE_OAUTH_CLIENT_CONFIG_JSON": self.google_oauth_client_config_json,
                "SESSION_PEPPER": self.session_pepper,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(f"Missing production settings: {', '.join(missing)}")
            _require_https("WORKER_BASE_URL", self.worker_base_url)


def load_saas_settings(env: Mapping[str, str] | None = None) -> SaaSSettings:
    source = env or os.environ
    settings = SaaSSettings(
        app_env=source.get("APP_ENV", "development"),
        app_base_url=source.get("APP_BASE_URL", "http://localhost:8501"),
        api_base_url=source.get("API_BASE_URL", "http://localhost:8080"),
        gcp_project_id=source.get("GCP_PROJECT_ID", ""),
        firestore_database=source.get("FIRESTORE_DATABASE", "(default)"),
        kms_key_name=source.get("KMS_KEY_NAME", ""),
        gcs_bucket=source.get("GCS_BUCKET", ""),
        cloud_tasks_location=source.get("CLOUD_TASKS_LOCATION", ""),
        tasks_queue_name=source.get("TASKS_QUEUE_NAME", ""),
        worker_base_url=source.get("WORKER_BASE_URL", "http://localhost:8001"),
        cloud_tasks_service_account_email=source.get("CLOUD_TASKS_SERVICE_ACCOUNT_EMAIL", ""),
        google_oauth_client_config_json=source.get("GOOGLE_OAUTH_CLIENT_CONFIG_JSON", ""),
        session_pepper=source.get("SESSION_PEPPER", "development-only-pepper"),
    )
    settings.validate()
    return settings


@dataclass(frozen=True)
class WebSettings:
    """What the Streamlit web service needs: its own URL and the API it signs in through."""

    app_env: str
    app_base_url: str
    api_base_url: str

    @property
    def signs_in_through_api(self) -> bool:
        """Production always signs in through the API.

        The local demo sign-in, where Streamlit itself talks to Google and holds the tokens,
        is for a developer machine only (invariants 1 and 5).
        """
        return self.app_env == "production"


_WEB_PRODUCTION_URLS = ("APP_BASE_URL", "API_BASE_URL")


def load_web_settings(env: Mapping[str, str] | None = None) -> WebSettings:
    """Read the web settings; in production a missing or plain-HTTP URL is an error.

    Before this check a production web service without API_BASE_URL quietly switched to the
    local demo sign-in instead of failing.
    """
    source = os.environ if env is None else env
    app_env = source.get("APP_ENV", "development")
    _require_known_env(app_env)
    if app_env == "production":
        missing = [name for name in _WEB_PRODUCTION_URLS if not source.get(name)]
        if missing:
            raise ValueError(
                f"Missing production settings for the web service: {', '.join(missing)}"
            )
        for name in _WEB_PRODUCTION_URLS:
            _require_https(name, source[name])
    return WebSettings(
        app_env=app_env,
        app_base_url=(source.get("APP_BASE_URL") or "http://localhost:8501").rstrip("/"),
        api_base_url=(source.get("API_BASE_URL") or "http://localhost:8000").rstrip("/"),
    )


def _require_known_env(app_env: str) -> None:
    if app_env not in APP_ENVS:
        raise ValueError("APP_ENV must be one of: development, production, test.")


def _require_https(name: str, value: str) -> None:
    if not value.startswith("https://"):
        raise ValueError(f"{name} must use HTTPS in production.")
