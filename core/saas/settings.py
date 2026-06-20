"""Environment-backed SaaS settings with production safety checks."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class SaaSSettings:
    app_env: str
    app_base_url: str
    api_base_url: str
    gcp_project_id: str
    kms_key_name: str
    gcs_bucket: str
    tasks_queue_name: str
    google_oauth_client_config_json: str
    session_pepper: str

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    def validate(self) -> None:
        if self.app_env not in {"development", "production", "test"}:
            raise ValueError("APP_ENV must be one of: development, production, test.")
        if self.is_production:
            _require_https("APP_BASE_URL", self.app_base_url)
            _require_https("API_BASE_URL", self.api_base_url)
            required = {
                "GCP_PROJECT_ID": self.gcp_project_id,
                "KMS_KEY_NAME": self.kms_key_name,
                "GCS_BUCKET": self.gcs_bucket,
                "TASKS_QUEUE_NAME": self.tasks_queue_name,
                "GOOGLE_OAUTH_CLIENT_CONFIG_JSON": self.google_oauth_client_config_json,
                "SESSION_PEPPER": self.session_pepper,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(f"Missing production settings: {', '.join(missing)}")


def load_saas_settings(env: Mapping[str, str] | None = None) -> SaaSSettings:
    source = env or os.environ
    settings = SaaSSettings(
        app_env=source.get("APP_ENV", "development"),
        app_base_url=source.get("APP_BASE_URL", "http://localhost:8501"),
        api_base_url=source.get("API_BASE_URL", "http://localhost:8080"),
        gcp_project_id=source.get("GCP_PROJECT_ID", ""),
        kms_key_name=source.get("KMS_KEY_NAME", ""),
        gcs_bucket=source.get("GCS_BUCKET", ""),
        tasks_queue_name=source.get("TASKS_QUEUE_NAME", ""),
        google_oauth_client_config_json=source.get("GOOGLE_OAUTH_CLIENT_CONFIG_JSON", ""),
        session_pepper=source.get("SESSION_PEPPER", "development-only-pepper"),
    )
    settings.validate()
    return settings


def _require_https(name: str, value: str) -> None:
    if not value.startswith("https://"):
        raise ValueError(f"{name} must use HTTPS in production.")
