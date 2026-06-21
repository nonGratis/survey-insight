"""Cloud Storage artifact adapter for generated PDF reports."""

from __future__ import annotations

from datetime import timedelta
from urllib.parse import quote

from google.cloud import storage

from core.saas.models import Artifact
from core.saas.security import utcnow


class GcsArtifactStorage:
    def __init__(
        self,
        bucket_name: str,
        client: storage.Client | None = None,
        *,
        retention: timedelta = timedelta(days=30),
        signed_url_ttl: timedelta = timedelta(minutes=15),
    ) -> None:
        if not bucket_name:
            raise ValueError("GCS bucket name is required.")
        self.bucket_name = bucket_name
        self.client = client or storage.Client()
        self.retention = retention
        self.signed_url_ttl = signed_url_ttl

    def save_pdf(self, report_id: str, user_id: str, content: bytes) -> Artifact:
        created_at = utcnow()
        object_name = _report_object_name(user_id=user_id, report_id=report_id)
        blob = self.client.bucket(self.bucket_name).blob(object_name)
        blob.upload_from_string(content, content_type="application/pdf")
        return Artifact(
            id=f"artifact_{report_id}",
            report_id=report_id,
            user_id=user_id,
            gcs_uri=f"gs://{self.bucket_name}/{object_name}",
            content_type="application/pdf",
            created_at=created_at,
            expires_at=created_at + self.retention,
        )

    def signed_download_url(self, artifact: Artifact) -> str:
        bucket_name, object_name = _parse_gcs_uri(artifact.gcs_uri)
        blob = self.client.bucket(bucket_name).blob(object_name)
        return blob.generate_signed_url(
            version="v4",
            expiration=self.signed_url_ttl,
            method="GET",
            response_disposition=f'attachment; filename="{quote(artifact.report_id)}.pdf"',
        )


def _report_object_name(*, user_id: str, report_id: str) -> str:
    return f"reports/{_safe_path_segment(user_id)}/{_safe_path_segment(report_id)}.pdf"


def _safe_path_segment(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    return "".join(char if char in allowed else "_" for char in value)


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("Artifact URI must be a gs:// URI.")
    bucket_and_object = gcs_uri.removeprefix("gs://")
    bucket, separator, object_name = bucket_and_object.partition("/")
    if not bucket or separator != "/" or not object_name:
        raise ValueError("Artifact URI must include bucket and object name.")
    return bucket, object_name
