from __future__ import annotations

from datetime import timedelta

import pytest

from core.saas.adapters.storage import GcsArtifactStorage, _parse_gcs_uri, _report_object_name


class _FakeBlob:
    def __init__(self, bucket_name: str, name: str) -> None:
        self.bucket_name = bucket_name
        self.name = name
        self.content: bytes | None = None
        self.content_type: str | None = None
        self.signed_url_calls: list[dict] = []

    def upload_from_string(self, content: bytes, content_type: str) -> None:
        self.content = content
        self.content_type = content_type

    def generate_signed_url(self, **kwargs) -> str:
        self.signed_url_calls.append(kwargs)
        return f"https://storage.local/{self.bucket_name}/{self.name}?signed=1"


class _FakeBucket:
    def __init__(self, name: str) -> None:
        self.name = name
        self.blobs: dict[str, _FakeBlob] = {}

    def blob(self, name: str) -> _FakeBlob:
        if name not in self.blobs:
            self.blobs[name] = _FakeBlob(self.name, name)
        return self.blobs[name]


class _FakeStorageClient:
    def __init__(self) -> None:
        self.buckets: dict[str, _FakeBucket] = {}

    def bucket(self, name: str) -> _FakeBucket:
        if name not in self.buckets:
            self.buckets[name] = _FakeBucket(name)
        return self.buckets[name]


def test_gcs_artifact_storage_uploads_pdf_and_returns_metadata() -> None:
    client = _FakeStorageClient()
    storage = GcsArtifactStorage(
        "survey-insight-reports",
        client=client,
        retention=timedelta(days=7),
    )

    artifact = storage.save_pdf("report/1", "user@example.com", b"%PDF-1.4")

    assert artifact.gcs_uri == "gs://survey-insight-reports/reports/user_example_com/report_1.pdf"
    assert artifact.content_type == "application/pdf"
    assert artifact.expires_at is not None
    blob = client.bucket("survey-insight-reports").blob("reports/user_example_com/report_1.pdf")
    assert blob.content == b"%PDF-1.4"
    assert blob.content_type == "application/pdf"


def test_gcs_artifact_storage_generates_short_lived_signed_url() -> None:
    client = _FakeStorageClient()
    storage = GcsArtifactStorage(
        "survey-insight-reports",
        client=client,
        signed_url_ttl=timedelta(minutes=5),
    )
    artifact = storage.save_pdf("report_1", "user_1", b"%PDF-1.4")

    url = storage.signed_download_url(artifact)

    assert url.endswith("?signed=1")
    blob = client.bucket("survey-insight-reports").blob("reports/user_1/report_1.pdf")
    assert blob.signed_url_calls[0]["version"] == "v4"
    assert blob.signed_url_calls[0]["method"] == "GET"
    assert blob.signed_url_calls[0]["expiration"] == timedelta(minutes=5)


def test_gcs_artifact_storage_requires_bucket_name() -> None:
    with pytest.raises(ValueError, match="bucket name"):
        GcsArtifactStorage("", client=_FakeStorageClient())


def test_gcs_uri_parser_rejects_non_gcs_uri() -> None:
    with pytest.raises(ValueError, match="gs://"):
        _parse_gcs_uri("https://example.com/report.pdf")


def test_report_object_name_sanitises_path_segments() -> None:
    assert _report_object_name(user_id="user@example.com", report_id="report/1") == (
        "reports/user_example_com/report_1.pdf"
    )
