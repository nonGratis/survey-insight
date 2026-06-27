from __future__ import annotations

from core.forms_catalog import (
    CATALOG_SUMMARY_FIELDS,
    _parse_form,
    enrich_form,
    list_forms_with_drive_meta,
)


class _FakeExecute:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class _FakeForms:
    def __init__(self):
        self.calls = []

    def get(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeExecute(
            {
                "info": {"title": "Demo"},
                "items": [{"questionItem": {}}, {"pageBreakItem": {}}],
            }
        )


class _FakeFormsService:
    def __init__(self):
        self.forms_resource = _FakeForms()

    def forms(self):
        return self.forms_resource


class _FakeDriveFiles:
    def __init__(self):
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeExecute(
            {
                "files": [
                    {
                        "id": "form_1",
                        "name": "Form 1",
                        "owners": [{"emailAddress": "a@example.com", "displayName": "A"}],
                    },
                    {
                        "id": "form_2",
                        "name": "Form 2",
                        "owners": [{"emailAddress": "b@example.com", "displayName": "B"}],
                    },
                ],
                "nextPageToken": "next-page",
            }
        )


class _FakeDriveService:
    def __init__(self):
        self.files_resource = _FakeDriveFiles()

    def files(self):
        return self.files_resource


def test_parse_form_reads_publish_state() -> None:
    enrichment = _parse_form(
        {
            "info": {"title": "Demo", "description": "Test"},
            "items": [{"questionItem": {}}, {"pageBreakItem": {}}, {"questionItem": {}}],
            "linkedSheetId": "sheet-1",
            "publishSettings": {
                "publishState": {
                    "isPublished": True,
                    "isAcceptingResponses": False,
                }
            },
        }
    )

    assert enrichment.title == "Demo"
    assert enrichment.questions_count == 2
    assert enrichment.sections_count == 1
    assert enrichment.linked_sheet_id == "sheet-1"
    assert enrichment.is_published is True
    assert enrichment.accepting_responses is False


def test_parse_form_keeps_legacy_publish_state_unknown() -> None:
    enrichment = _parse_form({"info": {"title": "Legacy"}})

    assert enrichment.is_published is None
    assert enrichment.accepting_responses is None


def test_enrich_form_requests_only_catalog_summary_fields(monkeypatch) -> None:
    service = _FakeFormsService()
    monkeypatch.setattr("core.forms_catalog.build", lambda *args, **kwargs: service)

    enrichment = enrich_form(object(), "form_1")

    assert enrichment.title == "Demo"
    assert service.forms_resource.calls == [{"formId": "form_1", "fields": CATALOG_SUMMARY_FIELDS}]


def test_list_forms_with_drive_meta_stops_at_configured_limit(monkeypatch) -> None:
    service = _FakeDriveService()
    monkeypatch.setattr("core.forms_catalog.DRIVE_MAX_FORMS", 2)
    monkeypatch.setattr("core.forms_catalog.DRIVE_PAGE_SIZE", 100)
    monkeypatch.setattr("core.forms_catalog.build", lambda *args, **kwargs: service)

    forms = list_forms_with_drive_meta(object())

    assert [form.id for form in forms] == ["form_1", "form_2"]
    assert len(service.files_resource.calls) == 1
    assert service.files_resource.calls[0]["pageSize"] == 2
