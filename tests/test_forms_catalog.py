from __future__ import annotations

from core.forms_catalog import _parse_form


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
