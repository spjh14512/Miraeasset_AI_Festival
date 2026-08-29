import json
from pathlib import Path

from converters.event_extractor.category_mapping import (
    EventCategory,
    cleaned_report_name,
    load_event_categories,
)
from converters.event_extractor.event_extractor import extract_event
from converters.event_extractor.event_pipeline import latest_receipt_numbers


def _field(key: str, value: str) -> dict:
    return {"key_paths": [[key]], "raw_value": value}


def test_category_uses_cleaned_name_from_workbook_column_a():
    catalog = load_event_categories(Path("DOCS/Event_categories.xlsx"))

    assert cleaned_report_name(
        "[기재정정][첨부추가]주요사항보고서(유상증자결정)"
    ) == "유상증자결정"
    assert catalog.match(
        "[기재정정]주요사항보고서(유상증자결정)"
    ).event_subtype == "유상증자결정"
    assert catalog.match(
        "[첨부추가]단일판매ㆍ공급계약체결(자율공시)"
    ).event_subtype == "단일판매ㆍ공급계약체결"


def test_major_event_uses_core_fragment_and_relationship_directions():
    rcept_no = "20240102000001"
    core_evidence_id = f"evidence:{rcept_no}:src0:s0:e0"
    annex_evidence_id = f"evidence:{rcept_no}:src0:s1:e0"
    fragments = [
        {
            "section_id": f"section:{rcept_no}:src0:s0",
            "evidence_list": [
                {
                    "evidence_id": core_evidence_id,
                    "evidence_type": "TABLE",
                    "payload": {
                        "table_id": "t0",
                        "fields": [
                            _field("이사회결의일(결정일)", "2024년 01월 01일"),
                            _field("신주의 종류와 수 > 보통주식", "1,000"),
                        ],
                    },
                }
            ],
            "records": [],
        },
        {
            "section_id": f"section:{rcept_no}:src0:s1",
            "evidence_list": [
                {
                    "evidence_id": annex_evidence_id,
                    "evidence_type": "TEXT",
                    "payload": {"text": "긴 첨부 계약서 원문"},
                }
            ],
            "records": [],
        },
    ]

    result = extract_event(
        {
            "doc_id": f"major_{rcept_no}",
            "rcept_no": rcept_no,
            "rcept_dt": "20240102",
            "doc_group": "major",
            "listed_name": "테스트회사",
        },
        category=EventCategory("유상증자결정", "자본·증권발행", 2),
        fragments=fragments,
    )

    assert result.event.event_subtype == "유상증자결정"
    assert result.event.event_date == "2024-01-01"
    assert "1,000" in result.event.content
    assert "긴 첨부 계약서 원문" not in result.event.content
    assert result.reports.to_dict() == {
        "type": "REPORTS",
        "source_id": f"d{rcept_no}",
        "target_id": f"event:{rcept_no}",
    }
    assert [relation.target_id for relation in result.is_supported_by] == [
        f"d{rcept_no}:src0:s0:e0"
    ]


def test_exchange_event_excludes_correction_metadata_evidence():
    rcept_no = "20240202800001"
    correction_id = f"evidence:{rcept_no}:src0:s0:e0"
    event_id = f"evidence:{rcept_no}:src1:s0:e0"
    fragments = [
        {
            "section_id": f"section:{rcept_no}:src0:s0",
            "evidence_list": [
                {
                    "evidence_id": correction_id,
                    "evidence_type": "TABLE",
                    "payload": {
                        "table_id": "correction",
                        "title": "정정관련 공시서류",
                        "headers": [["정정사항", "계약금액"]],
                    },
                }
            ],
            "records": [
                {
                    "table_id": "correction",
                    "row_context": ["정정사항"],
                    "values": ["계약금액", "100"],
                }
            ],
        },
        {
            "section_id": f"section:{rcept_no}:src1:s0",
            "evidence_list": [
                {
                    "evidence_id": event_id,
                    "evidence_type": "TABLE",
                    "payload": {
                        "table_id": "main",
                        "headers": [["계약내용", "공급계약"]],
                    },
                }
            ],
            "records": [
                {
                    "table_id": "main",
                    "row_context": ["계약금액"],
                    "values": ["1,000억원"],
                },
                {
                    "table_id": "main",
                    "row_context": ["계약수주일자"],
                    "values": ["2024-02-01"],
                },
            ],
        },
    ]

    result = extract_event(
        {
            "doc_id": f"exchange_{rcept_no}",
            "rcept_no": rcept_no,
            "rcept_dt": "20240202",
            "doc_group": "exchange",
            "listed_name": "테스트회사",
        },
        category=EventCategory(
            "단일판매ㆍ공급계약체결", "계약·영업거래", 2
        ),
        fragments=fragments,
    )

    assert result.event.event_date == "2024-02-01"
    assert "1,000억원" in result.event.content
    assert "정정사항" not in result.event.content
    assert [relation.target_id for relation in result.is_supported_by] == [
        f"d{rcept_no}:src1:s0:e0"
    ]


def test_latest_receipts_exclude_only_resolved_correction_targets():
    manifest = [
        {"rcept_no": "20240101000001"},
        {"rcept_no": "20240202000002"},
        {"rcept_no": "20240303000003"},
    ]
    corrections = [
        {
            "source_document": {"rcept_no": "20240202000002"},
            "status": "FOUND",
            "correction": {"target_rcept_no": "20240101000001"},
        },
        {
            "source_document": {"rcept_no": "20240303000003"},
            "status": "NOT_FOUND",
            "correction": None,
        },
    ]

    assert latest_receipt_numbers(manifest, correction_rows=corrections) == {
        "20240202000002",
        "20240303000003",
    }


def test_latest_receipts_keep_only_leaf_of_multistep_correction_chain():
    manifest = [
        {"rcept_no": "20240101000001"},
        {"rcept_no": "20240202000002"},
        {"rcept_no": "20240303000003"},
    ]
    corrections = [
        {
            "source_document": {"rcept_no": "20240202000002"},
            "status": "FOUND",
            "correction": {"target_rcept_no": "20240101000001"},
        },
        {
            "source_document": {"rcept_no": "20240303000003"},
            "status": "RECOVERED",
            "correction": {"target_rcept_no": "20240101000001"},
        },
    ]

    assert latest_receipt_numbers(manifest, correction_rows=corrections) == {
        "20240303000003"
    }


def test_latest_receipts_ignore_unresolved_null_target():
    manifest = [
        {"rcept_no": "20240101000001"},
        {"rcept_no": "20240202000002"},
    ]
    corrections = [
        {
            "source_document": {"rcept_no": "20240202000002"},
            "status": "FOUND",
            "correction": {"target_rcept_no": None},
        }
    ]

    assert latest_receipt_numbers(manifest, correction_rows=corrections) == {
        "20240101000001",
        "20240202000002",
    }
