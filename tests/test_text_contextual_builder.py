from __future__ import annotations

import pytest

from vector_db.contextual_text_builders import build_text_contextual_text


def test_builds_document_section_context_and_original_text():
    evidence = {
        "evidence_id": "evidence:20250101000001:src0:s12:e0",
        "evidence_type": "TEXT",
        "order": 0,
        "payload": {
            "text": "당사는 2025년 평택 사업장에 신규 생산시설을 구축할 예정입니다.",
        },
    }

    result = build_text_contextual_text(
        evidence,
        corp_name="삼성전자",
        report_nm="사업보고서",
        section_path=["사업의 내용", "시설 및 설비"],
    )

    assert result == (
        "공시 : 삼성전자 사업보고서\n"
        "섹션 : 사업의 내용 > 시설 및 설비\n\n"
        "당사는 2025년 평택 사업장에 신규 생산시설을 구축할 예정입니다."
    )


def test_appends_heading_path_and_removes_adjacent_duplicates():
    evidence = {
        "evidence_type": "TEXT",
        "payload": {
            "text": "신규 시설을 구축합니다.",
            "heading_path": ["시설 및 설비", "생산 능력"],
        },
    }

    result = build_text_contextual_text(
        evidence,
        corp_name="삼성전자",
        report_nm="사업보고서",
        section_path=["사업의 내용", "시설 및 설비"],
    )

    assert result == (
        "공시 : 삼성전자 사업보고서\n"
        "섹션 : 사업의 내용 > 시설 및 설비 > 생산 능력\n\n"
        "신규 시설을 구축합니다."
    )


@pytest.mark.parametrize(
    ("evidence", "message"),
    [
        ({"evidence_type": "TABLE", "payload": {}}, "evidence_type"),
        ({"evidence_type": "TEXT", "payload": {}}, "payload.text"),
        (
            {
                "evidence_type": "TEXT",
                "payload": {"text": "본문", "heading_path": "소제목"},
            },
            "heading_path",
        ),
    ],
)
def test_rejects_invalid_text_evidence(evidence, message):
    with pytest.raises(ValueError, match=message):
        build_text_contextual_text(
            evidence,
            corp_name="삼성전자",
            report_nm="사업보고서",
            section_path=["사업의 내용"],
        )
