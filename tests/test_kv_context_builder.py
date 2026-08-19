from __future__ import annotations

import pytest

from vector_db.contextual_text_builders import build_kv_contextual_text


def test_builds_complete_kv_table_contextual_text():
    evidence = {
        "evidence_id": "evidence:20250101000001:src0:s12:e0",
        "evidence_type": "TABLE",
        "table_type": "KV_TABLE",
        "order": 0,
        "payload": {
            "title": "생산시설 현황",
            "captions": ["주요 생산시설 내역"],
            "units": ["백만원"],
            "notes": ["연결 기준", "2025년 말 현재"],
            "heading_path": ["시설 및 설비"],
            "fields": [
                {"key_paths": [["사업장", "소재지"]], "raw_value": "평택"},
                {
                    "key_paths": [["사업장", "신규 시설", "완공 예정"]],
                    "raw_value": "2025년",
                },
            ],
        },
    }

    result = build_kv_contextual_text(
        evidence,
        corp_name="삼성전자",
        report_nm="2025년 사업보고서",
        section_path=["사업의 내용", "시설 및 설비"],
    )

    assert result == (
        "회사 : 삼성전자\n"
        "공시 : 2025년 사업보고서\n"
        "섹션 : 사업의 내용 > 시설 및 설비\n\n"
        "표 제목 : 생산시설 현황\n"
        "표 설명 : 주요 생산시설 내역\n"
        "단위 : 백만원\n"
        "주석 : 연결 기준\n"
        "주석 : 2025년 말 현재\n\n"
        "사업장 > 소재지 : 평택\n"
        "사업장 > 신규 시설 > 완공 예정 : 2025년"
    )


def test_omits_missing_table_context_and_preserves_each_key_path():
    evidence = {
        "evidence_type": "TABLE",
        "table_type": "KV_TABLE",
        "payload": {
            "fields": [
                {
                    "key_paths": [["당기", "금액"], ["현재", "금액"]],
                    "raw_value": "100",
                }
            ]
        },
    }

    result = build_kv_contextual_text(
        evidence,
        corp_name="삼성전자",
        report_nm="사업보고서",
        section_path=["재무에 관한 사항"],
    )

    assert result == (
        "회사 : 삼성전자\n"
        "공시 : 사업보고서\n"
        "섹션 : 재무에 관한 사항\n\n"
        "당기 > 금액 : 100\n"
        "현재 > 금액 : 100"
    )


@pytest.mark.parametrize(
    ("evidence", "message"),
    [
        ({"evidence_type": "TEXT", "payload": {}}, "evidence_type"),
        (
            {"evidence_type": "TABLE", "table_type": "R_TABLE", "payload": {}},
            "table_type",
        ),
        (
            {
                "evidence_type": "TABLE",
                "table_type": "KV_TABLE",
                "payload": {"fields": "invalid"},
            },
            "payload.fields",
        ),
    ],
)
def test_rejects_invalid_kv_evidence(evidence, message):
    with pytest.raises(ValueError, match=message):
        build_kv_contextual_text(
            evidence,
            corp_name="삼성전자",
            report_nm="사업보고서",
            section_path=["사업의 내용"],
        )
