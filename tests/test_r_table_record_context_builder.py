from __future__ import annotations

import pytest

from vector_db.contextual_text_builders import (
    build_r_table_record_contextual_text,
)


def _evidence():
    return {
        "evidence_id": "evidence:20250318000123:src0:s12:e3",
        "evidence_type": "TABLE",
        "table_type": "R_TABLE",
        "storage_mode": "SECTION_RECORDS",
        "payload": {
            "table_id": "rtable:20250318000123:src0:s12:t0",
            "title": "사업장별 생산시설 현황",
            "captions": ["2025년 말 현재 주요 생산시설 내역"],
            "units": ["백만원"],
            "notes": ["연결 기준"],
            "heading_path": ["시설 및 설비"],
            "headers": [
                ["사업장"],
                ["주소", "시도"],
                ["가동 여부"],
                ["비고"],
                ["투자 금액"],
            ],
            "record_count": 3,
        },
    }


def test_builds_one_contextual_text_per_r_table_record():
    record = {
        "table_id": "rtable:20250318000123:src0:s12:t0",
        "record_index": 0,
        "row_type": "DATA",
        "row_context": [],
        "values": ["평택 사업장", "경기도", "가동", "-", "120000"],
    }

    result = build_r_table_record_contextual_text(
        _evidence(),
        record,
        corp_name="삼성전자",
        report_nm="2025년 사업보고서",
        section_path=["사업의 내용", "시설 및 설비"],
    )

    assert result == (
        "회사 : 삼성전자\n"
        "공시 : 2025년 사업보고서\n"
        "섹션 : 사업의 내용 > 시설 및 설비\n\n"
        "표 제목 : 사업장별 생산시설 현황\n"
        "표 설명 : 2025년 말 현재 주요 생산시설 내역\n"
        "단위 : 백만원\n"
        "주석 : 연결 기준\n\n"
        "사업장 : 평택 사업장\n"
        "주소 > 시도 : 경기도\n"
        "가동 여부 : 가동\n"
        "비고 : -\n"
        "투자 금액 : 120000"
    )


def test_omits_empty_values_but_preserves_explicit_source_values():
    record = {
        "table_id": "rtable:20250318000123:src0:s12:t0",
        "record_index": 1,
        "row_type": "DATA",
        "row_context": [],
        "values": ["평택", "", "해당없음", "-", "0"],
    }

    result = build_r_table_record_contextual_text(
        _evidence(),
        record,
        corp_name="삼성전자",
        report_nm="사업보고서",
        section_path=["사업의 내용"],
    )

    assert "주소 > 시도" not in result
    assert "가동 여부 : 해당없음" in result
    assert "비고 : -" in result
    assert "투자 금액 : 0" in result


def test_renders_non_data_row_type_and_row_context():
    record = {
        "table_id": "rtable:20250318000123:src0:s12:t0",
        "record_index": 2,
        "row_type": "TOTAL",
        "row_context": ["국내", "사업장"],
        "values": ["합계", "", "", "", "350000"],
    }

    result = build_r_table_record_contextual_text(
        _evidence(),
        record,
        corp_name="삼성전자",
        report_nm="사업보고서",
        section_path=["사업의 내용"],
    )

    assert "행 유형 : TOTAL\n행 문맥 : 국내 > 사업장" in result
    assert "사업장 : 합계" in result
    assert "투자 금액 : 350000" in result


@pytest.mark.parametrize(
    ("evidence", "record", "message"),
    [
        (
            {**_evidence(), "table_type": "KV_TABLE"},
            {
                "table_id": "rtable:20250318000123:src0:s12:t0",
                "row_type": "DATA",
                "row_context": [],
                "values": [],
            },
            "table_type",
        ),
        (
            _evidence(),
            {
                "table_id": "rtable:other",
                "row_type": "DATA",
                "row_context": [],
                "values": ["", "", "", "", ""],
            },
            "table_id",
        ),
        (
            _evidence(),
            {
                "table_id": "rtable:20250318000123:src0:s12:t0",
                "row_type": "DATA",
                "row_context": [],
                "values": ["too short"],
            },
            "same length",
        ),
    ],
)
def test_rejects_invalid_r_table_input(evidence, record, message):
    with pytest.raises(ValueError, match=message):
        build_r_table_record_contextual_text(
            evidence,
            record,
            corp_name="삼성전자",
            report_nm="사업보고서",
            section_path=["사업의 내용"],
        )
