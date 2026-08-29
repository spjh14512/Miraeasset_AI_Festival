from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from vector_db.contextual_text_builders import (
    build_r_table_descriptor_contextual_text,
)
from vector_db.r_table_column_profiler import profile_r_table_columns


TABLE_ID = "rtable:20250318000123:src0:s12:t0"


def _evidence(headers, record_count, *, include_context=True):
    payload = {
        "table_id": TABLE_ID,
        "headers": headers,
        "record_count": record_count,
    }
    if include_context:
        payload.update(
            {
                "title": "사업부문별 매출",
                "captions": ["주요 제품별 매출 내역"],
                "units": ["백만원"],
                "notes": ["연결 기준"],
                "heading_path": ["매출"],
            }
        )
    return {
        "evidence_id": "evidence:20250318000123:src0:s12:e0",
        "evidence_type": "TABLE",
        "table_type": "R_TABLE",
        "storage_mode": "SECTION_RECORDS",
        "payload": payload,
    }


def _records(rows):
    return [
        {
            "table_id": TABLE_ID,
            "record_index": index,
            "row_type": "DATA",
            "row_context": [],
            "values": row,
        }
        for index, row in enumerate(rows)
    ]


def test_builds_descriptor_from_context_headers_and_dimension_values():
    headers = [
        ["사업부문"],
        ["품목"],
        ["지역"],
        ["2025년", "매출액"],
        ["비중"],
    ]
    records = _records(
        [
            ["반도체", "DRAM", "국내", "1500", "35%"],
            ["반도체", "NAND", "미주", "950", "22%"],
            ["DX", "TV", "중국", "650", "15%"],
        ]
    )
    evidence = _evidence(headers, len(records))
    analysis = profile_r_table_columns(evidence, records)

    result = build_r_table_descriptor_contextual_text(
        evidence,
        analysis,
        corp_name="삼성전자",
        report_nm="2025년 사업보고서",
        section_path=["연결재무제표 주석"],
    )

    assert result == (
        "회사 : 삼성전자\n"
        "공시 : 2025년 사업보고서\n"
        "섹션 : 연결재무제표 주석 > 매출\n\n"
        "표 제목 : 사업부문별 매출\n"
        "표 설명 : 주요 제품별 매출 내역\n"
        "단위 : 백만원\n"
        "주석 : 연결 기준\n\n"
        "컬럼 구조 :\n"
        "사업부문\n"
        "품목\n"
        "지역\n"
        "2025년 > 매출액\n"
        "비중\n\n"
        "행 항목 :\n"
        "사업부문 : 반도체, DX\n"
        "품목 : DRAM, NAND, TV\n"
        "지역 : 국내, 미주, 중국\n\n"
        "검색어 :\n"
        "사업부문별 매출 | 사업부문 | 품목 | 지역 | 2025년 | 매출액 | 비중 | "
        "반도체 | DX | DRAM | NAND | TV | 국내 | 미주 | 중국 | 백만원"
    )
    assert "행 식별 컬럼" not in result
    assert "레코드 수" not in result
    assert "1500" not in result
    assert "35%" not in result


def test_omits_high_cardinality_and_non_dimension_values():
    rows = [
        [
            f"품목-{index}",
            f"성명-{index}",
            str(index * 100),
            f"2025-01-{index % 28 + 1:02d}",
        ]
        for index in range(55)
    ]
    evidence = _evidence(
        [["품목"], ["성명"], ["금액"], ["변동일"]],
        len(rows),
        include_context=False,
    )
    records = _records(rows)
    analysis = profile_r_table_columns(evidence, records)

    result = build_r_table_descriptor_contextual_text(
        evidence,
        analysis,
        corp_name="KB금융",
        report_nm="주식등의대량보유상황보고서",
        section_path=["주식등의 세부변동내역"],
    )

    assert "컬럼 구조 :\n품목\n성명\n금액\n변동일" in result
    assert "행 항목 :" not in result
    assert "품목-0" not in result
    assert "성명-0" not in result
    assert "2025-01-01" not in result


def test_omits_missing_optional_table_context_lines():
    evidence = _evidence([["구분"], ["금액"]], 1, include_context=False)
    records = _records([["합계", "100"]])
    analysis = profile_r_table_columns(evidence, records)

    result = build_r_table_descriptor_contextual_text(
        evidence,
        analysis,
        corp_name="삼성전자",
        report_nm="사업보고서",
        section_path=["재무제표 주석"],
    )

    assert "표 제목" not in result
    assert "표 설명" not in result
    assert "\n주석 :" not in result
    assert "컬럼 구조 :\n구분\n금액" in result
    assert "행 항목 :\n구분 : 합계" in result


def test_groups_merged_header_repetitions_for_retrieval_text_only():
    headers = [
        ["당기", "당기", "금액"],
        ["당기", "당기", "비율"],
        ["전기", "전기", "금액"],
        ["전기", "전기", "비율"],
    ]
    records = _records([["100", "10%", "90", "9%"]])
    evidence = _evidence(headers, 1, include_context=False)

    result = build_r_table_descriptor_contextual_text(
        evidence,
        profile_r_table_columns(evidence, records),
        corp_name="삼성전자",
        report_nm="사업보고서",
        section_path=["재무제표 주석"],
    )

    assert "컬럼 구조 :\n당기 : 금액 | 비율\n전기 : 금액 | 비율" in result
    assert "당기 > 당기" not in result
    assert "전기 > 전기" not in result
    assert evidence["payload"]["headers"] == headers


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda evidence, analysis: (
                evidence,
                replace(analysis, table_id="rtable:other"),
            ),
            "table_id must match",
        ),
        (
            lambda evidence, analysis: (
                evidence,
                replace(analysis, record_count=analysis.record_count + 1),
            ),
            "record_count must match",
        ),
        (
            lambda evidence, analysis: (
                deepcopy(evidence),
                analysis,
            ),
            "header paths must match",
        ),
    ],
)
def test_rejects_mismatched_column_analysis(mutate, message):
    evidence = _evidence([["구분"], ["금액"]], 1)
    records = _records([["합계", "100"]])
    analysis = profile_r_table_columns(evidence, records)
    changed_evidence, changed_analysis = mutate(evidence, analysis)
    if message == "header paths must match":
        changed_evidence["payload"]["headers"][0] = ["다른 구분"]

    with pytest.raises(ValueError, match=message):
        build_r_table_descriptor_contextual_text(
            changed_evidence,
            changed_analysis,
            corp_name="삼성전자",
            report_nm="사업보고서",
            section_path=["재무제표 주석"],
        )
