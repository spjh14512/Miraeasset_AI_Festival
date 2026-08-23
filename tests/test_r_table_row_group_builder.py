from __future__ import annotations

from copy import deepcopy

import pytest

from vector_db.contextual_text_builders.r_table_row_group_builder import (
    build_r_table_row_groups,
)


TABLE_ID = "rtable:20250318000123:src0:s12:t0"


def _evidence(record_count: int) -> dict:
    return {
        "evidence_id": "evidence:20250318000123:src0:s12:e0",
        "evidence_type": "TABLE",
        "table_type": "R_TABLE",
        "storage_mode": "SECTION_RECORDS",
        "payload": {
            "table_id": TABLE_ID,
            "heading_path": ["유형자산"],
            "title": "생산시설 현황",
            "captions": ["주요 생산시설 내역"],
            "units": ["백만원"],
            "notes": ["연결 기준"],
            "headers": [["사업장"], ["설명"]],
            "record_count": record_count,
        },
    }


def _records(count: int) -> list[dict]:
    return [
        {
            "table_id": TABLE_ID,
            "record_index": index,
            "row_type": "DATA",
            "row_context": [],
            "values": [f"사업장-{index}", f"설명-{index}"],
        }
        for index in range(count)
    ]


def _row_weighted_token_counter(text: str) -> int:
    return 300 + (600 * text.count("사업장 :"))


def test_groups_records_greedily_without_overlap_or_omission():
    records = list(reversed(_records(5)))

    groups = build_r_table_row_groups(
        _evidence(5),
        records,
        corp_name="삼성전자",
        report_nm="2025년 사업보고서",
        section_path=["연결재무제표 주석"],
        max_tokens=1500,
        token_counter=_row_weighted_token_counter,
    )

    assert [
        (group.group_index, group.row_start_index, group.row_end_index)
        for group in groups
    ] == [(0, 0, 1), (1, 2, 3), (2, 4, 4)]
    assert [group.token_count for group in groups] == [1500, 1500, 900]
    assert all(group.table_id == TABLE_ID for group in groups)

    covered_indexes = [
        record_index
        for group in groups
        for record_index in range(
            group.row_start_index,
            group.row_end_index + 1,
        )
    ]
    assert covered_indexes == list(range(5))
    assert len(covered_indexes) == len(set(covered_indexes))

    all_text = "\n".join(group.contextual_text for group in groups)
    for index in range(5):
        assert all_text.count(f"사업장 : 사업장-{index}") == 1
    for group in groups:
        assert "삼성전자" in group.contextual_text
        assert "2025년 사업보고서" in group.contextual_text
        assert "연결재무제표 주석 > 유형자산" in group.contextual_text
        assert "생산시설 현황" in group.contextual_text
        assert "주요 생산시설 내역" in group.contextual_text
        assert "백만원" in group.contextual_text
        assert "연결 기준" in group.contextual_text
        assert "사업장 | 설명" in group.contextual_text

    assert groups[0].to_dict() == {
        "group_index": 0,
        "table_id": TABLE_ID,
        "row_start_index": 0,
        "row_end_index": 1,
        "contextual_text": groups[0].contextual_text,
        "token_count": 1500,
    }


def test_uses_configured_default_row_group_limit():
    groups = build_r_table_row_groups(
        _evidence(3),
        _records(3),
        corp_name="삼성전자",
        report_nm="사업보고서",
        section_path=[],
        token_counter=_row_weighted_token_counter,
    )

    assert [
        (group.row_start_index, group.row_end_index) for group in groups
    ] == [(0, 1), (2, 2)]


def test_keeps_one_oversized_record_as_a_singleton_group():
    groups = build_r_table_row_groups(
        _evidence(1),
        _records(1),
        corp_name="삼성전자",
        report_nm="사업보고서",
        section_path=[],
        max_tokens=1500,
        token_counter=lambda text: 1501,
    )

    assert len(groups) == 1
    assert groups[0].row_start_index == 0
    assert groups[0].row_end_index == 0
    assert groups[0].token_count == 1501


def test_empty_table_produces_no_groups_without_counting_tokens():
    def unexpected_counter(text: str) -> int:
        raise AssertionError("empty tables must not invoke the token counter")

    assert build_r_table_row_groups(
        _evidence(0),
        [],
        corp_name="삼성전자",
        report_nm="사업보고서",
        section_path=[],
        token_counter=unexpected_counter,
    ) == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda evidence, records: records[0].update(table_id="rtable:other"),
            "table_id must match",
        ),
        (
            lambda evidence, records: records[0].update(values=["too short"]),
            "same length",
        ),
        (
            lambda evidence, records: evidence["payload"].update(record_count=2),
            "records length",
        ),
        (
            lambda evidence, records: records.append(deepcopy(records[0])),
            "record_index must be unique",
        ),
        (
            lambda evidence, records: records[0].update(record_index=1),
            "contiguous",
        ),
    ],
)
def test_rejects_invalid_table_structure(mutation, message):
    evidence = _evidence(1)
    records = _records(1)
    mutation(evidence, records)
    if message == "record_index must be unique":
        evidence["payload"]["record_count"] = 2

    with pytest.raises(ValueError, match=message):
        build_r_table_row_groups(
            evidence,
            records,
            corp_name="삼성전자",
            report_nm="사업보고서",
            section_path=[],
            max_tokens=1500,
            token_counter=_row_weighted_token_counter,
        )


@pytest.mark.parametrize("token_count", [-1, 1.5, True])
def test_rejects_invalid_token_counter_result(token_count):
    with pytest.raises(ValueError, match="non-negative integer"):
        build_r_table_row_groups(
            _evidence(1),
            _records(1),
            corp_name="삼성전자",
            report_nm="사업보고서",
            section_path=[],
            max_tokens=1500,
            token_counter=lambda text: token_count,
        )
