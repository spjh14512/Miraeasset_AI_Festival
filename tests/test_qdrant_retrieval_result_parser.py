from __future__ import annotations

import pytest
from qdrant_client.http.models import QueryResponse, ScoredPoint

from agent_graph.retrieval_result_parser import (
    parse_qdrant_point,
    parse_qdrant_response,
    parse_r_table_headers,
    parse_r_table_point,
    parse_r_table_records,
)


def _point(
    point_kind: str,
    canonical: dict,
    *,
    point_id: str = "00000000-0000-0000-0000-000000000001",
    score: float = 0.9,
    extra_metadata: dict | None = None,
) -> ScoredPoint:
    retrieval_metadata = {
        "point_kind": point_kind,
        "evidence_id": "d1:src0:s0:e0",
        "corp_name": "삼성전자",
        "corp_code": "00126380",
        "report_nm": "사업보고서 (2023.12)",
        **(extra_metadata or {}),
    }
    return ScoredPoint(
        id=point_id,
        version=1,
        score=score,
        payload={
            "retrieval_metadata": retrieval_metadata,
            "contextual_text": (
                "회사 : 삼성전자\n"
                "공시 : 사업보고서 (2023.12)\n"
                "섹션 : III. 재무에 관한 사항 > 1. 요약재무정보\n\n"
                "검색 문맥"
            ),
            "canonical": canonical,
        },
    )


def _r_table_point() -> ScoredPoint:
    return _point(
        "R_TABLE",
        {
            "table_metadata": {"title": "특별관계자 현황"},
            "headers": [["성명"], ["관계"], ["관계"]],
            "records": [
                {"record_index": 0, "values": ["홍길동", "최대주주", "본인"]},
                {"record_index": 1, "values": ["김영희", "임원", "등기"]},
            ],
        },
        extra_metadata={"table_id": "table-1"},
    )


def test_parse_r_table_headers_makes_duplicate_names_unique():
    assert parse_r_table_headers([["구분"], ["금액", "당기"], ["금액", "당기"]]) == [
        "구분",
        "금액 > 당기",
        "금액 > 당기 [2]",
    ]


def test_parse_r_table_records_maps_values_to_columns_and_selects_indexes():
    headers = ["성명", "관계"]
    records = [
        {"record_index": 0, "values": ["홍길동", "최대주주"]},
        {"record_index": 1, "values": ["김영희", "임원"]},
    ]

    assert parse_r_table_records(headers, records, record_indexes={1}) == [
        {
            "record_index": 1,
            "values": {"성명": "김영희", "관계": "임원"},
        }
    ]


def test_r_table_defaults_to_summary_without_records():
    parsed = parse_r_table_point(_r_table_point())

    assert parsed["type"] == "r_table"
    assert parsed["metadata"]["retrieval_context"]["corp_name"] == "삼성전자"
    assert parsed["metadata"]["table_id"] == "table-1"
    assert "reference" not in parsed
    assert "retrieval_context" not in parsed
    assert parsed["columns"] == ["성명", "관계", "관계 [2]"]
    assert parsed["available_record_count"] == 2
    assert "records" not in parsed


def test_r_table_records_mode_includes_only_selected_records():
    parsed = parse_r_table_point(
        _r_table_point(),
        detail="records",
        record_indexes={1},
    )

    assert parsed["records"] == [
        {
            "record_index": 1,
            "values": {
                "성명": "김영희",
                "관계": "임원",
                "관계 [2]": "등기",
            },
        }
    ]
    assert parsed["included_record_count"] == 1
    assert parsed["omitted_record_count"] == 1


def test_parse_qdrant_point_dispatches_text_and_kv_table():
    text = parse_qdrant_point(_point("TEXT", {"text": "신규 시설을 구축합니다."}))
    kv_table = parse_qdrant_point(
        _point(
            "KV_TABLE",
            {
                "table_metadata": {"title": "시설 현황"},
                "entries": [{"key": "사업장", "value": "평택"}],
            },
        )
    )

    assert text["type"] == "text"
    assert text["content"] == "신규 시설을 구축합니다."
    assert kv_table["type"] == "kv_table"
    assert kv_table["entries"] == [{"key": "사업장", "value": "평택"}]
    assert kv_table["metadata"]["retrieval_context"]["corp_name"] == "삼성전자"
    assert "reference" not in kv_table
    assert "retrieval_context" not in kv_table


def test_parse_qdrant_response_selects_kv_entries_by_point_id():
    point = _point(
        "KV_TABLE",
        {
            "entries": [
                {"key": "유동자산", "value": "100"},
                {"key": "비유동자산", "value": "200"},
            ]
        },
    )

    result = parse_qdrant_response(
        QueryResponse(points=[point]),
        plan_id="plan_1",
        query="유동자산",
        selected_item_ids={str(point.id): {0}},
    )

    assert result.items[0]["entries"] == [
        {"key": "유동자산", "value": "100"}
    ]


def test_qdrant_item_compacts_common_context_and_reference():
    item = parse_qdrant_point(
        _point("TEXT", {"text": "신규 시설을 구축합니다."})
    )

    assert item["metadata"]["retrieval_context"] == {
        "corp_name": "삼성전자",
        "report_name": "사업보고서 (2023.12)",
        "section_path": ["III. 재무에 관한 사항", "1. 요약재무정보"],
    }
    assert "point_id" not in item
    assert "context" not in item
    assert "reference" not in item
    assert "retrieval_context" not in item
    assert "text" not in item
    assert "point_kind" not in item["metadata"]
    assert "corp_name" not in item["metadata"]
    assert "report_nm" not in item["metadata"]


def test_qdrant_reference_combines_base_year_and_month():
    reference = parse_qdrant_point(
        _point(
            "TEXT",
            {"text": "신규 시설을 구축합니다."},
            extra_metadata={"base_year": 2025, "base_month": 3},
        )
    )["metadata"]

    assert reference["base_date"] == "2025-03"
    assert "base_year" not in reference
    assert "base_month" not in reference


def test_qdrant_reference_omits_null_base_date():
    reference = parse_qdrant_point(
        _point(
            "TEXT",
            {"text": "신규 시설을 구축합니다."},
            extra_metadata={"base_year": None, "base_month": None},
        )
    )["metadata"]

    assert "base_date" not in reference
    assert "base_year" not in reference
    assert "base_month" not in reference


def test_qdrant_reference_rejects_partial_base_date():
    with pytest.raises(ValueError, match="함께 존재"):
        parse_qdrant_point(
            _point(
                "TEXT",
                {"text": "신규 시설을 구축합니다."},
                extra_metadata={"base_year": 2025, "base_month": None},
            )
        )


def test_parse_qdrant_point_derives_citation_parent_ids():
    point = _point(
        "TEXT",
        {"text": "유동자산은 4,964,158백만원입니다."},
        extra_metadata={
            "evidence_id": "d20240306000686:src0:s27:e8",
        },
    )

    reference = parse_qdrant_point(point)["metadata"]

    assert reference["disclosure_id"] == "d20240306000686"
    assert reference["section_id"] == "d20240306000686:src0:s27"
    assert reference["evidence_id"] == "d20240306000686:src0:s27:e8"


def test_parse_qdrant_response_returns_one_result_and_uses_table_selection():
    response = QueryResponse(points=[_r_table_point()])

    result = parse_qdrant_response(
        response,
        plan_id="plan_2",
        query="삼성전자 특별관계자",
        r_table_detail="records",
        selected_item_ids={
            "00000000-0000-0000-0000-000000000001": {0}
        },
    )

    assert result.result_id == "retrieval:plan_2"
    assert result.source == "qdrant"
    assert result.result_count == 1
    assert [row["record_index"] for row in result.items[0]["records"]] == [0]
    assert result.metadata == {
        "returned_point_count": 1,
        "r_table_detail": "records",
    }
    result.model_dump_json()
