from __future__ import annotations

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
        **(extra_metadata or {}),
    }
    return ScoredPoint(
        id=point_id,
        version=1,
        score=score,
        payload={
            "retrieval_metadata": retrieval_metadata,
            "contextual_text": "삼성전자 사업보고서 검색 문맥",
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
    assert text["text"] == "신규 시설을 구축합니다."
    assert kv_table["type"] == "kv_table"
    assert kv_table["entries"] == [{"key": "사업장", "value": "평택"}]


def test_parse_qdrant_point_derives_citation_parent_ids():
    point = _point(
        "TEXT",
        {"text": "유동자산은 4,964,158백만원입니다."},
        extra_metadata={
            "evidence_id": "d20240306000686:src0:s27:e8",
        },
    )

    reference = parse_qdrant_point(point)["reference"]

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
        r_table_record_indexes={"table-1": {0}},
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
