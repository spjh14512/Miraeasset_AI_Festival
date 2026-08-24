from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Literal

from neo4j import Record
from neo4j.graph import Node, Path, Relationship
from neo4j.spatial import Point
from qdrant_client.http.models import (
    QueryResponse,
    Record as QdrantRecord,
    ScoredPoint,
)

from .state import RetrievalResult


QdrantPoint = ScoredPoint | QdrantRecord | Mapping[str, Any]
RTableDetail = Literal["summary", "records"]


def _parse_scalar(value: Any) -> Any:
    """JSON으로 바로 표현하기 어려운 scalar를 LLM 친화적인 값으로 변환합니다.

    입력 예시:
        neo4j.time.Date(2025, 3, 31)

    출력 예시:
        "2025-03-31"
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()

    iso_format = getattr(value, "iso_format", None)
    if callable(iso_format):
        return iso_format()

    return str(value)


def parse_node(node: Node) -> dict[str, Any]:
    """Neo4j Node를 label과 property가 보존된 dictionary로 변환합니다.

    입력 예시:
        <Node labels={'Company'} properties={'corp_name': '삼성전자'}>

    출력 예시:
        {
            "type": "node",
            "labels": ["Company"],
            "properties": {"corp_name": "삼성전자"},
        }
    """

    return {
        "type": "node",
        "labels": sorted(node.labels),
        "properties": {
            str(key): parse_value(value)
            for key, value in node.items()
        },
    }


def parse_relationship(relationship: Relationship) -> dict[str, Any]:
    """Neo4j Relationship를 관계 유형, 양 끝 노드, property로 변환합니다.

    입력 예시:
        (Company {corp_name: '삼성전자'})-[:PUBLISHES]->(Disclosure {id: 'd1'})

    출력 예시:
        {
            "type": "relationship",
            "relationship_type": "PUBLISHES",
            "start_node": {"type": "node", ...},
            "end_node": {"type": "node", ...},
            "properties": {},
        }
    """

    return {
        "type": "relationship",
        "relationship_type": relationship.type,
        "start_node": (
            parse_node(relationship.start_node)
            if relationship.start_node is not None
            else None
        ),
        "end_node": (
            parse_node(relationship.end_node)
            if relationship.end_node is not None
            else None
        ),
        "properties": {
            str(key): parse_value(value)
            for key, value in relationship.items()
        },
    }


def parse_path(path: Path) -> dict[str, Any]:
    """Neo4j Path를 순서가 보존된 node와 relationship 목록으로 변환합니다.

    입력 예시:
        (Company)-[:PUBLISHES]->(Disclosure)-[:HAS_SECTION]->(Section)

    출력 예시:
        {
            "type": "path",
            "nodes": [{"type": "node", ...}, ...],
            "relationships": [{"type": "relationship", ...}, ...],
        }
    """

    return {
        "type": "path",
        "nodes": [parse_node(node) for node in path.nodes],
        "relationships": [
            parse_relationship(relationship)
            for relationship in path.relationships
        ],
    }


def parse_aggregate(value: Any) -> Any:
    """Aggregate 결과의 list, map, scalar를 재귀적으로 JSON 호환 값으로 변환합니다.

    Neo4j 응답에는 값이 aggregate 함수에서 만들어졌는지 표시되지 않으므로
    이 함수는 aggregate 의미를 추론하지 않고 반환된 값의 구조만 보존합니다.

    입력 예시:
        [<Node labels={'Company'} ...>, 3, {"total": 10}]

    출력 예시:
        [{"type": "node", ...}, 3, {"total": 10}]
    """

    if isinstance(value, Mapping):
        return {
            str(key): parse_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [parse_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [parse_value(item) for item in sorted(value, key=repr)]
    return _parse_scalar(value)


def parse_record(record: Record | Mapping[str, Any]) -> dict[str, Any]:
    """Neo4j Record의 RETURN alias와 값을 보존한 dictionary를 생성합니다.

    입력 예시:
        <Record company=<Node ...> disclosure_count=3>

    출력 예시:
        {
            "type": "record",
            "fields": {
                "company": {"type": "node", ...},
                "disclosure_count": 3,
            },
        }
    """

    return {
        "type": "record",
        "fields": {
            str(key): parse_value(value)
            for key, value in record.items()
        },
    }


def parse_value(value: Any) -> Any:
    """Neo4j 반환값의 runtime type에 맞는 parser를 선택합니다.

    입력 예시:
        <Node ...>, <Relationship ...>, <Path ...>, list, dict 또는 scalar

    출력 예시:
        graph 객체는 type 정보가 포함된 dict로, collection은 재귀적인 JSON 값으로
        변환됩니다.
    """

    if isinstance(value, Node):
        return parse_node(value)
    if isinstance(value, Relationship):
        return parse_relationship(value)
    if isinstance(value, Path):
        return parse_path(value)
    if isinstance(value, Record):
        return parse_record(value)
    if isinstance(value, Point):
        return {
            "type": "point",
            "srid": value.srid,
            "coordinates": list(value),
        }
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return parse_aggregate(value)
    return _parse_scalar(value)


def parse_neo4j_response(
    response: Iterable[Record] | Any,
    *,
    plan_id: str,
    query: str,
    parameters: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    result_id: str | None = None,
) -> RetrievalResult:
    """Neo4j query 응답 전체를 하나의 RetrievalResult로 변환합니다.

    ``session.run()``의 Result, ``driver.execute_query()``의 EagerResult 또는
    ``list[Record]``를 입력으로 받을 수 있습니다.

    입력 예시:
        parse_neo4j_response(
            records,
            plan_id="plan_1",
            query="MATCH (c:Company) RETURN c LIMIT 2",
            parameters={},
        )

    출력 예시:
        RetrievalResult(
            result_id="retrieval:plan_1",
            plan_id="plan_1",
            source="neo4j",
            query="MATCH (c:Company) RETURN c LIMIT 2",
            items=[{"type": "record", "fields": {...}}, ...],
            result_count=2,
            metadata={"parameters": {}},
        )
    """

    records = getattr(response, "records", response)
    items = [parse_record(record) for record in records]

    result_metadata = {
        str(key): parse_value(value)
        for key, value in (metadata or {}).items()
    }
    if parameters is not None:
        result_metadata["parameters"] = parse_aggregate(parameters)

    return RetrievalResult(
        result_id=result_id or f"retrieval:{plan_id}",
        plan_id=plan_id,
        source="neo4j",
        query=query,
        items=items,
        result_count=len(items),
        metadata=result_metadata,
    )


def _qdrant_point_value(
    point: QdrantPoint,
    field: str,
    default: Any = None,
) -> Any:
    """Qdrant model 또는 dictionary에서 같은 방식으로 field를 읽습니다.

    입력 예시:
        _qdrant_point_value(scored_point, "score")

    출력 예시:
        0.87
    """

    if isinstance(point, Mapping):
        return point.get(field, default)
    return getattr(point, field, default)


def _qdrant_payload(point: QdrantPoint) -> Mapping[str, Any]:
    """Qdrant point의 payload가 object인지 검증하여 반환합니다.

    입력 예시:
        <ScoredPoint payload={"retrieval_metadata": {...}}>

    출력 예시:
        {"retrieval_metadata": {...}, "contextual_text": "...", "canonical": {...}}
    """

    payload = _qdrant_point_value(point, "payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Qdrant point payload must be a mapping")
    return payload


def _parse_qdrant_reference(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Qdrant retrieval_metadata를 출처 추적용 reference로 정규화합니다.

    입력 예시:
        {"retrieval_metadata": {"evidence_id": "d1", "corp_name": "삼성전자"}}

    출력 예시:
        {"evidence_id": "d1", "corp_name": "삼성전자"}
    """

    retrieval_metadata = payload.get("retrieval_metadata")
    if not isinstance(retrieval_metadata, Mapping):
        raise ValueError("Qdrant payload.retrieval_metadata must be a mapping")
    reference = {
        str(key): parse_value(value)
        for key, value in retrieval_metadata.items()
    }
    evidence_id = reference.get("evidence_id")
    if isinstance(evidence_id, str):
        match = re.fullmatch(
            r"(?P<disclosure_id>d\d{14})"
            r"(?P<section_suffix>:src\d+:s\d+)"
            r":e\d+",
            evidence_id,
        )
        if match is not None:
            disclosure_id = match.group("disclosure_id")
            reference.setdefault("disclosure_id", disclosure_id)
            reference.setdefault(
                "section_id",
                disclosure_id + match.group("section_suffix"),
            )
    return reference


def _qdrant_point_base(
    point: QdrantPoint,
    *,
    item_type: str,
) -> dict[str, Any]:
    """모든 Qdrant item에 공통으로 들어갈 식별자와 검색 문맥을 생성합니다.

    입력 예시:
        _qdrant_point_base(point, item_type="text")

    출력 예시:
        {
            "type": "text",
            "point_id": "uuid",
            "score": 0.87,
            "reference": {"evidence_id": "d1", ...},
            "context": "삼성전자 사업보고서 ...",
        }
    """

    payload = _qdrant_payload(point)
    point_id = _qdrant_point_value(point, "id")
    if point_id is None:
        raise ValueError("Qdrant point id is required")

    contextual_text = payload.get("contextual_text")
    if not isinstance(contextual_text, str):
        raise ValueError("Qdrant payload.contextual_text must be a string")

    parsed = {
        "type": item_type,
        "point_id": str(point_id),
        "reference": _parse_qdrant_reference(payload),
        "context": contextual_text,
    }
    score = _qdrant_point_value(point, "score")
    if score is not None:
        parsed["score"] = float(score)
    return parsed


def parse_text_point(point: QdrantPoint) -> dict[str, Any]:
    """Qdrant TEXT point를 검색 문맥과 원문이 분리된 item으로 변환합니다.

    입력 예시:
        <ScoredPoint payload={"canonical": {"text": "신규 시설을 구축합니다."}, ...}>

    출력 예시:
        {
            "type": "text",
            "point_id": "uuid",
            "score": 0.91,
            "reference": {...},
            "context": "삼성전자 사업보고서 ...",
            "text": "신규 시설을 구축합니다.",
        }
    """

    payload = _qdrant_payload(point)
    canonical = payload.get("canonical")
    if not isinstance(canonical, Mapping):
        raise ValueError("Qdrant payload.canonical must be a mapping")
    text = canonical.get("text")
    if not isinstance(text, str):
        raise ValueError("TEXT canonical.text must be a string")

    return {
        **_qdrant_point_base(point, item_type="text"),
        "text": text,
    }


def parse_kv_table_point(point: QdrantPoint) -> dict[str, Any]:
    """Qdrant KV_TABLE point를 table metadata와 key-value entry로 변환합니다.

    입력 예시:
        <ScoredPoint payload={"canonical": {"entries": [{"key": "자산", "value": "100"}]}, ...}>

    출력 예시:
        {
            "type": "kv_table",
            "table_metadata": {"title": "재무 현황"},
            "entries": [{"key": "자산", "value": "100"}],
            ...,
        }
    """

    payload = _qdrant_payload(point)
    canonical = payload.get("canonical")
    if not isinstance(canonical, Mapping):
        raise ValueError("Qdrant payload.canonical must be a mapping")
    entries = canonical.get("entries")
    if not isinstance(entries, list):
        raise ValueError("KV_TABLE canonical.entries must be a list")

    parsed_entries: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"KV_TABLE canonical.entries[{index}] must be a mapping")
        parsed_entries.append(
            {
                str(key): parse_value(value)
                for key, value in entry.items()
            }
        )

    parsed = {
        **_qdrant_point_base(point, item_type="kv_table"),
        "entries": parsed_entries,
    }
    table_metadata = canonical.get("table_metadata")
    if table_metadata is not None:
        if not isinstance(table_metadata, Mapping):
            raise ValueError("KV_TABLE canonical.table_metadata must be a mapping")
        parsed["table_metadata"] = parse_aggregate(table_metadata)
    return parsed


def parse_r_table_headers(headers: Any) -> list[str]:
    """R_TABLE의 column path 배열을 중복 없는 LLM용 column 이름으로 변환합니다.

    입력 예시:
        [["구분"], ["금액", "당기"], ["금액", "당기"]]

    출력 예시:
        ["구분", "금액 > 당기", "금액 > 당기 [2]"]
    """

    if not isinstance(headers, list):
        raise ValueError("R_TABLE canonical.headers must be a list")

    parsed_headers: list[str] = []
    occurrences: dict[str, int] = {}
    for index, header in enumerate(headers):
        if not isinstance(header, list) or not all(
            isinstance(part, str) for part in header
        ):
            raise ValueError(
                f"R_TABLE canonical.headers[{index}] must be a list of strings"
            )
        name = " > ".join(part.strip() for part in header if part.strip())
        name = name or f"column_{index}"
        occurrences[name] = occurrences.get(name, 0) + 1
        occurrence = occurrences[name]
        parsed_headers.append(name if occurrence == 1 else f"{name} [{occurrence}]")
    return parsed_headers


def parse_r_table_records(
    headers: Sequence[str],
    records: Any,
    *,
    record_indexes: Collection[int] | None = None,
) -> list[dict[str, Any]]:
    """R_TABLE record의 위치 기반 values를 column 이름 기반 dictionary로 변환합니다.

    입력 예시:
        headers=["성명", "관계"]
        records=[{"record_index": 0, "values": ["홍길동", "최대주주"]}]
        record_indexes={0}

    출력 예시:
        [{
            "record_index": 0,
            "values": {"성명": "홍길동", "관계": "최대주주"},
        }]
    """

    if not isinstance(records, list):
        raise ValueError("R_TABLE canonical.records must be a list")
    selected_indexes = set(record_indexes) if record_indexes is not None else None

    parsed_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"R_TABLE canonical.records[{index}] must be a mapping")
        record_index = record.get("record_index")
        if not isinstance(record_index, int) or isinstance(record_index, bool):
            raise ValueError(
                f"R_TABLE canonical.records[{index}].record_index must be an integer"
            )
        values = record.get("values")
        if not isinstance(values, list) or len(values) != len(headers):
            raise ValueError(
                f"R_TABLE canonical.records[{index}].values must match header count"
            )
        if selected_indexes is not None and record_index not in selected_indexes:
            continue
        parsed_records.append(
            {
                "record_index": record_index,
                "values": {
                    header: parse_value(value)
                    for header, value in zip(headers, values, strict=True)
                },
            }
        )
    return parsed_records


def parse_r_table_point(
    point: QdrantPoint,
    *,
    detail: RTableDetail = "summary",
    record_indexes: Collection[int] | None = None,
) -> dict[str, Any]:
    """Qdrant R_TABLE point를 summary 또는 선택된 record가 포함된 item으로 변환합니다.

    입력 예시:
        parse_r_table_point(point, detail="records", record_indexes={3, 7})

    출력 예시:
        {
            "type": "r_table",
            "columns": ["성명", "관계"],
            "scope": {"kind": "table"},
            "available_record_count": 100,
            "included_record_count": 2,
            "records": [...],
            ...,
        }
    """

    if detail not in {"summary", "records"}:
        raise ValueError("R_TABLE detail must be 'summary' or 'records'")
    if detail == "summary" and record_indexes is not None:
        raise ValueError("record_indexes requires detail='records'")

    payload = _qdrant_payload(point)
    canonical = payload.get("canonical")
    if not isinstance(canonical, Mapping):
        raise ValueError("Qdrant payload.canonical must be a mapping")

    headers = parse_r_table_headers(canonical.get("headers"))
    raw_records = canonical.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("R_TABLE canonical.records must be a list")

    reference = _parse_qdrant_reference(payload)
    row_start_index = reference.get("row_start_index")
    row_end_index = reference.get("row_end_index")
    scope = (
        {
            "kind": "row_group",
            "row_start_index": row_start_index,
            "row_end_index": row_end_index,
        }
        if row_start_index is not None or row_end_index is not None
        else {"kind": "table"}
    )

    parsed = {
        **_qdrant_point_base(point, item_type="r_table"),
        "columns": headers,
        "scope": scope,
        "available_record_count": len(raw_records),
    }
    table_metadata = canonical.get("table_metadata")
    if table_metadata is not None:
        if not isinstance(table_metadata, Mapping):
            raise ValueError("R_TABLE canonical.table_metadata must be a mapping")
        parsed["table_metadata"] = parse_aggregate(table_metadata)

    if detail == "records":
        parsed_records = parse_r_table_records(
            headers,
            raw_records,
            record_indexes=record_indexes,
        )
        parsed["records"] = parsed_records
        parsed["included_record_count"] = len(parsed_records)
        parsed["omitted_record_count"] = len(raw_records) - len(parsed_records)
    return parsed


def parse_qdrant_point(
    point: QdrantPoint,
    *,
    r_table_detail: RTableDetail = "summary",
    r_table_record_indexes: Collection[int] | None = None,
) -> dict[str, Any]:
    """retrieval_metadata.point_kind에 따라 Qdrant point parser를 선택합니다.

    입력 예시:
        parse_qdrant_point(point, r_table_detail="summary")

    출력 예시:
        TEXT는 {"type": "text", ...}, KV_TABLE은 {"type": "kv_table", ...},
        R_TABLE은 {"type": "r_table", ...} 형태로 반환됩니다.
    """

    payload = _qdrant_payload(point)
    reference = _parse_qdrant_reference(payload)
    point_kind = reference.get("point_kind")
    if point_kind == "TEXT":
        return parse_text_point(point)
    if point_kind == "KV_TABLE":
        return parse_kv_table_point(point)
    if point_kind == "R_TABLE":
        return parse_r_table_point(
            point,
            detail=r_table_detail,
            record_indexes=r_table_record_indexes,
        )
    raise ValueError(f"Unsupported Qdrant point_kind: {point_kind}")


def _extract_qdrant_points(response: Any) -> Iterable[QdrantPoint]:
    """QueryResponse, point iterable 또는 scroll 응답에서 point 목록을 꺼냅니다.

    입력 예시:
        QueryResponse(points=[point1, point2]) 또는 ([point1, point2], next_offset)

    출력 예시:
        [point1, point2]
    """

    if isinstance(response, QueryResponse):
        return response.points
    points = getattr(response, "points", None)
    if points is not None:
        return points
    if (
        isinstance(response, tuple)
        and len(response) == 2
        and isinstance(response[0], list)
    ):
        return response[0]
    return response


def parse_qdrant_response(
    response: Any,
    *,
    plan_id: str,
    query: str,
    r_table_detail: RTableDetail = "summary",
    r_table_record_indexes: Mapping[str, Collection[int]] | None = None,
    metadata: Mapping[str, Any] | None = None,
    result_id: str | None = None,
) -> RetrievalResult:
    """Qdrant 응답 전체를 Plan 하나에 대응하는 RetrievalResult로 변환합니다.

    기본값은 큰 R_TABLE records를 제외하는 ``summary`` mode입니다.
    ``records`` mode에서는 table_id별 record index 선택을 전달할 수 있습니다.

    입력 예시:
        parse_qdrant_response(
            response,
            plan_id="plan_2",
            query="삼성전자 특별관계자",
            r_table_detail="records",
            r_table_record_indexes={"table-1": {3, 7}},
        )

    출력 예시:
        RetrievalResult(
            result_id="retrieval:plan_2",
            plan_id="plan_2",
            source="qdrant",
            query="삼성전자 특별관계자",
            items=[{"type": "r_table", "records": [...], ...}],
            result_count=1,
            metadata={"r_table_detail": "records"},
        )
    """

    if r_table_detail not in {"summary", "records"}:
        raise ValueError("R_TABLE detail must be 'summary' or 'records'")

    points = list(_extract_qdrant_points(response))
    items: list[dict[str, Any]] = []
    for point in points:
        payload = _qdrant_payload(point)
        reference = _parse_qdrant_reference(payload)
        table_id = reference.get("table_id")
        selected_indexes = (
            r_table_record_indexes.get(str(table_id))
            if r_table_record_indexes is not None and table_id is not None
            else None
        )
        items.append(
            parse_qdrant_point(
                point,
                r_table_detail=r_table_detail,
                r_table_record_indexes=selected_indexes,
            )
        )

    result_metadata = {
        str(key): parse_value(value)
        for key, value in (metadata or {}).items()
    }
    result_metadata.update(
        {
            "returned_point_count": len(points),
            "r_table_detail": r_table_detail,
        }
    )
    return RetrievalResult(
        result_id=result_id or f"retrieval:{plan_id}",
        plan_id=plan_id,
        source="qdrant",
        query=query,
        items=items,
        result_count=len(items),
        metadata=result_metadata,
    )
