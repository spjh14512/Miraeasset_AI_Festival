from __future__ import annotations

import json

import pytest
from qdrant_client.http.models import ScoredPoint

from agent_graph.compactor import (
    CompactorOutput,
    compact_qdrant_point,
    point_requires_compaction,
)
from agent_graph.state import Plan


class _CompactorLlm:
    def __init__(self, item_ids: list[int]):
        self.item_ids = item_ids
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return CompactorOutput(item_ids=self.item_ids)


def _point(point_kind: str, canonical: dict) -> ScoredPoint:
    return ScoredPoint(
        id="00000000-0000-0000-0000-000000000001",
        version=1,
        score=0.9,
        payload={
            "point_kind": point_kind,
            "contextual_text": (
                "회사 : 삼성전자\n"
                "공시 : 사업보고서 (2023.12)\n"
                "섹션 : VII. 주주에 관한 사항\n\n"
                "검색 내용"
            ),
            "canonical": canonical,
        },
    )


def _plan() -> Plan:
    return Plan(
        plan_id="plan_1",
        source="qdrant",
        query="최대주주인 특별관계자",
        purpose="특별관계자 성명 확인",
        dependencies=[],
    )


def test_only_long_table_points_require_compaction():
    kv_point = _point(
        "KV_TABLE",
        {"entries": [{"key": "유동자산", "value": "100"}]},
    )
    text_point = _point("TEXT", {"text": "긴 본문" * 100})

    assert point_requires_compaction(
        kv_point,
        content_character_limit=1,
    )
    assert not point_requires_compaction(
        kv_point,
        content_character_limit=10_000,
    )
    assert not point_requires_compaction(
        text_point,
        content_character_limit=1,
    )


def test_compactor_selects_kv_entry_indexes():
    point = _point(
        "KV_TABLE",
        {
            "entries": [
                {"key": "유동자산", "value": "100"},
                {"key": "비유동자산", "value": "200"},
            ]
        },
    )
    llm = _CompactorLlm([1])

    assert compact_qdrant_point(point, _plan(), llm=llm) == [1]
    payload = json.loads(llm.messages[1].content)
    assert payload["items"] == [
        {"item_id": 0, "key": "유동자산", "value": "100"},
        {"item_id": 1, "key": "비유동자산", "value": "200"},
    ]


def test_compactor_preserves_r_table_source_order():
    point = _point(
        "R_TABLE",
        {
            "headers": [["성명"], ["관계"]],
            "records": [
                {"record_index": 10, "values": ["홍길동", "최대주주"]},
                {"record_index": 11, "values": ["김영희", "임원"]},
            ],
        },
    )

    assert compact_qdrant_point(
        point,
        _plan(),
        llm=_CompactorLlm([11, 10]),
    ) == [10, 11]


def test_compactor_rejects_unknown_item_id():
    point = _point(
        "KV_TABLE",
        {"entries": [{"key": "유동자산", "value": "100"}]},
    )

    with pytest.raises(ValueError, match="point에 없는 item ID"):
        compact_qdrant_point(point, _plan(), llm=_CompactorLlm([99]))


def test_compactor_output_rejects_duplicate_item_ids():
    with pytest.raises(ValueError, match="중복"):
        CompactorOutput(item_ids=[1, 1])
