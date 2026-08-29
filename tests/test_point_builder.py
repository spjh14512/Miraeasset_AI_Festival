from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from uuid import uuid5

import pytest

from vector_db.point_builder import (
    POINT_ID_NAMESPACE,
    assemble_qdrant_points,
    build_point_inputs,
    build_qdrant_points,
    embed_point_inputs,
    to_neo4j_evidence_id,
)
from vector_db.r_table_strategy_selector import (
    RTableEmbeddingStrategySelector,
    RTableStrategyConfig,
)
from vector_db.text2vector import HybridEmbedding, SparseEmbedding


TABLE_ID = "rtable:20250318000123:src0:s12:t0"
EVIDENCE_ID = "evidence:20250318000123:src0:s12:e2"
POINT_PAYLOAD_FIELDS = {
    "point_kind",
    "disclosure_id",
    "section_id",
    "evidence_id",
    "corp_name",
    "report_name",
    "section_name",
    "rcept_date",
    "is_latest_version",
    "contextual_text",
    "canonical",
}

DOCUMENT_CONTEXT = {
    "corp_name": "삼성전자",
    "corp_code": "00126380",
    "industry": "IT",
    "sector": "반도체와 반도체장비",
    "report_nm": "2025년 사업보고서",
    "rcept_date": "20250318",
    "is_latest_version": True,
}

SECTION_CONTEXT = {
    "section_id": "section:20250318000123:src0:s12",
    "section_path": ["사업의 내용", "시설 및 설비"],
}


def _hybrid(*dense: float) -> HybridEmbedding:
    return HybridEmbedding(
        dense=tuple(dense),
        sparse=SparseEmbedding(indices=(1, 42), values=(0.2, 0.8)),
    )


def _fragment(
    rows: list[list[str | None]] | None = None,
    *,
    headers: list[list[str]] | None = None,
) -> dict:
    resolved_rows = (
        [["평택", "100"], ["화성", "200"]] if rows is None else rows
    )
    resolved_headers = (
        [["사업장"], ["투자 금액"]] if headers is None else headers
    )
    return {
        "schema_version": "evidence-fragment.v2",
        "section_id": "section:20250318000123:src0:s12",
        "evidence_list": [
            {
                "evidence_id": "evidence:20250318000123:src0:s12:e0",
                "evidence_type": "TEXT",
                "order": 0,
                "payload": {"text": "신규 생산시설을 구축할 예정입니다."},
            },
            {
                "evidence_id": "evidence:20250318000123:src0:s12:e1",
                "evidence_type": "TABLE",
                "table_type": "KV_TABLE",
                "order": 1,
                "payload": {
                    "title": "시설 개요",
                    "fields": [
                        {"key_paths": [["사업장"]], "raw_value": "평택"}
                    ],
                },
            },
            {
                "evidence_id": EVIDENCE_ID,
                "evidence_type": "TABLE",
                "table_type": "R_TABLE",
                "storage_mode": "SECTION_RECORDS",
                "order": 2,
                "payload": {
                    "table_id": TABLE_ID,
                    "title": "투자 현황",
                    "headers": resolved_headers,
                    "record_count": len(resolved_rows),
                },
            },
        ],
        "records": [
            {
                "table_id": TABLE_ID,
                "record_index": index,
                "row_type": "DATA",
                "row_context": [],
                "values": values,
            }
            for index, values in enumerate(resolved_rows)
        ],
    }


def _selector(
    token_counter,
    *,
    whole_table_max_tokens: int = 1500,
    row_group_max_tokens: int = 1500,
    canonical_max_bytes: int = 32 * 1024,
    canonical_hard_max_bytes: int = 64 * 1024,
    long_text_min_characters: int = 40,
    semantic_text_ratio_threshold: float = 0.2,
) -> RTableEmbeddingStrategySelector:
    return RTableEmbeddingStrategySelector(
        config=RTableStrategyConfig(
            whole_table_max_tokens=whole_table_max_tokens,
            row_group_max_tokens=row_group_max_tokens,
            canonical_max_bytes=canonical_max_bytes,
            canonical_hard_max_bytes=canonical_hard_max_bytes,
            long_text_min_characters=long_text_min_characters,
            semantic_text_ratio_threshold=semantic_text_ratio_threshold,
        ),
        token_counter=token_counter,
    )


def _whole_table_selector() -> RTableEmbeddingStrategySelector:
    return _selector(lambda text: 100)


def test_builds_text_kv_and_one_point_for_a_small_r_table():
    embedded_texts = []

    points = build_qdrant_points(
        _fragment(),
        document_context=DOCUMENT_CONTEXT,
        section_context=SECTION_CONTEXT,
        vectorizer=lambda text: embedded_texts.append(text) or _hybrid(0.1, 0.2),
        r_table_strategy_selector=_whole_table_selector(),
    )

    assert [
        point["payload"]["point_kind"] for point in points
    ] == ["TEXT", "KV_TABLE", "R_TABLE"]
    assert len(embedded_texts) == 3
    assert all(
        point["vector"] == {
            "evidence_dense": [0.1, 0.2],
            "evidence_sparse": {
                "indices": [1, 42],
                "values": [0.2, 0.8],
            },
        }
        for point in points
    )
    assert len({point["id"] for point in points}) == 3
    assert [
        point["payload"]["evidence_id"]
        for point in points
    ] == [
        "d20250318000123:src0:s12:e0",
        "d20250318000123:src0:s12:e1",
        "d20250318000123:src0:s12:e2",
    ]
    assert all(
        point["payload"]["disclosure_id"] == "d20250318000123"
        for point in points
    )
    assert all(
        point["payload"]["section_id"] == "d20250318000123:src0:s12"
        for point in points
    )
    assert "사업장 : 평택" in points[2]["payload"]["contextual_text"]
    assert "사업장 : 화성" in points[2]["payload"]["contextual_text"]


def test_payload_has_flat_retrieval_fields_and_minimal_canonical_data():
    points = build_qdrant_points(
        _fragment(),
        document_context=DOCUMENT_CONTEXT,
        section_context=SECTION_CONTEXT,
        vectorizer=lambda text: _hybrid(0.0),
        r_table_strategy_selector=_whole_table_selector(),
    )

    assert all(set(point["payload"]) == POINT_PAYLOAD_FIELDS for point in points)
    assert points[0]["payload"]["canonical"] == {
        "text": "신규 생산시설을 구축할 예정입니다."
    }
    assert points[1]["payload"]["canonical"] == {
        "table_metadata": {"title": "시설 개요"},
        "entries": [{"key": "사업장", "value": "평택"}],
    }
    assert points[2]["payload"]["canonical"] == {
        "table_metadata": {"title": "투자 현황"},
        "headers": [["사업장"], ["투자 금액"]],
        "records": [
            {"record_index": 0, "values": ["평택", "100"]},
            {"record_index": 1, "values": ["화성", "200"]},
        ],
    }
    assert all("chunking" not in point["payload"] for point in points)
    assert all(
        point["payload"]["section_name"] == "사업의 내용 > 시설 및 설비"
        for point in points
    )
    assert all(point["payload"]["rcept_date"] == "20250318" for point in points)
    assert all(point["payload"]["is_latest_version"] is True for point in points)


def test_payload_preserves_false_latest_version_flag():
    document_context = {**DOCUMENT_CONTEXT, "is_latest_version": False}

    points = build_point_inputs(
        _fragment(),
        document_context=document_context,
        section_context=SECTION_CONTEXT,
        r_table_strategy_selector=_whole_table_selector(),
    )

    assert all(point.payload["is_latest_version"] is False for point in points)


def test_payload_rejects_missing_latest_version_flag():
    document_context = dict(DOCUMENT_CONTEXT)
    document_context.pop("is_latest_version")

    with pytest.raises(ValueError, match="is_latest_version must be a boolean"):
        build_point_inputs(
            _fragment(),
            document_context=document_context,
            section_context=SECTION_CONTEXT,
            r_table_strategy_selector=_whole_table_selector(),
        )


def test_section_name_appends_evidence_heading_path():
    fragment = _fragment()
    fragment["evidence_list"][0]["payload"]["heading_path"] = [
        "시설 및 설비",
        "신규 투자",
    ]

    point_inputs = build_point_inputs(
        fragment,
        document_context=DOCUMENT_CONTEXT,
        section_context=SECTION_CONTEXT,
        r_table_strategy_selector=_whole_table_selector(),
    )

    assert point_inputs[0].payload["section_name"] == (
        "사업의 내용 > 시설 및 설비 > 신규 투자"
    )


def test_kv_canonical_preserves_key_value_pairs_and_omits_empty_metadata():
    fragment = _fragment()
    kv_payload = fragment["evidence_list"][1]["payload"]
    kv_payload.pop("title")
    kv_payload.update(
        {
            "captions": [],
            "units": [""],
            "notes": None,
            "fields": [
                {
                    "key_paths": [["당기", "금액"], ["현재", "금액"]],
                    "raw_value": "100",
                }
            ],
        }
    )

    point_inputs = build_point_inputs(
        fragment,
        document_context=DOCUMENT_CONTEXT,
        section_context=SECTION_CONTEXT,
        r_table_strategy_selector=_whole_table_selector(),
    )

    assert point_inputs[1].payload["canonical"] == {
        "entries": [
            {"key": "당기 > 금액", "value": "100"},
            {"key": "현재 > 금액", "value": "100"},
        ]
    }


def test_r_table_metadata_omits_fields_without_values():
    fragment = _fragment()
    r_payload = fragment["evidence_list"][2]["payload"]
    r_payload.update(
        {
            "title": " ",
            "captions": [],
            "units": ["백만원", ""],
            "notes": None,
        }
    )

    point_inputs = build_point_inputs(
        fragment,
        document_context=DOCUMENT_CONTEXT,
        section_context=SECTION_CONTEXT,
        r_table_strategy_selector=_whole_table_selector(),
    )

    assert point_inputs[2].payload["canonical"]["table_metadata"] == {
        "units": ["백만원"]
    }


def test_structured_oversized_r_table_builds_descriptor_with_all_records():
    point_inputs = build_point_inputs(
        _fragment(),
        document_context=DOCUMENT_CONTEXT,
        section_context=SECTION_CONTEXT,
        r_table_strategy_selector=_selector(lambda text: 1501),
    )

    descriptor = point_inputs[2]
    assert descriptor.payload["point_kind"] == "R_TABLE"
    assert "chunking" not in descriptor.payload
    assert descriptor.payload["canonical"]["records"] == [
        {"record_index": 0, "values": ["평택", "100"]},
        {"record_index": 1, "values": ["화성", "200"]},
    ]
    assert "투자 현황" in descriptor.contextual_text
    assert "사업장" in descriptor.contextual_text
    assert "투자 금액" in descriptor.contextual_text
    assert descriptor.id == str(uuid5(POINT_ID_NAMESPACE, f"{EVIDENCE_ID}:descriptor"))


def test_large_structured_r_table_builds_chunk_specific_descriptors():
    rows = [
        [f"사업부문-{index}", str(index + 1) * 140]
        for index in range(4)
    ]
    point_inputs = build_point_inputs(
        _fragment(rows, headers=[["구분"], ["금액"]]),
        document_context=DOCUMENT_CONTEXT,
        section_context=SECTION_CONTEXT,
        r_table_strategy_selector=_selector(
            lambda text: 1501,
            canonical_max_bytes=360,
        ),
    )

    chunks = point_inputs[2:]
    assert len(chunks) == 4
    assert [
        item.payload["chunking"]["chunk_index"] for item in chunks
    ] == [0, 1, 2, 3]
    assert all(
        item.payload["chunking"]["chunk_count"] == 4
        for item in chunks
    )
    assert all(
        item.payload["chunking"]["table_id"] == TABLE_ID
        for item in chunks
    )
    assert [
        (
            item.payload["chunking"]["row_start_index"],
            item.payload["chunking"]["row_end_index"],
        )
        for item in chunks
    ] == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert [
        record["record_index"]
        for item in chunks
        for record in item.payload["canonical"]["records"]
    ] == [0, 1, 2, 3]

    for index, item in enumerate(chunks):
        assert f"구분 : 사업부문-{index}" in item.contextual_text
        assert rows[index][1] not in item.contextual_text
        assert len(
            json.dumps(
                item.payload["canonical"],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ) <= 360
    assert len({item.embedding_cache_key for item in chunks}) == 4


def test_canonical_chunks_use_minimum_count_with_balanced_sizes():
    rows = [["공통 항목", "100"] for _ in range(10)]
    fragment = _fragment(rows, headers=[["구분"], ["금액"]])
    r_payload = fragment["evidence_list"][2]["payload"]
    four_record_canonical = {
        "table_metadata": {"title": r_payload["title"]},
        "headers": r_payload["headers"],
        "records": [
            {"record_index": index, "values": rows[index]}
            for index in range(4)
        ],
    }
    four_record_size = len(
        json.dumps(
            four_record_canonical,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    point_inputs = build_point_inputs(
        fragment,
        document_context=DOCUMENT_CONTEXT,
        section_context=SECTION_CONTEXT,
        r_table_strategy_selector=_selector(
            lambda text: 1501,
            canonical_max_bytes=four_record_size,
        ),
    )

    chunks = point_inputs[2:]
    assert [
        len(item.payload["canonical"]["records"]) for item in chunks
    ] == [3, 4, 3]
    assert [
        record["record_index"]
        for item in chunks
        for record in item.payload["canonical"]["records"]
    ] == list(range(10))
    assert all(
        len(
            json.dumps(
                item.payload["canonical"],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        <= four_record_size
        for item in chunks
    )


def test_header_heavy_table_uses_one_indivisible_fallback_point():
    rows = [["항목-0", "100"], ["항목-1", "200"], ["항목-2", "300"]]
    fragment = _fragment(
        rows,
        headers=[["구분"], ["매우 긴 계층형 헤더 " * 20]],
    )

    point_inputs = build_point_inputs(
        fragment,
        document_context=DOCUMENT_CONTEXT,
        section_context=SECTION_CONTEXT,
        r_table_strategy_selector=_selector(
            lambda text: 1501,
            canonical_max_bytes=200,
            canonical_hard_max_bytes=4096,
        ),
    )

    replacement = point_inputs[2]
    canonical_size = len(
        json.dumps(
            replacement.payload["canonical"],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert len(point_inputs[2:]) == 1
    assert 200 < canonical_size <= 4096
    assert "chunking" not in replacement.payload


def test_semantic_oversized_r_table_stores_only_each_row_group_records():
    long_text = "생산시설 증설 계획과 공정 전환 일정에 관한 상세 설명입니다. " * 2
    fragment = _fragment(
        [[f"사업장-{index}", f"{long_text}{index}"] for index in range(4)],
        headers=[["사업장"], ["설명"]],
    )

    def row_weighted_counter(text: str) -> int:
        return 300 + (500 * text.count("사업장 :"))

    point_inputs = build_point_inputs(
        fragment,
        document_context=DOCUMENT_CONTEXT,
        section_context=SECTION_CONTEXT,
        r_table_strategy_selector=_selector(row_weighted_counter),
    )

    row_groups = point_inputs[2:]
    assert len(row_groups) == 2
    assert [
        (
            item.payload["chunking"]["row_start_index"],
            item.payload["chunking"]["row_end_index"],
        )
        for item in row_groups
    ] == [(0, 1), (2, 3)]
    assert all(
        item.payload["chunking"]["table_id"] == TABLE_ID
        for item in row_groups
    )
    assert [
        [record["record_index"] for record in item.payload["canonical"]["records"]]
        for item in row_groups
    ] == [[0, 1], [2, 3]]

    combined_text = "\n".join(item.contextual_text for item in row_groups)
    for index in range(4):
        assert combined_text.count(f"사업장 : 사업장-{index}") == 1
    assert [item.id for item in row_groups] == [
        str(uuid5(POINT_ID_NAMESPACE, f"{EVIDENCE_ID}:row_group:0:1")),
        str(uuid5(POINT_ID_NAMESPACE, f"{EVIDENCE_ID}:row_group:2:3")),
    ]
    assert [
        item.payload["chunking"]["chunk_index"]
        for item in row_groups
    ] == [0, 1]
    assert all(
        item.payload["chunking"]["chunk_count"] == 2
        for item in row_groups
    )


def test_point_ids_are_deterministic_and_evidence_specific():
    kwargs = {
        "document_context": DOCUMENT_CONTEXT,
        "section_context": SECTION_CONTEXT,
        "vectorizer": lambda text: _hybrid(0.0),
        "r_table_strategy_selector": _whole_table_selector(),
    }

    first = build_qdrant_points(_fragment(), **kwargs)
    second = build_qdrant_points(_fragment(), **kwargs)

    assert [point["id"] for point in first] == [point["id"] for point in second]
    assert len(first) == 3


def test_converts_source_evidence_id_to_neo4j_evidence_id():
    assert to_neo4j_evidence_id(
        "evidence:20241227000631:src0:s18:e361"
    ) == "d20241227000631:src0:s18:e361"


@pytest.mark.parametrize("source_evidence_id", [None, "", "section:123", "evidence:"])
def test_rejects_invalid_source_evidence_id(source_evidence_id):
    with pytest.raises(ValueError, match="source_evidence_id"):
        to_neo4j_evidence_id(source_evidence_id)


def test_embedding_and_point_assembly_are_separate_and_cache_ready():
    point_inputs = build_point_inputs(
        _fragment(),
        document_context=DOCUMENT_CONTEXT,
        section_context=SECTION_CONTEXT,
        r_table_strategy_selector=_whole_table_selector(),
    )
    duplicate_input = replace(point_inputs[0], id="another-point-id")
    calls = []

    embeddings = embed_point_inputs(
        [point_inputs[0], duplicate_input],
        vectorizer=lambda text: calls.append(text) or _hybrid(0.5),
    )
    points = assemble_qdrant_points(
        [point_inputs[0], duplicate_input],
        embeddings,
    )

    assert len(calls) == 1
    assert point_inputs[0].embedding_cache_key == duplicate_input.embedding_cache_key
    assert [point["vector"] for point in points] == [
        {
            "evidence_dense": [0.5],
            "evidence_sparse": {"indices": [1, 42], "values": [0.2, 0.8]},
        },
        {
            "evidence_dense": [0.5],
            "evidence_sparse": {"indices": [1, 42], "values": [0.2, 0.8]},
        },
    ]


def test_rejects_mismatched_section_context():
    section = {**SECTION_CONTEXT, "section_id": "section:other"}

    with pytest.raises(ValueError, match="section_id must match"):
        build_qdrant_points(
            _fragment(),
            document_context=DOCUMENT_CONTEXT,
            section_context=section,
            vectorizer=lambda text: _hybrid(0.0),
            r_table_strategy_selector=_whole_table_selector(),
        )


def test_rejects_orphan_records():
    fragment = deepcopy(_fragment())
    fragment["records"].append(
        {
            "table_id": "rtable:orphan",
            "record_index": 0,
            "row_type": "DATA",
            "row_context": [],
            "values": ["orphan"],
        }
    )

    with pytest.raises(ValueError, match="without matching R_TABLE"):
        build_qdrant_points(
            fragment,
            document_context=DOCUMENT_CONTEXT,
            section_context=SECTION_CONTEXT,
            vectorizer=lambda text: _hybrid(0.0),
            r_table_strategy_selector=_whole_table_selector(),
        )
