from __future__ import annotations

from copy import deepcopy

import pytest

from vector_db.r_table_strategy_selector import (
    RTableEmbeddingStrategy,
    RTableEmbeddingStrategySelector,
    RTableStrategyConfig,
    calculate_r_table_features,
    load_r_table_strategy_config,
)


TABLE_ID = "rtable:20250318000123:src0:s12:t0"


def _evidence(headers, record_count):
    return {
        "evidence_id": "evidence:20250318000123:src0:s12:e0",
        "evidence_type": "TABLE",
        "table_type": "R_TABLE",
        "storage_mode": "SECTION_RECORDS",
        "payload": {
            "table_id": TABLE_ID,
            "headers": headers,
            "record_count": record_count,
        },
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


def _config(
    *,
    whole_table_max_tokens=1500,
    row_group_max_tokens=1500,
    long_text_min_characters=40,
    semantic_text_ratio_threshold=0.2,
):
    return RTableStrategyConfig(
        whole_table_max_tokens=whole_table_max_tokens,
        row_group_max_tokens=row_group_max_tokens,
        long_text_min_characters=long_text_min_characters,
        semantic_text_ratio_threshold=semantic_text_ratio_threshold,
    )


def test_short_numeric_table_selects_whole_table():
    evidence = _evidence(
        [["품목"], ["2025년", "매출"], ["비중"], ["기준일"]],
        2,
    )
    records = _records(
        [
            ["DRAM", "1,500", "35%", "2025-12-31"],
            ["NAND", "950", "22%", "2025-12-31"],
        ]
    )
    selector = RTableEmbeddingStrategySelector(
        config=_config(),
        token_counter=lambda text: 100,
    )

    decision = selector.select(evidence, records, "short numeric table")

    assert decision.strategy is RTableEmbeddingStrategy.WHOLE_TABLE
    assert decision.reason == "within_whole_table_token_limit"
    assert decision.features.numeric_cell_ratio == pytest.approx(4 / 8)
    assert decision.features.date_cell_ratio == pytest.approx(2 / 8)
    assert decision.features.column_count == 4


def test_short_long_text_table_still_selects_whole_table():
    long_text = "차세대 고대역폭 메모리 제품 개발을 추진하고 시장 수요에 대응합니다."
    evidence = _evidence([["구분"], ["주요 내용"]], 2)
    records = _records([["HBM4", long_text], ["AI 서버", long_text]])
    selector = RTableEmbeddingStrategySelector(
        config=_config(long_text_min_characters=20),
        token_counter=lambda text: 1499,
    )

    decision = selector.select(evidence, records, "short semantic table")

    assert decision.strategy is RTableEmbeddingStrategy.WHOLE_TABLE
    assert decision.features.long_text_cell_ratio == pytest.approx(0.5)


def test_oversized_numeric_table_selects_descriptor():
    evidence = _evidence([["품목"], ["매출"], ["비중"]], 2)
    records = _records([["DRAM", "1,500", "35%"], ["NAND", "950", "22%"]])
    selector = RTableEmbeddingStrategySelector(
        config=_config(),
        token_counter=lambda text: 1501,
    )

    decision = selector.select(evidence, records, "oversized numeric table")

    assert decision.strategy is RTableEmbeddingStrategy.DESCRIPTOR
    assert decision.reason == "structured_oversized"


def test_oversized_long_text_table_selects_row_group():
    long_text = "차세대 고대역폭 메모리 제품 개발을 추진하고 시장 수요에 대응합니다."
    evidence = _evidence([["구분"], ["주요 내용"]], 2)
    records = _records([["HBM4", long_text], ["AI 서버", long_text]])
    selector = RTableEmbeddingStrategySelector(
        config=_config(
            long_text_min_characters=20,
            semantic_text_ratio_threshold=0.4,
        ),
        token_counter=lambda text: 1501,
    )

    decision = selector.select(evidence, records, "oversized semantic table")

    assert decision.strategy is RTableEmbeddingStrategy.ROW_GROUP
    assert decision.reason == "semantic_oversized"


def test_empty_cells_are_excluded_but_explicit_values_are_preserved():
    evidence = _evidence(
        [["구분"], ["금액", "당기"], ["기준일"]],
        2,
    )
    records = _records(
        [
            ["-", "", "2025-12-31"],
            ["해당없음", "0", None],
        ]
    )

    features = calculate_r_table_features(
        evidence,
        records,
        "table context",
        config=_config(),
        token_counter=lambda text: 10,
    )

    assert features.row_count == 2
    assert features.column_count == 3
    assert features.non_empty_cell_count == 4
    assert features.numeric_cell_ratio == pytest.approx(0.25)
    assert features.date_cell_ratio == pytest.approx(0.25)
    assert features.long_text_cell_ratio == 0
    assert features.max_cell_length == len("2025-12-31")


def test_config_threshold_changes_the_selected_strategy():
    evidence = _evidence([["구분"], ["금액"]], 1)
    records = _records([["합계", "100"]])

    whole_table = RTableEmbeddingStrategySelector(
        config=_config(whole_table_max_tokens=100),
        token_counter=lambda text: 100,
    ).select(evidence, records, "same table")
    descriptor = RTableEmbeddingStrategySelector(
        config=_config(whole_table_max_tokens=99),
        token_counter=lambda text: 100,
    ).select(evidence, records, "same table")

    assert whole_table.strategy is RTableEmbeddingStrategy.WHOLE_TABLE
    assert descriptor.strategy is RTableEmbeddingStrategy.DESCRIPTOR


def test_decision_can_be_serialized_for_future_analysis():
    evidence = _evidence([["구분"], ["금액"]], 1)
    records = _records([["합계", "100"]])
    decision = RTableEmbeddingStrategySelector(
        config=_config(),
        token_counter=lambda text: 42,
    ).select(evidence, records, "table context")

    serialized = decision.to_dict()

    assert serialized["embedding_strategy"] == "whole_table"
    assert serialized["embedding_token_count"] == 42
    assert serialized["embedding_strategy_reason"] == (
        "within_whole_table_token_limit"
    )
    assert serialized["features"]["row_count"] == 1
    assert "token_count" not in serialized["features"]


def test_loads_default_thresholds_from_yaml():
    config = load_r_table_strategy_config()

    assert config == _config()


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
    ],
)
def test_rejects_invalid_table_structure(mutation, message):
    evidence = _evidence([["구분"], ["금액"]], 1)
    records = _records([["합계", "100"]])
    mutation(evidence, records)
    if message == "record_index must be unique":
        evidence["payload"]["record_count"] = 2

    with pytest.raises(ValueError, match=message):
        calculate_r_table_features(
            evidence,
            records,
            "table context",
            config=_config(),
            token_counter=lambda text: 10,
        )


@pytest.mark.parametrize("token_count", [-1, 1.5, True])
def test_rejects_invalid_token_counter_result(token_count):
    evidence = _evidence([["구분"]], 1)
    records = _records([["합계"]])

    with pytest.raises(ValueError, match="non-negative integer"):
        calculate_r_table_features(
            evidence,
            records,
            "table context",
            config=_config(),
            token_counter=lambda text: token_count,
        )
