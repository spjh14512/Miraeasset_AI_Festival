from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from agent_graph.state import RetrievalResult
from agent_graph.tools import CalculationOperation, TableTarget, calculate_table_statistic


def test_table_target_accepts_valid_reference():
    target = TableTarget(result_id="retrieval:plan_1", item_index=0)

    assert target.result_id == "retrieval:plan_1"
    assert target.item_index == 0


def test_table_target_rejects_blank_result_id():
    with pytest.raises(ValidationError):
        TableTarget(result_id="", item_index=0)


def test_table_target_rejects_whitespace_only_result_id():
    with pytest.raises(ValidationError):
        TableTarget(result_id="   ", item_index=0)


def test_table_target_strips_result_id_whitespace():
    target = TableTarget(result_id="  retrieval:plan_1  ", item_index=0)

    assert target.result_id == "retrieval:plan_1"


def test_table_target_rejects_negative_item_index():
    with pytest.raises(ValidationError):
        TableTarget(result_id="retrieval:plan_1", item_index=-1)


def test_table_target_rejects_bool_item_index():
    with pytest.raises(ValidationError):
        TableTarget(result_id="retrieval:plan_1", item_index=True)


def test_table_target_rejects_string_item_index():
    with pytest.raises(ValidationError):
        TableTarget(result_id="retrieval:plan_1", item_index="2")


def test_calculation_operation_accepts_supported_values():
    class _Probe(BaseModel):
        operation: CalculationOperation

    for operation in ("sum", "mean", "median", "max", "min", "mode"):
        assert _Probe(operation=operation).operation == operation


def test_calculation_operation_rejects_unsupported_value():
    class _Probe(BaseModel):
        operation: CalculationOperation

    with pytest.raises(ValidationError):
        _Probe(operation="stddev")


def _r_table_item(
    *,
    values: list[str | None],
    column: str = "금액",
    disclosure_id: str | None = "d20240101000001",
    section_id: str = "d20240101000001:src0:s1",
    evidence_id: str = "d20240101000001:src0:s1:e1",
    chunk: tuple[str, int, int, int, int] | None = None,
    table_units: list[str] | None = None,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """calculate_table_statistic 테스트용 R_TABLE item을 만듭니다.

    chunk가 주어지면 (table_id, chunk_index, chunk_count, row_start,
    row_end) 형태로 청킹된 item을, 아니면 청킹되지 않은 전체 표 item을
    만듭니다. records를 직접 넘기지 않으면 values 길이와 일치하는
    정상적인 record를 자동 생성합니다. disclosure_id=None이면 인용
    정보가 없는 item(citation_check 실패 케이스)을 만듭니다.
    """

    resolved_records = (
        [
            {"record_index": index, "values": {"구분": f"row{index}", column: value}}
            for index, value in enumerate(values)
        ]
        if records is None
        else records
    )
    metadata: dict[str, Any] = {}
    if disclosure_id is not None:
        metadata.update(
            {
                "disclosure_id": disclosure_id,
                "section_id": section_id,
                "evidence_id": evidence_id,
            }
        )
    if chunk is None:
        scope = {"kind": "table"}
    else:
        table_id, chunk_index, chunk_count, row_start, row_end = chunk
        scope = {
            "kind": "row_group",
            "row_start_index": row_start,
            "row_end_index": row_end,
        }
        metadata.update(
            {
                "table_id": table_id,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "row_start_index": row_start,
                "row_end_index": row_end,
            }
        )
    item: dict[str, Any] = {
        "type": "r_table",
        "columns": ["구분", column],
        "scope": scope,
        "records": resolved_records,
        "available_record_count": len(resolved_records),
        "included_record_count": len(resolved_records),
        "omitted_record_count": 0,
        "metadata": metadata,
    }
    if table_units is not None:
        item["table_metadata"] = {"units": table_units}
    return item


def _retrieval_result(
    result_id: str,
    items: list[dict[str, Any]],
    *,
    status: str = "SUCCESS",
) -> RetrievalResult:
    return RetrievalResult(
        result_id=result_id,
        plan_id=result_id.removeprefix("retrieval:"),
        source="qdrant",
        status=status,
        query="검색 쿼리",
        items=items,
        result_count=len(items),
    )


def _state(*results: RetrievalResult, next_plan_seq: int = 1) -> dict[str, Any]:
    return {
        "question_id": "question-1",
        "question_text": "매출액 합계를 알려줘",
        "next_plan_seq": next_plan_seq,
        "retrieval_results": list(results),
    }


def _invoke(**kwargs: Any) -> dict:
    return calculate_table_statistic.invoke(kwargs)


def test_sum_of_whole_table_column_succeeds():
    item = _r_table_item(values=["1,200", "800"])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "SUCCESS"
    assert result.source == "derived"
    assert result.result_id == "derived:plan_1"
    assert update["next_plan_seq"] == 2
    fields = result.items[0]["fields"]
    assert fields["value"] == "2000"
    assert fields["operation"] == "sum"
    assert fields["input_count"] == 2
    assert result.metadata["result_kind"] == "numeric_scalar"


def test_mean_of_whole_table_column_succeeds():
    item = _r_table_item(values=["10", "20", "30"])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="평균",
        operation="mean",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    assert update["retrieval_results"][0].items[0]["fields"]["value"] == "20"


def test_median_of_whole_table_column_succeeds():
    item = _r_table_item(values=["1", "2", "100"])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="중앙값",
        operation="median",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    assert update["retrieval_results"][0].items[0]["fields"]["value"] == "2"


def test_max_and_min_succeed():
    item = _r_table_item(values=["5", "9", "1"])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    max_update = _invoke(
        variable_name="최댓값",
        operation="max",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )
    min_update = _invoke(
        variable_name="최솟값",
        operation="min",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    assert max_update["retrieval_results"][0].items[0]["fields"]["value"] == "9"
    assert min_update["retrieval_results"][0].items[0]["fields"]["value"] == "1"


def test_mode_with_single_winner_succeeds():
    item = _r_table_item(values=["3", "3", "5"])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="최빈값",
        operation="mode",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    assert update["retrieval_results"][0].items[0]["fields"]["value"] == "3"


def test_mode_with_tie_is_invalid_input():
    item = _r_table_item(values=["3", "3", "5", "5"])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="최빈값",
        operation="mode",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "calculation"


def test_missing_values_are_excluded_but_calculation_still_succeeds():
    item = _r_table_item(values=["100", "-", "", None, "200"])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "SUCCESS"
    assert result.items[0]["fields"]["value"] == "300"
    assert result.items[0]["fields"]["input_count"] == 2


def test_non_numeric_value_makes_invalid_input():
    item = _r_table_item(values=["100", "해당없음"])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "numeric_parsing"
    assert result.items == []


def test_mixed_units_makes_invalid_input():
    item = _r_table_item(values=["100%", "200 백만원"])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "unit_check"


def test_all_missing_values_makes_invalid_input():
    item = _r_table_item(values=["-", "", None])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "numeric_parsing"


def test_incomplete_table_makes_invalid_input_without_touching_values():
    # chunk_count=2인데 chunk 하나만 검색된 상태.
    item = _r_table_item(
        values=["100"],
        chunk=("table-1", 0, 2, 0, 0),
    )
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "completeness_check"


def test_two_chunks_of_same_table_are_combined():
    chunk_a = _r_table_item(values=["100"], chunk=("table-1", 0, 2, 0, 0))
    chunk_b = _r_table_item(values=["200"], chunk=("table-1", 1, 2, 1, 1))
    state = _state(
        _retrieval_result("retrieval:plan_1", [chunk_a]),
        _retrieval_result("retrieval:plan_2", [chunk_b]),
    )

    update = _invoke(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[
            {"result_id": "retrieval:plan_1", "item_index": 0},
            {"result_id": "retrieval:plan_2", "item_index": 0},
        ],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "SUCCESS"
    assert result.items[0]["fields"]["value"] == "300"


def test_nonexistent_result_id_raises_value_error():
    state = _state()

    with pytest.raises(ValueError, match="찾지 못했습니다"):
        _invoke(
            variable_name="합계",
            operation="sum",
            column="금액",
            targets=[{"result_id": "retrieval:unknown", "item_index": 0}],
            state=state,
        )


def test_out_of_range_item_index_raises_value_error():
    item = _r_table_item(values=["100"])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    with pytest.raises(ValueError, match="item_index"):
        _invoke(
            variable_name="합계",
            operation="sum",
            column="금액",
            targets=[{"result_id": "retrieval:plan_1", "item_index": 5}],
            state=state,
        )


def test_empty_targets_raises_value_error():
    state = _state()

    with pytest.raises(ValueError, match="targets"):
        _invoke(
            variable_name="합계",
            operation="sum",
            column="금액",
            targets=[],
            state=state,
        )


def test_blank_variable_name_raises_value_error():
    item = _r_table_item(values=["100"])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    with pytest.raises(ValueError, match="variable_name"):
        _invoke(
            variable_name="   ",
            operation="sum",
            column="금액",
            targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
            state=state,
        )


def test_source_references_are_collected_and_deduplicated():
    item = _r_table_item(values=["100", "200"])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    references = update["retrieval_results"][0].items[0]["source_references"]
    assert references == [
        {
            "disclosure_id": "d20240101000001",
            "section_id": "d20240101000001:src0:s1",
            "evidence_id": "d20240101000001:src0:s1:e1",
        }
    ]


def test_non_success_source_result_raises_value_error():
    item = _r_table_item(values=["100"])
    state = _state(_retrieval_result("retrieval:plan_1", [item], status="INVALID_QUERY"))

    with pytest.raises(ValueError, match="SUCCESS"):
        _invoke(
            variable_name="합계",
            operation="sum",
            column="금액",
            targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
            state=state,
        )


def test_non_r_table_target_raises_value_error():
    item = _r_table_item(values=["100"])
    item["type"] = "kv_table"
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    with pytest.raises(ValueError, match="R_TABLE"):
        _invoke(
            variable_name="합계",
            operation="sum",
            column="금액",
            targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
            state=state,
        )


def test_missing_column_in_target_raises_value_error():
    item = _r_table_item(values=["100"])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    with pytest.raises(ValueError, match="열이 없습니다"):
        _invoke(
            variable_name="합계",
            operation="sum",
            column="존재하지않는열",
            targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
            state=state,
        )


def test_malformed_record_missing_values_key_is_invalid_input_not_exception():
    item = _r_table_item(values=["100"], records=[{"record_index": 0}])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "record_shape"


def test_malformed_record_values_not_a_dict_is_invalid_input():
    item = _r_table_item(values=["100"], records=[{"record_index": 0, "values": ["not", "a", "dict"]}])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "record_shape"


def test_no_source_reference_makes_invalid_input_not_success():
    item = _r_table_item(values=["100", "200"], disclosure_id=None)
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "citation_check"
    assert result.items == []


def test_whitespace_only_disclosure_id_makes_invalid_input():
    item = _r_table_item(values=["100", "200"])
    item["metadata"]["disclosure_id"] = "   "
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "citation_check"


def test_evidence_id_without_section_id_makes_invalid_input():
    item = _r_table_item(values=["100", "200"])
    del item["metadata"]["section_id"]
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "citation_check"


def test_partial_citation_across_chunks_makes_invalid_input():
    # 두 chunk 중 하나만 인용 정보를 갖고 있으면, 나머지 chunk의 값까지
    # 인용된 것처럼 SUCCESS로 내보내면 안 된다.
    chunk_a = _r_table_item(values=["100"], chunk=("table-1", 0, 2, 0, 0))
    chunk_b = _r_table_item(values=["200"], chunk=("table-1", 1, 2, 1, 1), disclosure_id=None)
    state = _state(
        _retrieval_result("retrieval:plan_1", [chunk_a]),
        _retrieval_result("retrieval:plan_2", [chunk_b]),
    )

    update = _invoke(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[
            {"result_id": "retrieval:plan_1", "item_index": 0},
            {"result_id": "retrieval:plan_2", "item_index": 0},
        ],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "citation_check"


def test_non_list_columns_raises_value_error_not_typeerror():
    item = _r_table_item(values=["100"])
    item["columns"] = 123
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    with pytest.raises(ValueError, match="열이 없습니다"):
        _invoke(
            variable_name="합계",
            operation="sum",
            column="금액",
            targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
            state=state,
        )


def test_table_level_unit_fills_in_when_cells_have_no_unit():
    item = _r_table_item(values=["100", "200"], table_units=["백만원"])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    fields = update["retrieval_results"][0].items[0]["fields"]
    assert fields["value"] == "300"
    assert fields["unit"] == "백만원"


def test_ambiguous_table_level_units_make_invalid_input():
    # 셀 자체에는 단위가 없고, 표 단위 후보가 2개라 어떤 게 이 열에
    # 적용되는지 확정할 수 없음 -> 단위를 모르는 숫자를 SUCCESS로
    # 내보내지 않는다.
    item = _r_table_item(values=["100", "200"], table_units=["백만원", "주"])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "unit_check"


def test_explicit_cell_unit_takes_priority_over_table_level_unit():
    item = _r_table_item(values=["100%", "200%"], table_units=["백만원"])
    state = _state(_retrieval_result("retrieval:plan_1", [item]))

    update = _invoke(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    fields = update["retrieval_results"][0].items[0]["fields"]
    assert fields["unit"] == "%"


def test_next_plan_seq_uses_state_counter_not_plan_from_plan_draft():
    item = _r_table_item(values=["100"])
    state = _state(_retrieval_result("retrieval:plan_1", [item]), next_plan_seq=7)

    update = _invoke(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.plan_id == "plan_7"
    assert result.result_id == "derived:plan_7"
    assert update["next_plan_seq"] == 8
