from __future__ import annotations

from typing import Any

import pytest

from agent_graph.state import RetrievalResult
from agent_graph.tools import execute_tool_call, validate_retriever_tool_call


def _r_table_item(*, values: list[str], column: str = "금액") -> dict[str, Any]:
    records = [
        {"record_index": index, "values": {"구분": f"row{index}", column: value}}
        for index, value in enumerate(values)
    ]
    return {
        "type": "r_table",
        "columns": ["구분", column],
        "scope": {"kind": "table"},
        "records": records,
        "available_record_count": len(records),
        "included_record_count": len(records),
        "omitted_record_count": 0,
        "metadata": {
            "disclosure_id": "d20240101000001",
            "section_id": "d20240101000001:src0:s1",
            "evidence_id": "d20240101000001:src0:s1:e1",
        },
    }


def _state(*results: RetrievalResult) -> dict[str, Any]:
    return {
        "question_id": "question-1",
        "question_text": "합계를 알려줘",
        "next_plan_seq": 1,
        "retrieval_results": list(results),
    }


def _calculate_tool_call(**args: Any) -> dict:
    return {"name": "calculate_table_statistic", "args": args}


def _combine_tool_call(**args: Any) -> dict:
    return {"name": "combine_numeric_results", "args": args}


def test_validate_accepts_valid_calculate_call():
    item = _r_table_item(values=["100", "200"])
    state = _state(RetrievalResult(
        result_id="retrieval:plan_1", plan_id="plan_1", source="qdrant",
        query="q", items=[item], result_count=1,
    ))
    tool_call = _calculate_tool_call(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
    )

    validate_retriever_tool_call(state, tool_call)  # 예외 없이 통과해야 함


def test_validate_rejects_calculate_call_with_unknown_result_id():
    state = _state()
    tool_call = _calculate_tool_call(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:unknown", "item_index": 0}],
    )

    with pytest.raises(ValueError, match="찾지 못했습니다"):
        validate_retriever_tool_call(state, tool_call)


def test_validate_rejects_calculate_call_with_blank_variable_name():
    item = _r_table_item(values=["100"])
    state = _state(RetrievalResult(
        result_id="retrieval:plan_1", plan_id="plan_1", source="qdrant",
        query="q", items=[item], result_count=1,
    ))
    tool_call = _calculate_tool_call(
        variable_name="   ",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
    )

    with pytest.raises(ValueError, match="variable_name"):
        validate_retriever_tool_call(state, tool_call)


def test_validate_then_execute_calculate_call_succeeds_end_to_end():
    item = _r_table_item(values=["100", "200"])
    state = _state(RetrievalResult(
        result_id="retrieval:plan_1", plan_id="plan_1", source="qdrant",
        query="q", items=[item], result_count=1,
    ))
    tool_call = _calculate_tool_call(
        variable_name="합계",
        operation="sum",
        column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
    )

    validate_retriever_tool_call(state, tool_call)
    update = execute_tool_call(state, tool_call)

    assert update["retrieval_results"][0].status == "SUCCESS"
    assert update["retrieval_results"][0].items[0]["fields"]["value"] == "300"


def test_validate_accepts_valid_combine_call():
    item_a = _r_table_item(values=["100"])
    item_b = _r_table_item(values=["200"])
    calc_state = _state(
        RetrievalResult(result_id="retrieval:plan_1", plan_id="plan_1", source="qdrant", query="q", items=[item_a], result_count=1),
        RetrievalResult(result_id="retrieval:plan_2", plan_id="plan_2", source="qdrant", query="q", items=[item_b], result_count=1),
    )
    calc_a = execute_tool_call(calc_state, _calculate_tool_call(
        variable_name="A", operation="sum", column="금액",
        targets=[{"result_id": "retrieval:plan_1", "item_index": 0}],
    ))["retrieval_results"][0]
    calc_state["retrieval_results"].append(calc_a)
    calc_state["next_plan_seq"] += 1
    calc_b = execute_tool_call(calc_state, _calculate_tool_call(
        variable_name="B", operation="sum", column="금액",
        targets=[{"result_id": "retrieval:plan_2", "item_index": 0}],
    ))["retrieval_results"][0]
    calc_state["retrieval_results"].append(calc_b)

    tool_call = _combine_tool_call(
        variable_name="합계",
        operation="sum",
        targets=[{"result_id": calc_a.result_id}, {"result_id": calc_b.result_id}],
    )

    validate_retriever_tool_call(calc_state, tool_call)  # 예외 없이 통과해야 함
    update = execute_tool_call(calc_state, tool_call)
    assert update["retrieval_results"][0].status == "SUCCESS"


def test_validate_rejects_combine_call_with_wrong_operand_count():
    state = _state()
    tool_call = _combine_tool_call(
        variable_name="차이",
        operation="difference",
        targets=[{"result_id": "derived:plan_1"}],
    )

    with pytest.raises(ValueError, match="difference"):
        validate_retriever_tool_call(state, tool_call)


def test_validate_rejects_combine_call_with_duplicate_targets():
    state = _state()
    tool_call = _combine_tool_call(
        variable_name="합계",
        operation="sum",
        targets=[{"result_id": "derived:plan_1"}, {"result_id": "derived:plan_1"}],
    )

    with pytest.raises(ValueError, match="중복"):
        validate_retriever_tool_call(state, tool_call)


def test_execute_rejects_unsupported_tool_name():
    state = _state()

    with pytest.raises(ValueError, match="지원하지 않는"):
        execute_tool_call(state, {"name": "unknown_tool", "args": {}})


def test_validate_rejects_unsupported_tool_name():
    state = _state()

    with pytest.raises(ValueError, match="지원하지 않는"):
        validate_retriever_tool_call(state, {"name": "unknown_tool", "args": {}})
