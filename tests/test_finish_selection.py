from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agent_graph.state import QuestionAnalysis, RetrievalResult
from agent_graph.tools import (
    build_answer_generator_human_message,
    execute_tool_call,
    finish,
)


def _result(result_id: str, item_count: int = 2) -> RetrievalResult:
    return RetrievalResult(
        result_id=result_id,
        plan_id=result_id.removeprefix("retrieval:"),
        source="qdrant",
        query="검색 쿼리",
        items=[{"value": index} for index in range(item_count)],
        result_count=item_count,
        metadata={"plan_purpose": "질문의 핵심 정보 확인"}
    )


def _state():
    return {
        "question_id": "question-1",
        "question_text": "삼성전자의 관련 정보를 알려줘",
        "question_analysis": QuestionAnalysis(
            decision="retrieve",
            normalized_question="삼성전자 관련 정보",
            decision_reason="공시 근거가 필요합니다."
        ),
        "retrieval_results": [
            _result("retrieval:plan_1"),
            _result("retrieval:plan_2")
        ],
    }


def test_finish_stores_validated_evidence_selection():
    update = finish.invoke({
        "status": "COMPLETE",
        "reason": "필요한 근거를 확보했습니다.",
        "selected_evidence": [{
            "result_id": "retrieval:plan_1",
            "item_indexes": [1],
            "reason": "질문의 핵심 정보를 포함합니다."
        }],
        "state": _state(),
    })

    assert update["retrieval_status"] == "COMPLETE"
    assert update["selected_evidence"][0].item_indexes == [1]


def test_finish_requires_selection_for_complete():
    with pytest.raises(ValueError, match="최소 하나"):
        finish.invoke({
            "status": "COMPLETE",
            "reason": "완료",
            "selected_evidence": [],
            "state": _state(),
        })


def test_finish_allows_empty_selection_for_insufficient():
    update = execute_tool_call(
        _state(),
        {
            "name": "finish",
            "args": {
                "status": "INSUFFICIENT",
                "reason": "근거를 찾지 못했습니다.",
                "selected_evidence": [],
            },
        },
    )

    assert update["selected_evidence"] == []


def test_finish_rejects_unknown_result_and_out_of_range_item():
    with pytest.raises(ValueError, match="찾지 못했습니다"):
        finish.invoke({
            "status": "COMPLETE",
            "reason": "완료",
            "selected_evidence": [{
                "result_id": "retrieval:unknown",
                "item_indexes": None,
                "reason": "근거"
            }],
            "state": _state(),
        })

    with pytest.raises(ValueError, match="범위를 벗어났습니다"):
        finish.invoke({
            "status": "COMPLETE",
            "reason": "완료",
            "selected_evidence": [{
                "result_id": "retrieval:plan_1",
                "item_indexes": [2],
                "reason": "근거"
            }],
            "state": _state(),
        })


def test_evidence_selection_rejects_duplicate_indexes():
    with pytest.raises(ValidationError, match="중복"):
        finish.invoke({
            "status": "COMPLETE",
            "reason": "완료",
            "selected_evidence": [{
                "result_id": "retrieval:plan_1",
                "item_indexes": [0, 0],
                "reason": "근거"
            }],
            "state": _state(),
        })


def test_answer_message_contains_only_selected_items():
    state = _state()
    state.update(finish.invoke({
        "status": "COMPLETE",
        "reason": "필요한 근거를 확보했습니다.",
        "selected_evidence": [{
            "result_id": "retrieval:plan_2",
            "item_indexes": [1],
            "reason": "두 번째 결과의 두 번째 item이 필요합니다."
        }],
        "state": state,
    }))

    payload = json.loads(build_answer_generator_human_message(state).content)

    assert len(payload["selected_retrieval_results"]) == 1
    selected = payload["selected_retrieval_results"][0]
    assert selected["result_id"] == "retrieval:plan_2"
    assert selected["selection_reason"] == "두 번째 결과의 두 번째 item이 필요합니다."
    assert selected["items"] == [{
        "value": 1,
        "item_reference_id": "R1-I2",
        "item_index": 1,
    }]
