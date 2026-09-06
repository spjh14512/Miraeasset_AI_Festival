from __future__ import annotations

import json

import pytest

from agent_graph.state import QuestionAnalysis, RetrievalResult
from agent_graph.tools import finish
from agent_graph.utils import (
    build_answer_generator_human_message,
    execute_tool_call,
)


def _message_payload(state: dict) -> dict:
    content = build_answer_generator_human_message(state).content
    json_dump = content.split("[입력]\n\n", 1)[1].split("\n\n\n[출력]", 1)[0]
    return json.loads(json_dump)


def _result(result_id: str, item_count: int = 2) -> RetrievalResult:
    return RetrievalResult(
        result_id=result_id,
        source="qdrant",
        query="검색 쿼리",
        items=[{"value": index} for index in range(item_count)],
        result_count=item_count,
    )


def _failed_result(result_id: str, status: str = "NO_RESULTS") -> RetrievalResult:
    return RetrievalResult(
        result_id=result_id,
        source="qdrant",
        status=status,
        query="검색 쿼리",
        items=[],
        result_count=0,
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


def test_finish_stores_validated_result_ids():
    update = finish.invoke({
        "status": "COMPLETE",
        "reason": "필요한 근거를 확보했습니다.",
        "selected_result_ids": ["retrieval:plan_1"],
        "state": _state(),
    })

    assert update["retrieval_status"] == "COMPLETE"
    assert update["selected_result_ids"] == ["retrieval:plan_1"]


def test_finish_requires_selection_for_complete():
    with pytest.raises(ValueError, match="최소 하나"):
        finish.invoke({
            "status": "COMPLETE",
            "reason": "완료",
            "selected_result_ids": [],
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
                "selected_result_ids": [],
            },
        },
    )

    assert update["selected_result_ids"] == []


def test_finish_rejects_unknown_result():
    with pytest.raises(ValueError, match="찾지 못했습니다"):
        finish.invoke({
            "status": "COMPLETE",
            "reason": "완료",
            "selected_result_ids": ["retrieval:unknown"],
            "state": _state(),
        })


def test_finish_rejects_duplicate_result_ids():
    with pytest.raises(ValueError, match="중복"):
        finish.invoke({
            "status": "COMPLETE",
            "reason": "완료",
            "selected_result_ids": [
                "retrieval:plan_1",
                "retrieval:plan_1",
            ],
            "state": _state(),
        })


def test_finish_rejects_non_success_selection_for_complete():
    state = _state()
    state["retrieval_results"].append(_failed_result("retrieval:plan_3"))

    with pytest.raises(ValueError, match="SUCCESS 상태가 아닌"):
        finish.invoke({
            "status": "COMPLETE",
            "reason": "완료",
            "selected_result_ids": ["retrieval:plan_3"],
            "state": state,
        })


def test_finish_rejects_non_success_selection_for_insufficient():
    state = _state()
    state["retrieval_results"].append(_failed_result("retrieval:plan_3"))

    with pytest.raises(ValueError, match="SUCCESS 상태가 아닌"):
        finish.invoke({
            "status": "INSUFFICIENT",
            "reason": "근거를 찾지 못했습니다.",
            "selected_result_ids": ["retrieval:plan_3"],
            "state": state,
        })


def test_finish_allows_success_selection_alongside_failed_result_in_state():
    state = _state()
    state["retrieval_results"].append(_failed_result("retrieval:plan_3"))

    update = finish.invoke({
        "status": "INSUFFICIENT",
        "reason": "일부 근거만 확보했습니다.",
        "selected_result_ids": ["retrieval:plan_1"],
        "state": state,
    })

    assert update["selected_result_ids"] == ["retrieval:plan_1"]


def test_answer_message_contains_all_items_from_selected_result():
    state = _state()
    state.update(finish.invoke({
        "status": "COMPLETE",
        "reason": "필요한 근거를 확보했습니다.",
        "selected_result_ids": ["retrieval:plan_2"],
        "state": state,
    }))

    payload = _message_payload(state)

    assert payload["retrieval_results"] == [
        {
            "result_id": "answer_result_1",
            "context": "",
            "content": {"value": 0},
        },
        {
            "result_id": "answer_result_2",
            "context": "",
            "content": {"value": 1},
        },
    ]
