from __future__ import annotations

import pytest

from agent_graph.state import Plan, RetrievalResult
from agent_graph.tools import execute_tool_call, modify_plan


def _state():
    return {
        "question_id": "question-1",
        "question_text": "삼성전자 관련 정보를 알려줘",
        "plans": [
            Plan(
                plan_id="plan_1",
                source="neo4j",
                query="삼성전자 관련 공시 탐색",
                purpose="관련 공시 식별"
            ),
            Plan(
                plan_id="plan_2",
                source="qdrant",
                query="삼성전자 공시 내용",
                purpose="관련 근거 검색"
            ),
        ],
    }


def test_modify_plan_preserves_id_position_and_input_state():
    state = _state()

    update = modify_plan.invoke({
        "plan_id": "plan_1",
        "modified_plan": {
            "source": "qdrant",
            "query": "삼성전자 특별관계자 목록",
            "purpose": "특별관계자 명단 확인"
        },
        "state": state,
    })

    assert [plan.plan_id for plan in update["plans"]] == ["plan_1", "plan_2"]
    assert update["plans"][0] == Plan(
        plan_id="plan_1",
        source="qdrant",
        query="삼성전자 특별관계자 목록",
        purpose="특별관계자 명단 확인"
    )
    assert state == _state()


def test_modify_plan_rejects_unknown_plan_id():
    with pytest.raises(ValueError, match="plan_99"):
        modify_plan.invoke({
            "plan_id": "plan_99",
            "modified_plan": {
                "source": "qdrant",
                "query": "검색어",
                "purpose": "검색 목적"
            },
            "state": _state(),
        })


def test_execute_tool_call_dispatches_modify_plan():
    update = execute_tool_call(
        _state(),
        {
            "name": "modify_plan",
            "args": {
                "plan_id": "plan_2",
                "modified_plan": {
                    "source": "neo4j",
                    "query": "삼성전자 공시 관계",
                    "purpose": "공시 구조 확인"
                },
            },
        },
    )

    assert update["plans"][1].source == "neo4j"
    assert update["plans"][1].plan_id == "plan_2"


def test_modify_plan_accepts_existing_retrieval_result_dependency():
    state = _state()
    state["retrieval_results"] = [RetrievalResult(
        result_id="retrieval:plan_0",
        plan_id="plan_0",
        source="neo4j",
        query="MATCH ...",
        items=[],
        result_count=0,
    )]

    update = modify_plan.invoke({
        "plan_id": "plan_1",
        "modified_plan": {
            "source": "qdrant",
            "query": "확인된 Evidence 검색",
            "purpose": "실제 내용 확인",
            "dependencies": ["retrieval:plan_0"],
        },
        "state": state,
    })

    assert update["plans"][0].dependencies == ["retrieval:plan_0"]


def test_modify_plan_rejects_unknown_dependency():
    with pytest.raises(ValueError, match="retrieval:missing"):
        modify_plan.invoke({
            "plan_id": "plan_1",
            "modified_plan": {
                "source": "qdrant",
                "query": "확인된 Evidence 검색",
                "purpose": "실제 내용 확인",
                "dependencies": ["retrieval:missing"],
            },
            "state": _state(),
        })
