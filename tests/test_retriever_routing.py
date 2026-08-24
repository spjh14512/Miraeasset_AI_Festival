from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_graph import graph as graph_module
from agent_graph.state import QuestionAnalysis, RetrievalResult


class _FakeRetrieverLlm:
    def __init__(self):
        self.bound_tools = None
        self.invoke_count = 0

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, _):
        self.invoke_count += 1
        return SimpleNamespace(tool_calls=[{
            "name": "finish",
            "args": {
                "status": "COMPLETE",
                "reason": "필요한 근거를 확보했습니다.",
                "selected_evidence": [{
                    "result_id": "retrieval:plan_1",
                    "item_indexes": [0],
                    "reason": "질문에 필요한 값을 포함합니다."
                }]
            },
        }])


class _RetryingRetrieverLlm(_FakeRetrieverLlm):
    def __init__(self, invalid_tool_calls):
        super().__init__()
        self.invalid_tool_calls = invalid_tool_calls

    def invoke(self, messages):
        if self.invoke_count < len(self.invalid_tool_calls):
            tool_calls = self.invalid_tool_calls[self.invoke_count]
            self.invoke_count += 1
            return SimpleNamespace(tool_calls=tool_calls)
        return super().invoke(messages)


class _InvalidRetrieverLlm(_FakeRetrieverLlm):
    def invoke(self, _):
        self.invoke_count += 1
        return SimpleNamespace(tool_calls=[])


def _state_without_plans():
    return {
        "question_id": "question-1",
        "question_text": "삼성전자의 영문 기업명을 알려줘",
        "question_analysis": QuestionAnalysis(
            decision="retrieve",
            normalized_question="삼성전자의 영문 기업명",
            decision_reason="기업 속성 조회가 필요합니다."
        ),
        "plans": [],
        "retrieval_status": "CONTINUE",
        "retrieval_results": [
            RetrievalResult(
                result_id="retrieval:plan_1",
                plan_id="plan_1",
                source="neo4j",
                query="MATCH ...",
                items=[{"corp_eng_name": "SAMSUNG ELECTRONICS CO., LTD."}],
                result_count=1,
                metadata={"plan_purpose": "영문 기업명 확인"}
            )
        ],
    }


def test_route_after_retrieval_returns_registered_retriever_key():
    assert graph_module.route_after_retrieval({
        "retrieval_status": "CONTINUE"
    }) == "retriever"


def test_retriever_can_finish_when_plan_list_is_empty():
    llm = _FakeRetrieverLlm()

    update = graph_module.retriever(_state_without_plans(), llm=llm)

    assert update["retrieval_status"] == "COMPLETE"
    assert update["selected_evidence"][0].result_id == "retrieval:plan_1"
    assert {tool.name for tool in llm.bound_tools} == {
        "retrieve_search",
        "create_plan",
        "delete_plan",
        "modify_plan",
        "finish",
    }
    assert llm.invoke_count == 1


def test_retriever_retries_when_tool_call_is_missing():
    llm = _RetryingRetrieverLlm([[]])

    update = graph_module.retriever(_state_without_plans(), llm=llm)

    assert update["retrieval_status"] == "COMPLETE"
    assert llm.invoke_count == 2


def test_retriever_reports_invalid_tool_calls_after_retries():
    llm = _InvalidRetrieverLlm()

    with pytest.raises(
        ValueError,
        match="재시도 후에도.*호출 개수: 0",
    ):
        graph_module.retriever(_state_without_plans(), llm=llm)

    assert llm.invoke_count == graph_module.MAX_TOOL_CALL_RETRIES + 1
