from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_graph import graph as graph_module
from agent_graph.state import QuestionAnalysis, RetrievalResult, Scope


class _FakeRetrieverLlm:
    def __init__(self):
        self.bound_tools = None
        self.invoke_count = 0
        self.calls = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.invoke_count += 1
        self.calls.append(list(messages))
        return SimpleNamespace(tool_calls=[{
            "name": "finish",
            "args": {
                "status": "COMPLETE",
                "reason": "필요한 근거를 확보했습니다.",
                "selected_result_ids": ["retrieval:plan_1"]
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
            self.calls.append(list(messages))
            return SimpleNamespace(tool_calls=tool_calls)
        return super().invoke(messages)


class _InvalidRetrieverLlm(_FakeRetrieverLlm):
    def invoke(self, _):
        self.invoke_count += 1
        return SimpleNamespace(tool_calls=[])


def _retrieval_state():
    return {
        "question_id": "question-1",
        "question_text": "삼성전자의 영문 기업명을 알려줘",
        "question_analysis": QuestionAnalysis(
            decision="retrieve",
            normalized_question="삼성전자의 영문 기업명",
            decision_reason="기업 속성 조회가 필요합니다."
        ),
        "retrieval_status": "CONTINUE",
        "scope_candidates": [Scope(
            scope_id="scope_1",
            subquestion_id="subquestion_1",
            level="GLOBAL",
            corp_names=[],
            corp_codes=[],
            disclosure_ids=[],
            section_ids=[],
            reason="검색 범위를 더 좁힐 수 없습니다.",
        )],
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


def test_retriever_can_finish_with_tool_interface():
    llm = _FakeRetrieverLlm()

    update = graph_module.retriever(_retrieval_state(), llm=llm)

    assert update["retrieval_status"] == "COMPLETE"
    assert update["selected_result_ids"] == ["retrieval:plan_1"]
    assert {tool.name for tool in llm.bound_tools} == {
        "retrieve_search",
        "retrieve_correction_history",
        "calculate_table_statistic",
        "combine_numeric_results",
        "finish",
    }
    assert llm.invoke_count == 1


def test_retriever_retries_when_tool_call_is_missing():
    llm = _RetryingRetrieverLlm([[]])

    update = graph_module.retriever(_retrieval_state(), llm=llm)

    assert update["retrieval_status"] == "COMPLETE"
    assert llm.invoke_count == 2


def test_retriever_retries_unknown_plan_dependency():
    llm = _RetryingRetrieverLlm([[
        {
            "name": "retrieve_search",
            "args": {
                "plan": {
                    "source": "qdrant",
                    "query": "확인된 공시의 판매전략",
                    "purpose": "판매전략 근거 확인",
                    "dependencies": ["d20240306000686"],
                    "scope_id": "scope_1",
                },
                "limit": 5,
            },
        }
    ]])

    update = graph_module.retriever(_retrieval_state(), llm=llm)

    assert update["retrieval_status"] == "COMPLETE"
    assert llm.invoke_count == 2
    assert "Plan dependency RetrievalResult" in llm.calls[1][-1].content


def test_retriever_retries_retrieve_search_without_scope_id():
    llm = _RetryingRetrieverLlm([[
        {
            "name": "retrieve_search",
            "args": {
                "plan": {
                    "source": "qdrant",
                    "query": "삼성전자 판매전략",
                    "purpose": "판매전략 근거 확인",
                    "dependencies": [],
                },
                "limit": 5,
            },
        }
    ]])

    update = graph_module.retriever(_retrieval_state(), llm=llm)

    assert update["retrieval_status"] == "COMPLETE"
    assert llm.invoke_count == 2
    assert "scope_id" in llm.calls[1][-1].content


def test_retriever_retries_unknown_finish_result_id():
    llm = _RetryingRetrieverLlm([[
        {
            "name": "finish",
            "args": {
                "status": "COMPLETE",
                "reason": "근거를 확보했습니다.",
                "selected_result_ids": ["retrieval:unknown"],
            },
        }
    ]])

    update = graph_module.retriever(_retrieval_state(), llm=llm)

    assert update["retrieval_status"] == "COMPLETE"
    assert llm.invoke_count == 2
    assert "RetrievalResult를 찾지 못했습니다" in llm.calls[1][-1].content


def test_retriever_reports_invalid_tool_calls_after_retries():
    llm = _InvalidRetrieverLlm()

    with pytest.raises(
        ValueError,
        match="재시도 후에도.*호출 개수: 0",
    ):
        graph_module.retriever(_retrieval_state(), llm=llm)

    assert llm.invoke_count == graph_module.MAX_TOOL_CALL_RETRIES + 1
