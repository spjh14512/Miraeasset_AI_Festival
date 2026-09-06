from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_graph import graph as graph_module
from agent_graph.state import Plan, QuestionAnalysis, RetrievalResult, Scope
from agent_graph import utils


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
                source="neo4j",
                query="MATCH ...",
                items=[{"corp_eng_name": "SAMSUNG ELECTRONICS CO., LTD."}],
                result_count=1,
            )
        ],
    }


def test_retriever_tools_constant_includes_correction_history():
    """_build_retriever_llm()(운영 경로)과 retriever()의 llm 인자 경로(테스트

    경로)가 서로 다른 tool 목록을 쓰면, 테스트는 통과하는데 운영에서만
    tool이 빠지는 결함이 생긴다. 두 경로가 이 상수 하나를 공유하는지
    확인한다.
    """

    assert {tool.name for tool in graph_module.RETRIEVER_TOOLS} == {
        "retrieve_search",
        "retrieve_correction_history",
        "calculate_table_statistic",
        "combine_numeric_results",
        "finish",
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
    assert [event.name for event in update["think_trace_events"]] == [
        "retriever",
        "finish",
    ]
    assert update["think_trace_events"][1].message == "필요한 근거를 확보했습니다."
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
                "breadth": "initial",
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
                "breadth": "initial",
            },
        }
    ]])

    update = graph_module.retriever(_retrieval_state(), llm=llm)

    assert update["retrieval_status"] == "COMPLETE"
    assert llm.invoke_count == 2
    assert "scope_id" in llm.calls[1][-1].content


def test_retriever_retries_retrieve_search_after_maximum_count():
    llm = _RetryingRetrieverLlm([[
        {
            "name": "retrieve_search",
            "args": {
                "plan": {
                    "source": "qdrant",
                    "query": "삼성전자 판매전략",
                    "purpose": "판매전략 근거 확인",
                    "dependencies": [],
                    "scope_id": "scope_1",
                },
                "breadth": "initial",
            },
        }
    ]])
    state = {**_retrieval_state(), "retrieval_search_count": 15}

    update = graph_module.retriever(state, llm=llm)

    assert update["retrieval_status"] == "COMPLETE"
    assert llm.invoke_count == 2
    assert "최대 호출 횟수(15회)" in llm.calls[1][-1].content


def test_retriever_rejects_plan_after_two_matching_failures():
    plan_args = {
        "source": "neo4j",
        "query": "AMD 기업 정보",
        "purpose": "AMD의 기업 정보를 찾기 위함",
        "dependencies": [],
        "scope_id": "scope_1",
    }
    signature = utils._retrieval_plan_signature(Plan(
        plan_id="plan_2",
        **plan_args,
    ))
    state = _retrieval_state()
    state["retrieval_results"].extend([
        RetrievalResult(
            result_id=f"retrieval:plan_{index}",
            source="neo4j",
            status="INVALID_QUERY",
            query='MATCH (c:Company) RETURN c.corp_code AS "corp_code"',
            items=[],
            result_count=0,
            metadata={"plan_signature": signature},
        )
        for index in (2, 3)
    ])
    llm = _RetryingRetrieverLlm([[
        {
            "name": "retrieve_search",
            "args": {"plan": plan_args, "breadth": "initial"},
        }
    ]])

    update = graph_module.retriever(state, llm=llm)

    assert update["retrieval_status"] == "COMPLETE"
    assert llm.invoke_count == 2
    assert "동일한 retrieval Plan이 이미 2회 실패" in llm.calls[1][-1].content


def test_retriever_ignores_finish_when_another_tool_is_called(monkeypatch):
    llm = _RetryingRetrieverLlm([[
        {
            "name": "finish",
            "args": {
                "status": "INSUFFICIENT",
                "reason": "검색을 종료합니다.",
                "selected_result_ids": [],
            },
        },
        {
            "name": "retrieve_search",
            "args": {
                "plan": {
                    "source": "qdrant",
                    "query": "삼성전자 판매전략",
                    "purpose": "판매전략 근거 확인",
                    "dependencies": [],
                    "scope_id": "scope_1",
                },
                "breadth": "initial",
            },
        },
    ]])
    selected_calls = []

    def execute(_state, tool_call):
        selected_calls.append(tool_call)
        return {"retrieval_status": "CONTINUE"}

    monkeypatch.setattr(graph_module, "execute_tool_call", execute)

    update = graph_module.retriever(_retrieval_state(), llm=llm)

    assert llm.invoke_count == 1
    assert selected_calls[0]["name"] == "retrieve_search"
    assert update["retrieval_status"] == "CONTINUE"


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
