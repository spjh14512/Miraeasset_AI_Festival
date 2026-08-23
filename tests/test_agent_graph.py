from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_graph.graph import analyze_question, route_after_analysis
from agent_graph.state import QueryPlan, QuestionAnalysis


def _plan() -> QueryPlan:
    return QueryPlan(
        plan_id="plan-1",
        source="qdrant",
        query="삼성전자 2025년 시설 투자",
        purpose="관련 Evidence를 검색합니다.",
        filters={"corp_name": "삼성전자", "base_year": 2025},
    )


def test_retrieve_decision_requires_at_least_one_plan():
    with pytest.raises(ValidationError, match="query plan"):
        QuestionAnalysis(
            decision="retrieve",
            normalized_question="삼성전자 시설 투자를 알려줘",
            decision_reason="공시 Evidence가 필요합니다.",
        )


def test_direct_decision_rejects_query_plans():
    with pytest.raises(ValidationError, match="없어야 합니다"):
        QuestionAnalysis(
            decision="direct",
            normalized_question="안녕하세요",
            decision_reason="일반적인 인사입니다.",
            query_plans=[_plan()],
        )


def test_clarify_decision_requires_a_clarification_question():
    with pytest.raises(ValidationError, match="clarification_question"):
        QuestionAnalysis(
            decision="clarify",
            normalized_question="매출을 알려줘",
            decision_reason="대상 기업과 기간이 필요합니다.",
        )


@pytest.mark.parametrize(
    ("analysis", "expected"),
    [
        (
            QuestionAnalysis(
                decision="retrieve",
                normalized_question="삼성전자 시설 투자를 알려줘",
                decision_reason="공시 Evidence가 필요합니다.",
                query_plans=[_plan()],
            ),
            "execute_retrieval",
        ),
        (
            QuestionAnalysis(
                decision="direct",
                normalized_question="안녕하세요",
                decision_reason="일반적인 인사입니다.",
            ),
            "answer_directly",
        ),
        (
            QuestionAnalysis(
                decision="clarify",
                normalized_question="매출을 알려줘",
                decision_reason="검색 조건이 부족합니다.",
                clarification_question="어느 기업의 어느 기간 매출인가요?",
            ),
            "request_clarification",
        ),
    ],
)
def test_route_after_analysis(analysis, expected):
    state = {
        "question_id": "question-1",
        "question_text": "질문",
        "question_analysis": analysis,
    }

    assert route_after_analysis(state) == expected


class _FakeStructuredPlanner:
    def __init__(self, result: QuestionAnalysis):
        self.result = result
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.result


class _FakeLlm:
    def __init__(self, result: QuestionAnalysis):
        self.structured = _FakeStructuredPlanner(result)
        self.schema = None
        self.method = None

    def with_structured_output(self, schema, *, method):
        self.schema = schema
        self.method = method
        return self.structured


def test_analyze_question_returns_a_partial_state_update():
    expected = QuestionAnalysis(
        decision="retrieve",
        normalized_question="삼성전자 시설 투자를 알려줘",
        decision_reason="공시 Evidence가 필요합니다.",
        query_plans=[_plan()],
    )
    llm = _FakeLlm(expected)

    update = analyze_question(
        {
            "question_id": "question-1",
            "question_text": "  삼성전자 시설 투자를 알려줘  ",
        },
        llm=llm,
    )

    assert update == {"question_analysis": expected}
    assert llm.schema is QuestionAnalysis
    assert llm.method == "json_schema"
    assert llm.structured.messages[-1] == (
        "human",
        "삼성전자 시설 투자를 알려줘",
    )


def test_analyze_question_rejects_an_empty_question():
    with pytest.raises(ValueError, match="비어 있을 수 없습니다"):
        analyze_question(
            {"question_id": "question-1", "question_text": "  "},
            llm=object(),
        )
