from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_graph.graph import planner, route_after_analysis
from agent_graph.state import (
    PlanDraft,
    PlannerOutput,
    QuestionAnalysis,
)


def _plan_draft() -> PlanDraft:
    return PlanDraft(
        source="qdrant",
        query="삼성전자 2025년 시설 투자",
        purpose="관련 Evidence를 검색합니다.",
        dependencies=[],
    )


def test_retrieve_decision_does_not_create_a_plan():
    output = PlannerOutput(
        question_analysis=QuestionAnalysis(
            decision="retrieve",
            normalized_question="삼성전자 시설 투자를 알려줘",
            decision_reason="공시 Evidence가 필요합니다.",
        ),
    )

    assert output.model_dump() == {
        "question_analysis": output.question_analysis.model_dump()
    }


def test_planner_output_rejects_plans_for_every_decision():
    with pytest.raises(ValidationError, match="plans"):
        PlannerOutput(
            question_analysis=QuestionAnalysis(
                decision="direct",
                normalized_question="안녕하세요",
                decision_reason="일반적인 인사입니다.",
            ),
            plans=[_plan_draft()],
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
            ),
            "retrieve",
        ),
        (
            QuestionAnalysis(
                decision="direct",
                normalized_question="안녕하세요",
                decision_reason="일반적인 인사입니다.",
            ),
            "direct",
        ),
        (
            QuestionAnalysis(
                decision="clarify",
                normalized_question="매출을 알려줘",
                decision_reason="검색 조건이 부족합니다.",
                clarification_question="어느 기업의 어느 기간 매출인가요?",
            ),
            "clarify",
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
    def __init__(self, result: PlannerOutput):
        self.result = result
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.result


class _FakeLlm:
    def __init__(self, result: PlannerOutput):
        self.structured = _FakeStructuredPlanner(result)
        self.schema = None
        self.method = None

    def with_structured_output(self, schema, *, method):
        self.schema = schema
        self.method = method
        return self.structured


def test_planner_returns_analysis_without_plans():
    expected_analysis = QuestionAnalysis(
        decision="retrieve",
        normalized_question="삼성전자 시설 투자를 알려줘",
        decision_reason="공시 Evidence가 필요합니다.",
    )
    expected = PlannerOutput(
        question_analysis=expected_analysis,
    )
    llm = _FakeLlm(expected)

    update = planner(
        {
            "question_id": "question-1",
            "question_text": "  삼성전자 시설 투자를 알려줘  ",
        },
        llm=llm,
    )

    assert update == {
        "question_analysis": expected_analysis,
        "next_plan_seq": 1,
        "retrieval_status": "CONTINUE",
    }
    assert llm.schema is PlannerOutput
    assert llm.method == "function_calling"
    assert llm.structured.messages[-1].content == "삼성전자 시설 투자를 알려줘"


def test_planner_rejects_an_empty_question():
    with pytest.raises(ValueError, match="비어 있을 수 없습니다"):
        planner(
            {"question_id": "question-1", "question_text": "  "},
            llm=object(),
        )


def test_planner_rejects_initial_plan_payload():
    with pytest.raises(ValidationError, match="plans"):
        PlannerOutput(
            question_analysis=QuestionAnalysis(
                decision="retrieve",
                normalized_question="삼성전자 시설 투자를 알려줘",
                decision_reason="공시 Evidence가 필요합니다.",
            ),
            plans=[PlanDraft(
                source="qdrant",
                query="삼성전자 시설 투자",
                purpose="관련 Evidence 검색",
                dependencies=["retrieval:missing"],
            )],
        )

