from __future__ import annotations

from datetime import date
import json

import pytest
from pydantic import ValidationError

from agent_graph.graph import question_analyzer, route_after_analysis
from agent_graph.state import (
    PlanDraft,
    QuestionAnalyzerOutput,
    QuestionAnalysis,
    SubQuestion,
)
from agent_graph.utils import build_retriever_human_message


def _plan_draft() -> PlanDraft:
    return PlanDraft(
        source="qdrant",
        query="삼성전자 2025년 시설 투자",
        purpose="관련 Evidence를 검색합니다.",
        dependencies=[],
    )


def _sub_question() -> SubQuestion:
    return SubQuestion(
        question="삼성전자의 시설 투자 내용은 무엇인가?",
        entities=[{
            "mention": "삼성전자",
            "roles": ["ISSUER"],
            "canonical_name": "삼성전자",
            "match_status": "MATCHED",
        }],
        events=[{
            "event_type": "시설 투자",
            "candidate_event_types": [],
            "confidence": "HIGH",
        }],
        intents=["DETAIL"],
        periods=[],
        requested_facts=["시설 투자 내용"],
    )


def test_retrieve_decision_does_not_create_a_plan():
    output = QuestionAnalyzerOutput(
        question_analysis=QuestionAnalysis(
            decision="retrieve",
            normalized_question="삼성전자 시설 투자를 알려줘",
            decision_reason="공시 Evidence가 필요합니다.",
            sub_questions=[_sub_question()],
        ),
    )

    assert output.model_dump() == {
        "question_analysis": output.question_analysis.model_dump()
    }


def test_question_analyzer_output_rejects_plans_for_every_decision():
    with pytest.raises(ValidationError, match="plans"):
        QuestionAnalyzerOutput(
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


def test_retrieve_output_requires_a_sub_question():
    with pytest.raises(ValidationError, match="sub_questions"):
        QuestionAnalyzerOutput(
            question_analysis=QuestionAnalysis(
                decision="retrieve",
                normalized_question="삼성전자 시설 투자를 알려줘",
                decision_reason="공시 Evidence가 필요합니다.",
            )
        )


def test_multiple_sub_questions_require_synthesis_requirement():
    with pytest.raises(ValidationError, match="synthesis_requirement"):
        QuestionAnalyzerOutput(
            question_analysis=QuestionAnalysis(
                decision="retrieve",
                normalized_question="삼성전자와 SK하이닉스의 투자를 비교해줘",
                decision_reason="두 기업의 공시 Evidence가 필요합니다.",
                sub_questions=[_sub_question(), _sub_question()],
            )
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


class _FakeStructuredQuestionAnalyzer:
    def __init__(self, result: QuestionAnalyzerOutput):
        self.result = result
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.result


class _FakeLlm:
    def __init__(self, result: QuestionAnalyzerOutput):
        self.structured = _FakeStructuredQuestionAnalyzer(result)
        self.schema = None
        self.method = None

    def with_structured_output(self, schema, *, method):
        self.schema = schema
        self.method = method
        return self.structured


def test_question_analyzer_returns_analysis_without_plans():
    expected_analysis = QuestionAnalysis(
        decision="retrieve",
        normalized_question="삼성전자 시설 투자를 알려줘",
        decision_reason="공시 Evidence가 필요합니다.",
        sub_questions=[_sub_question()],
    )
    expected = QuestionAnalyzerOutput(
        question_analysis=expected_analysis,
    )
    llm = _FakeLlm(expected)

    update = question_analyzer(
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
    assert llm.schema["title"] == "QuestionAnalyzerOutput"
    assert llm.method == "json_schema"
    assert json.loads(llm.structured.messages[-1].content) == {
        "current_date": date.today().isoformat(),
        "user_question": "삼성전자 시설 투자를 알려줘",
    }


def test_question_analyzer_rejects_an_empty_question():
    with pytest.raises(ValueError, match="비어 있을 수 없습니다"):
        question_analyzer(
            {"question_id": "question-1", "question_text": "  "},
            llm=object(),
        )


def test_question_analyzer_rejects_initial_plan_payload():
    with pytest.raises(ValidationError, match="plans"):
        QuestionAnalyzerOutput(
            question_analysis=QuestionAnalysis(
                decision="retrieve",
                normalized_question="삼성전자 시설 투자를 알려줘",
                decision_reason="공시 Evidence가 필요합니다.",
                sub_questions=[_sub_question()],
            ),
            plans=[PlanDraft(
                source="qdrant",
                query="삼성전자 시설 투자",
                purpose="관련 Evidence 검색",
                dependencies=["retrieval:missing"],
            )],
        )


def test_retriever_human_message_contains_full_question_analysis():
    analysis = QuestionAnalysis(
        decision="retrieve",
        normalized_question="삼성전자 시설 투자를 알려줘",
        decision_reason="공시 Evidence가 필요합니다.",
        sub_questions=[_sub_question()],
    )

    message = build_retriever_human_message({
        "question_id": "question-1",
        "question_text": "삼성전자 시설 투자를 알려줘",
        "question_analysis": analysis,
        "retrieval_results": [],
    })
    payload = json.loads(message.content.split("\n\n", 1)[1])

    assert payload["question_analysis"] == analysis.model_dump(mode="json")
