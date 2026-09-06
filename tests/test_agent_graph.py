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
    ThinkTraceEvent,
    merge_think_trace_events,
)
from agent_graph.utils import (
    build_retriever_human_message,
    enforce_question_clarification_policy,
    load_issuer_universe,
    normalize_question_analysis_entities,
)


def _plan_draft() -> PlanDraft:
    return PlanDraft(
        source="qdrant",
        query="삼성전자 2025년 시설 투자",
        purpose="관련 Evidence를 검색합니다.",
        dependencies=[],
        scope_id="scope_1",
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


def test_think_trace_events_preserve_execution_order():
    first = ThinkTraceEvent(
        type="node",
        name="question_analyzer",
        message="질문을 분석했습니다.",
    )
    second = ThinkTraceEvent(
        type="tool",
        name="retrieve_search",
        message="근거를 검색했습니다.",
    )

    assert merge_think_trace_events([first], [second]) == [first, second]


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

    assert update["next_plan_seq"] == 1
    assert update["retrieval_status"] == "CONTINUE"
    assert update["question_analysis"].normalized_question == expected_analysis.normalized_question
    assert update["question_analysis"].sub_questions[0].subquestion_id == "subquestion_1"
    trace = update["think_trace_events"][0]
    assert trace.name == "question_analyzer"
    assert trace.message == "공시 Evidence가 필요합니다."
    assert trace.details["decision"] == "retrieve"
    assert llm.schema["title"] == "QuestionAnalyzerOutput"
    assert llm.method == "json_schema"
    payload = json.loads(llm.structured.messages[-1].content)
    assert payload["current_date"] == date.today().isoformat()
    assert payload["user_question"] == "삼성전자 시설 투자를 알려줘"
    assert payload["issuer_universe_tsv"].startswith(
        "corp_code\tstock_code\tcorp_name"
    )
    assert len(payload["issuer_universe_tsv"].splitlines()) == 71


def test_question_analyzer_clarifies_financial_amount_without_issuer_or_period():
    output = QuestionAnalyzerOutput(question_analysis=QuestionAnalysis(
        decision="retrieve",
        normalized_question="연결 기준 유동부채가 얼마인지 알려줘",
        decision_reason="공시 조회가 필요합니다.",
        sub_questions=[SubQuestion(
            question="연결 기준 유동부채 금액",
            entities=[],
            events=[],
            intents=["AMOUNT"],
            periods=[],
            requested_facts=["연결 기준 유동부채"],
        )],
    ))

    update = question_analyzer(
        {
            "question_id": "question-1",
            "question_text": "연결 기준 유동부채가 얼마인지 알려줘",
        },
        llm=_FakeLlm(output),
    )

    analysis = update["question_analysis"]
    assert analysis.decision == "clarify"
    assert "기업명과 보고기간" in analysis.clarification_question
    assert analysis.sub_questions[0].subquestion_id == "subquestion_1"


def test_clarification_policy_keeps_complete_financial_amount_retrieval():
    analysis = QuestionAnalysis(
        decision="retrieve",
        normalized_question="삼성전자의 2024년 연결 기준 유동부채",
        decision_reason="공시 조회가 필요합니다.",
        sub_questions=[SubQuestion(
            question="삼성전자의 2024년 연결 기준 유동부채 금액",
            entities=[{"mention": "삼성전자", "roles": ["ISSUER"]}],
            events=[],
            intents=["AMOUNT"],
            periods=[{
                "expression": "2024년",
                "kind": "REPORTING_PERIOD",
                "normalized_value": "2024",
                "granularity": "YEAR",
            }],
            requested_facts=["연결 기준 유동부채"],
        )],
    )

    assert enforce_question_clarification_policy(analysis).decision == "retrieve"


def test_clarification_policy_does_not_require_period_for_event_amount():
    analysis = QuestionAnalysis(
        decision="retrieve",
        normalized_question="삼성전자의 계약 금액",
        decision_reason="공시 조회가 필요합니다.",
        sub_questions=[SubQuestion(
            question="삼성전자의 계약 금액",
            entities=[{"mention": "삼성전자", "roles": ["ISSUER"]}],
            events=[{"event_type": "계약", "confidence": "HIGH"}],
            intents=["AMOUNT"],
            periods=[],
            requested_facts=["계약 금액"],
        )],
    )

    assert enforce_question_clarification_policy(analysis).decision == "retrieve"


def test_entity_normalization_uses_registry_for_issuer():
    analysis = QuestionAnalysis(
        decision="retrieve",
        normalized_question="삼성전자 정보를 알려줘",
        decision_reason="검색이 필요합니다.",
        sub_questions=[SubQuestion(
            question="삼성전자 정보를 알려줘",
            entities=[{
                "mention": "삼성전자",
                "roles": ["ISSUER"],
                "canonical_name": "Samsung Electronics Co., Ltd.",
                "match_status": "MATCHED",
            }],
            intents=["DETAIL"],
            requested_facts=["삼성전자 정보"],
        )],
    )

    normalized = normalize_question_analysis_entities(analysis)
    entity = normalized.sub_questions[0].entities[0]

    assert entity.canonical_name == "삼성전자"
    assert entity.match_status == "MATCHED"
    assert len(load_issuer_universe()) == 70


def test_entity_normalization_marks_only_unknown_issuer_out_of_universe():
    analysis = QuestionAnalysis(
        decision="retrieve",
        normalized_question="OpenAI와 삼성전자의 계약",
        decision_reason="검색이 필요합니다.",
        sub_questions=[SubQuestion(
            question="OpenAI와 삼성전자의 계약을 알려줘",
            entities=[
                {"mention": "OpenAI", "roles": ["ISSUER"]},
                {"mention": "Arm", "roles": ["TARGET"]},
            ],
            intents=["DETAIL"],
            requested_facts=["계약 내용"],
        )],
    )

    normalized = normalize_question_analysis_entities(analysis)
    issuer, target = normalized.sub_questions[0].entities

    assert issuer.match_status == "OUT_OF_UNIVERSE"
    assert issuer.canonical_name is None
    assert target.match_status == "UNKNOWN"
    assert target.canonical_name is None


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
                scope_id="scope_1",
            )],
        )


def test_retriever_human_message_contains_sub_questions():
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
    payload = json.loads(message.content)

    assert payload["sub_questions"] == [
        subquestion.model_dump(mode="json")
        for subquestion in analysis.sub_questions
    ]
    assert payload["scope_candidates"] == []
    assert payload["retrieval_search_count"] == 0
    assert payload["max_retrieval_search_count"] == 15
