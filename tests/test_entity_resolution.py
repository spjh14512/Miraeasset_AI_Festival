from __future__ import annotations

import pytest

from agent_graph import graph as graph_module
from agent_graph.llm import MAX_LLM_RETRIES
from agent_graph.state import (
    QuestionAnalysis,
    QuestionAnalyzerOutput,
    SubQuestion,
)
from agent_graph.utils import (
    AmbiguousEntityConflict,
    apply_deterministic_entity_resolution,
    build_fallback_clarification,
    resolve_company_mention,
    resolve_question_analyzer_output,
)


# resolve_company_mention -----------------------------------------------


def test_resolve_company_mention_exact_name_match():
    status, canonical, candidates = resolve_company_mention("삼성전자")

    assert (status, canonical, candidates) == ("MATCHED", "삼성전자", [])


def test_resolve_company_mention_exact_stock_code_match():
    status, canonical, candidates = resolve_company_mention("005930")

    assert (status, canonical, candidates) == ("MATCHED", "삼성전자", [])


def test_resolve_company_mention_unique_prefix_match():
    status, canonical, candidates = resolve_company_mention("현대자동차")

    assert (status, canonical, candidates) == ("MATCHED", "현대자동차", [])


def test_resolve_company_mention_group_name_is_ambiguous():
    status, canonical, candidates = resolve_company_mention("현대")

    assert status == "AMBIGUOUS"
    assert canonical is None
    assert "현대자동차" in candidates
    assert "현대건설" in candidates
    assert len(candidates) >= 5


def test_resolve_company_mention_not_in_registry():
    status, canonical, candidates = resolve_company_mention("존재하지않는기업")

    assert (status, canonical, candidates) == ("NOT_FOUND", None, [])


def test_resolve_company_mention_blank_mention_is_not_found():
    assert resolve_company_mention("   ") == ("NOT_FOUND", None, [])


def test_resolve_company_mention_does_not_substring_match_generic_word():
    # "전자"는 "삼성전자"에 포함되지만 접두어가 아니므로 매칭되지 않아야 한다.
    assert resolve_company_mention("전자") == ("NOT_FOUND", None, [])


# apply_deterministic_entity_resolution ----------------------------------


def _sub_question(mention: str, match_status: str) -> SubQuestion:
    return SubQuestion(
        question=f"{mention}의 2024년 매출액은 얼마인가?",
        entities=[{
            "mention": mention,
            "roles": ["ISSUER"],
            "match_status": match_status,
        }],
        events=[],
        intents=["AMOUNT"],
        periods=[],
        requested_facts=["매출액"],
    )


def test_apply_deterministic_entity_resolution_overrides_llm_matched():
    analysis = QuestionAnalysis(
        decision="retrieve",
        normalized_question="현대의 2024년 매출액은 얼마인가?",
        decision_reason="검색이 필요합니다.",
        sub_questions=[_sub_question("현대", "MATCHED")],
    )

    corrected = apply_deterministic_entity_resolution(analysis)

    entity = corrected.sub_questions[0].entities[0]
    assert entity.match_status == "AMBIGUOUS"
    assert entity.canonical_name is None


def test_apply_deterministic_entity_resolution_fixes_llm_unknown_for_real_company():
    """실제 HyperCLOVA X 호출에서 '현대자동차', '두산'이 UNKNOWN으로 잘못
    나온 사례를 재현한다. LLM의 UNKNOWN 판단을 신뢰해 건너뛰면 이 오류가
    그대로 남는다."""

    analysis = QuestionAnalysis(
        decision="retrieve",
        normalized_question="현대자동차의 2024년 매출액은 얼마인가?",
        decision_reason="검색이 필요합니다.",
        sub_questions=[_sub_question("현대자동차", "UNKNOWN")],
    )

    corrected = apply_deterministic_entity_resolution(analysis)

    entity = corrected.sub_questions[0].entities[0]
    assert entity.match_status == "MATCHED"
    assert entity.canonical_name == "현대자동차"


# resolve_question_analyzer_output ---------------------------------------


def test_resolve_question_analyzer_output_raises_on_ambiguous_issuer_conflict():
    output = QuestionAnalyzerOutput(question_analysis=QuestionAnalysis(
        decision="retrieve",
        normalized_question="한화의 2024년 매출액은 얼마인가?",
        decision_reason="검색이 필요합니다.",
        sub_questions=[_sub_question("한화", "UNKNOWN")],
    ))

    with pytest.raises(AmbiguousEntityConflict) as exc_info:
        resolve_question_analyzer_output(output)

    assert exc_info.value.mention == "한화"
    assert "한화솔루션" in exc_info.value.candidates


def test_resolve_question_analyzer_output_passes_when_decision_is_clarify():
    output = QuestionAnalyzerOutput(question_analysis=QuestionAnalysis(
        decision="clarify",
        normalized_question="한화의 2024년 매출액은 얼마인가?",
        decision_reason="기업을 특정할 수 없습니다.",
        clarification_question="한화솔루션, 한화에어로스페이스, 한화오션 중 어느 기업인가요?",
        sub_questions=[_sub_question("한화", "UNKNOWN")],
    ))

    resolved = resolve_question_analyzer_output(output)

    assert resolved.question_analysis.decision == "clarify"
    assert (
        resolved.question_analysis.sub_questions[0].entities[0].match_status
        == "AMBIGUOUS"
    )


def test_resolve_question_analyzer_output_passes_for_matched_entity():
    output = QuestionAnalyzerOutput(question_analysis=QuestionAnalysis(
        decision="retrieve",
        normalized_question="삼성전자의 2024년 매출액은 얼마인가?",
        decision_reason="검색이 필요합니다.",
        sub_questions=[_sub_question("삼성전자", "UNKNOWN")],
    ))

    resolved = resolve_question_analyzer_output(output)

    entity = resolved.question_analysis.sub_questions[0].entities[0]
    assert entity.match_status == "MATCHED"
    assert entity.canonical_name == "삼성전자"


# build_fallback_clarification --------------------------------------------


def test_build_fallback_clarification_lists_real_candidates():
    analysis = build_fallback_clarification("두산", [
        "두산로보틱스", "두산에너빌리티", "두산퓨얼셀",
    ])

    assert analysis.decision == "clarify"
    assert analysis.clarification_question is not None
    assert "두산로보틱스" in analysis.clarification_question
    assert "두산퓨얼셀" in analysis.clarification_question


# graph.question_analyzer end-to-end --------------------------------------


class _SequenceLlm:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output


def _ambiguous_retrieve_output(mention: str) -> QuestionAnalyzerOutput:
    return QuestionAnalyzerOutput(question_analysis=QuestionAnalysis(
        decision="retrieve",
        normalized_question=f"{mention}의 2024년 매출액은 얼마인가?",
        decision_reason="검색이 필요합니다.",
        sub_questions=[_sub_question(mention, "UNKNOWN")],
    ))


def test_question_analyzer_retries_and_accepts_corrected_clarify():
    llm = _SequenceLlm([
        _ambiguous_retrieve_output("현대"),
        QuestionAnalyzerOutput(question_analysis=QuestionAnalysis(
            decision="clarify",
            normalized_question="현대의 2024년 매출액은 얼마인가?",
            decision_reason="기업을 특정할 수 없습니다.",
            clarification_question="현대자동차, 현대모비스 등 중 어느 기업인가요?",
            sub_questions=[_sub_question("현대", "UNKNOWN")],
        )),
    ])

    update = graph_module.question_analyzer({
        "question_id": "question-1",
        "question_text": "현대의 2024년 매출액 알려줘",
    }, llm=llm)

    assert update["question_analysis"].decision == "clarify"
    assert len(llm.calls) == 2
    assert "특정할 수 없어" in llm.calls[1][-1].content


def test_question_analyzer_falls_back_to_safe_clarify_without_raising():
    """모델이 재시도에서도 계속 같은 실수를 반복하면, 500 대신 코드가

    만든 안전한 clarify 응답으로 끝나야 한다."""

    llm = _SequenceLlm([
        _ambiguous_retrieve_output("한화")
        for _ in range(MAX_LLM_RETRIES + 1)
    ])

    update = graph_module.question_analyzer({
        "question_id": "question-1",
        "question_text": "한화의 2024년 매출액 알려줘",
    }, llm=llm)

    analysis = update["question_analysis"]
    assert analysis.decision == "clarify"
    assert "한화솔루션" in analysis.clarification_question
    assert len(llm.calls) == MAX_LLM_RETRIES + 1


def test_question_analyzer_passes_through_unambiguous_company():
    llm = _SequenceLlm([
        QuestionAnalyzerOutput(question_analysis=QuestionAnalysis(
            decision="retrieve",
            normalized_question="현대자동차의 2024년 매출액은 얼마인가?",
            decision_reason="검색이 필요합니다.",
            sub_questions=[_sub_question("현대자동차", "UNKNOWN")],
        )),
    ])

    update = graph_module.question_analyzer({
        "question_id": "question-1",
        "question_text": "현대자동차의 2024년 매출액 알려줘",
    }, llm=llm)

    analysis = update["question_analysis"]
    assert analysis.decision == "retrieve"
    assert len(llm.calls) == 1
    entity = analysis.sub_questions[0].entities[0]
    assert entity.match_status == "MATCHED"
    assert entity.canonical_name == "현대자동차"
