from __future__ import annotations

from agent_graph import graph as graph_module
from agent_graph.state import (
    AiAnswer,
    AnswerGeneratorOutput,
    AnswerValidatorOutput,
    QuestionAnalysis,
    RetrievalResult,
    SubQuestion,
)
from agent_graph.utils import (
    apply_answer_repairs,
    build_answer_repair_guidance,
    validate_and_repair_answer,
)


def _state(items: list[dict], requested_facts: list[str] | None = None) -> dict:
    sub_questions = []
    if requested_facts:
        sub_questions = [SubQuestion(
            question="삼성전자의 정보는 무엇인가?",
            entities=[],
            events=[],
            intents=["DETAIL"],
            periods=[],
            requested_facts=requested_facts,
        )]
    return {
        "question_id": "question-1",
        "question_text": "삼성전자의 정보를 알려줘",
        "question_analysis": QuestionAnalysis(
            decision="retrieve",
            normalized_question="삼성전자 정보",
            decision_reason="공시 근거가 필요합니다.",
            sub_questions=sub_questions,
        ),
        "retrieval_results": [
            RetrievalResult(
                result_id="retrieval:plan_1",
                plan_id="plan_1",
                source="qdrant",
                query="검색 쿼리",
                items=items,
                result_count=len(items),
            )
        ],
        "retrieval_status": "COMPLETE",
        "retrieval_finish_reason": "필요한 결과를 찾았습니다.",
        "selected_result_ids": ["retrieval:plan_1"],
    }


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


# apply_answer_repairs --------------------------------------------------


def test_apply_answer_repairs_strips_unsupported_claims_and_appends_notes():
    ai_answer = AiAnswer(
        answer="매출액은 100억원입니다. 영업이익은 20억원입니다.",
        citation=[],
    )
    issues = AnswerValidatorOutput(
        unsupported_claims=["영업이익은 20억원입니다."],
        missing_requested_facts=["당기순이익"],
    )

    repaired = apply_answer_repairs(ai_answer, issues)

    assert "영업이익" not in repaired.answer
    assert "매출액은 100억원입니다." in repaired.answer
    assert "당기순이익" in repaired.answer
    assert repaired.citation == ai_answer.citation


def test_apply_answer_repairs_falls_back_when_answer_becomes_empty():
    ai_answer = AiAnswer(answer="영업이익은 20억원입니다.", citation=[])
    issues = AnswerValidatorOutput(unsupported_claims=["영업이익은 20억원입니다."])

    repaired = apply_answer_repairs(ai_answer, issues)

    assert repaired.answer == "제공된 근거만으로는 확인 가능한 답변을 생성할 수 없습니다."


def test_apply_answer_repairs_notes_incomplete_evidence():
    ai_answer = AiAnswer(answer="A사의 매출액은 100억원입니다.", citation=[])
    issues = AnswerValidatorOutput(
        incomplete_evidence_note="B사의 매출액 근거가 검색되지 않았습니다.",
    )

    repaired = apply_answer_repairs(ai_answer, issues)

    assert "A사의 매출액은 100억원입니다." in repaired.answer
    assert "B사의 매출액 근거가 검색되지 않았습니다." in repaired.answer


# build_answer_repair_guidance -------------------------------------------


def test_build_answer_repair_guidance_lists_all_issue_types():
    issues = AnswerValidatorOutput(
        unsupported_claims=["근거없는문장"],
        missing_requested_facts=["누락사실"],
        incomplete_evidence_note="근거부족사유",
    )

    guidance = build_answer_repair_guidance(issues)

    assert "근거없는문장" in guidance
    assert "누락사실" in guidance
    assert "근거부족사유" in guidance


# validate_and_repair_answer ----------------------------------------------


def test_validate_and_repair_answer_passes_through_when_no_issues():
    state = _state([{"disclosure_id": "d1", "content": "매출액은 100억원이다."}])
    ai_answer = AiAnswer(answer="매출액은 100억원입니다.", citation=[])
    llm = _SequenceLlm([AnswerValidatorOutput()])

    result = validate_and_repair_answer(state, ai_answer, llm=llm)

    assert result == ai_answer
    assert len(llm.calls) == 1


def test_validate_and_repair_answer_regenerates_and_accepts_clean_result():
    state = _state([{"disclosure_id": "d1", "content": "매출액은 100억원이다."}])
    ai_answer = AiAnswer(answer="매출액은 200억원입니다.", citation=[])
    llm = _SequenceLlm([
        AnswerValidatorOutput(unsupported_claims=["매출액은 200억원입니다."]),
        AnswerGeneratorOutput(answer="매출액은 100억원입니다.", used_result_ids=[]),
        AnswerValidatorOutput(),
    ])

    result = validate_and_repair_answer(state, ai_answer, llm=llm)

    assert result.answer == "매출액은 100억원입니다."
    assert len(llm.calls) == 3


def test_validate_and_repair_answer_falls_back_when_regeneration_still_bad():
    state = _state([{"disclosure_id": "d1", "content": "매출액은 100억원이다."}])
    ai_answer = AiAnswer(answer="매출액은 200억원입니다.", citation=[])
    llm = _SequenceLlm([
        AnswerValidatorOutput(unsupported_claims=["매출액은 200억원입니다."]),
        AnswerGeneratorOutput(answer="매출액은 200억원입니다.", used_result_ids=[]),
        AnswerValidatorOutput(unsupported_claims=["매출액은 200억원입니다."]),
    ])

    result = validate_and_repair_answer(state, ai_answer, llm=llm)

    assert result.answer == "제공된 근거만으로는 확인 가능한 답변을 생성할 수 없습니다."
    assert len(llm.calls) == 3


# graph.answer_validator node ----------------------------------------------


def test_answer_validator_node_updates_ai_answer():
    state = _state([{"disclosure_id": "d1", "content": "매출액은 100억원이다."}])
    state["ai_answer"] = AiAnswer(answer="매출액은 100억원입니다.", citation=[])
    llm = _SequenceLlm([AnswerValidatorOutput()])

    update = graph_module.answer_validator(state, llm=llm)

    assert update["ai_answer"].answer == "매출액은 100억원입니다."
