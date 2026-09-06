from __future__ import annotations

import json

import pytest

from agent_graph import graph as graph_module
from agent_graph.state import (
    AiAnswer,
    AnswerGeneratorOutput,
    AnswerValidatorOutput,
    Citation,
    QuestionAnalysis,
    RetrievalResult,
    SubQuestion,
)
from agent_graph.utils import (
    MARKET_CAP_AS_OF_NOTE,
    UNSUPPORTED_ANSWER_FALLBACK,
    append_limitation_notes,
    apply_answer_repairs,
    build_answer_repair_guidance,
    build_answer_validator_human_message,
    normalize_validator_issues,
    validate_and_repair_answer,
)


# 실제 Qdrant TEXT 결과 구조. 식별자는 metadata 안에 중첩되며,
# Answer Generator payload에서는 제거된다.
def _text_item(
    content: str = "2025년 삼성전자의 연결 매출액은 300조원이다.",
    disclosure_id: str = "d20260306000123",
) -> dict:
    return {
        "type": "text",
        "content": content,
        "metadata": {
            "disclosure_id": disclosure_id,
            "section_id": f"{disclosure_id}:src0:s1",
            "evidence_id": f"{disclosure_id}:src0:s1:e1",
            "retrieval_context": {
                "corp_name": "삼성전자",
                "report_name": "사업보고서",
            },
        },
    }


# 실제 Neo4j 기업 metadata 결과 구조. 공시 인용 대상이 아니다.
def _company_record_item() -> dict:
    return {
        "type": "record",
        "fields": {"corp_name": "삼성전자", "market_cap": 14586465},
    }


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
        "question_text": "삼성전자의 2025년 매출액을 알려줘",
        "question_analysis": QuestionAnalysis(
            decision="retrieve",
            normalized_question="삼성전자 2025년 매출액",
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


def _validator_payload(state: dict, ai_answer: AiAnswer) -> dict:
    return json.loads(build_answer_validator_human_message(state, ai_answer).content)


# Validator 입력 -----------------------------------------------------------


def test_validator_payload_includes_source_ids_for_real_text_item():
    """Answer Generator payload는 식별자를 제거하므로, Validator 전용

    payload가 source_ids를 따로 실어주지 않으면 출처 검증이 불가능하다."""

    state = _state([_text_item()])
    ai_answer = AiAnswer(answer="매출액은 300조원입니다.", citation=[])

    payload = _validator_payload(state, ai_answer)

    source_ids = payload["retrieval_results"][0]["source_ids"]
    assert source_ids == [{
        "disclosure_id": "d20260306000123",
        "section_id": "d20260306000123:src0:s1",
        "evidence_id": "d20260306000123:src0:s1:e1",
    }]


def test_validator_payload_includes_draft_citations():
    state = _state([_text_item()])
    ai_answer = AiAnswer(
        answer="매출액은 300조원입니다.",
        citation=[Citation(disclosure_id="d20260306000123")],
    )

    payload = _validator_payload(state, ai_answer)

    assert payload["draft_citations"] == [{"disclosure_id": "d20260306000123"}]


def test_validator_payload_marks_company_metadata_as_uncitable():
    """기업 metadata는 공시 인용 대상이 아니므로 source_ids가 비어야 한다."""

    state = _state([_company_record_item()])
    ai_answer = AiAnswer(answer="시가총액은 14,586,465억원입니다.", citation=[])

    payload = _validator_payload(state, ai_answer)

    assert payload["retrieval_results"][0]["source_ids"] == []


def test_validator_payload_omits_constant_application_notes():
    state = _state([_company_record_item()])
    ai_answer = AiAnswer(answer=f"시가총액입니다. {MARKET_CAP_AS_OF_NOTE}", citation=[])

    payload = _validator_payload(state, ai_answer)

    assert "trusted_application_notes" not in payload
    assert "trusted_field_semantics" not in payload


# LLM 판정 정규화 ----------------------------------------------------------


def test_market_cap_evidence_carries_unit_and_as_of_date():
    """단위 없는 숫자만 넘기면 답변 생성 LLM은 단위를 지어내고, 검증 LLM은

    답변의 "억원"을 근거 없는 표현으로 판정한다. 근거 자체를
    자기설명적으로 만들어 두 문제를 함께 없앤다."""

    state = _state([_company_record_item()])
    ai_answer = AiAnswer(answer="시가총액은 14,586,465억원입니다.", citation=[])

    payload = _validator_payload(state, ai_answer)

    assert (
        payload["retrieval_results"][0]["content"]["market_cap"]
        == "14,586,465억원 (2026-07-24 기준)"
    )


def test_keeps_unsupported_claim_when_number_is_absent_from_evidence():
    state = _state([_text_item()])
    ai_answer = AiAnswer(answer="매출액은 500조원입니다.", citation=[])
    issues = AnswerValidatorOutput(unsupported_claims=["매출액은 500조원입니다."])

    normalized = normalize_validator_issues(state, ai_answer, issues)

    assert normalized.unsupported_claims == ["매출액은 500조원입니다."]


def test_keeps_cross_company_value_swap():
    """A사 값을 B사에 붙인 답변. 숫자 자체는 근거 어딘가에 존재한다."""

    state = _state([
        _text_item("A사의 매출액은 100억원이다.", "d20260306000001"),
        _text_item("B사의 매출액은 200억원이다.", "d20260306000002"),
    ])
    ai_answer = AiAnswer(answer="A사의 매출액은 200억원입니다.", citation=[])
    issues = AnswerValidatorOutput(
        unsupported_claims=["A사의 매출액은 200억원입니다."],
    )

    normalized = normalize_validator_issues(state, ai_answer, issues)

    assert normalized.unsupported_claims == ["A사의 매출액은 200억원입니다."]


def test_keeps_unit_swap():
    """조원을 억원으로 바꾼 답변. 숫자 300은 근거에 그대로 있다."""

    state = _state([_text_item("2025년 매출액은 300조원이다.")])
    ai_answer = AiAnswer(answer="2025년 매출액은 300억원입니다.", citation=[])
    issues = AnswerValidatorOutput(
        unsupported_claims=["2025년 매출액은 300억원입니다."],
    )

    normalized = normalize_validator_issues(state, ai_answer, issues)

    assert normalized.unsupported_claims == ["2025년 매출액은 300억원입니다."]


def test_keeps_decimal_swap():
    """9.42%를 42.9%로 뒤집은 답변. 숫자 토큰만 보면 구분되지 않는다."""

    state = _state([_text_item("ROA는 9.42%이다.")])
    ai_answer = AiAnswer(answer="ROA는 42.9%입니다.", citation=[])
    issues = AnswerValidatorOutput(unsupported_claims=["ROA는 42.9%입니다."])

    normalized = normalize_validator_issues(state, ai_answer, issues)

    assert normalized.unsupported_claims == ["ROA는 42.9%입니다."]


def test_keeps_wrong_metric_with_same_number():
    """순위 숫자는 같지만 지표가 다른 답변."""

    state = _state([_text_item("매출액 순위 1위는 A사이다.")])
    ai_answer = AiAnswer(answer="영업이익 순위 1위는 A사입니다.", citation=[])
    issues = AnswerValidatorOutput(
        unsupported_claims=["영업이익 순위 1위는 A사입니다."],
    )

    normalized = normalize_validator_issues(state, ai_answer, issues)

    assert normalized.unsupported_claims == ["영업이익 순위 1위는 A사입니다."]


def test_keeps_missing_citation_verdict_even_when_numbers_match():
    """수치는 맞지만 출처가 붙지 않은 답변. 지표8(근거 표시) 위반이다."""

    state = _state([_text_item("2025년 매출액은 300조원이다.")])
    ai_answer = AiAnswer(answer="2025년 매출액은 300조원입니다.", citation=[])
    issues = AnswerValidatorOutput(
        unsupported_claims=["2025년 매출액은 300조원입니다."],
    )

    normalized = normalize_validator_issues(state, ai_answer, issues)

    assert normalized.unsupported_claims == ["2025년 매출액은 300조원입니다."]


def test_keeps_requested_fact_when_tokens_are_scattered_across_sentences():
    """"2025년 매출액"과 "2024년 영업이익"을 각각 말한 답변이

    "2025년 영업이익"을 다뤘다고 오판되면 안 된다."""

    state = _state([_text_item()])
    ai_answer = AiAnswer(
        answer="2025년 매출액은 300억원입니다. 영업이익은 2024년에 20억원입니다.",
        citation=[],
    )
    issues = AnswerValidatorOutput(missing_requested_facts=["2025년 영업이익"])

    normalized = normalize_validator_issues(state, ai_answer, issues)

    assert normalized.missing_requested_facts == ["2025년 영업이익"]


def test_drops_note_that_does_not_describe_a_limitation():
    """실제 HyperCLOVA X가 근거 부족 설명 자리에 "확인할 수 있습니다"라는

    정반대 문장을 돌려준 사례가 있었다. 그대로 붙이면 답변이 모순된다."""

    state = _state([_text_item()])
    ai_answer = AiAnswer(answer="매출액은 300조원입니다.", citation=[])
    issues = AnswerValidatorOutput(
        incomplete_evidence_note="현재 제공된 정보로는 시가총액을 확인할 수 있습니다.",
    )

    normalized = normalize_validator_issues(state, ai_answer, issues)

    assert normalized.incomplete_evidence_note is None


def test_keeps_note_that_describes_a_limitation():
    state = _state([_text_item()])
    ai_answer = AiAnswer(answer="A사의 매출액은 300조원입니다.", citation=[])
    issues = AnswerValidatorOutput(
        incomplete_evidence_note="B사의 매출액 근거가 검색되지 않았습니다.",
    )

    normalized = normalize_validator_issues(state, ai_answer, issues)

    assert normalized.incomplete_evidence_note is not None


def test_drops_claim_that_does_not_appear_in_answer():
    state = _state([_text_item()])
    ai_answer = AiAnswer(answer="매출액은 300조원입니다.", citation=[])
    issues = AnswerValidatorOutput(unsupported_claims=["답변에 없는 문장입니다."])

    normalized = normalize_validator_issues(state, ai_answer, issues)

    assert normalized.unsupported_claims == []


def test_drops_requested_fact_already_disclosed_as_unavailable():
    """이미 "확인되지 않았다"고 밝힌 항목을 다시 요구하면 같은 답변만 반복된다."""

    state = _state([_text_item()])
    ai_answer = AiAnswer(
        answer="매출액은 300조원입니다. 2025년 영업이익은 공시에서 확인되지 않았습니다.",
        citation=[],
    )
    issues = AnswerValidatorOutput(missing_requested_facts=["2025년 영업이익"])

    normalized = normalize_validator_issues(state, ai_answer, issues)

    assert normalized.missing_requested_facts == []


def test_normalizes_literal_null_string_note():
    """모델이 null 대신 문자열 "null"을 반환하면 참값이라 실제 고지로 취급된다."""

    state = _state([_text_item()])
    ai_answer = AiAnswer(answer="매출액은 300조원입니다.", citation=[])
    issues = AnswerValidatorOutput(incomplete_evidence_note="null")

    normalized = normalize_validator_issues(state, ai_answer, issues)

    assert normalized.incomplete_evidence_note is None


# 결정론적 보정 -------------------------------------------------------------


def test_unsupported_claim_replaces_whole_answer_and_clears_citations():
    """부분 문자열 삭제는 문장을 훼손하므로 전체를 안전 문구로 교체하고,

    근거가 남지 않았으므로 citation도 함께 비운다."""

    ai_answer = AiAnswer(
        answer="- 매출액은 100억원입니다.\n- 영업이익은 20억원입니다.",
        citation=[Citation(disclosure_id="d20240306000686")],
    )
    issues = AnswerValidatorOutput(unsupported_claims=["영업이익은 20억원입니다."])

    repaired = apply_answer_repairs(ai_answer, issues)

    assert repaired.answer == UNSUPPORTED_ANSWER_FALLBACK
    assert repaired.citation == []


def test_missing_facts_keep_answer_and_citations_intact():
    ai_answer = AiAnswer(
        answer="매출액은 300조원입니다.",
        citation=[Citation(disclosure_id="d20240306000686")],
    )
    issues = AnswerValidatorOutput(missing_requested_facts=["영업이익"])

    repaired = apply_answer_repairs(ai_answer, issues)

    assert repaired.answer.startswith("매출액은 300조원입니다.")
    assert "영업이익" in repaired.answer
    assert repaired.citation == ai_answer.citation


def test_limitation_notes_are_not_duplicated():
    """답변이 이미 같은 한계를 고지했다면 다시 덧붙이지 않는다."""

    ai_answer = AiAnswer(
        answer="매출액은 300조원입니다. 영업이익은 확인되지 않았습니다.",
        citation=[],
    )
    issues = AnswerValidatorOutput(missing_requested_facts=["영업이익"])

    repaired = append_limitation_notes(ai_answer, issues)

    assert repaired.answer == ai_answer.answer


def test_incomplete_evidence_note_is_appended_once():
    ai_answer = AiAnswer(answer="A사의 매출액은 100억원입니다.", citation=[])
    issues = AnswerValidatorOutput(
        incomplete_evidence_note="B사의 매출액 근거가 검색되지 않았습니다.",
    )

    repaired = append_limitation_notes(ai_answer, issues)

    assert "B사의 매출액 근거가 검색되지 않았습니다." in repaired.answer
    assert repaired.answer.count("B사의 매출액 근거가") == 1


def test_build_answer_repair_guidance_marks_evidence_gap_as_context_only():
    issues = AnswerValidatorOutput(
        unsupported_claims=["근거없는문장"],
        incomplete_evidence_note="근거부족사유",
    )

    guidance = build_answer_repair_guidance(issues)

    assert "근거없는문장" in guidance
    assert "고칠 대상 아님" in guidance


# 오케스트레이션 ------------------------------------------------------------


def test_passes_through_when_no_issues():
    state = _state([_text_item()])
    ai_answer = AiAnswer(answer="매출액은 300조원입니다.", citation=[])
    llm = _SequenceLlm([AnswerValidatorOutput()])

    result = validate_and_repair_answer(state, ai_answer, llm=llm)

    assert result == ai_answer
    assert len(llm.calls) == 1


def test_evidence_gap_alone_does_not_trigger_rewrite():
    """검색 결과의 공백은 다시 써도 해소되지 않으므로 재작성하지 않는다."""

    state = _state([_text_item()])
    ai_answer = AiAnswer(answer="A사의 매출액은 300조원입니다.", citation=[])
    llm = _SequenceLlm([AnswerValidatorOutput(
        incomplete_evidence_note="B사의 근거가 없습니다.",
    )])

    result = validate_and_repair_answer(state, ai_answer, llm=llm)

    assert len(llm.calls) == 1
    assert result.answer.startswith("A사의 매출액은 300조원입니다.")
    assert "B사의 근거가 없습니다." in result.answer


def test_regenerates_once_and_accepts_clean_result():
    state = _state([_text_item()])
    ai_answer = AiAnswer(answer="매출액은 500조원입니다.", citation=[])
    llm = _SequenceLlm([
        AnswerValidatorOutput(unsupported_claims=["매출액은 500조원입니다."]),
        AnswerGeneratorOutput(answer="매출액은 300조원입니다.", used_result_ids=[]),
        AnswerValidatorOutput(),
    ])

    result = validate_and_repair_answer(state, ai_answer, llm=llm)

    assert result.answer == "매출액은 300조원입니다."
    assert len(llm.calls) == 3


def test_falls_back_safely_when_rewrite_still_unsupported():
    state = _state([_text_item()])
    ai_answer = AiAnswer(
        answer="매출액은 500조원입니다.",
        citation=[Citation(disclosure_id="d20240306000686")],
    )
    llm = _SequenceLlm([
        AnswerValidatorOutput(unsupported_claims=["매출액은 500조원입니다."]),
        AnswerGeneratorOutput(answer="매출액은 500조원입니다.", used_result_ids=[]),
        AnswerValidatorOutput(unsupported_claims=["매출액은 500조원입니다."]),
    ])

    result = validate_and_repair_answer(state, ai_answer, llm=llm)

    assert result.answer == UNSUPPORTED_ANSWER_FALLBACK
    assert result.citation == []


# 노드 동작 ----------------------------------------------------------------


def test_node_returns_validated_answer():
    state = _state([_text_item()])
    state["ai_answer"] = AiAnswer(answer="매출액은 300조원입니다.", citation=[])
    llm = _SequenceLlm([AnswerValidatorOutput()])

    update = graph_module.answer_validator(state, llm=llm)

    assert update["ai_answer"].answer == "매출액은 300조원입니다."
    assert update["answer_validation_status"] == "PASSED"


def test_node_keeps_original_answer_when_validation_fails():
    """검증 LLM 장애가 이미 정상 생성된 답변을 API 500으로 만들면 안 된다."""

    state = _state([_text_item()])
    original = AiAnswer(
        answer="매출액은 300조원입니다.",
        citation=[Citation(disclosure_id="d20240306000686")],
    )
    state["ai_answer"] = original
    llm = _SequenceLlm([TimeoutError("validator timeout")])

    update = graph_module.answer_validator(state, llm=llm)

    assert update["ai_answer"] == original
    assert update["answer_validation_status"] == "SKIPPED"
    assert any("answer_validator 실패" in error for error in update["errors"])


def test_node_does_not_swallow_programming_errors():
    """모든 예외를 삼키면 코드 결함이 "검증 생략"으로 조용히 묻힌다."""

    state = _state([_text_item()])
    state["ai_answer"] = AiAnswer(answer="매출액은 300조원입니다.", citation=[])
    llm = _SequenceLlm([KeyError("내부 계약 위반")])

    with pytest.raises(KeyError):
        graph_module.answer_validator(state, llm=llm)
