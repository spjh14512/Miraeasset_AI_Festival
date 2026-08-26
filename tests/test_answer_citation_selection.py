from __future__ import annotations

import json

import pytest

from agent_graph.state import (
    AnswerGeneratorOutput,
    EvidenceSelection,
    QuestionAnalysis,
    RetrievalResult,
)
from agent_graph.tools import (
    build_answer_generator_human_message,
    build_citation_candidates,
    resolve_answer_draft,
)


def _message_payload(state: dict) -> dict:
    content = build_answer_generator_human_message(state).content
    json_dump = content.split("[입력]\n\n", 1)[1].split("\n\n\n[출력]", 1)[0]
    return json.loads(json_dump)


def _state(items: list[dict]):
    return {
        "question_id": "question-1",
        "question_text": "삼성전자의 정보를 알려줘",
        "question_analysis": QuestionAnalysis(
            decision="retrieve",
            normalized_question="삼성전자 정보",
            decision_reason="공시 근거가 필요합니다.",
        ),
        "retrieval_results": [
            RetrievalResult(
                result_id="retrieval:plan_1",
                plan_id="plan_1",
                source="neo4j",
                query="MATCH ...",
                items=items,
                result_count=len(items),
            )
        ],
        "retrieval_status": "COMPLETE",
        "retrieval_finish_reason": "필요한 결과를 찾았습니다.",
        "selected_evidence": [
            EvidenceSelection(
                result_id="retrieval:plan_1",
                item_indexes=None,
                reason="답변에 필요한 결과입니다.",
            )
        ],
    }


def test_builds_candidates_only_from_actual_dart_ids():
    state = _state([
        {
            "type": "record",
            "fields": {
                "disclosure_id": "d1",
                "section_id": "s1",
                "evidence_id": "e1",
            },
        },
        {
            "type": "record",
            "fields": {
                "corp_eng_name": "SAMSUNG ELECTRONICS CO., LTD.",
                "stock_code": "005930",
            },
        },
    ])

    assert build_citation_candidates(state) == [{
        "reference_id": "C1",
        "source_item_reference_id": "R1-I1",
        "citation": {
            "disclosure_id": "d1",
            "section_id": "s1",
            "evidence_id": "e1",
        },
    }]


def test_accepts_disclosure_only_candidate():
    state = _state([{"disclosure_id": "d1"}])

    assert build_citation_candidates(state)[0]["citation"] == {
        "disclosure_id": "d1"
    }


def test_does_not_treat_internal_reference_as_citation():
    state = _state([{
        "item_reference_id": "R1-I1",
        "result_id": "retrieval:plan_1",
        "evidence_id": "e1",
    }])

    assert build_citation_candidates(state) == []
    payload = _message_payload(state)
    assert payload["citation_candidates"] == []


def test_resolves_selected_reference_to_citation():
    state = _state([{
        "disclosure_id": "d1",
        "section_id": "s1",
    }])

    answer = resolve_answer_draft(
        state,
        AnswerGeneratorOutput(
            answer="확인했습니다.",
            citation_reference_ids=["C1"],
        ),
    )

    assert answer.answer == "확인했습니다."
    assert answer.citation[0].model_dump(exclude_none=True) == {
        "disclosure_id": "d1",
        "section_id": "s1",
    }


def test_rejects_unknown_reference_id():
    state = _state([{"disclosure_id": "d1"}])

    with pytest.raises(ValueError, match="허용되지 않은 citation reference"):
        resolve_answer_draft(
            state,
            AnswerGeneratorOutput(
                answer="확인했습니다.",
                citation_reference_ids=["C99"],
            ),
        )


def test_deduplicates_selected_reference_ids():
    state = _state([{"disclosure_id": "d1"}])

    answer = resolve_answer_draft(
        state,
        AnswerGeneratorOutput(
            answer="확인했습니다.",
            citation_reference_ids=["C1", "C1"],
        ),
    )

    assert len(answer.citation) == 1


def test_company_properties_without_dart_ids_return_empty_citations():
    state = _state([{
        "type": "record",
        "fields": {
            "corp_eng_name": "SAMSUNG ELECTRONICS CO., LTD.",
            "stock_code": "005930",
        },
    }])

    answer = resolve_answer_draft(
        state,
        AnswerGeneratorOutput(
            answer="영문명과 종목 코드입니다.",
            citation_reference_ids=["R1-I1"],
        ),
    )

    assert answer.citation == []
