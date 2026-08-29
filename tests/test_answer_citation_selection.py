from __future__ import annotations

import json

import pytest

from agent_graph.state import (
    AnswerGeneratorOutput,
    QuestionAnalysis,
    RetrievalResult,
)
from agent_graph.tools import (
    build_answer_generator_human_message,
    build_answer_result_map,
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


def test_flattens_items_and_builds_llm_friendly_payloads():
    state = _state([
        {
            "type": "text",
            "metadata": {
                "retrieval_context": {
                    "corp_name": "삼성전자",
                    "report_name": "사업보고서",
                    "section_path": ["재무에 관한 사항", "재무제표"],
                },
                "disclosure_id": "d1",
                "section_id": "s1",
                "evidence_id": "e1",
            },
            "score": 0.91,
            "content": "유동자산은 100입니다.",
        },
        {
            "type": "r_table",
            "metadata": {
                "retrieval_context": {
                    "corp_name": "삼성전자",
                    "report_name": "사업보고서",
                    "section_path": ["주석"],
                },
            },
            "table_metadata": {"captions": ["자산 현황"], "units": ["백만원"]},
            "columns": ["항목", "금액"],
            "records": [{"record_index": 0, "values": {"항목": "자산", "금액": "100"}}],
            "scope": {"kind": "row_group"},
            "available_record_count": 10,
            "included_record_count": 1,
            "omitted_record_count": 9,
        },
    ])

    payload = _message_payload(state)

    assert set(payload) == {
        "user_question",
        "retrieval_finish_reason",
        "retrieval_results",
    }
    assert payload["retrieval_results"] == [
        {
            "result_id": "answer_result_1",
            "context": "삼성전자 > 사업보고서 > 재무에 관한 사항 > 재무제표",
            "content": "유동자산은 100입니다.",
        },
        {
            "result_id": "answer_result_2",
            "context": "삼성전자 > 사업보고서 > 주석",
            "content": {
                "records": [{"record_index": 0, "values": {"항목": "자산", "금액": "100"}}],
            },
            "table_info": {
                "captions": ["자산 현황"],
                "units": ["백만원"],
                "scope": {"kind": "row_group"},
                "available_record_count": 10,
                "included_record_count": 1,
                "omitted_record_count": 9,
            },
        },
    ]


def test_answer_result_map_preserves_original_item_for_citation_resolution():
    item = {
        "type": "record",
        "fields": {
            "disclosure_id": "d1",
            "section_id": "s1",
            "evidence_id": "e1",
        },
    }

    result_map = build_answer_result_map(_state([item]))

    assert result_map["answer_result_1"]["item"] == item


def test_answer_message_requires_output_when_retrieval_is_insufficient():
    state = _state([])
    state["retrieval_status"] = "INSUFFICIENT"
    state["retrieval_finish_reason"] = "유효한 추가 검색 전략이 없습니다."
    state["selected_result_ids"] = []

    content = build_answer_generator_human_message(state).content
    payload = _message_payload(state)

    assert "retrieval_results가 비어 있어도" in content
    assert "used_result_ids는 빈 목록으로 반환하세요" in content
    assert payload["retrieval_results"] == []


def test_resolves_used_result_to_all_citations_in_its_original_item():
    state = _state([{
        "type": "record",
        "fields": {
            "primary": {"disclosure_id": "d1", "section_id": "s1"},
            "secondary": {"disclosure_id": "d2"},
        },
    }])

    answer = resolve_answer_draft(
        state,
        AnswerGeneratorOutput(
            answer="확인했습니다.",
            used_result_ids=["answer_result_1"],
        ),
    )

    assert [citation.model_dump(exclude_none=True) for citation in answer.citation] == [
        {"disclosure_id": "d1", "section_id": "s1"},
        {"disclosure_id": "d2"},
    ]


def test_rejects_unknown_result_id_even_when_no_citation_exists():
    state = _state([{"type": "record", "fields": {"stock_code": "005930"}}])

    with pytest.raises(ValueError, match="허용되지 않은 answer result ID"):
        resolve_answer_draft(
            state,
            AnswerGeneratorOutput(
                answer="확인했습니다.",
                used_result_ids=["answer_result_99"],
            ),
        )


def test_deduplicates_used_result_ids_and_citations():
    state = _state([{"disclosure_id": "d1"}])

    answer = resolve_answer_draft(
        state,
        AnswerGeneratorOutput(
            answer="확인했습니다.",
            used_result_ids=["answer_result_1", "answer_result_1"],
        ),
    )

    assert len(answer.citation) == 1


def test_company_properties_without_dart_ids_return_empty_citations():
    state = _state([{
        "type": "record",
        "fields": {
            "corp_name": "삼성전자",
            "corp_eng_name": "SAMSUNG ELECTRONICS CO., LTD.",
            "stock_code": "005930",
        },
    }])

    answer = resolve_answer_draft(
        state,
        AnswerGeneratorOutput(
            answer="영문명과 종목 코드입니다.",
            used_result_ids=["answer_result_1"],
        ),
    )

    assert answer.citation == []
