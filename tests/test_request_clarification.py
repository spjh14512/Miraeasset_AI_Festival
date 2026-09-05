from __future__ import annotations

import pytest

from agent_graph.graph import request_clarification
from agent_graph.state import AiAnswer, QuestionAnalysis


def _clarify_analysis(
    clarification_question: str = "어느 회계연도를 기준으로 계산할까요?",
) -> QuestionAnalysis:
    return QuestionAnalysis(
        decision="clarify",
        normalized_question="삼성전자의 PER을 알려줘",
        decision_reason="기준 회계연도가 지정되지 않았습니다.",
        clarification_question=clarification_question,
    )


def test_returns_clarification_question_as_ai_answer():
    state = {
        "question_id": "question-1",
        "question_text": "삼성전자 PER 알려줘",
        "question_analysis": _clarify_analysis(),
    }

    update = request_clarification(state)

    assert update == {
        "ai_answer": AiAnswer(
            answer="어느 회계연도를 기준으로 계산할까요?",
            citation=[],
        )
    }


def test_accepts_question_analysis_given_as_mapping():
    state = {
        "question_id": "question-1",
        "question_text": "삼성전자 PER 알려줘",
        "question_analysis": _clarify_analysis().model_dump(),
    }

    update = request_clarification(state)

    assert update["ai_answer"].answer == "어느 회계연도를 기준으로 계산할까요?"


def test_rejects_state_without_question_analysis():
    with pytest.raises(ValueError, match="question_analysis가 생성되지 않았습니다"):
        request_clarification({
            "question_id": "question-1",
            "question_text": "삼성전자 PER 알려줘",
        })


def test_rejects_non_clarify_decision():
    state = {
        "question_id": "question-1",
        "question_text": "삼성전자 매출액 알려줘",
        "question_analysis": QuestionAnalysis(
            decision="direct",
            normalized_question="안녕하세요",
            decision_reason="일반 대화입니다.",
        ),
    }

    with pytest.raises(ValueError, match="clarify 결정이 아닌 질문은"):
        request_clarification(state)
