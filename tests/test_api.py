from fastapi.testclient import TestClient

import main
from agent_graph.state import AiAnswer, Citation, QuestionAnalysis, RetrievalResult


class _FakeGraph:
    def invoke(self, state):
        assert state["question_id"] == "Q-001"
        assert state["question_text"] == "삼성전자의 설비 투자를 알려줘"
        return {
            **state,
            "question_analysis": QuestionAnalysis(
                decision="retrieve",
                normalized_question="삼성전자 설비 투자",
                decision_reason="공시 근거 검색이 필요합니다.",
            ),
            "retrieval_status": "COMPLETE",
            "retrieval_finish_reason": "관련 근거를 찾았습니다.",
            "selected_result_ids": ["retrieval:plan_1"],
            "retrieval_results": [RetrievalResult(
                result_id="retrieval:plan_1",
                plan_id="plan_1",
                source="qdrant",
                query="삼성전자 설비 투자",
                items=[{"text": "검색 문서 본문"}],
                result_count=1,
            )],
            "ai_answer": AiAnswer(
                answer="최종 답변",
                citation=[Citation(
                    disclosure_id="d20240306000686",
                    section_id="d20240306000686:src0:s27",
                    evidence_id="d20240306000686:src0:s27:e8",
                )],
            ),
        }


def test_answer_endpoint_returns_evaluation_contract(monkeypatch):
    monkeypatch.setattr(main, "graph", _FakeGraph())
    monkeypatch.setattr(
        main,
        "format_citations",
        lambda citations, *, style: [
            "[사업보고서 (2023.12)(20240306000686) > "
            "IV. 이사의 경영진단 및 분석의견 > 가. 연결 재무상태]"
        ],
    )
    client = TestClient(main.app)

    response = client.get(
        "/answer",
        params={
            "question_id": "Q-001",
            "question": "삼성전자의 설비 투자를 알려줘",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()
    assert payload["question_id"] == "Q-001"
    assert payload["question"] == "삼성전자의 설비 투자를 알려줘"
    assert payload["answer"] == "최종 답변"
    assert payload["retrieved_context"] == (
        "[사업보고서 (2023.12)(20240306000686) > "
        "IV. 이사의 경영진단 및 분석의견 > 가. 연결 재무상태]"
    )
    assert payload["think_trace"] == (
        '{"query_text": "삼성전자의 설비 투자를 알려줘", '
        '"reason": "관련 근거를 찾았습니다."}'
    )
    assert all(isinstance(value, str) for value in payload.values())


class _FakeDirectGraph:
    """answer_directly가 실제로 반환하는 상태 형태(ai_answer, citation=[])를 흉내낸다."""

    def invoke(self, state):
        return {
            **state,
            "question_analysis": QuestionAnalysis(
                decision="direct",
                normalized_question=state["question_text"],
                decision_reason="공시 근거가 필요 없는 일반 대화입니다.",
            ),
            "ai_answer": AiAnswer(answer="안녕하세요! 무엇을 도와드릴까요?", citation=[]),
        }


def test_answer_endpoint_returns_200_for_direct_decision(monkeypatch):
    """이전에는 answer_directly가 상태에 없는 키를 반환해 500이 났다."""

    monkeypatch.setattr(main, "graph", _FakeDirectGraph())
    client = TestClient(main.app)

    response = client.get(
        "/answer",
        params={"question_id": "Q-002", "question": "안녕하세요"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "안녕하세요! 무엇을 도와드릴까요?"


def test_answer_endpoint_requires_both_query_parameters():
    client = TestClient(main.app)

    response = client.get("/answer", params={"question_id": "Q-001"})

    assert response.status_code == 422
