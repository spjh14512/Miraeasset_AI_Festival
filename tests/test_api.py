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


def test_answer_endpoint_requires_both_query_parameters():
    client = TestClient(main.app)

    response = client.get("/answer", params={"question_id": "Q-001"})

    assert response.status_code == 422
