import json

from fastapi.testclient import TestClient

import main
from agent_graph.state import (
    AiAnswer,
    Citation,
    QuestionAnalysis,
    RetrievalResult,
    ThinkTraceEvent,
)


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
            "think_trace_events": [
                ThinkTraceEvent(
                    type="node",
                    name="question_analyzer",
                    message="공시 근거 검색이 필요합니다.",
                    details={"decision": "retrieve"},
                ),
                ThinkTraceEvent(
                    type="tool",
                    name="finish",
                    message="관련 근거를 찾았습니다.",
                    details={"status": "COMPLETE"},
                ),
            ],
            "retrieval_results": [RetrievalResult(
                result_id="retrieval:plan_1",
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
            "answer_validation_status": "PASSED",
        }


def test_answer_endpoint_returns_evaluation_contract(monkeypatch):
    monkeypatch.setattr(main, "graph", _FakeGraph())
    monkeypatch.setattr(
        main,
        "format_citations",
        lambda citations, *, style: [
            "[근거: 사업보고서 (2023.12), 2024-03-06]"
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
    assert json.loads(payload["think_trace"]) == {
        "query_text": "삼성전자의 설비 투자를 알려줘",
        "steps": [
            {
                "type": "node",
                "name": "question_analyzer",
                "message": "공시 근거 검색이 필요합니다.",
                "details": {"decision": "retrieve"},
            },
            {
                "type": "tool",
                "name": "finish",
                "message": "관련 근거를 찾았습니다.",
                "details": {"status": "COMPLETE"},
            },
        ],
    }
    assert all(isinstance(value, str) for value in payload.values())


def test_think_trace_includes_calculation_and_sanitizes_errors():
    trace = json.loads(main._build_think_trace({
        "question_text": "매출액 합계를 구해줘",
        "retrieval_results": [RetrievalResult(
            result_id="derived:plan_2",
            source="derived",
            status="SUCCESS",
            query='{"operation":"sum"}',
            items=[{
                "type": "record",
                "fields": {
                    "variable_name": "매출액 합계",
                    "operation": "sum",
                    "value": "300",
                    "unit": "억원",
                    "input_count": 2,
                },
            }],
            result_count=1,
        )],
        "errors": ["password=secret internal endpoint"],
    }))

    assert trace["calculation_history"][0]["operation"] == "sum"
    assert trace["calculation_history"][0]["value"] == "300"
    assert trace["warnings"]["count"] == 1
    assert "secret" not in json.dumps(trace)


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
    assert json.loads(response.json()["retrieved_context"]) == {
        "citations": [],
        "results": [],
    }


def test_answer_endpoint_requires_both_query_parameters():
    client = TestClient(main.app)

    response = client.get("/answer", params={"question_id": "Q-001"})

    assert response.status_code == 422


def test_answer_endpoint_preserves_contract_when_graph_fails(monkeypatch):
    class FailingGraph:
        def invoke(self, _state):
            raise ConnectionError("secret upstream detail")

    monkeypatch.setattr(main, "graph", FailingGraph())
    client = TestClient(main.app)

    response = client.get(
        "/answer",
        params={"question_id": "Q-ERR", "question": "오류 상황 질문"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "question_id", "question", "retrieved_context", "think_trace", "answer",
    }
    assert payload["question_id"] == "Q-ERR"
    assert payload["question"] == "오류 상황 질문"
    assert "확인할 수 없습니다" in payload["answer"]
    assert "secret upstream detail" not in payload["think_trace"]
    assert json.loads(payload["retrieved_context"]) == {
        "citations": [], "results": [],
    }
    trace = json.loads(payload["think_trace"])
    assert trace["final_basis"]["retrieval_status"] == "ERROR"
    assert trace["warnings"]["type"] == "ConnectionError"
