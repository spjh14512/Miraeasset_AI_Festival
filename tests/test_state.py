from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_graph.state import PlanDraft, RetrievalResult


def _plan_draft(source: str) -> PlanDraft:
    return PlanDraft(
        source=source,
        query="삼성전자 매출액",
        purpose="질문에 답하기 위한 근거 확보",
        dependencies=[],
        scope_id="scope_1",
    )


def _retrieval_result(
    source: str,
    status: str | None = None,
    result_count: int = 1,
) -> RetrievalResult:
    # 실제 파서는 source와 무관하게 "retrieval:" 접두사를 쓰고, "derived:"는
    # 계산 결과 전용이다 (retrieval_result_parser.parse_qdrant_response 등 참고).
    prefix = "derived" if source == "derived" else "retrieval"
    return RetrievalResult(
        result_id=f"{prefix}:plan_1",
        plan_id="plan_1",
        source=source,
        status=status,
        query="검색 쿼리",
        items=[{"value": 1}] if result_count else [],
        result_count=result_count,
    )


def test_plan_draft_accepts_qdrant_and_neo4j_sources():
    assert _plan_draft("qdrant").source == "qdrant"
    assert _plan_draft("neo4j").source == "neo4j"


def test_plan_draft_rejects_derived_source():
    with pytest.raises(ValidationError):
        _plan_draft("derived")


def test_retrieval_result_accepts_qdrant_and_neo4j_sources():
    qdrant_result = _retrieval_result("qdrant")
    neo4j_result = _retrieval_result("neo4j")
    assert qdrant_result.source == "qdrant"
    assert neo4j_result.source == "neo4j"
    assert qdrant_result.result_id == "retrieval:plan_1"
    assert neo4j_result.result_id == "retrieval:plan_1"


def test_retrieval_result_accepts_derived_source():
    result = _retrieval_result("derived")
    assert result.source == "derived"
    assert result.status == "SUCCESS"


def test_retrieval_result_accepts_invalid_input_status():
    result = _retrieval_result("derived", status="INVALID_INPUT", result_count=0)
    assert result.status == "INVALID_INPUT"
    assert result.result_id == "derived:plan_1"


def test_retrieval_result_rejects_unknown_source():
    with pytest.raises(ValidationError):
        _retrieval_result("web")


def test_retrieval_result_rejects_unknown_status():
    with pytest.raises(ValidationError):
        _retrieval_result("qdrant", status="UNKNOWN_STATUS")
