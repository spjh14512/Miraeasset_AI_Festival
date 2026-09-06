from __future__ import annotations

from typing import Any

import pytest
from neo4j import Record as Neo4jRecord
from pydantic import ValidationError

from agent_graph import tools as tool_module
from agent_graph import utils
from agent_graph.state import RetrievalResult
from agent_graph.tools import retrieve_correction_history
from agent_graph.utils import execute_tool_call, validate_retriever_tool_call


class _Neo4jSession:
    def __init__(self, records: list[Neo4jRecord]):
        self.records = records
        self.call = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def run(self, cypher, parameters):
        self.call = (cypher, parameters)
        return self.records


class _Neo4jDriver:
    def __init__(self, records: list[Neo4jRecord]):
        self.opened_session = _Neo4jSession(records)

    def session(self):
        return self.opened_session


def _state() -> dict[str, Any]:
    return {
        "question_id": "question-1",
        "question_text": "이 공시의 정정내용을 알려줘",
        "next_plan_seq": 3,
        "retrieval_results": [],
    }


def _successful_result(disclosure_id: str, plan_id: str) -> RetrievalResult:
    return RetrievalResult(
        result_id=f"retrieval:{plan_id}",
        source="neo4j",
        query="fixed correction query",
        items=[{
            "type": "record",
            "fields": {
                "disclosure_id": disclosure_id,
                "correction_content": "정정 내용",
            },
        }],
        result_count=1,
    )


def test_correction_history_executor_uses_fixed_parameterized_query(monkeypatch):
    records = [
        Neo4jRecord([
            ("disclosure_id", "d20240201000002"),
            ("correction_date", "2024-02-01"),
            ("correction_content", "금액 정정"),
        ]),
        Neo4jRecord([
            ("disclosure_id", "d20240301000003"),
            ("correction_date", "2024-03-01"),
            ("correction_content", "기간 정정"),
        ]),
    ]
    driver = _Neo4jDriver(records)
    monkeypatch.setattr(utils, "neo4j_driver", driver)

    result = utils.correction_history_executor(
        "d20240301000003",
        "plan_3",
    )

    assert driver.opened_session.call == (
        utils.CORRECTION_HISTORY_QUERY,
        {"disclosure_id": "d20240301000003"},
    )
    assert "[:CORRECTS*1..]" in result.query
    assert "latest.is_latest_version = true" in result.query
    assert result.source == "neo4j"
    assert result.result_count == 2
    assert [
        item["fields"]["correction_content"] for item in result.items
    ] == ["금액 정정", "기간 정정"]
    assert result.metadata["history_order"] == "oldest_to_latest"


def test_correction_history_tool_adds_retrieval_result(monkeypatch):
    monkeypatch.setattr(
        tool_module,
        "correction_history_executor",
        _successful_result,
    )

    update = retrieve_correction_history.invoke({
        "disclosure_id": "d20240301000003",
        "state": _state(),
    })

    assert update["next_plan_seq"] == 4
    result = update["retrieval_results"][0]
    assert result.result_id == "retrieval:plan_3"
    # RetrievalResult는 Plan 관련 정보를 담지 않는다.
    assert "plan_purpose" not in result.metadata


def test_dispatch_validates_and_executes_correction_history(monkeypatch):
    monkeypatch.setattr(
        tool_module,
        "correction_history_executor",
        _successful_result,
    )
    state = _state()
    tool_call = {
        "name": "retrieve_correction_history",
        "args": {"disclosure_id": "d20240301000003"},
    }

    validate_retriever_tool_call(state, tool_call)
    update = execute_tool_call(state, tool_call)

    assert update["retrieval_results"][0].source == "neo4j"


@pytest.mark.parametrize("disclosure_id", [
    "20240301000003",
    "d2024030100000",
    "d2024030100000x",
    " d20240301000003",
])
def test_correction_history_tool_rejects_invalid_disclosure_id(disclosure_id):
    with pytest.raises(ValidationError):
        retrieve_correction_history.invoke({
            "disclosure_id": disclosure_id,
            "state": _state(),
        })


def test_correction_history_tool_returns_no_results(monkeypatch):
    monkeypatch.setattr(
        tool_module,
        "correction_history_executor",
        lambda disclosure_id, plan_id: RetrievalResult(
            result_id=f"retrieval:{plan_id}",
            source="neo4j",
            query="fixed correction query",
            items=[],
            result_count=0,
        ),
    )

    update = retrieve_correction_history.invoke({
        "disclosure_id": "d20240301000003",
        "state": _state(),
    })

    assert update["retrieval_results"][0].status == "NO_RESULTS"


def test_correction_history_tool_preserves_database_error(monkeypatch):
    def fail(_disclosure_id: str, _plan_id: str):
        raise RuntimeError("Neo4j 정정이력 조회 중 오류 발생!")

    monkeypatch.setattr(tool_module, "correction_history_executor", fail)

    update = retrieve_correction_history.invoke({
        "disclosure_id": "d20240301000003",
        "state": _state(),
    })

    result = update["retrieval_results"][0]
    assert result.status == "ERROR"
    assert result.metadata["failure_stage"] == "query_executor"
