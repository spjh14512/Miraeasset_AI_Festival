from __future__ import annotations

from neo4j import Record as Neo4jRecord
from qdrant_client.http.models import QueryResponse, ScoredPoint

from agent_graph import tools
from agent_graph.state import Plan, RetrievalResult
from agent_graph.tools import CypherQuery, QdrantQuery


class _Neo4jSession:
    def __init__(self):
        self.call = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def run(self, cypher, parameters):
        self.call = (cypher, parameters)
        return [Neo4jRecord([("company_count", 3)])]


class _Neo4jDriver:
    def __init__(self):
        self.opened_session = _Neo4jSession()

    def session(self):
        return self.opened_session


class _QdrantClient:
    def __init__(self, response):
        self.response = response
        self.query_call = None
        self.scroll_call = None

    def query_points(self, **kwargs):
        self.query_call = kwargs
        return self.response

    def scroll(self, **kwargs):
        self.scroll_call = kwargs
        return self.response


def _r_table_point():
    return ScoredPoint(
        id="00000000-0000-0000-0000-000000000001",
        version=1,
        score=0.9,
        payload={
            "retrieval_metadata": {
                "point_kind": "R_TABLE",
                "evidence_id": "d1:src0:s0:e0",
                "corp_name": "삼성전자",
                "corp_code": "00126380",
                "table_id": "table-1",
            },
            "contextual_text": "삼성전자 특별관계자 현황",
            "canonical": {
                "headers": [["성명"], ["관계"]],
                "records": [
                    {"record_index": 0, "values": ["홍길동", "최대주주"]}
                ],
            },
        },
    )


def test_cypher_executor_returns_one_retrieval_result(monkeypatch):
    driver = _Neo4jDriver()
    monkeypatch.setattr(tools, "neo4j_driver", driver)
    query = CypherQuery(
        cypher="MATCH (c:Company) RETURN count(c) AS company_count",
        parameters={}
    )

    result = tools.cypher_executor(query, "plan_1")

    assert result.plan_id == "plan_1"
    assert result.source == "neo4j"
    assert result.items[0]["fields"]["company_count"] == 3
    assert driver.opened_session.call == (query.cypher, {})


def test_query_executor_runs_vector_query(monkeypatch):
    client = _QdrantClient(QueryResponse(points=[_r_table_point()]))
    monkeypatch.setattr(tools, "qdrant_client", client)
    query = QdrantQuery(
        mode="vector",
        query_text="삼성전자 특별관계자",
        query_vector=[0.1, 0.2],
        point_kinds=["R_TABLE"],
        filters=[
            {
                "key": "retrieval_metadata.corp_name",
                "match": "삼성전자"
            }
        ],
        limit=5
    )

    result = tools.query_executor(query, "plan_2")

    assert result.plan_id == "plan_2"
    assert result.source == "qdrant"
    assert "records" not in result.items[0]
    assert client.query_call["query"] == [0.1, 0.2]
    assert client.query_call["using"] == "evidence_dense"


def test_query_executor_uses_scroll_and_includes_exact_table_records(monkeypatch):
    client = _QdrantClient(([_r_table_point()], None))
    monkeypatch.setattr(tools, "qdrant_client", client)
    query = QdrantQuery(
        mode="filter",
        point_kinds=["R_TABLE"],
        filters=[
            {
                "key": "retrieval_metadata.table_id",
                "match": "table-1"
            }
        ],
        limit=10
    )

    result = tools.query_executor(query, "plan_3")

    assert result.items[0]["records"][0]["values"] == {
        "성명": "홍길동",
        "관계": "최대주주",
    }
    assert client.scroll_call["limit"] == 10


def test_retrieve_search_preserves_plan_purpose(monkeypatch):
    plan = Plan(
        plan_id="plan_4",
        source="qdrant",
        query="삼성전자 특별관계자",
        purpose="특별관계자 명단 확인"
    )
    query = QdrantQuery(
        mode="vector",
        query_text="삼성전자 특별관계자",
        query_vector=[0.1],
        point_kinds=["R_TABLE"]
    )
    retrieval_result = RetrievalResult(
        result_id="retrieval:plan_4",
        plan_id="plan_4",
        source="qdrant",
        query="{}",
        items=[],
        result_count=0,
        metadata={"mode": "vector"}
    )
    monkeypatch.setattr(tools, "query_builder", lambda _: query)
    monkeypatch.setattr(
        tools,
        "query_executor",
        lambda *_: retrieval_result
    )

    update = tools.retrieve_search.invoke({
        "plan_id": "plan_4",
        "state": {
            "question_id": "question-1",
            "question_text": "삼성전자의 특별관계자 목록을 알려줘",
            "plans": [plan],
            "retrieval_results": []
        },
    })

    assert update["retrieval_results"][0].metadata == {
        "mode": "vector",
        "plan_purpose": "특별관계자 명단 확인"
    }
    assert retrieval_result.metadata == {"mode": "vector"}
