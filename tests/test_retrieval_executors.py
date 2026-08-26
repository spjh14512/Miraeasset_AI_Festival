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


def _r_table_point(
    point_id: str = "00000000-0000-0000-0000-000000000001",
):
    return ScoredPoint(
        id=point_id,
        version=1,
        score=0.9,
        payload={
            "retrieval_metadata": {
                "point_kind": "R_TABLE",
                "evidence_id": "d1:src0:s0:e0",
                "corp_name": "삼성전자",
                "corp_code": "00126380",
                "report_nm": "사업보고서 (2023.12)",
                "table_id": "table-1",
            },
            "contextual_text": (
                "회사 : 삼성전자\n"
                "공시 : 사업보고서 (2023.12)\n"
                "섹션 : VII. 주주에 관한 사항\n\n"
                "특별관계자 현황"
            ),
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
    raw_result = QueryResponse(points=[_r_table_point()])
    client = _QdrantClient(raw_result)
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

    result = tools.query_executor(query)

    assert result is raw_result
    assert client.query_call["query"] == [0.1, 0.2]
    assert client.query_call["using"] == "evidence_dense"


def test_query_executor_uses_scroll_and_returns_raw_result(monkeypatch):
    raw_result = ([_r_table_point()], None)
    client = _QdrantClient(raw_result)
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

    result = tools.query_executor(query)

    assert result is raw_result
    assert client.scroll_call["limit"] == 10


def test_qdrant_query_limit_is_owned_by_application():
    schema = QdrantQuery.model_json_schema()
    query = QdrantQuery(
        mode="vector",
        query_text="삼성전자 특별관계자",
        point_kinds=["R_TABLE"],
    )

    assert "limit" not in schema["properties"]
    assert query.limit == 3


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
    raw_result = QueryResponse(points=[])
    monkeypatch.setattr(tools, "query_builder", lambda _, **__: query)
    monkeypatch.setattr(
        tools,
        "query_executor",
        lambda *_: raw_result
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
        "requested_limit": 3,
        "raw_point_count": 0,
        "duplicate_point_count": 0,
        "returned_point_count": 0,
        "r_table_detail": "records",
        "plan_purpose": "특별관계자 명단 확인"
    }
    assert update["retrieved_qdrant_point_ids"] == []


def test_retrieve_search_applies_point_compactor_selection(monkeypatch):
    plan = Plan(
        plan_id="plan_5",
        source="qdrant",
        query="삼성전자 특별관계자",
        purpose="특별관계자 명단 확인",
    )
    query = QdrantQuery(
        mode="vector",
        query_text="삼성전자 특별관계자",
        query_vector=[0.1],
        point_kinds=["R_TABLE"],
    )
    point = _r_table_point()
    monkeypatch.setattr(tools, "query_builder", lambda _, **__: query)
    monkeypatch.setattr(
        tools,
        "query_executor",
        lambda *_: QueryResponse(points=[point]),
    )
    monkeypatch.setattr(tools, "point_requires_compaction", lambda _: True)
    monkeypatch.setattr(tools, "compact_qdrant_point", lambda *_: [])

    update = tools.retrieve_search.invoke({
        "plan_id": "plan_5",
        "state": {
            "question_id": "question-1",
            "question_text": "삼성전자의 특별관계자 목록을 알려줘",
            "plans": [plan],
            "retrieval_results": [],
        },
    })

    item = update["retrieval_results"][0].items[0]
    assert item["records"] == []
    assert item["included_record_count"] == 0
    assert item["omitted_record_count"] == 1


def test_retrieve_search_expands_limit_and_returns_only_new_points(monkeypatch):
    plans = [
        Plan(
            plan_id="plan_1",
            source="qdrant",
            query="삼성전자 특별관계자",
            purpose="초기 근거 검색",
        ),
        Plan(
            plan_id="plan_2",
            source="qdrant",
            query="삼성전자 특별관계자",
            purpose="검색 범위 확대",
        ),
    ]
    points = [
        _r_table_point(f"00000000-0000-0000-0000-{index:012d}")
        for index in range(1, 6)
    ]
    raw_results = [
        QueryResponse(points=points[:3]),
        QueryResponse(points=points),
    ]
    executed_limits = []

    monkeypatch.setattr(
        tools,
        "query_builder",
        lambda _, **__: QdrantQuery(
            mode="vector",
            query_text="삼성전자 특별관계자",
            query_vector=[0.1],
            point_kinds=["R_TABLE"],
        ),
    )

    def execute(query):
        executed_limits.append(query.limit)
        return raw_results.pop(0)

    monkeypatch.setattr(tools, "query_executor", execute)

    first_update = tools.retrieve_search.invoke({
        "plan_id": "plan_1",
        "limit": 3,
        "state": {
            "question_id": "question-1",
            "question_text": "삼성전자의 특별관계자 목록을 알려줘",
            "plans": plans,
            "retrieval_results": [],
        },
    })
    first_result = first_update["retrieval_results"][0]

    second_update = tools.retrieve_search.invoke({
        "plan_id": "plan_2",
        "limit": 5,
        "state": {
            "question_id": "question-1",
            "question_text": "삼성전자의 특별관계자 목록을 알려줘",
            "plans": first_update["plans"],
            "retrieval_results": [first_result],
            "retrieved_qdrant_point_ids": first_update[
                "retrieved_qdrant_point_ids"
            ],
        },
    })
    second_result = second_update["retrieval_results"][0]

    assert executed_limits == [3, 5]
    assert len(first_result.items) == 3
    assert first_result.metadata["duplicate_point_count"] == 0
    assert first_result.metadata["returned_point_count"] == 3
    assert len(second_result.items) == 2
    assert second_result.metadata["requested_limit"] == 5
    assert second_result.metadata["raw_point_count"] == 5
    assert second_result.metadata["duplicate_point_count"] == 3
    assert second_result.metadata["returned_point_count"] == 2
    assert second_update["retrieved_qdrant_point_ids"] == [
        str(point.id) for point in points
    ]


def test_retrieve_search_rejects_non_progressive_qdrant_limit():
    plan = Plan(
        plan_id="plan_1",
        source="qdrant",
        query="삼성전자 특별관계자",
        purpose="근거 검색",
    )

    try:
        tools.retrieve_search.invoke({
            "plan_id": "plan_1",
            "limit": 4,
            "state": {
                "question_id": "question-1",
                "question_text": "삼성전자의 특별관계자 목록을 알려줘",
                "plans": [plan],
                "retrieval_results": [],
            },
        })
    except ValueError as error:
        assert "Qdrant limit" in str(error)
    else:
        raise AssertionError("progressive limit 규칙을 벗어난 값이 허용되었습니다.")


def test_retrieve_search_passes_only_plan_dependencies_to_builder(monkeypatch):
    used_dependency = RetrievalResult(
        result_id="retrieval:plan_1",
        plan_id="plan_1",
        source="neo4j",
        query="MATCH ...",
        items=[{"fields": {"evidence_id": "evidence-1"}}],
        result_count=1,
    )
    unused_result = RetrievalResult(
        result_id="retrieval:other",
        plan_id="other",
        source="neo4j",
        query="MATCH ...",
        items=[{"fields": {"evidence_id": "evidence-other"}}],
        result_count=1,
    )
    plan = Plan(
        plan_id="plan_2",
        source="qdrant",
        query="확인된 Evidence의 실제 내용",
        purpose="답변 근거 확인",
        dependencies=["retrieval:plan_1"],
    )
    captured = {}

    def build_query(received_plan, **kwargs):
        captured["plan"] = received_plan
        captured.update(kwargs)
        return QdrantQuery(
            mode="filter",
            point_kinds=["TEXT", "KV_TABLE", "R_TABLE"],
            filters=[{
                "key": "retrieval_metadata.evidence_id",
                "match": "evidence-1",
            }],
        )

    monkeypatch.setattr(tools, "query_builder", build_query)
    monkeypatch.setattr(
        tools,
        "query_executor",
        lambda *_: ([], None),
    )

    tools.retrieve_search.invoke({
        "plan_id": "plan_2",
        "state": {
            "question_id": "question-1",
            "question_text": "해당 Evidence의 실제 내용을 알려줘",
            "plans": [plan],
            "retrieval_results": [used_dependency, unused_result],
        },
    })

    assert captured["plan"] == plan
    assert captured["user_question"] == "해당 Evidence의 실제 내용을 알려줘"
    assert captured["dependencies"] == [used_dependency]


def test_retrieve_search_rejects_missing_dependency():
    plan = Plan(
        plan_id="plan_2",
        source="qdrant",
        query="확인된 Evidence의 실제 내용",
        purpose="답변 근거 확인",
        dependencies=["retrieval:missing"],
    )

    try:
        tools.retrieve_search.invoke({
            "plan_id": "plan_2",
            "state": {
                "question_id": "question-1",
                "question_text": "해당 Evidence의 실제 내용을 알려줘",
                "plans": [plan],
                "retrieval_results": [],
            },
        })
    except ValueError as error:
        assert "retrieval:missing" in str(error)
    else:
        raise AssertionError("존재하지 않는 dependency가 허용되었습니다.")
