from __future__ import annotations

import json

import pytest
from neo4j import Record as Neo4jRecord
from qdrant_client import models
from qdrant_client.http.models import QueryResponse, ScoredPoint

from agent_graph import utils as tools
from agent_graph.tools import retrieve_search
from agent_graph.state import Plan, RetrievalResult, Scope
from agent_graph.utils import CypherQuery, QdrantQuery, QdrantQueryToolArgs
from vector_db.text2vector import HybridEmbedding, SparseEmbedding


def _hybrid() -> HybridEmbedding:
    return HybridEmbedding(
        dense=(0.1, 0.2),
        sparse=SparseEmbedding(indices=(1, 42), values=(0.2, 0.8)),
    )


def _global_scope() -> Scope:
    return Scope(
        scope_id="scope_1",
        subquestion_id="subquestion_1",
        level="GLOBAL",
        corp_names=[],
        corp_codes=[],
        disclosure_ids=[],
        section_ids=[],
        reason="검색 범위를 더 좁힐 수 없습니다.",
    )


def _query_tool_args(query: QdrantQuery) -> QdrantQueryToolArgs:
    return QdrantQueryToolArgs(
        mode=query.mode,
        query_text=query.query_text or "",
        filters_json=json.dumps([
            item.model_dump(mode="json", exclude_none=True)
            for item in query.filters
        ]),
        score_threshold_json=json.dumps(query.score_threshold),
    )


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


class _QueryBuilderLlm:
    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, messages):
        self.messages = messages
        return _query_tool_args(QdrantQuery(
            mode="vector",
            query_text="삼성전자 특별관계자",
        ))


class _SequenceQueryBuilderLlm:
    def __init__(self, queries):
        self.queries = list(queries)
        self.calls = []

    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        query = self.queries.pop(0)
        return _query_tool_args(query) if isinstance(query, QdrantQuery) else query


def _r_table_point(
    point_id: str = "00000000-0000-0000-0000-000000000001",
):
    return ScoredPoint(
        id=point_id,
        version=1,
        score=0.9,
        payload={
            "point_kind": "R_TABLE",
            "disclosure_id": "d1",
            "section_id": "d1:src0:s0",
            "evidence_id": "d1:src0:s0:e0",
            "corp_name": "삼성전자",
            "report_name": "사업보고서 (2023.12)",
            "section_name": "VII. 주주에 관한 사항",
            "rcept_date": "20240306",
            "chunking": {
                "table_id": "table-1",
                "chunk_index": 0,
                "chunk_count": 2,
                "row_start_index": 0,
                "row_end_index": 0,
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
        query_vector=_hybrid(),
        filters=[
            {
                "key": "evidence_id",
                "match": "evidence-1"
            }
        ],
        limit=5
    )

    result = tools.query_executor(query)

    assert result is raw_result
    assert isinstance(client.query_call["query"], models.FusionQuery)
    assert client.query_call["query"].fusion == models.Fusion.RRF
    assert len(client.query_call["prefetch"]) == 2
    dense_prefetch, sparse_prefetch = client.query_call["prefetch"]
    assert dense_prefetch.query == [0.1, 0.2]
    assert dense_prefetch.using == "evidence_dense"
    assert sparse_prefetch.query == models.SparseVector(
        indices=[1, 42], values=[0.2, 0.8]
    )
    assert sparse_prefetch.using == "evidence_sparse"
    assert dense_prefetch.limit == 25
    assert sparse_prefetch.limit == 25
    assert dense_prefetch.filter == sparse_prefetch.filter
    assert [condition.key for condition in dense_prefetch.filter.must] == [
        "is_latest_version",
        "evidence_id"
    ]
    assert dense_prefetch.filter.must[0].match.value is True
    assert client.query_call["with_payload"] is True


def test_query_executor_adds_latest_version_filter_to_vector_query(monkeypatch):
    client = _QdrantClient(QueryResponse(points=[]))
    monkeypatch.setattr(tools, "qdrant_client", client)
    query = QdrantQuery(
        mode="vector",
        query_text="삼성전자 특별관계자",
        query_vector=_hybrid(),
    )

    tools.query_executor(query)

    for prefetch in client.query_call["prefetch"]:
        assert [condition.key for condition in prefetch.filter.must] == [
            "is_latest_version"
        ]
        assert prefetch.filter.must[0].match.value is True


def test_query_executor_uses_scroll_and_returns_raw_result(monkeypatch):
    raw_result = ([_r_table_point()], None)
    client = _QdrantClient(raw_result)
    monkeypatch.setattr(tools, "qdrant_client", client)
    query = QdrantQuery(
        mode="filter",
        filters=[
            {
                "key": "chunking.table_id",
                "match": "table-1"
            }
        ],
        limit=10
    )

    result = tools.query_executor(query)

    assert result is raw_result
    assert client.scroll_call["limit"] == 10
    assert client.scroll_call["with_payload"] is True
    assert [
        condition.key
        for condition in client.scroll_call["scroll_filter"].must
    ] == ["is_latest_version", "chunking.table_id"]
    assert client.scroll_call["scroll_filter"].must[0].match.value is True


def test_qdrant_query_limit_is_owned_by_application():
    schema = QdrantQuery.model_json_schema()
    query = QdrantQuery(
        mode="vector",
        query_text="삼성전자 특별관계자",
    )

    assert "limit" not in schema["properties"]
    assert query.limit == 5


def test_qdrant_query_tool_args_use_clova_compatible_flat_schema():
    schema = QdrantQueryToolArgs.model_json_schema()
    properties = schema["properties"]

    assert set(properties) == {
        "mode",
        "query_text",
        "filters_json",
        "score_threshold_json",
    }
    assert all(property_schema.get("type") == "string" for name, property_schema in properties.items() if name != "mode")
    assert "$defs" not in schema
    assert "anyOf" not in json.dumps(schema)


def test_qdrant_query_tool_args_convert_to_validated_query():
    query = QdrantQueryToolArgs(
        mode="filter",
        query_text="",
        filters_json='[{"key":"evidence_id","match":"evidence-1"}]',
        score_threshold_json="null",
    ).to_qdrant_query()

    assert query.mode == "filter"
    assert query.query_text is None
    assert query.filters[0].key == "evidence_id"


@pytest.mark.parametrize("field", [
    "retrieval_metadata.corp_name",
    "retrieval_metadata.corp_code",
    "retrieval_metadata.industry",
    "retrieval_metadata.sector",
])
def test_qdrant_filter_rejects_removed_company_fields(field):
    with pytest.raises(ValueError):
        tools.QdrantFilter(key=field, match="value")


def test_query_builder_logs_human_message(monkeypatch, capsys):
    llm = _QueryBuilderLlm()
    plan = Plan(
        plan_id="plan_1",
        source="qdrant",
        query="삼성전자 특별관계자",
        purpose="특별관계자 확인",
        dependencies=[],
        scope_id="scope_1",
    )
    monkeypatch.setattr(tools, "text_to_hybrid_vector", lambda _: _hybrid())

    tools.query_builder(
        plan,
        user_question="삼성전자의 특별관계자를 알려줘",
        dependencies=[],
        qdrant_schema="schema",
        llm=llm,
    )

    output = capsys.readouterr().out
    assert "[query builder human message]:" in output
    assert '"user_question": "삼성전자의 특별관계자를 알려줘"' in output
    assert '"plan_id": "plan_1"' in output
    assert '"previous_results": []' in output
    assert '"dependencies"' not in output


def test_query_builder_regenerates_repeated_no_results_filter(monkeypatch):
    failed_result = RetrievalResult(
        result_id="retrieval:plan_1",
        plan_id="plan_1",
        source="qdrant",
        status="NO_RESULTS",
        query=(
            '{"mode":"vector","query_text":"삼성생명보험 취득자금",'
            '"filters":[{"key":"evidence_id",'
            '"match":"evidence-1"}],"score_threshold":null}'
        ),
        items=[],
        result_count=0,
    )
    repeated_query = QdrantQuery(
        mode="vector",
        query_text="삼성생명보험 취득자금 원천",
        filters=[{
            "key": "evidence_id",
            "match": "evidence-1",
        }],
    )
    relaxed_query = QdrantQuery(
        mode="vector",
        query_text="삼성생명보험 취득자금 원천",
        filters=[],
    )
    llm = _SequenceQueryBuilderLlm([repeated_query, relaxed_query])
    monkeypatch.setattr(tools, "text_to_hybrid_vector", lambda _: _hybrid())

    query = tools.query_builder(
        Plan(
            plan_id="plan_2",
            source="qdrant",
            query="삼성생명보험 취득자금 원천 재검색",
            purpose="실패한 기업명 filter를 완화하여 재검색",
            dependencies=["retrieval:plan_1"],
            scope_id="scope_1",
        ),
        user_question="삼성생명보험의 취득자금 원천을 알려줘",
        dependencies=[failed_result],
        qdrant_schema="schema",
        llm=llm,
    )

    assert query.filters == []
    assert len(llm.calls) == 2
    assert "filter 조건의 제거 또는 완화" in llm.calls[1][-1].content


def test_retrieve_search_records_repeated_filter_as_invalid_query(monkeypatch):
    failed_result = RetrievalResult(
        result_id="retrieval:plan_1",
        plan_id="plan_1",
        source="qdrant",
        status="NO_RESULTS",
        query='{"filters":[]}',
        items=[],
        result_count=0,
    )
    plan = Plan(
        plan_id="plan_2",
        source="qdrant",
        query="필터를 완화한 후속 검색",
        purpose="NO_RESULTS 검색 보정",
        dependencies=["retrieval:plan_1"],
        scope_id="scope_1",
    )

    def reject_repeated_filter(*_args, **_kwargs):
        raise tools.RepeatedNoResultsFilterError("동일 filter 조합")

    monkeypatch.setattr(tools, "query_builder", reject_repeated_filter)

    update = retrieve_search.invoke({
        "plan": plan,
        "state": {
            "question_id": "question-1",
            "question_text": "후속 검색",
            "scope_candidates": [_global_scope()],
            "retrieval_results": [failed_result],
            "next_plan_seq": 2,
        },
    })

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_QUERY"
    assert result.metadata["failure_stage"] == "query_builder"


def test_retrieve_search_preserves_plan_purpose(monkeypatch):
    plan = Plan(
        plan_id="plan_4",
        source="qdrant",
        query="삼성전자 특별관계자",
        purpose="특별관계자 명단 확인",
        dependencies=[],
        scope_id="scope_1",
    )
    query = QdrantQuery(
        mode="vector",
        query_text="삼성전자 특별관계자",
        query_vector=_hybrid(),
    )
    raw_result = QueryResponse(points=[])
    monkeypatch.setattr(tools, "query_builder", lambda _, **__: query)
    monkeypatch.setattr(
        tools,
        "query_executor",
        lambda *_, **__: raw_result
    )

    update = retrieve_search.invoke({
        "plan": plan,
        "state": {
            "question_id": "question-1",
            "question_text": "삼성전자의 특별관계자 목록을 알려줘",
            "scope_candidates": [_global_scope()],
            "retrieval_results": [],
            "next_plan_seq": 4,
        },
    })

    assert update["retrieval_results"][0].metadata == {
        "mode": "vector",
        "requested_limit": 5,
        "raw_point_count": 0,
        "duplicate_point_count": 0,
        "returned_point_count": 0,
        "r_table_detail": "records",
        "plan_purpose": "특별관계자 명단 확인"
    }
    assert update["retrieved_qdrant_point_ids"] == []
    assert update["retrieval_results"][0].status == "NO_RESULTS"


def test_retrieve_search_marks_duplicate_only_result(monkeypatch):
    plan = Plan(
        plan_id="plan_4",
        source="qdrant",
        query="삼성전자 특별관계자",
        purpose="중복 결과 확인",
        dependencies=[],
        scope_id="scope_1",
    )
    point = _r_table_point()
    monkeypatch.setattr(
        tools,
        "query_builder",
        lambda *_args, **_kwargs: QdrantQuery(
            mode="vector",
            query_text="삼성전자 특별관계자",
            query_vector=_hybrid(),
        ),
    )
    monkeypatch.setattr(
        tools,
        "query_executor",
        lambda *_, **__: QueryResponse(points=[point]),
    )

    update = retrieve_search.invoke({
        "plan": plan,
        "state": {
            "question_id": "question-1",
            "question_text": "삼성전자의 특별관계자 목록을 알려줘",
            "scope_candidates": [_global_scope()],
            "retrieval_results": [],
            "retrieved_qdrant_point_ids": [str(point.id)],
            "next_plan_seq": 4,
        },
    })

    result = update["retrieval_results"][0]
    assert result.status == "DUPLICATES_ONLY"
    assert result.result_count == 0


def test_retrieve_search_preserves_timeout_as_result(monkeypatch):
    plan = Plan(
        plan_id="plan_4",
        source="qdrant",
        query="삼성전자 특별관계자",
        purpose="timeout 기록 확인",
        dependencies=[],
        scope_id="scope_1",
    )
    monkeypatch.setattr(
        tools,
        "query_builder",
        lambda *_args, **_kwargs: QdrantQuery(
            mode="vector",
            query_text="삼성전자 특별관계자",
            query_vector=_hybrid(),
        ),
    )

    def raise_timeout(_query, **_kwargs):
        try:
            raise TimeoutError("timed out")
        except TimeoutError as error:
            raise RuntimeError("Qdrant 쿼리 실행 중 오류 발생!") from error

    monkeypatch.setattr(tools, "query_executor", raise_timeout)

    update = retrieve_search.invoke({
        "plan": plan,
        "state": {
            "question_id": "question-1",
            "question_text": "삼성전자의 특별관계자 목록을 알려줘",
            "scope_candidates": [_global_scope()],
            "retrieval_results": [],
            "next_plan_seq": 4,
        },
    })

    result = update["retrieval_results"][0]
    assert result.status == "TIMEOUT"
    assert result.result_count == 0
    assert '"query_text":"삼성전자 특별관계자"' in result.query
    assert "point_kinds" not in result.query


def test_retrieve_search_applies_point_compactor_selection(monkeypatch):
    plan = Plan(
        plan_id="plan_5",
        source="qdrant",
        query="삼성전자 특별관계자",
        purpose="특별관계자 명단 확인",
        dependencies=[],
        scope_id="scope_1",
    )
    query = QdrantQuery(
        mode="vector",
        query_text="삼성전자 특별관계자",
        query_vector=_hybrid(),
    )
    point = _r_table_point()
    monkeypatch.setattr(tools, "query_builder", lambda _, **__: query)
    monkeypatch.setattr(
        tools,
        "query_executor",
        lambda *_, **__: QueryResponse(points=[point]),
    )
    monkeypatch.setattr(tools, "point_requires_compaction", lambda _: True)

    def compact(compactor_point, _plan):
        assert "contextual_text" not in compactor_point.payload
        assert compactor_point.payload["retrieval_context"]["section_path"] == [
            "VII. 주주에 관한 사항"
        ]
        return []

    monkeypatch.setattr(tools, "compact_qdrant_point", compact)

    update = retrieve_search.invoke({
        "plan": plan,
        "state": {
            "question_id": "question-1",
            "question_text": "삼성전자의 특별관계자 목록을 알려줘",
            "scope_candidates": [_global_scope()],
            "retrieval_results": [],
            "next_plan_seq": 5,
        },
    })

    item = update["retrieval_results"][0].items[0]
    assert "contextual_text" not in point.payload
    assert item["metadata"]["retrieval_context"]["section_path"] == [
        "VII. 주주에 관한 사항"
    ]
    assert item["records"] == []
    assert item["included_record_count"] == 0
    assert item["omitted_record_count"] == 1


def test_retrieve_search_expands_limit_and_returns_only_new_points(monkeypatch):
    first_plan = Plan(
        plan_id="plan_1",
        source="qdrant",
        query="삼성전자 특별관계자",
        purpose="초기 근거 검색",
        dependencies=[],
        scope_id="scope_1",
    )
    second_plan = Plan(
        plan_id="plan_2",
        source="qdrant",
        query="삼성전자 특별관계자",
        purpose="검색 범위 확대",
        dependencies=[],
        scope_id="scope_1",
    )
    points = [
        _r_table_point(f"00000000-0000-0000-0000-{index:012d}")
        for index in range(1, 11)
    ]
    raw_results = [
        QueryResponse(points=points[:5]),
        QueryResponse(points=points),
    ]
    executed_limits = []

    monkeypatch.setattr(
        tools,
        "query_builder",
        lambda _, **__: QdrantQuery(
            mode="vector",
            query_text="삼성전자 특별관계자",
            query_vector=_hybrid(),
        ),
    )

    def execute(query, **_kwargs):
        executed_limits.append(query.limit)
        return raw_results.pop(0)

    monkeypatch.setattr(tools, "query_executor", execute)

    first_update = retrieve_search.invoke({
        "plan": first_plan,
        "limit": 5,
        "state": {
            "question_id": "question-1",
            "question_text": "삼성전자의 특별관계자 목록을 알려줘",
            "scope_candidates": [_global_scope()],
            "retrieval_results": [],
            "next_plan_seq": 1,
        },
    })
    first_result = first_update["retrieval_results"][0]

    second_update = retrieve_search.invoke({
        "plan": second_plan,
        "limit": 10,
        "state": {
            "question_id": "question-1",
            "question_text": "삼성전자의 특별관계자 목록을 알려줘",
            "scope_candidates": [_global_scope()],
            "retrieval_results": [first_result],
            "retrieved_qdrant_point_ids": first_update[
                "retrieved_qdrant_point_ids"
            ],
            "next_plan_seq": first_update["next_plan_seq"],
        },
    })
    second_result = second_update["retrieval_results"][0]

    assert executed_limits == [5, 10]
    assert len(first_result.items) == 5
    assert first_result.metadata["duplicate_point_count"] == 0
    assert first_result.metadata["returned_point_count"] == 5
    assert len(second_result.items) == 5
    assert second_result.metadata["requested_limit"] == 10
    assert second_result.metadata["raw_point_count"] == 10
    assert second_result.metadata["duplicate_point_count"] == 5
    assert second_result.metadata["returned_point_count"] == 5
    assert second_update["retrieved_qdrant_point_ids"] == [
        str(point.id) for point in points
    ]


def test_retrieve_search_rejects_non_progressive_qdrant_limit():
    plan = Plan(
        plan_id="plan_1",
        source="qdrant",
        query="삼성전자 특별관계자",
        purpose="근거 검색",
        dependencies=[],
        scope_id="scope_1",
    )

    try:
        retrieve_search.invoke({
            "plan": plan,
            "limit": 4,
            "state": {
                "question_id": "question-1",
                "question_text": "삼성전자의 특별관계자 목록을 알려줘",
                "scope_candidates": [_global_scope()],
                "retrieval_results": [],
                "next_plan_seq": 1,
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
    failed_result = RetrievalResult(
        result_id="retrieval:failed",
        plan_id="failed",
        source="qdrant",
        status="NO_RESULTS",
        query='{"filters":[{"key":"rcept_date","match":"20250318"}]}',
        items=[],
        result_count=0,
    )
    plan = Plan(
        plan_id="plan_2",
        source="qdrant",
        query="확인된 Evidence의 실제 내용",
        purpose="답변 근거 확인",
        dependencies=["retrieval:plan_1"],
        scope_id="scope_1",
    )
    captured = {}

    def build_query(received_plan, **kwargs):
        captured["plan"] = received_plan
        captured.update(kwargs)
        return QdrantQuery(
            mode="filter",
            filters=[{
                "key": "evidence_id",
                "match": "evidence-1",
            }],
        )

    monkeypatch.setattr(tools, "query_builder", build_query)
    monkeypatch.setattr(
        tools,
        "query_executor",
        lambda *_, **__: ([], None),
    )

    retrieve_search.invoke({
        "plan": plan,
        "state": {
            "question_id": "question-1",
            "question_text": "해당 Evidence의 실제 내용을 알려줘",
            "scope_candidates": [_global_scope()],
            "retrieval_results": [used_dependency, unused_result, failed_result],
            "next_plan_seq": 2,
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
        scope_id="scope_1",
    )

    try:
        retrieve_search.invoke({
            "plan": plan,
            "state": {
                "question_id": "question-1",
                "question_text": "해당 Evidence의 실제 내용을 알려줘",
                "scope_candidates": [_global_scope()],
                "retrieval_results": [],
                "next_plan_seq": 2,
            },
        })
    except ValueError as error:
        assert "retrieval:missing" in str(error)
    else:
        raise AssertionError("존재하지 않는 dependency가 허용되었습니다.")
