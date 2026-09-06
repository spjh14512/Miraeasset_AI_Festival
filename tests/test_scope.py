from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError
from qdrant_client import models
from qdrant_client.http.models import QueryResponse

from agent_graph import graph as graph_module
from agent_graph import utils
from agent_graph.state import (
    PlanDraft,
    QuestionAnalysis,
    Scope,
    ScopeDraft,
    SubQuestion,
)
from agent_graph.utils import (
    CypherQuery,
    DisclosureSelection,
    QdrantQuery,
    SectionSelection,
)
from vector_db.text2vector import HybridEmbedding, SparseEmbedding


def _subquestion() -> SubQuestion:
    return SubQuestion(
        subquestion_id="subquestion_1",
        question="삼성전자의 2025년 합병 상대방은 누구인가?",
        entities=[{
            "mention": "삼성전자",
            "roles": ["ISSUER"],
            "canonical_name": "삼성전자",
            "corp_code": "00126380",
            "match_status": "MATCHED",
        }],
        events=[{
            "event_type": "합병",
            "candidate_event_types": ["기업결합"],
            "confidence": "HIGH",
        }],
        intents=["DETAIL"],
        periods=[{
            "expression": "2025년",
            "kind": "EVENT_DATE",
            "normalized_value": "2025",
            "granularity": "YEAR",
        }],
        requested_facts=["합병 상대방"],
    )


def _state() -> dict[str, Any]:
    return {
        "question_id": "question-1",
        "question_text": "삼성전자의 2025년 합병 상대방은 누구인가?",
        "question_analysis": QuestionAnalysis(
            decision="retrieve",
            normalized_question="삼성전자의 2025년 합병 상대방은 누구인가?",
            decision_reason="공시 검색이 필요합니다.",
            sub_questions=[_subquestion()],
        ),
        "scope_candidates": [],
        "next_plan_seq": 1,
        "retrieval_results": [],
    }


def _section_scope() -> Scope:
    return Scope(
        scope_id="scope_subquestion_1",
        subquestion_id="subquestion_1",
        level="SECTION",
        corp_names=["삼성전자"],
        corp_codes=["00126380"],
        disclosure_ids=["d1"],
        section_ids=["s1", "s2"],
        reason="관련 공시의 근거 섹션을 확인했습니다.",
    )


def _hybrid() -> HybridEmbedding:
    return HybridEmbedding(
        dense=(0.1, 0.2),
        sparse=SparseEmbedding(indices=(1,), values=(0.5,)),
    )


class _StructuredSequence:
    def __init__(self, actions: list[Any]):
        self.actions = list(actions)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.actions.pop(0)


class _Llm:
    def __init__(self, actions: list[Any]):
        self.structured = _StructuredSequence(actions)
        self.schema = None
        self.method = None

    def with_structured_output(self, schema, *, method):
        self.schema = schema
        self.method = method
        return self.structured


class _QdrantClient:
    def __init__(self):
        self.query_call = None

    def query_points(self, **kwargs):
        self.query_call = kwargs
        return QueryResponse(points=[])


def test_plan_draft_requires_scope_id():
    plan = PlanDraft(
        source="qdrant",
        query="합병 상대방",
        purpose="합병 상대방 확인",
        dependencies=[],
        scope_id="scope_1",
    )

    assert plan.scope_id == "scope_1"

    with pytest.raises(ValidationError, match="scope_id"):
        PlanDraft(
            source="qdrant",
            query="합병 상대방",
            purpose="합병 상대방 확인",
            dependencies=[],
        )


def test_plan_draft_rejects_blank_scope_id():
    with pytest.raises(ValidationError, match="scope_id"):
        PlanDraft(
            source="qdrant",
            query="합병 상대방",
            purpose="합병 상대방 확인",
            dependencies=[],
            scope_id="  ",
        )


def test_scope_level_contract_rejects_missing_parent_ids():
    with pytest.raises(ValidationError, match="disclosure_ids와 section_ids"):
        ScopeDraft(
            level="SECTION",
            corp_names=["삼성전자"],
            corp_codes=[],
            disclosure_ids=[],
            section_ids=["s1"],
            reason="섹션 후보",
        )


def test_knowledge_query_combines_required_subquestion_fields():
    query = utils._scope_knowledge_query_text(_subquestion())

    for value in ("삼성전자의", "DETAIL", "합병 상대방", "합병", "기업결합"):
        assert value in query


def test_scope_event_date_expands_disclosure_rcept_date_by_one_month():
    rcept_date_range = utils._scope_rcept_date_range(_subquestion())
    query = utils._disclosure_scope_query(
        ["삼성전자"],
        ["major"],
        rcept_date_range,
    )

    assert rcept_date_range == ("2024-12-01", "2026-01-31")
    assert "d.rcept_date >= date($start_date)" in query.cypher
    assert "d.rcept_date <= date($end_date)" in query.cypher
    assert query.parameters["start_date"] == "2024-12-01"
    assert query.parameters["end_date"] == "2026-01-31"


def test_scope_reporting_period_does_not_filter_rcept_date():
    subquestion = SubQuestion.model_validate({
        **_subquestion().model_dump(mode="json"),
        "periods": [{
            "expression": "2023년",
            "kind": "REPORTING_PERIOD",
            "normalized_value": "2023",
            "granularity": "YEAR",
        }],
    })

    assert utils._scope_rcept_date_range(subquestion) is None


def test_scope_knowledge_search_keeps_llm_fields_and_omits_empty_values(monkeypatch):
    points = [
        SimpleNamespace(
            payload={
                "knowledge_type": "METRIC",
                "name": "매출액",
                "aliases": ["매출", "수익"],
                "description": "영업활동으로 얻은 수익입니다.",
                "datasets": ["periodic"],
                "categories": ["재무제표·회계·감사"],
                "unused_field": "제외 대상",
            },
            score=0.91,
        ),
        SimpleNamespace(
            payload={
                "knowledge_type": "TERM",
                "name": "사업보고서",
                "aliases": [],
                "description": "   ",
                "datasets": [],
                "categories": [],
            },
            score=0.8,
        ),
    ]
    client = SimpleNamespace(
        query_points=lambda **_kwargs: SimpleNamespace(points=points),
    )
    monkeypatch.setattr(utils, "qdrant_client", client)
    monkeypatch.setattr(utils, "text_to_hybrid_vector", lambda _text: _hybrid())

    hints = utils.search_scope_knowledge(_subquestion())

    assert hints == [
        {
            "knowledge_type": "METRIC",
            "name": "매출액",
            "aliases": ["매출", "수익"],
            "description": "영업활동으로 얻은 수익입니다.",
            "datasets": ["periodic"],
            "categories": ["재무제표·회계·감사"],
            "score": 0.91,
        },
        {
            "knowledge_type": "TERM",
            "name": "사업보고서",
            "score": 0.8,
        },
    ]


def test_narrow_scope_selects_disclosure_then_section(monkeypatch, capsys):
    disclosure_selection = DisclosureSelection(
        selected_disclosure_ids=["d1"],
        reason="합병 관련 공시를 선택했습니다.",
    )
    section_selection = SectionSelection(
        selected_section_ids=["s1"],
        reason="합병 관련 Section을 선택했습니다.",
    )
    llm = _Llm([disclosure_selection, section_selection])
    monkeypatch.setattr(
        utils,
        "search_scope_knowledge",
        lambda _subquestion: [{
            "name": "합병",
            "aliases": ["기업결합"],
            "description": "둘 이상의 기업이 하나로 결합하는 사건",
            "datasets": ["major"],
            "categories": ["인수합병"],
            "score": 0.9,
        }],
    )
    monkeypatch.setattr(
        utils,
        "_execute_scope_cypher",
        lambda _query, stage: [{
            "type": "record",
            "fields": (
                {
                    "corp_name": "삼성전자",
                    "corp_code": "00126380",
                    "disclosure_id": "d1",
                    "report_name": "주요사항보고서(회사합병결정)",
                    "doc_group": "major",
                    "rcept_date": "2025-01-02",
                }
                if stage == "disclosures"
                else {
                    "corp_name": "삼성전자",
                    "corp_code": "00126380",
                    "disclosure_id": "d1",
                    "report_name": "주요사항보고서(회사합병결정)",
                    "section_id": "s1",
                    "section_title": "합병 상대방",
                    "section_path": ["합병에 관한 사항", "합병 상대방"],
                    "order_in_doc": 1,
                }
            ),
        }],
    )

    scope = utils.run_narrow_scope_agent(
        "subquestion_1",
        _state(),
        llm=llm,
    )

    assert scope.scope_id == "scope_subquestion_1"
    assert scope.level == "SECTION"
    assert scope.section_ids == ["s1"]
    assert llm.method == "json_schema"
    output = capsys.readouterr().out
    assert output.count("[narrow_scope human message]:") == 2
    assert output.count('"aliases": [') == 2
    assert '"description": "둘 이상의 기업이 하나로 결합하는 사건"' in output
    assert '"categories": [' in output
    assert '"stage": "DISCLOSURE_SELECTION"' in output
    assert '"stage": "SECTION_SELECTION"' in output
    assert "[narrow_scope] Knowledge Base 검색어:" not in output
    assert "[narrow_scope] LLM action:" not in output
    assert "[narrow_scope] Neo4j 검색" not in output
    assert "[narrow_scope] 완료:" not in output


def test_scope_resolver_adds_application_assigned_scope(monkeypatch):
    expected = _section_scope()
    monkeypatch.setattr(
        graph_module,
        "run_narrow_scope_agent",
        lambda subquestion_id, state: expected,
    )

    update = graph_module.scope_resolver(_state())

    assert update["scope_candidates"] == [expected]
    trace = update["think_trace_events"][0]
    assert trace.name == "scope_resolver"
    assert trace.message == expected.reason
    assert trace.details["scope_id"] == expected.scope_id
    assert trace.details["level"] == "SECTION"


def test_scope_resolver_processes_every_subquestion_in_order(monkeypatch):
    state = _state()
    first = _subquestion()
    second = first.model_copy(update={
        "subquestion_id": "subquestion_2",
        "question": "삼성전자의 합병일은 언제인가?",
        "requested_facts": ["합병일"],
    })
    state["question_analysis"] = state["question_analysis"].model_copy(update={
        "sub_questions": [first, second],
        "synthesis_requirement": "상대방과 합병일을 함께 설명합니다.",
    })
    calls = []

    def resolve(subquestion_id, _state):
        calls.append(subquestion_id)
        return Scope.from_scope_draft(
            ScopeDraft(level="GLOBAL", reason="전역 검색"),
            subquestion_id=subquestion_id,
        )

    monkeypatch.setattr(graph_module, "run_narrow_scope_agent", resolve)

    update = graph_module.scope_resolver(state)

    assert calls == ["subquestion_1", "subquestion_2"]
    assert [scope.scope_id for scope in update["scope_candidates"]] == [
        "scope_subquestion_1",
        "scope_subquestion_2",
    ]


def test_scope_resolver_falls_back_to_global_scope(monkeypatch, capsys):
    def fail(_subquestion_id, _state):
        raise RuntimeError("Neo4j unavailable")

    monkeypatch.setattr(graph_module, "run_narrow_scope_agent", fail)

    update = graph_module.scope_resolver(_state())

    scope = update["scope_candidates"][0]
    assert scope.scope_id == "scope_subquestion_1"
    assert scope.subquestion_id == "subquestion_1"
    assert scope.level == "GLOBAL"
    assert "RuntimeError: Neo4j unavailable" in scope.reason
    assert "GLOBAL Scope로 대체" in capsys.readouterr().out


def test_scope_resolver_skips_out_of_universe_issuer(monkeypatch, capsys):
    state = _state()
    subquestion = state["question_analysis"].sub_questions[0]
    unsupported_entity = subquestion.entities[0].model_copy(update={
        "canonical_name": None,
        "corp_code": None,
        "match_status": "OUT_OF_UNIVERSE",
    })
    unsupported_subquestion = subquestion.model_copy(update={
        "entities": [unsupported_entity],
    })
    state["question_analysis"] = state["question_analysis"].model_copy(update={
        "sub_questions": [unsupported_subquestion],
    })
    calls = []
    monkeypatch.setattr(
        graph_module,
        "run_narrow_scope_agent",
        lambda subquestion_id, _state: calls.append(subquestion_id),
    )

    update = graph_module.scope_resolver(state)

    assert update["scope_candidates"] == []
    assert update["think_trace_events"] == []
    assert calls == []
    assert "universe에서 선택되지 않아 Scope 생성을 건너뜁니다" in capsys.readouterr().out


def test_scope_selection_rejects_ids_not_returned_by_neo4j():
    draft = ScopeDraft(
        level="DISCLOSURE",
        corp_names=["삼성전자"],
        corp_codes=[],
        disclosure_ids=["unknown"],
        section_ids=[],
        reason="공시 후보",
    )

    with pytest.raises(ValueError, match="확인되지 않은 disclosure_ids"):
        utils._validate_scope_selection(
            draft,
            [{"fields": {"corp_name": "삼성전자", "disclosure_id": "d1"}}],
        )


def test_retriever_tool_validation_rejects_unknown_scope():
    with pytest.raises(ValueError, match="Plan Scope를 찾지 못했습니다"):
        utils.validate_retriever_tool_call(
            _state(),
            {
                "name": "retrieve_search",
                "args": {
                    "plan": {
                        "source": "qdrant",
                        "query": "합병 상대방",
                        "purpose": "합병 상대방 확인",
                        "dependencies": [],
                        "scope_id": "scope_missing",
                    },
                    "breadth": "initial",
                },
            },
        )


def test_cypher_scope_validation_requires_section_id_constraint():
    scope = _section_scope()
    utils._validate_cypher_scope_filter(
        CypherQuery(
            cypher="MATCH (s:Section) WHERE s.id IN $section_ids RETURN s",
            parameters={"section_ids": ["s1", "s2"]},
        ),
        scope,
    )

    with pytest.raises(ValueError, match="Scope가 지정된 Cypher"):
        utils._validate_cypher_scope_filter(
            CypherQuery(cypher="MATCH (s:Section) RETURN s", parameters={}),
            scope,
        )


def test_cypher_relationship_validation_rejects_wrong_reports_direction():
    with pytest.raises(ValueError, match="REPORTS.*Disclosure -> Event"):
        utils._validate_cypher_relationships(CypherQuery(
            cypher=(
                "MATCH (c:Company)<-[:REPORTS]-(e:Event) "
                "RETURN e.content AS result LIMIT 10"
            ),
            parameters={},
        ))


def test_cypher_relationship_validation_accepts_schema_direction():
    utils._validate_cypher_relationships(CypherQuery(
        cypher=(
            "MATCH (c:Company)-[:PUBLISHES]->(d:Disclosure) "
            "MATCH (d)-[:REPORTS]->(e:Event) "
            "RETURN e.content AS result LIMIT 10"
        ),
        parameters={},
    ))


@pytest.mark.parametrize(
    ("scope", "expected_key", "expected_values"),
    [
        (
            Scope(
                scope_id="scope_1",
                subquestion_id="subquestion_1",
                level="COMPANY",
                corp_names=["삼성전자"],
                corp_codes=["00126380"],
                disclosure_ids=[],
                section_ids=[],
                reason="회사 범위",
            ),
            "corp_name",
            ["삼성전자"],
        ),
        (_section_scope(), "section_id", ["s1", "s2"]),
    ],
)
def test_query_executor_applies_hierarchical_scope(
    monkeypatch,
    scope,
    expected_key,
    expected_values,
):
    client = _QdrantClient()
    monkeypatch.setattr(utils, "qdrant_client", client)

    utils.query_executor(
        QdrantQuery(
            mode="vector",
            query_text="합병 상대방",
            query_vector=_hybrid(),
        ),
        scope=scope,
    )

    conditions = client.query_call["prefetch"][0].filter.must
    assert [condition.key for condition in conditions] == [
        "is_latest_version",
        expected_key,
    ]
    assert isinstance(conditions[1].match, models.MatchAny)
    assert conditions[1].match.any == expected_values
