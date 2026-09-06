from __future__ import annotations

from agent_graph import graph as graph_module
from agent_graph import utils as tools
from agent_graph.compactor import CompactorOutput, compact_qdrant_point
from agent_graph.state import (
    AnswerGeneratorOutput,
    Plan,
    QuestionAnalyzerOutput,
    QuestionAnalysis,
    RetrievalResult,
)
from agent_graph.utils import QdrantQueryToolArgs
from qdrant_client.http.models import ScoredPoint


class _SequenceLlm:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output


def _plan(source: str = "qdrant") -> Plan:
    return Plan(
        plan_id="plan_1",
        source=source,
        query="삼성전자 관련 정보",
        purpose="질문에 필요한 근거 확인",
        dependencies=[],
        scope_id="scope_1",
    )


def _answer_state() -> dict:
    return {
        "question_id": "question-1",
        "question_text": "삼성전자의 정보를 알려줘",
        "question_analysis": QuestionAnalysis(
            decision="retrieve",
            normalized_question="삼성전자 정보",
            decision_reason="검색이 필요합니다.",
        ),
        "retrieval_results": [RetrievalResult(
            result_id="retrieval:plan_1",
            plan_id="plan_1",
            source="neo4j",
            query="MATCH ...",
            items=[{
                "disclosure_id": "d1",
                "section_id": "s1",
            }],
            result_count=1,
        )],
        "retrieval_status": "COMPLETE",
        "retrieval_finish_reason": "근거를 확보했습니다.",
        "selected_result_ids": ["retrieval:plan_1"],
    }


def test_question_analyzer_retries_invalid_structured_output():
    llm = _SequenceLlm([
        {
            "question_analysis": {
                "decision": "retrieve",
                "normalized_question": "",
                "decision_reason": "검색이 필요합니다.",
            }
        },
        QuestionAnalyzerOutput(question_analysis=QuestionAnalysis(
            decision="retrieve",
            normalized_question="삼성전자 정보",
            decision_reason="검색이 필요합니다.",
            sub_questions=[{
                "question": "삼성전자의 정보는 무엇인가?",
                "entities": [{
                    "mention": "삼성전자",
                    "roles": ["ISSUER"],
                    "canonical_name": "삼성전자",
                    "corp_code": "00126380",
                    "match_status": "MATCHED",
                }],
                "events": [],
                "intents": ["DETAIL"],
                "periods": [],
                "requested_facts": ["기업 정보"],
            }],
        )),
    ])

    update = graph_module.question_analyzer({
        "question_id": "question-1",
        "question_text": "삼성전자 정보를 알려줘",
    }, llm=llm)

    assert update["question_analysis"].normalized_question == "삼성전자 정보"
    assert len(llm.calls) == 2
    assert "application 검증" in llm.calls[1][-1].content


def test_question_analyzer_retries_issuer_pair_outside_universe():
    def output(corp_code: str, match_status: str = "MATCHED") -> QuestionAnalyzerOutput:
        return QuestionAnalyzerOutput(question_analysis=QuestionAnalysis(
            decision="retrieve",
            normalized_question="삼성전자 정보",
            decision_reason="검색이 필요합니다.",
            sub_questions=[{
                "question": "삼성전자의 정보는 무엇인가?",
                "entities": [{
                    "mention": "삼성전자",
                    "roles": ["ISSUER"],
                    "canonical_name": "삼성전자",
                    "corp_code": corp_code,
                    "match_status": match_status,
                }],
                "events": [],
                "intents": ["DETAIL"],
                "periods": [],
                "requested_facts": ["기업 정보"],
            }],
        ))

    llm = _SequenceLlm([
        output("00164779"),
        output("00126380", match_status="UNKNOWN"),
    ])

    update = graph_module.question_analyzer({
        "question_id": "question-1",
        "question_text": "삼성전자 정보를 알려줘",
    }, llm=llm)

    entity = update["question_analysis"].sub_questions[0].entities[0]
    assert entity.corp_code == "00126380"
    assert entity.match_status == "UNKNOWN"
    assert len(llm.calls) == 2
    assert "동일한 행" in llm.calls[1][-1].content
    assert "canonical_name='삼성전자'" in llm.calls[1][-1].content
    assert "corp_name은 'SK하이닉스'" in llm.calls[1][-1].content


def test_cypher_builder_retries_invalid_parameters_json():
    llm = _SequenceLlm([
        {"cypher": "MATCH (c) RETURN c", "parameters_json": "[]"},
        {"cypher": "MATCH (c) RETURN c", "parameters_json": "{}"},
    ])

    query = tools.cypher_builder(
        _plan("neo4j"),
        user_question="삼성전자 정보를 알려줘",
        dependencies=[],
        neo4j_schema="schema",
        llm=llm,
    )

    assert query.parameters == {}
    assert len(llm.calls) == 2
    assert "parameters_json" in llm.calls[1][-1].content


def test_cypher_builder_normalizes_double_quoted_simple_aliases():
    llm = _SequenceLlm([{
        "cypher": (
            "MATCH (c:Company) "
            'RETURN c.corp_code AS "corp_code", '
            'c.stock_code AS "stock_code" LIMIT 10'
        ),
        "parameters_json": "{}",
    }])

    query = tools.cypher_builder(
        _plan("neo4j"),
        user_question="기업 정보를 알려줘",
        dependencies=[],
        neo4j_schema="schema",
        llm=llm,
    )

    assert query.cypher.endswith(
        "RETURN c.corp_code AS corp_code, c.stock_code AS stock_code LIMIT 10"
    )


def test_cypher_builder_retries_inline_and_out_of_domain_company_values():
    llm = _SequenceLlm([
        {
            "cypher": (
                "MATCH (c:Company) "
                "WHERE c.industry = 'IT' OR c.industry = '소프트웨어' "
                "OR c.industry = '정보통신' RETURN c LIMIT 100"
            ),
            "parameters_json": "{}",
        },
        {
            "cypher": (
                "MATCH (c:Company) WHERE c.industry IN $industries "
                "RETURN c LIMIT $limit"
            ),
            "parameters_json": '{"industries":["IT"],"limit":100}',
        },
    ])

    query = tools.cypher_builder(
        _plan("neo4j"),
        user_question="IT 기업들의 목록을 알려줘",
        dependencies=[],
        neo4j_schema="schema",
        llm=llm,
    )

    assert query.parameters["industries"] == ["IT"]
    assert len(llm.calls) == 2
    retry_message = llm.calls[1][-1].content
    assert "inline literal" in retry_message
    assert "소프트웨어" in retry_message
    assert "정보통신" in retry_message
    assert "허용값" in retry_message


def test_cypher_builder_retries_parameterized_out_of_domain_company_value():
    llm = _SequenceLlm([
        {
            "cypher": (
                "MATCH (c:Company) WHERE c.sector IN $sectors "
                "RETURN c LIMIT $limit"
            ),
            "parameters_json": '{"sectors":["소프트웨어"],"limit":100}',
        },
        {
            "cypher": (
                "MATCH (c:Company) WHERE c.sector IN $sectors "
                "RETURN c LIMIT $limit"
            ),
            "parameters_json": (
                '{"sectors":["AI소프트웨어·플랫폼"],"limit":100}'
            ),
        },
    ])

    query = tools.cypher_builder(
        _plan("neo4j"),
        user_question="AI 소프트웨어 기업들의 목록을 알려줘",
        dependencies=[],
        neo4j_schema="schema",
        llm=llm,
    )

    assert query.parameters["sectors"] == ["AI소프트웨어·플랫폼"]
    assert len(llm.calls) == 2
    assert "Company.sector domain에 없는 값" in llm.calls[1][-1].content


def test_cypher_builder_adds_missing_latest_disclosure_filter():
    llm = _SequenceLlm([
        {
            "cypher": "MATCH (d:Disclosure) RETURN d.id AS disclosure_id",
            "parameters_json": "{}",
        },
        {
            "cypher": (
                "MATCH (d:Disclosure) "
                "WHERE d.is_latest_version = $is_latest_version "
                "RETURN d.id AS disclosure_id"
            ),
            "parameters_json": '{"is_latest_version":true}',
        },
    ])

    query = tools.cypher_builder(
        _plan("neo4j"),
        user_question="최신 공시를 알려줘",
        dependencies=[],
        neo4j_schema="schema",
        llm=llm,
    )

    assert "(d:Disclosure {is_latest_version: true})" in query.cypher
    assert len(llm.calls) == 1


def test_cypher_builder_retries_invalid_relationship_direction():
    llm = _SequenceLlm([
        {
            "cypher": (
                "MATCH (c:Company)<-[:REPORTS]-(e:Event) "
                "RETURN e.content AS result LIMIT 10"
            ),
            "parameters_json": "{}",
        },
        {
            "cypher": (
                "MATCH (c:Company)-[:PUBLISHES]->(d:Disclosure) "
                "MATCH (d)-[:REPORTS]->(e:Event) "
                "WHERE d.is_latest_version = $is_latest_version "
                "RETURN e.content AS result LIMIT $limit"
            ),
            "parameters_json": '{"is_latest_version":true,"limit":10}',
        },
    ])

    query = tools.cypher_builder(
        _plan("neo4j"),
        user_question="삼성전자의 합병 사건을 알려줘",
        dependencies=[],
        neo4j_schema="schema",
        llm=llm,
    )

    assert "(d)-[:REPORTS]->(e:Event)" in query.cypher
    assert len(llm.calls) == 2
    assert "REPORTS" in llm.calls[1][-1].content


def test_query_builder_retries_invalid_structured_output(monkeypatch):
    llm = _SequenceLlm([
        {"mode": "vector", "query_text": None},
        QdrantQueryToolArgs(
            mode="vector",
            query_text="삼성전자 관련 정보",
            filters_json="[]",
            score_threshold_json="null",
        ),
    ])
    monkeypatch.setattr(tools, "text_to_hybrid_vector", lambda _: None)

    query = tools.query_builder(
        _plan(),
        user_question="삼성전자 정보를 알려줘",
        dependencies=[],
        qdrant_schema="schema",
        llm=llm,
    )

    assert query.query_text == "삼성전자 관련 정보"
    assert len(llm.calls) == 2
    assert "QdrantQuery" in llm.calls[1][-1].content


def test_compactor_retries_unknown_item_id():
    point = ScoredPoint(
        id="00000000-0000-0000-0000-000000000001",
        version=1,
        score=0.9,
        payload={
            "point_kind": "KV_TABLE",
            "canonical": {
                "entries": [{"key": "유동자산", "value": "100"}]
            },
        },
    )
    llm = _SequenceLlm([
        CompactorOutput(item_ids=[99]),
        CompactorOutput(item_ids=[0]),
    ])

    assert compact_qdrant_point(point, _plan(), llm=llm) == [0]
    assert len(llm.calls) == 2
    assert "point에 없는 item ID" in llm.calls[1][-1].content


def test_answer_generator_retries_unknown_answer_result_id():
    llm = _SequenceLlm([
        AnswerGeneratorOutput(
            answer="첫 번째 답변",
            used_result_ids=["answer_result_99"],
        ),
        AnswerGeneratorOutput(
            answer="수정된 답변",
            used_result_ids=["answer_result_1"],
        ),
    ])

    update = graph_module.answer_generator(_answer_state(), llm=llm)

    assert update["ai_answer"].answer == "수정된 답변"
    assert len(update["ai_answer"].citation) == 1
    trace = update["think_trace_events"][0]
    assert trace.name == "answer_generator"
    assert trace.details["used_result_ids"] == ["answer_result_1"]
    assert len(llm.calls) == 2
    assert "허용되지 않은 answer result ID" in llm.calls[1][-1].content
