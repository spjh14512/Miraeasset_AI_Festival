from __future__ import annotations

from agent_graph import graph as graph_module
from agent_graph import tools
from agent_graph.compactor import CompactorOutput, compact_qdrant_point
from agent_graph.state import (
    AnswerGeneratorOutput,
    Plan,
    PlannerOutput,
    QuestionAnalysis,
    RetrievalResult,
)
from agent_graph.tools import QdrantQueryToolArgs
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


def test_planner_retries_invalid_structured_output():
    llm = _SequenceLlm([
        {
            "question_analysis": {
                "decision": "retrieve",
                "normalized_question": "",
                "decision_reason": "검색이 필요합니다.",
            }
        },
        PlannerOutput(question_analysis=QuestionAnalysis(
            decision="retrieve",
            normalized_question="삼성전자 정보",
            decision_reason="검색이 필요합니다.",
        )),
    ])

    update = graph_module.planner({
        "question_id": "question-1",
        "question_text": "삼성전자 정보를 알려줘",
    }, llm=llm)

    assert update["question_analysis"].normalized_question == "삼성전자 정보"
    assert len(llm.calls) == 2
    assert "application 검증" in llm.calls[1][-1].content


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
    assert len(llm.calls) == 2
    assert "허용되지 않은 answer result ID" in llm.calls[1][-1].content
