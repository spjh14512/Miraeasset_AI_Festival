from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agent_graph import graph as graph_module
from agent_graph.state import (
    AnswerGeneratorOutput,
    QuestionAnalysis,
    RetrievalResult,
    merge_results,
)


class _FakeStructuredLlm:
    """answer_generator()가 요구하는 with_structured_output(...).invoke(...) 형태를 흉내냅니다."""

    def __init__(self, result: AnswerGeneratorOutput):
        self.result = result

    def invoke(self, _messages):
        return self.result


class _FakeAnswerLlm:
    def __init__(self, result: AnswerGeneratorOutput):
        self._structured = _FakeStructuredLlm(result)

    def with_structured_output(self, _schema, *, method):
        assert method == "json_schema"
        return self._structured


class _ScriptedRetrieverLlm:
    """미리 정해둔 tool_call을 호출 순서대로 반환하는 가짜 Retriever LLM입니다."""

    def __init__(self, tool_call_sequence: list[dict[str, Any]]):
        self.bound_tools = None
        self.invoke_count = 0
        self._sequence = tool_call_sequence

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, _messages):
        tool_call = self._sequence[self.invoke_count]
        self.invoke_count += 1
        return SimpleNamespace(tool_calls=[tool_call])


def _r_table_item(
    *,
    values: list[str],
    column: str = "매출액",
    disclosure_id: str,
) -> dict[str, Any]:
    records = [
        {"record_index": index, "values": {"구분": f"row{index}", column: value}}
        for index, value in enumerate(values)
    ]
    return {
        "type": "r_table",
        "columns": ["구분", column],
        "scope": {"kind": "table"},
        "records": records,
        "available_record_count": len(records),
        "included_record_count": len(records),
        "omitted_record_count": 0,
        "metadata": {
            "disclosure_id": disclosure_id,
            "section_id": f"{disclosure_id}:src0:s1",
            "evidence_id": f"{disclosure_id}:src0:s1:e1",
        },
    }


def _run_retriever_until_finished(state: dict[str, Any], llm) -> dict[str, Any]:
    """실제 LangGraph 실행에서 route_after_retrieval이 하는 병합·반복을 재현합니다."""

    while state.get("retrieval_status") == "CONTINUE":
        update = graph_module.retriever(state, llm=llm)
        if "retrieval_results" in update:
            state["retrieval_results"] = merge_results(
                state.get("retrieval_results", []),
                update["retrieval_results"],
            )
        for key, value in update.items():
            if key != "retrieval_results":
                state[key] = value
    return state


def test_full_pipeline_calculate_combine_finish_answer_from_preloaded_results():
    # 두 기업의 매출액 표가 이미 검색되어 있다고 가정하고 시작한다
    # (retrieve_search 자체는 실제 DB 연결이 필요해 다른 테스트에서 커버됨).
    item_a = _r_table_item(values=["100"], disclosure_id="d20240101000001")
    item_b = _r_table_item(values=["200"], disclosure_id="d20240201000002")

    state: dict[str, Any] = {
        "question_id": "question-1",
        "question_text": "A사와 B사의 매출액 합계를 알려줘",
        "question_analysis": QuestionAnalysis(
            decision="retrieve",
            normalized_question="A사와 B사의 매출액 합계",
            decision_reason="공시 표 값 확인이 필요합니다.",
        ),
        "retrieval_status": "CONTINUE",
        "next_plan_seq": 3,
        "retrieval_results": [
            RetrievalResult(
                result_id="retrieval:plan_1", plan_id="plan_1", source="qdrant",
                query="A사 매출액", items=[item_a], result_count=1,
            ),
            RetrievalResult(
                result_id="retrieval:plan_2", plan_id="plan_2", source="qdrant",
                query="B사 매출액", items=[item_b], result_count=1,
            ),
        ],
    }

    tool_call_sequence = [
        {
            "name": "calculate_table_statistic",
            "args": {
                "variable_name": "A사 매출액 합계",
                "operation": "sum",
                "column": "매출액",
                "targets": [{"result_id": "retrieval:plan_1", "item_index": 0}],
            },
        },
        {
            "name": "calculate_table_statistic",
            "args": {
                "variable_name": "B사 매출액 합계",
                "operation": "sum",
                "column": "매출액",
                "targets": [{"result_id": "retrieval:plan_2", "item_index": 0}],
            },
        },
        {
            "name": "combine_numeric_results",
            "args": {
                "variable_name": "A사+B사 매출액 합계",
                "operation": "sum",
                "targets": [
                    {"result_id": "derived:plan_3"},
                    {"result_id": "derived:plan_4"},
                ],
            },
        },
        {
            "name": "finish",
            "args": {
                "status": "COMPLETE",
                "reason": "두 기업의 매출액 합계를 계산했습니다.",
                "selected_result_ids": ["derived:plan_5"],
            },
        },
    ]
    retriever_llm = _ScriptedRetrieverLlm(tool_call_sequence)

    state = _run_retriever_until_finished(state, retriever_llm)

    assert retriever_llm.invoke_count == 4
    assert state["retrieval_status"] == "COMPLETE"
    assert state["selected_result_ids"] == ["derived:plan_5"]

    results_by_id = {result.result_id: result for result in state["retrieval_results"]}
    calc_a = results_by_id["derived:plan_3"]
    calc_b = results_by_id["derived:plan_4"]
    combined = results_by_id["derived:plan_5"]
    assert calc_a.status == "SUCCESS"
    assert calc_a.items[0]["fields"]["value"] == "100"
    assert calc_b.status == "SUCCESS"
    assert calc_b.items[0]["fields"]["value"] == "200"
    assert combined.status == "SUCCESS"
    assert combined.metadata["result_kind"] == "numeric_scalar"
    assert combined.items[0]["fields"]["value"] == "300"

    answer_llm = _FakeAnswerLlm(AnswerGeneratorOutput(
        answer="A사와 B사의 매출액 합계는 300입니다.",
        used_result_ids=["answer_result_1"],
    ))
    answer_update = graph_module.answer_generator(state, llm=answer_llm)

    ai_answer = answer_update["ai_answer"]
    assert ai_answer.answer == "A사와 B사의 매출액 합계는 300입니다."
    assert {
        (citation.disclosure_id, citation.section_id, citation.evidence_id)
        for citation in ai_answer.citation
    } == {
        ("d20240101000001", "d20240101000001:src0:s1", "d20240101000001:src0:s1:e1"),
        ("d20240201000002", "d20240201000002:src0:s1", "d20240201000002:src0:s1:e1"),
    }


def test_full_pipeline_incomplete_table_routes_back_to_retriever_before_finishing():
    # chunk_count=2인데 chunk 하나만 있는 불완전한 표 -> calculate가
    # INVALID_INPUT을 내면 retriever가 계속 루프를 돌다가 다른 대상을
    # finish로 선택하는 흐름을 검증한다.
    incomplete_chunk = {
        "type": "r_table",
        "columns": ["구분", "매출액"],
        "scope": {"kind": "row_group", "row_start_index": 0, "row_end_index": 0},
        "records": [{"record_index": 0, "values": {"구분": "row0", "매출액": "100"}}],
        "available_record_count": 1,
        "included_record_count": 1,
        "omitted_record_count": 0,
        "metadata": {
            "disclosure_id": "d20240101000001",
            "section_id": "d20240101000001:src0:s1",
            "evidence_id": "d20240101000001:src0:s1:e1",
            "table_id": "table-1",
            "chunk_index": 0,
            "chunk_count": 2,
            "row_start_index": 0,
            "row_end_index": 0,
        },
    }
    state: dict[str, Any] = {
        "question_id": "question-1",
        "question_text": "매출액을 알려줘",
        "question_analysis": QuestionAnalysis(
            decision="retrieve",
            normalized_question="매출액",
            decision_reason="공시 표 값 확인이 필요합니다.",
        ),
        "retrieval_status": "CONTINUE",
        "next_plan_seq": 2,
        "retrieval_results": [
            RetrievalResult(
                result_id="retrieval:plan_1", plan_id="plan_1", source="qdrant",
                query="매출액", items=[incomplete_chunk], result_count=1,
            ),
        ],
    }

    tool_call_sequence = [
        {
            "name": "calculate_table_statistic",
            "args": {
                "variable_name": "매출액 합계",
                "operation": "sum",
                "column": "매출액",
                "targets": [{"result_id": "retrieval:plan_1", "item_index": 0}],
            },
        },
        {
            "name": "finish",
            "args": {
                "status": "INSUFFICIENT",
                "reason": "표가 일부만 검색되어 계산할 수 없습니다.",
                "selected_result_ids": [],
            },
        },
    ]
    retriever_llm = _ScriptedRetrieverLlm(tool_call_sequence)

    state = _run_retriever_until_finished(state, retriever_llm)

    assert retriever_llm.invoke_count == 2
    assert state["retrieval_status"] == "INSUFFICIENT"
    assert state["selected_result_ids"] == []

    derived = next(
        result for result in state["retrieval_results"]
        if result.result_id == "derived:plan_2"
    )
    assert derived.status == "INVALID_INPUT"
    assert derived.metadata["failure_stage"] == "completeness_check"
