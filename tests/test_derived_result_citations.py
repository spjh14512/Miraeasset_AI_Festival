from __future__ import annotations

from typing import Any

from agent_graph.state import AnswerGeneratorOutput, Citation, RetrievalResult
from agent_graph.tools import (
    calculate_table_statistic,
    combine_numeric_results,
)
from agent_graph.utils import resolve_answer_draft


def _r_table_item(
    *,
    values: list[str],
    column: str = "금액",
    disclosure_id: str,
    section_id: str,
    evidence_id: str,
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
            "section_id": section_id,
            "evidence_id": evidence_id,
        },
    }


def _state(*results: RetrievalResult, **extra: Any) -> dict[str, Any]:
    return {
        "question_id": "question-1",
        "question_text": "매출액 합계를 알려줘",
        "next_plan_seq": 1,
        "retrieval_results": list(results),
        **extra,
    }


def test_calculate_result_citation_reaches_final_answer():
    item = _r_table_item(
        values=["100", "200"],
        disclosure_id="d20240101000001",
        section_id="d20240101000001:src0:s1",
        evidence_id="d20240101000001:src0:s1:e1",
    )
    raw_result = RetrievalResult(
        result_id="retrieval:plan_1",
        plan_id="plan_1",
        source="qdrant",
        query="검색 쿼리",
        items=[item],
        result_count=1,
    )
    state = _state(raw_result)

    update = calculate_table_statistic.invoke({
        "variable_name": "합계",
        "operation": "sum",
        "column": "금액",
        "targets": [{"result_id": "retrieval:plan_1", "item_index": 0}],
        "state": state,
    })
    derived_result = update["retrieval_results"][0]
    assert derived_result.status == "SUCCESS"

    state["retrieval_results"].append(derived_result)
    state["selected_result_ids"] = [derived_result.result_id]

    ai_answer = resolve_answer_draft(
        state,
        AnswerGeneratorOutput(answer="합계는 300입니다.", used_result_ids=["answer_result_1"]),
    )

    assert ai_answer.citation == [
        Citation(
            disclosure_id="d20240101000001",
            section_id="d20240101000001:src0:s1",
            evidence_id="d20240101000001:src0:s1:e1",
        )
    ]


def test_combine_result_merges_citations_from_both_calculate_inputs():
    item_a = _r_table_item(
        values=["100"],
        disclosure_id="d20240101000001",
        section_id="d20240101000001:src0:s1",
        evidence_id="d20240101000001:src0:s1:e1",
    )
    item_b = _r_table_item(
        values=["200"],
        disclosure_id="d20240201000002",
        section_id="d20240201000002:src0:s2",
        evidence_id="d20240201000002:src0:s2:e2",
    )
    raw_a = RetrievalResult(
        result_id="retrieval:plan_1", plan_id="plan_1", source="qdrant",
        query="q", items=[item_a], result_count=1,
    )
    raw_b = RetrievalResult(
        result_id="retrieval:plan_2", plan_id="plan_2", source="qdrant",
        query="q", items=[item_b], result_count=1,
    )
    state = _state(raw_a, raw_b)

    calc_a = calculate_table_statistic.invoke({
        "variable_name": "A사 매출액",
        "operation": "sum",
        "column": "금액",
        "targets": [{"result_id": "retrieval:plan_1", "item_index": 0}],
        "state": state,
    })["retrieval_results"][0]
    state["retrieval_results"].append(calc_a)
    state["next_plan_seq"] += 1

    calc_b = calculate_table_statistic.invoke({
        "variable_name": "B사 매출액",
        "operation": "sum",
        "column": "금액",
        "targets": [{"result_id": "retrieval:plan_2", "item_index": 0}],
        "state": state,
    })["retrieval_results"][0]
    state["retrieval_results"].append(calc_b)
    state["next_plan_seq"] += 1

    combined = combine_numeric_results.invoke({
        "variable_name": "합계",
        "operation": "sum",
        "targets": [
            {"result_id": calc_a.result_id},
            {"result_id": calc_b.result_id},
        ],
        "state": state,
    })["retrieval_results"][0]
    assert combined.status == "SUCCESS"

    state["retrieval_results"].append(combined)
    state["selected_result_ids"] = [combined.result_id]

    ai_answer = resolve_answer_draft(
        state,
        AnswerGeneratorOutput(answer="합계는 300입니다.", used_result_ids=["answer_result_1"]),
    )

    assert set((c.disclosure_id, c.section_id, c.evidence_id) for c in ai_answer.citation) == {
        ("d20240101000001", "d20240101000001:src0:s1", "d20240101000001:src0:s1:e1"),
        ("d20240201000002", "d20240201000002:src0:s2", "d20240201000002:src0:s2:e2"),
    }


def test_ordering_result_citations_also_reach_final_answer():
    item_a = _r_table_item(
        values=["300"],
        disclosure_id="d20240101000001",
        section_id="d20240101000001:src0:s1",
        evidence_id="d20240101000001:src0:s1:e1",
    )
    item_b = _r_table_item(
        values=["100"],
        disclosure_id="d20240201000002",
        section_id="d20240201000002:src0:s2",
        evidence_id="d20240201000002:src0:s2:e2",
    )
    raw_a = RetrievalResult(
        result_id="retrieval:plan_1", plan_id="plan_1", source="qdrant",
        query="q", items=[item_a], result_count=1,
    )
    raw_b = RetrievalResult(
        result_id="retrieval:plan_2", plan_id="plan_2", source="qdrant",
        query="q", items=[item_b], result_count=1,
    )
    state = _state(raw_a, raw_b)

    calc_a = calculate_table_statistic.invoke({
        "variable_name": "A사 매출액", "operation": "sum", "column": "금액",
        "targets": [{"result_id": "retrieval:plan_1", "item_index": 0}],
        "state": state,
    })["retrieval_results"][0]
    state["retrieval_results"].append(calc_a)
    state["next_plan_seq"] += 1

    calc_b = calculate_table_statistic.invoke({
        "variable_name": "B사 매출액", "operation": "sum", "column": "금액",
        "targets": [{"result_id": "retrieval:plan_2", "item_index": 0}],
        "state": state,
    })["retrieval_results"][0]
    state["retrieval_results"].append(calc_b)
    state["next_plan_seq"] += 1

    ordering = combine_numeric_results.invoke({
        "variable_name": "순위",
        "operation": "ordering",
        "targets": [{"result_id": calc_a.result_id}, {"result_id": calc_b.result_id}],
        "state": state,
    })["retrieval_results"][0]
    assert ordering.status == "SUCCESS"
    assert ordering.metadata["result_kind"] == "numeric_ordering"

    state["retrieval_results"].append(ordering)
    state["selected_result_ids"] = [ordering.result_id]

    ai_answer = resolve_answer_draft(
        state,
        AnswerGeneratorOutput(answer="A사가 더 큽니다.", used_result_ids=["answer_result_1"]),
    )

    assert {(c.disclosure_id, c.evidence_id) for c in ai_answer.citation} == {
        ("d20240101000001", "d20240101000001:src0:s1:e1"),
        ("d20240201000002", "d20240201000002:src0:s2:e2"),
    }


def test_invalid_input_result_produces_no_citations_even_if_selected():
    # finish는 SUCCESS가 아닌 결과의 선택을 거부하지만(1.5단계), 그 가드를
    # 우회해서 INVALID_INPUT 결과가 실수로 선택되더라도 items가 비어
    # 있어 인용이 전혀 생성되지 않는다는 것을 방어적으로 확인한다.
    item = _r_table_item(
        values=["100", "해당없음"],
        disclosure_id="d20240101000001",
        section_id="d20240101000001:src0:s1",
        evidence_id="d20240101000001:src0:s1:e1",
    )
    raw_result = RetrievalResult(
        result_id="retrieval:plan_1", plan_id="plan_1", source="qdrant",
        query="q", items=[item], result_count=1,
    )
    state = _state(raw_result)

    derived_result = calculate_table_statistic.invoke({
        "variable_name": "합계",
        "operation": "sum",
        "column": "금액",
        "targets": [{"result_id": "retrieval:plan_1", "item_index": 0}],
        "state": state,
    })["retrieval_results"][0]
    assert derived_result.status == "INVALID_INPUT"
    assert derived_result.items == []

    state["retrieval_results"].append(derived_result)
    state["selected_result_ids"] = [derived_result.result_id]

    ai_answer = resolve_answer_draft(
        state,
        AnswerGeneratorOutput(answer="확인할 수 없습니다.", used_result_ids=[]),
    )

    assert ai_answer.citation == []
