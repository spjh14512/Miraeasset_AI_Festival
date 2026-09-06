from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from agent_graph.graph import graph
from agent_graph.state import AgentState, Citation
from agent_graph.utils import format_citations


class AnswerResponse(BaseModel):
    """Response contract required by the competition evaluator."""

    question_id: str
    question: str
    retrieved_context: str
    think_trace: str
    answer: str


app = FastAPI(title="DART Disclosure Analyst API")


def _as_json_string(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _retrieval_trace(result: Any) -> dict[str, Any]:
    """RetrievalResult에서 외부에 공개할 실행 기록만 추출합니다."""

    payload = _model_dump(result)
    if not isinstance(payload, dict):
        return {"result_id": "", "source": "", "status": "ERROR"}

    trace = {
        "result_id": str(payload.get("result_id", "")),
        "source": str(payload.get("source", "")),
        "status": str(payload.get("status", "")),
        "query": str(payload.get("query", "")),
        "result_count": int(payload.get("result_count", 0)),
    }
    metadata = payload.get("metadata")
    if trace["status"] != "SUCCESS" and isinstance(metadata, dict):
        trace["failure"] = {
            "stage": str(metadata.get("failure_stage", "")),
            "type": str(metadata.get("error_type", "")),
        }
    return trace


def _calculation_trace(result: Any) -> dict[str, Any]:
    """계산 RetrievalResult에서 연산과 결과를 추출합니다."""

    payload = _model_dump(result)
    items = payload.get("items", []) if isinstance(payload, dict) else []
    fields: dict[str, Any] = {}
    if items and isinstance(items[0], dict):
        candidate = items[0].get("fields")
        if isinstance(candidate, dict):
            fields = candidate
    return {
        "result_id": str(payload.get("result_id", "")),
        "status": str(payload.get("status", "")),
        "variable_name": fields.get("variable_name"),
        "operation": fields.get("operation"),
        "value": fields.get("value", fields.get("ordered_results")),
        "unit": fields.get("unit"),
        "input_count": fields.get("input_count"),
    }


def _build_retrieved_context(output_state: dict[str, Any]) -> str:
    ai_answer = output_state.get("ai_answer")
    if hasattr(ai_answer, "citation"):
        raw_citations = ai_answer.citation
    elif isinstance(ai_answer, dict):
        raw_citations = ai_answer.get("citation", [])
    else:
        raw_citations = []

    citations = [
        citation
        if isinstance(citation, Citation)
        else Citation.model_validate(citation)
        for citation in raw_citations
    ]
    return "\n".join(format_citations(citations, style="path"))


def _build_think_trace(output_state: dict[str, Any]) -> str:
    """Return an auditable execution summary without private chain-of-thought."""

    analysis = output_state.get("question_analysis")
    analysis_payload = _model_dump(analysis) if analysis is not None else {}
    if not isinstance(analysis_payload, dict):
        analysis_payload = {}

    sub_questions = analysis_payload.get("sub_questions", [])
    requested_facts = list(dict.fromkeys(
        fact
        for sub_question in sub_questions
        if isinstance(sub_question, dict)
        for fact in sub_question.get("requested_facts", [])
        if isinstance(fact, str)
    ))
    retrieval_results = output_state.get("retrieval_results", [])
    search_results = [
        result
        for result in retrieval_results
        if _model_dump(result).get("source") != "derived"
    ]
    calculation_results = [
        result
        for result in retrieval_results
        if _model_dump(result).get("source") == "derived"
    ]
    ai_answer = output_state.get("ai_answer")
    answer_payload = _model_dump(ai_answer) if ai_answer is not None else {}
    citations = (
        answer_payload.get("citation", [])
        if isinstance(answer_payload, dict)
        else []
    )
    trace = {
        "query_text": str(output_state.get("question_text", "")),
        "question_analysis": {
            "decision": str(analysis_payload.get("decision", "")),
            "normalized_question": str(
                analysis_payload.get("normalized_question", "")
            ),
            "decision_reason": str(analysis_payload.get("decision_reason", "")),
            "requested_facts": requested_facts,
        },
        "retrieval_history": [
            _retrieval_trace(result)
            for result in search_results
        ],
        "calculation_history": [
            _calculation_trace(result)
            for result in calculation_results
        ],
        "selected_evidence": {
            "result_ids": list(output_state.get("selected_result_ids", [])),
            "citations": citations,
        },
        "answer_validation": {
            "status": str(
                output_state.get("answer_validation_status", "NOT_RUN")
            ),
        },
        "final_basis": {
            "retrieval_status": str(output_state.get("retrieval_status", "")),
            "retrieval_finish_reason": str(
                output_state.get("retrieval_finish_reason", "")
            ),
        },
    }
    # 상세 예외 문구는 접속 정보를 노출할 수 있으므로 건수만 공개합니다.
    errors = output_state.get("errors")
    if errors:
        trace["warnings"] = {
            "count": len(errors),
            "message": "일부 처리 단계가 실패하거나 생략됐습니다.",
        }
    return _as_json_string(trace)


def _extract_answer(output_state: dict[str, Any]) -> str:
    ai_answer = output_state.get("ai_answer")
    if ai_answer is not None:
        if hasattr(ai_answer, "answer"):
            return str(ai_answer.answer)
        if isinstance(ai_answer, dict) and "answer" in ai_answer:
            return str(ai_answer["answer"])

    # Keep the API compatible with direct/clarification graph branches.
    if "answer" in output_state:
        return str(output_state["answer"])
    raise ValueError("Agent output does not contain an answer.")


@app.get("/answer", response_model=AnswerResponse)
async def answer(
    question_id: str = Query(..., min_length=1),
    question: str = Query(..., min_length=1),
) -> AnswerResponse:
    input_state = AgentState(
        question_id=question_id,
        question_text=question,
    )
    output_state = await run_in_threadpool(graph.invoke, input_state)
    retrieved_context = await run_in_threadpool(
        _build_retrieved_context,
        output_state,
    )

    return AnswerResponse(
        question_id=question_id,
        question=question,
        retrieved_context=retrieved_context,
        think_trace=_build_think_trace(output_state),
        answer=_extract_answer(output_state),
    )


def main() -> None:
    """Run the API server for local development."""

    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
