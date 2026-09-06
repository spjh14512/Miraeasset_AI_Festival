from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent_graph.graph import graph
from agent_graph.state import AgentState, Citation, ThinkTraceEvent
from agent_graph.utils import format_citations


class AnswerResponse(BaseModel):
    """Response contract required by the competition evaluator."""

    question_id: str
    question: str
    retrieved_context: str
    think_trace: str
    answer: str


app = FastAPI(title="DART Disclosure Analyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _as_json_string(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


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
    """Return an execution summary without exposing private chain-of-thought."""

    events = [
        event
        if isinstance(event, ThinkTraceEvent)
        else ThinkTraceEvent.model_validate(event)
        for event in output_state.get("think_trace_events", [])
    ]
    if not events:
        analysis = output_state.get("question_analysis")
        analysis_payload = _model_dump(analysis) if analysis is not None else {}
        decision_reason = (
            analysis_payload.get("decision_reason", "")
            if isinstance(analysis_payload, dict)
            else ""
        )
        fallback_reason = str(
            output_state.get("retrieval_finish_reason") or decision_reason
        ).strip()
        if fallback_reason:
            events.append(ThinkTraceEvent(
                type="node",
                name="graph",
                message=fallback_reason,
            ))

    trace = {
        "query_text": str(output_state.get("question_text", "")),
        "steps": [event.model_dump(mode="json") for event in events],
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
