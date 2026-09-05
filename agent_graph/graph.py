import json
from datetime import date
from functools import lru_cache
from typing import Any

from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from . import system_prompts as sp
from .llm import (
    ANSWER_GENERATOR_MAX_COMPLETION_TOKENS,
    MAX_LLM_RETRIES,
    QUESTION_ANALYZER_MAX_COMPLETION_TOKENS,
    RETRIEVER_MAX_TOKENS,
    bind_structured_output,
    build_output_retry_message,
    get_llm,
    invoke_with_rate_limit_retry,
)
from .state import (
    AgentState,
    AnswerGeneratorOutput,
    QuestionAnalyzerOutput,
    QuestionAnalysis,
    Scope,
    ScopeDraft,
)
from .tools import (
    calculate_table_statistic,
    combine_numeric_results,
    finish,
    retrieve_correction_history,
    retrieve_search,
)
from .utils import (
    assign_subquestion_ids,
    build_retriever_human_message,
    build_answer_generator_human_message,
    resolve_answer_draft,
    execute_tool_call,
    run_narrow_scope_agent,
    validate_retriever_tool_call,
)


graph_builder = StateGraph(AgentState)

# DB retrieval 최대 반복 횟수
MAX_RETRIEVAL_COUNT = 5
MAX_TOOL_CALL_RETRIES = MAX_LLM_RETRIES


@lru_cache(maxsize=1)
def _build_question_analyzer_llm() -> Any:
    """공용 LLM에 Question Analyzer structured output을 한 번 binding합니다."""

    return bind_structured_output(
        get_llm(QUESTION_ANALYZER_MAX_COMPLETION_TOKENS),
        QuestionAnalyzerOutput,
    )


@lru_cache(maxsize=1)
def _build_retriever_llm() -> Any:
    """공용 LLM에 retrieval query 생성 tool을 한 번 binding한다."""

    return get_llm(
        RETRIEVER_MAX_TOKENS,
        output_token_parameter="max_tokens",
    ).bind_tools(
        [retrieve_search, calculate_table_statistic, combine_numeric_results, finish],
    )


@lru_cache(maxsize=1)
def _build_answer_generator_llm() -> Any:
    """공용 LLM에 answer generator structured output을 한 번 binding한다."""

    return bind_structured_output(
        get_llm(ANSWER_GENERATOR_MAX_COMPLETION_TOKENS),
        AnswerGeneratorOutput,
    )


# Actual Node


def question_analyzer(
    state: AgentState,
    *,
    llm: Any | None = None,
) -> dict:
    """질문을 작은 정보 요구로 분석하고 routing 결정을 생성합니다."""

    print("-- question_analyzer 노드 호출 --")

    question = state["question_text"].strip()
    if not question:
        raise ValueError("question_text는 비어 있을 수 없습니다.")

    question_analyzer_llm = (
        _build_question_analyzer_llm()
        if llm is None
        else bind_structured_output(llm, QuestionAnalyzerOutput)
    )
    messages = [
        SystemMessage(content=sp.QUESTION_ANALYZER_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(
            {
                "current_date": date.today().isoformat(),
                "user_question": question,
            },
            ensure_ascii=False,
            indent=2,
        )),
    ]
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            response = invoke_with_rate_limit_retry(question_analyzer_llm, messages)
            analyzer_output = (
                response
                if isinstance(response, QuestionAnalyzerOutput)
                else QuestionAnalyzerOutput.model_validate(response)
            )
            break
        except (ValidationError, ValueError, TypeError, AttributeError) as error:
            if attempt == MAX_LLM_RETRIES:
                raise
            messages.append(HumanMessage(content=build_output_retry_message(
                "QuestionAnalyzerOutput",
                error,
            )))

    print("질문 분석 결과:\n", analyzer_output, "\n" + "\n\n")
    question_analysis = assign_subquestion_ids(analyzer_output.question_analysis)
    return {
        "question_analysis": question_analysis,
        "next_plan_seq": state.get("next_plan_seq", 1),
        "retrieval_status": "CONTINUE"
    }


def scope_resolver(state: AgentState) -> dict:
    """모든 SubQuestion의 Scope를 순차 생성하고 실패하면 GLOBAL로 대체합니다."""

    analysis_value = state.get("question_analysis")
    if analysis_value is None:
        raise ValueError("question_analysis가 없습니다.")
    analysis = QuestionAnalysis.model_validate(analysis_value)
    scopes = []
    for subquestion in analysis.sub_questions:
        subquestion_id = subquestion.subquestion_id
        if subquestion_id is None:
            raise ValueError("SubQuestion에 subquestion_id가 없습니다.")
        try:
            scope = run_narrow_scope_agent(subquestion_id, state)
        except Exception as error:
            print(
                f"[scope_resolver] {subquestion_id} narrow_scope 실패, "
                f"GLOBAL Scope로 대체합니다: {type(error).__name__}: {error}"
            )
            scope = Scope.from_scope_draft(
                ScopeDraft(
                    level="GLOBAL",
                    reason=(
                        "narrow_scope 재시도 후에도 범위를 확인하지 못해 "
                        f"GLOBAL Scope로 대체했습니다: {type(error).__name__}: {error}"
                    ),
                ),
                subquestion_id=subquestion_id,
            )
        scopes.append(scope)
    return {"scope_candidates": scopes}


def retriever(
    state: AgentState,
    *,
    llm: Any | None = None,
) -> dict:
    print("retriever 노드 호출")

    tools = [
        retrieve_search,
        retrieve_correction_history,
        calculate_table_statistic,
        combine_numeric_results,
        finish,
    ]

    retriever_llm = (
        _build_retriever_llm()
        if llm is None
        else llm.bind_tools(tools)
    )

    retriever_human_message = build_retriever_human_message(state)

    print(f"[retriever human message]:\n{retriever_human_message.content.replace("\\n", "\n").replace('\\"', '"')}\n\n")

    messages = [
        SystemMessage(content=sp.RETRIEVER_SYSTEM_PROMPT),
        retriever_human_message
    ]
    for attempt in range(MAX_TOOL_CALL_RETRIES + 1):
        try:
            response = invoke_with_rate_limit_retry(retriever_llm, messages)
            tool_calls = response.tool_calls
            if len(tool_calls) != 1:
                raise ValueError(
                    "응답 본문을 작성하지 말고 현재 상태에 적합한 tool을 "
                    "정확히 하나만 호출하세요. "
                    f"호출 개수: {len(tool_calls)}"
                )
            validate_retriever_tool_call(state, tool_calls[0])
            break
        except (ValidationError, ValueError, TypeError, AttributeError) as error:
            if attempt == MAX_TOOL_CALL_RETRIES:
                raise ValueError(
                    "재시도 후에도 유효한 tool을 정확히 하나 생성하지 못했습니다. "
                    f"마지막 오류: {error}"
                ) from error
            messages.append(HumanMessage(content=build_output_retry_message(
                "Retriever tool call",
                error,
            )))

    # 실제 함수 실행
    return execute_tool_call(state, tool_calls[0])


def answer_generator(
        state: AgentState,
        *,
        llm: Any | None = None
) -> dict:
    print("-- answer_genartor 노드 호출 --")

    answer_generator_human_message = build_answer_generator_human_message(state)
    print(f"[retriever human message]:\n{answer_generator_human_message.content.replace("\\n", "\n")}\n\n")

    answer_generator_llm = (
        _build_answer_generator_llm()
        if llm is None
        else bind_structured_output(llm, AnswerGeneratorOutput)
    )

    messages = [
        SystemMessage(content=sp.ANSWER_GENERATOR_SYSTEM_PROMPT),
        answer_generator_human_message,
    ]
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            response = invoke_with_rate_limit_retry(
                answer_generator_llm,
                messages,
            )
            answer_generator_output = (
                response
                if isinstance(response, AnswerGeneratorOutput)
                else AnswerGeneratorOutput.model_validate(response)
            )
            ai_answer = resolve_answer_draft(state, answer_generator_output)
            break
        except (ValidationError, ValueError, TypeError, AttributeError) as error:
            if attempt == MAX_LLM_RETRIES:
                raise
            messages.append(HumanMessage(content=build_output_retry_message(
                "AnswerGeneratorOutput",
                error,
            )))

    return {"ai_answer": ai_answer}


def answer_validator(
        state: AgentState,
        *,
        llm: Any | None = None
) -> dict:
    """
    answer_generator가 생성한 answer와 출처가 된 공시 원문을 직접 비교하여 답변의 신뢰도를 검증한다.
    """
    print ("-- answer_genartor 노드 호출 --")

    ai_answer = state.get("ai_answer").answer
    citation = state.get("ai_answer").citation

    



def answer_directly(state: AgentState) -> dict:
    print("answer_directly 노드 호출")
    return {"answer": "임시 답변", "citations": ["임시 인용 정보"]}

def request_clarification(state: AgentState) -> dict:
    print("request_clarification 노드 호출")
    return {"answer": " 임시 답변", "citations": ["임시 인용 정보"]}


# Conditional Routing Function

def route_after_analysis(state: AgentState) -> str:
    """질문 분석 결과에 대응하는 다음 LangGraph node 이름을 반환한다."""

    analysis = state.get("question_analysis")
    if analysis is None:
        raise ValueError("question_analysis가 생성되지 않았습니다.")

    decision = (
        analysis.decision
        if isinstance(analysis, QuestionAnalysis)
        else QuestionAnalysis.model_validate(analysis).decision
    )
    return decision


def route_after_retrieval(state: AgentState) -> str:
    """Retrieval 결과에 대응하는 다음 LangGraph node 이름을 반환한다."""

    status = state.get("retrieval_status")
    if status == "CONTINUE":
        return "retriever"
    elif status in {"COMPLETE", "INSUFFICIENT"}:
        return "answer_generator"

    raise ValueError(f"지원하지 않는 retrieval status: {status}")


# Node

graph_builder.add_node("question_analyzer", question_analyzer)
graph_builder.add_node("scope_resolver", scope_resolver)
graph_builder.add_node("retriever", retriever)
graph_builder.add_node("answer_generator", answer_generator)
graph_builder.add_node("answer_directly", answer_directly)
graph_builder.add_node("request_clarification", request_clarification)


# Edge

graph_builder.set_entry_point("question_analyzer")
graph_builder.add_conditional_edges(
    "question_analyzer",
    route_after_analysis,
    {
        "retrieve": "scope_resolver",
        "direct": "answer_directly",
        "clarify": "request_clarification",
    }
)
graph_builder.add_edge("scope_resolver", "retriever")
graph_builder.add_conditional_edges(
    "retriever",
    route_after_retrieval,
    {
        "retriever": "retriever",
        "answer_generator": "answer_generator"
    }
)

# 임시 END 노드 설정
graph_builder.set_finish_point("answer_generator")
graph_builder.set_finish_point("answer_directly")
graph_builder.set_finish_point("request_clarification")


# Graph Compile

graph = graph_builder.compile()




# 그래프 시각화 (Mermaid) 지원
if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())
