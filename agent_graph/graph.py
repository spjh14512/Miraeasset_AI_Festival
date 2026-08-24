from functools import lru_cache
from typing import Any

from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage

from . import system_prompts as sp
from .llm import get_llm
from .state import AgentState, AnswerDraft, Plan, PlannerOutput, QuestionAnalysis
from .tools import (
    build_retriever_human_message,
    build_answer_generator_human_message,
    resolve_answer_draft,
    create_plan,
    delete_plan,
    execute_tool_call,
    modify_plan,
    retrieve_search,
    finish
)


graph_builder = StateGraph(AgentState)

# DB retrieval 최대 반복 횟수
MAX_RETRIEVAL_COUNT = 5
MAX_TOOL_CALL_RETRIES = 2


@lru_cache(maxsize=1)
def _build_planner_llm() -> Any:
    """공용 LLM에 planner structured output을 한 번 binding한다."""

    return get_llm().with_structured_output(
        PlannerOutput,
        method="function_calling",
    )
@lru_cache(maxsize=1)
def _build_retriever_llm() -> Any:
    """공용 LLM에 retrieval query 생성 tool을 한 번 binding한다."""

    return get_llm().bind_tools(
        [retrieve_search, create_plan, delete_plan, modify_plan, finish],
    )
@lru_cache(maxsize=1)
def _build_answer_generator_llm() -> Any:
    """공용 LLM에 answer generator structured output을 한 번 binding한다."""

    return get_llm().with_structured_output(
        AnswerDraft,
        method="function_calling"
    )


# Actual Node


def planner(
    state: AgentState,
    *,
    llm: Any | None = None,
) -> dict[str, QuestionAnalysis | list[Plan] | int]:
    """
    사용자의 질문을 분석하고 routing 결정과 plan을 생성한다.
    """

    print("-- planner 노드 호출 --")

    question = state["question_text"].strip()
    if not question:
        raise ValueError("question_text는 비어 있을 수 없습니다.")

    planner_llm = (
        _build_planner_llm()
        if llm is None
        else llm.with_structured_output(
            PlannerOutput,
            method="function_calling",
        )
    )
    response = planner_llm.invoke(
        [
            SystemMessage(content = sp.PLANNER_SYSTEM_PROMPT),
            HumanMessage(content = question),
        ]
    )

    print("질문 분석 결과:\n", response, "\n" + "\n\n")

    planner_output = (
        response
        if isinstance(response, PlannerOutput)
        else PlannerOutput.model_validate(response)
    )
    next_plan_seq = state.get("next_plan_seq", 1)
    plans: list[Plan] = []
    for draft in planner_output.plans:
        plan = Plan.from_plan_draft(draft, next_plan_seq)
        next_plan_seq += 1
        plans.append(plan)

    return {
        "question_analysis": planner_output.question_analysis,
        "plans": plans,
        "next_plan_seq": next_plan_seq,
        "retrieval_status": "CONTINUE"
    }


def retriever(
    state: AgentState,
    *,
    llm: Any | None = None,
) -> dict:
    print("retriever 노드 호출")

    tools = [retrieve_search, create_plan, delete_plan, modify_plan, finish]

    retriever_llm = (
        _build_retriever_llm()
        if llm is None
        else llm.bind_tools(tools)
    )

    retriever_human_message = build_retriever_human_message(state)

    print(f"[retriever human message]:\n{retriever_human_message}\n\n")

    messages = [
        SystemMessage(content=sp.RETRIEVER_SYSTEM_PROMPT),
        retriever_human_message
    ]
    for attempt in range(MAX_TOOL_CALL_RETRIES + 1):
        response = retriever_llm.invoke(messages)
        tool_calls = response.tool_calls
        if len(tool_calls) == 1:
            break
        if attempt < MAX_TOOL_CALL_RETRIES:
            messages.append(HumanMessage(
                content=(
                    "응답 본문을 작성하지 말고 현재 상태에 적합한 tool을 "
                    "정확히 하나만 호출하세요."
                )
            ))
    else:
        tool_names = [
            tool_call.get("name", "<unknown>")
            for tool_call in tool_calls
        ]
        raise ValueError(
            "재시도 후에도 정확히 하나의 tool을 호출하지 않았습니다. "
            f"호출 개수: {len(tool_calls)}, tool: {tool_names}"
        )

    # 실제 함수 실행
    return execute_tool_call(state, tool_calls[0])


def answer_generator(
        state: AgentState,
        *,
        llm: Any | None = None
) -> dict:
    print("-- answer_genartor 노드 호출 --")

    answer_generator_human_message = build_answer_generator_human_message(state)
    print(f"[retriever human message]:\n{answer_generator_human_message}\n\n")

    answer_generator_llm = (
        _build_answer_generator_llm()
        if llm is None
        else llm.with_structured_output(
            AnswerDraft,
            method="function_calling"
        )   
    )

    response = answer_generator_llm.invoke(
        [
            SystemMessage(content=sp.ANSWER_GENERATOR_SYSTEM_PROMPT),
            answer_generator_human_message
        ]
    )

    answer_generator_output = (
        response
        if isinstance(response, AnswerDraft)
        else AnswerDraft.model_validate(response)
    )

    return {
        "ai_answer": resolve_answer_draft(state, answer_generator_output)
    }


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

graph_builder.add_node("planner", planner)
graph_builder.add_node("retriever", retriever)
graph_builder.add_node("answer_generator", answer_generator)
graph_builder.add_node("answer_directly", answer_directly)
graph_builder.add_node("request_clarification", request_clarification)


# Edge

graph_builder.set_entry_point("planner")
graph_builder.add_conditional_edges(
    "planner",
    route_after_analysis,
    {
        "retrieve": "retriever",
        "direct": "answer_directly",
        "clarify": "request_clarification",
    }
)
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
