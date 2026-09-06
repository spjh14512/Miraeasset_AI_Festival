import json
from datetime import date
from functools import lru_cache
from typing import Any

from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from . import system_prompts as sp
from .llm import (
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
    AiAnswer,
    QuestionAnalyzerOutput,
    QuestionAnalysis,
    Scope,
    ScopeDraft,
    ThinkTraceEvent,
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
    enforce_question_clarification_policy,
    generate_answer,
    execute_tool_call,
    load_issuer_universe_for_llm_tsv,
    run_narrow_scope_agent,
    RepeatedFailedPlanError,
    UNSUPPORTED_ANSWER_FALLBACK,
    validate_and_repair_answer,
    subquestion_has_out_of_universe_issuer,
    validate_question_analysis_issuers,
    validate_retriever_tool_call,
)


graph_builder = StateGraph(AgentState)

# 검증 실패를 답변 실패로 번지게 하지 않을 예외들. 외부 요인(LLM 응답,
# 네트워크)에서 오는 것만 포함하며, 코드 결함은 일부러 제외한다.
ANSWER_VALIDATION_FAILURES = (
    ValidationError,
    ValueError,
    TimeoutError,
    ConnectionError,
    OSError,
)
MAX_TOOL_CALL_RETRIES = MAX_LLM_RETRIES


@lru_cache(maxsize=1)
def _build_question_analyzer_llm() -> Any:
    """공용 LLM에 Question Analyzer structured output을 한 번 binding합니다."""

    return bind_structured_output(
        get_llm(QUESTION_ANALYZER_MAX_COMPLETION_TOKENS),
        QuestionAnalyzerOutput,
    )


# retriever가 LLM에 제시하는 tool 목록의 유일한 정의입니다. 이 목록과
# retriever() 노드 내부 tool 목록이 어긋나면, 캐시된 LLM 경로(운영)와
# llm 인자를 넘기는 경로(테스트)의 동작이 갈라져 테스트가 놓치는 운영
# 결함이 생깁니다.
RETRIEVER_TOOLS = [
    retrieve_search,
    retrieve_correction_history,
    calculate_table_statistic,
    combine_numeric_results,
    finish,
]


@lru_cache(maxsize=1)
def _build_retriever_llm() -> Any:
    """공용 LLM에 retrieval query 생성 tool을 한 번 binding한다."""

    return get_llm(
        RETRIEVER_MAX_TOKENS,
        output_token_parameter="max_tokens",
    ).bind_tools(RETRIEVER_TOOLS)


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
                "issuer_universe_tsv": load_issuer_universe_for_llm_tsv(),
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
            question_analysis = validate_question_analysis_issuers(
                analyzer_output.question_analysis
            )
            break
        except (ValidationError, ValueError, TypeError, AttributeError) as error:
            if attempt == MAX_LLM_RETRIES:
                raise
            messages.append(HumanMessage(content=build_output_retry_message(
                "QuestionAnalyzerOutput",
                error,
            )))

    question_analysis = enforce_question_clarification_policy(question_analysis)
    question_analysis = assign_subquestion_ids(question_analysis)
    print("질문 분석 결과:\n", question_analysis, "\n" + "\n\n")
    return {
        "question_analysis": question_analysis,
        "next_plan_seq": state.get("next_plan_seq", 1),
        "retrieval_search_count": state.get("retrieval_search_count", 0),
        "retrieval_status": "CONTINUE",
        "think_trace_events": [ThinkTraceEvent(
            type="node",
            name="question_analyzer",
            message=question_analysis.decision_reason,
            details=question_analysis.model_dump(
                mode="json",
                exclude={"decision_reason"},
            ),
        )],
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
        if subquestion_has_out_of_universe_issuer(subquestion):
            print(
                f"[scope_resolver] {subquestion_id}의 ISSUER 이름·기업코드가 "
                "universe에서 선택되지 않아 Scope 생성을 건너뜁니다."
            )
            continue
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
    return {
        "scope_candidates": scopes,
        "think_trace_events": [
            ThinkTraceEvent(
                type="node",
                name="scope_resolver",
                message=scope.reason,
                details=scope.model_dump(mode="json", exclude={"reason"}),
            )
            for scope in scopes
        ],
    }


def retriever(
    state: AgentState,
    *,
    llm: Any | None = None,
) -> dict:
    print("retriever 노드 호출")

    retriever_llm = (
        _build_retriever_llm()
        if llm is None
        else llm.bind_tools(RETRIEVER_TOOLS)
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
            if not tool_calls:
                raise ValueError(
                    "응답 본문을 작성하지 말고 현재 상태에 적합한 tool을 "
                    "정확히 하나만 호출하세요. "
                    "호출 개수: 0"
                )
            non_finish_calls = [
                tool_call
                for tool_call in tool_calls
                if tool_call.get("name") != "finish"
            ]
            tool_call = (
                non_finish_calls[0]
                if non_finish_calls
                else tool_calls[0]
            )
            validate_retriever_tool_call(state, tool_call)
            break
        except (ValidationError, ValueError, TypeError, AttributeError) as error:
            if attempt == MAX_TOOL_CALL_RETRIES:
                if isinstance(error, RepeatedFailedPlanError):
                    tool_call = {
                        "name": "finish",
                        "args": {
                            "status": "INSUFFICIENT",
                            "reason": (
                                "동일한 검색 계획이 반복 실패했고 새로운 유효한 "
                                "검색 전략을 생성하지 못했습니다."
                            ),
                            "selected_result_ids": [],
                        },
                    }
                    break
                if not state.get("scope_candidates"):
                    tool_call = {
                        "name": "finish",
                        "args": {
                            "status": "INSUFFICIENT",
                            "reason": (
                                "검색에 사용할 유효한 Scope가 없어 retrieval을 "
                                "진행할 수 없습니다."
                            ),
                            "selected_result_ids": [],
                        },
                    }
                    break
                tool_call = {
                    "name": "finish",
                    "args": {
                        "status": "INSUFFICIENT",
                        "reason": (
                            "Retriever가 재시도 후에도 유효한 다음 행동을 "
                            "생성하지 못해 안전하게 검색을 종료했습니다."
                        ),
                        "selected_result_ids": [],
                    },
                }
                break
            messages.append(HumanMessage(content=build_output_retry_message(
                "Retriever tool call",
                error,
            )))

    # 실제 함수 실행
    tool_update = execute_tool_call(state, tool_call)
    retriever_event = ThinkTraceEvent(
        type="node",
        name="retriever",
        message=f"{tool_call['name']} tool을 선택했습니다.",
        details={"selected_tool": tool_call["name"]},
    )
    return {
        **tool_update,
        "think_trace_events": [
            retriever_event,
            *tool_update.get("think_trace_events", []),
        ],
    }


def answer_generator(
        state: AgentState,
        *,
        llm: Any | None = None
) -> dict:
    print("-- answer_genartor 노드 호출 --")

    ai_answer, answer_generator_output = generate_answer(
        state,
        llm=llm,
        include_output=True,
    )

    return {
        "ai_answer": ai_answer,
        "think_trace_events": [ThinkTraceEvent(
            type="node",
            name="answer_generator",
            message="선택된 검색 결과를 근거로 최종 답변을 생성했습니다.",
            details={
                "used_result_ids": answer_generator_output.used_result_ids,
            },
        )],
    }


def answer_validator(
        state: AgentState,
        *,
        llm: Any | None = None
) -> dict:
    """answer_generator의 초안을 근거·요구사항과 대조해 검증하고 필요하면 고친다.

    검증 자체가 외부 요인으로 실패하면(LLM timeout, 연결 오류, 구조적
    출력 재시도 소진 등) 예외를 밖으로 던지지 않고 검증 전 답변을 그대로
    반환한다. 이 노드는 검색 기반 답변의 필수 종료 경로이므로, 여기서
    예외가 나가면 이미 정상적으로 생성된 답변까지 API 500이 된다.

    반대로 KeyError처럼 코드 결함에서 오는 예외는 잡지 않는다. 모두
    잡아버리면 계약 위반이 "검증 생략"으로 조용히 묻혀 테스트에서도
    드러나지 않는다.
    """

    print("-- answer_validator 노드 호출 --")

    ai_answer = state.get("ai_answer")
    if ai_answer is None:
        raise ValueError("ai_answer가 생성되지 않았습니다.")
    if not isinstance(ai_answer, AiAnswer):
        ai_answer = AiAnswer.model_validate(ai_answer)

    try:
        validated = validate_and_repair_answer(state, ai_answer, llm=llm)
    except ANSWER_VALIDATION_FAILURES as error:
        print(f"answer_validator 실패, 검증 전 답변을 그대로 사용합니다: {error}")
        errors = list(state.get("errors", []))
        errors.append(f"answer_validator 실패: {type(error).__name__}: {error}")
        return {
            "ai_answer": ai_answer,
            "answer_validation_status": "SKIPPED",
            "errors": errors,
        }

    if validated.answer.startswith(UNSUPPORTED_ANSWER_FALLBACK):
        validation_status = "FALLBACK"
    elif validated != ai_answer:
        validation_status = "REPAIRED"
    else:
        validation_status = "PASSED"
    return {
        "ai_answer": validated,
        "answer_validation_status": validation_status,
    }



@lru_cache(maxsize=1)
def _build_direct_answer_llm() -> Any:
    """공용 LLM에 direct 응답용 설정을 한 번 적용합니다."""

    return get_llm()


def answer_directly(
    state: AgentState,
    *,
    llm: Any | None = None,
) -> dict:
    """검색이 필요 없는 인사·기능 안내 질문에 짧게 답합니다."""

    print("-- answer_directly 노드 호출 --")

    question = state["question_text"].strip()
    if not question:
        raise ValueError("question_text는 비어 있을 수 없습니다.")

    direct_answer_llm = _build_direct_answer_llm() if llm is None else llm
    messages = [
        SystemMessage(content=sp.DIRECT_ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    response = invoke_with_rate_limit_retry(direct_answer_llm, messages)

    # 검색을 수행하지 않았으므로 인용할 근거가 없습니다.
    return {"ai_answer": AiAnswer(answer=str(response.content), citation=[])}

def request_clarification(state: AgentState) -> dict:
    """Question Analyzer가 만든 되묻기 질문을 그대로 사용자에게 전달합니다."""

    print("-- request_clarification 노드 호출 --")

    analysis = state.get("question_analysis")
    if analysis is None:
        raise ValueError("question_analysis가 생성되지 않았습니다.")
    if not isinstance(analysis, QuestionAnalysis):
        analysis = QuestionAnalysis.model_validate(analysis)
    if analysis.decision != "clarify":
        raise ValueError(
            "clarify 결정이 아닌 질문은 request_clarification으로 보낼 수 없습니다"
            f"(decision={analysis.decision})."
        )
    if not analysis.clarification_question:
        raise ValueError("clarify 결정에 clarification_question이 없습니다.")

    # 검색을 수행하지 않았으므로 인용할 근거가 없습니다.
    return {"ai_answer": AiAnswer(answer=analysis.clarification_question, citation=[])}


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
graph_builder.add_node("answer_validator", answer_validator)
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
graph_builder.add_edge("answer_generator", "answer_validator")
graph_builder.set_finish_point("answer_validator")
graph_builder.set_finish_point("answer_directly")
graph_builder.set_finish_point("request_clarification")


# Graph Compile

graph = graph_builder.compile()




# 그래프 시각화 (Mermaid) 지원
if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())
