import os
from typing import Any

from dotenv import load_dotenv

from langgraph.graph import StateGraph

if __package__:
    from . import system_prompts as sp
    from .state import AgentState, QuestionAnalysis
else:
    import system_prompts as sp
    from state import AgentState, QuestionAnalysis


graph_builder = StateGraph(AgentState)
load_dotenv()


def _build_llm() -> Any:
    """환경 설정으로 question planner용 ChatClovaX를 생성한다."""

    api_key = os.getenv("CLOVASTUDIO_API_KEY")
    if not api_key:
        raise RuntimeError("CLOVASTUDIO_API_KEY 환경변수가 필요합니다.")

    # Import가 무거우므로 실제 LLM 호출이 필요한 시점까지 지연한다.
    from langchain_naver import ChatClovaX

    return ChatClovaX(
        model=os.getenv("CLOVAX_MODEL_NAME", "HCX-005"),
        api_key=api_key,
        max_tokens=1024,
        temperature=0.0,
        top_p=0.8,
        top_k=0,
        repetition_penalty=1.0,
        disabled_params={"parallel_tool_calls": None},
    )


def planner(
    state: AgentState,
    *,
    llm: Any | None = None,
) -> dict[str, QuestionAnalysis]:
    """
    사용자의 질문을 분석하고 routing 결정과 query plan을 생성한다.
    """

    question = state["question_text"].strip()
    if not question:
        raise ValueError("question_text는 비어 있을 수 없습니다.")

    planner = (llm or _build_llm()).with_structured_output(
        QuestionAnalysis,
        method="function_calling",
    )
    result = planner.invoke(
        [
            ("system", sp.PLANNER_SYSTEM_PROMPT),
            ("human", question),
        ]
    )

    print("질문 분석 결과:\n", result, "\n" + "-" * 80)

    analysis = (
        result
        if isinstance(result, QuestionAnalysis)
        else QuestionAnalysis.model_validate(result)
    )
    return {"question_analysis": analysis}


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


def retriever(state: AgentState) -> dict:
    print("retriever 노드 호출")

    plans = state.get("question_analysis").query_plans
    if not plans:
        raise ValueError("query plan이 생성되지 않았습니다.")

    

    return {"answer": "임시 답변", "citations": ["임시 인용 정보"]}

def answer_directly(state: AgentState) -> dict:
    print("answer_directly 노드 호출")
    return {"answer": "임시 답변", "citations": ["임시 인용 정보"]}

def request_clarification(state: AgentState) -> dict:
    print("request_clarification 노드 호출")
    return {"answer": " 임시 답변", "citations": ["임시 인용 정보"]}


# Node

graph_builder.add_node("planner", planner)
graph_builder.add_node("retriever", retriever)
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

# 임시 END 노드 설정
graph_builder.set_finish_point("retriever")
graph_builder.set_finish_point("answer_directly")
graph_builder.set_finish_point("request_clarification")


# Graph Compile

graph = graph_builder.compile()




# 그래프 시각화 (Mermaid) 지원
if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())
