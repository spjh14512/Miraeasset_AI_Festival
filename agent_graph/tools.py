from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import ValidationError

from .state import AgentState, Plan, PlanDraft
from .utils import (
    FinishStatus,
    RepeatedNoResultsFilterError,
    _build_failed_retrieval_result,
    _classify_execution_error,
    _execute_retrieval_plan,
    _resolve_plan_dependencies,
    _validate_finish_selection,
)


@tool
def retrieve_search(
    plan: PlanDraft,
    state: Annotated[AgentState, InjectedState],
    limit: int = 5,
) -> dict:
    """전달받은 단일 plan을 즉시 실행하고 retrieval 결과를 state에 추가합니다.

    args:
        plan(PlanDraft): 즉시 실행할 source, query, purpose, dependencies
        limit(int): Qdrant에서 조회할 누적 상위 point 수. 최초 검색은 5, 추가 검색은 5 단위로 늘립니다. Neo4j 검색에는 적용하지 않습니다.

    return:
        dict: 자동 할당된 plan_id와 검색 결과를 포함한 state update
    """

    next_plan_seq = state.get("next_plan_seq", 1)
    executable_plan = Plan.from_plan_draft(plan, next_plan_seq)
    dependencies = _resolve_plan_dependencies(executable_plan, state)

    try:
        retrieval_result, previous_point_ids, new_point_ids = _execute_retrieval_plan(
            executable_plan,
            state=state,
            dependencies=dependencies,
            limit=limit,
        )
    except (ValidationError, RepeatedNoResultsFilterError) as error:
        retrieval_result = _build_failed_retrieval_result(
            executable_plan,
            status="INVALID_QUERY",
            error=error,
            stage="query_builder",
        )
        previous_point_ids, new_point_ids = [], []
    except RuntimeError as error:
        retrieval_result = _build_failed_retrieval_result(
            executable_plan,
            status=_classify_execution_error(error),
            error=error,
            stage="query_executor",
        )
        previous_point_ids = (
            list(dict.fromkeys(state.get("retrieved_qdrant_point_ids", [])))
            if executable_plan.source == "qdrant"
            else []
        )
        new_point_ids = []

    retrieval_result = retrieval_result.model_copy(update={
        "metadata": {
            **retrieval_result.metadata,
            "plan_purpose": executable_plan.purpose,
        }
    })

    update = {
        "next_plan_seq": next_plan_seq + 1,
        "retrieval_results": [retrieval_result],
    }
    if executable_plan.source == "qdrant":
        update["retrieved_qdrant_point_ids"] = previous_point_ids + new_point_ids
    return update


@tool
def finish(
    status: FinishStatus,
    reason: str,
    selected_result_ids: list[str],
    state: Annotated[AgentState, InjectedState],
) -> dict:
    """더 이상의 retrieval을 멈추고 답변을 생성합니다. 종료 원인은 다음 두 가지 중 하나입니다.
        - COMPLETE: 지금까지의 retrieval을 통해 사용자의 질문에 답변하기 위해 필요한 충분한 정보를 얻었음.
        - INSUFFICIENT: 충분한 retrieval을 수행했으나, 사용자의 질문에 답변하기 위한 신뢰도 있는 정보를 얻지 못함. 더 이상의 retrieval은 무의미하다 판단.

    args:
        status(FinishStatus): 'COMPLETE' 또는 'INSUFFICIENT'
        reason(str): status를 그렇게 판단한 이유를 설명하는 간략한 한국어 문장. INSUFFICIENT라면 추가로 유효한 검색 전략이 없는 이유를 설명하며, 추가 검색이 필요하다고 작성하지 않습니다.
        selected_result_ids(list[str]): Answer Generator가 사용할 RetrievalResult ID 목록

    return:
        dict: retrieval_status와 검증된 selected_result_ids를 포함한 state update
    """

    _validate_finish_selection(status, selected_result_ids, state)
    return {
        "retrieval_status": status,
        "retrieval_finish_reason": reason,
        "selected_result_ids": selected_result_ids,
    }
