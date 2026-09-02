from typing import Annotated
import json
import re
import statistics
from datetime import datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field, ValidationError, field_validator

from .calculation import check_table_completeness, parse_numeric_cell
from .state import AgentState, Plan, PlanDraft, RetrievalResult
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
        selected_result_ids(list[str]): Answer Generator가 사용할 RetrievalResult ID 목록.
            status와 무관하게 SUCCESS 상태의 RetrievalResult만 선택할 수 있습니다.

    return:
        dict: retrieval_status와 검증된 selected_result_ids를 포함한 state update
    """

    _validate_finish_selection(status, selected_result_ids, state)
    return {
        "retrieval_status": status,
        "retrieval_finish_reason": reason,
        "selected_result_ids": selected_result_ids,
    }


CalculationOperation = Literal["sum", "mean", "median", "max", "min", "mode", "stdev"]


class TableTarget(BaseModel):
    """calculate_table_statistic이 참조할 R_TABLE item 하나의 주소입니다.

    result_id는 대상 RetrievalResult, item_index는 그 RetrievalResult.items
    안에서 이 R_TABLE이 위치한 인덱스입니다. 하나의 표가 여러 chunk로 나뉘어
    서로 다른 result_id에 저장된 경우, 이 표를 여러 개 나열해 함께 참조합니다.
    """

    result_id: str = Field(
        ...,
        min_length=1,
        description="참조할 RetrievalResult의 result_id",
    )
    item_index: int = Field(
        ...,
        ge=0,
        strict=True,
        description="RetrievalResult.items 안에서 이 R_TABLE의 위치",
    )

    @field_validator("result_id")
    @classmethod
    def validate_result_id(cls, value: str) -> str:
        result_id = value.strip()
        if not result_id:
            raise ValueError("result_id는 비어 있을 수 없습니다.")
        return result_id


class TableRowSelector(BaseModel):
    """calculate_table_statistic이 표 전체가 아니라 특정 계정과목(행)만 계산하게 하는 선택자입니다.

    지정하지 않으면(기본값 None) 대상 표의 모든 행을 계산 대상으로
    삼습니다(기존 동작과 동일). 지정하면 label_column 열의 값이 공백만
    정규화한 뒤 정확히 일치하는 행만 골라 계산합니다. 부분 일치는
    지원하지 않습니다 — "유동부채"로 찾으면 그 글자를 포함할 뿐인
    "비유동부채" 같은 행은 매치되지 않습니다. labels를 여러 개 지정하면
    (예: ["유동부채", "비유동부채"]) 각각 정확히 한 행과 매치되어야 하며,
    그 값들에 operation(sum 등)을 적용합니다.
    """

    label_column: str = Field(
        ...,
        min_length=1,
        description="행을 식별하는 라벨 열 이름(예: '구분')",
    )
    labels: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "선택할 라벨 값 목록. 각 라벨은 대상 표 전체에서 정확히 "
            "하나의 행과만 일치해야 합니다."
        ),
    )

    @field_validator("label_column")
    @classmethod
    def validate_label_column(cls, value: str) -> str:
        label_column = value.strip()
        if not label_column:
            raise ValueError("label_column은 비어 있을 수 없습니다.")
        return label_column

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, value: list[str]) -> list[str]:
        normalized = [label.strip() for label in value]
        if any(not label for label in normalized):
            raise ValueError("labels에는 빈 문자열을 사용할 수 없습니다.")
        if len(normalized) != len(set(normalized)):
            raise ValueError("labels에는 중복 값을 사용할 수 없습니다.")
        return normalized


def _resolve_table_targets(
    targets: list[TableTarget],
    column: str,
    state: AgentState,
    row_selector: TableRowSelector | None = None,
) -> list[dict[str, Any]]:
    """TableTarget 목록을 실제 R_TABLE item dict 목록으로 해석합니다.

    다음 중 하나라도 어긋나면 ValueError를 발생시켜 Retriever LLM이 tool
    호출을 다시 만들게 합니다(state에 결과를 남기지 않음). 이 조건들은
    LLM이 애초에 존재하지 않거나 사용할 수 없는 대상을 지정한
    구조적 오류이기 때문입니다.

    - result_id가 state에 없음
    - 그 RetrievalResult의 status가 SUCCESS가 아님(finish의 SUCCESS 전용
      선택 규칙과 동일한 원칙)
    - item_index가 범위를 벗어남
    - 가리킨 item이 R_TABLE이 아님
    - 가리킨 item에 column이 없음
    - row_selector가 주어졌는데 가리킨 item에 label_column이 없음

    column/label_column의 "존재 여부"는 item마다 즉시 확인할 수 있는
    구조적 성질이라 여기서 검증합니다. 반면 특정 label이 실제로 몇 개의
    행과 일치하는지는 chunk가 전부 모여야 결론 낼 수 있으므로(다른 chunk에
    있을 수 있음) 여기서 검증하지 않고, check_table_completeness로 완전성을
    확인한 뒤 calculate_table_statistic 본문에서 처리합니다.
    """

    results_by_id = {
        result.result_id: result
        for result in state.get("retrieval_results", [])
    }
    resolved: list[dict[str, Any]] = []
    for target in targets:
        result = results_by_id.get(target.result_id)
        if result is None:
            raise ValueError(f"RetrievalResult를 찾지 못했습니다: {target.result_id}")
        if result.status != "SUCCESS":
            raise ValueError(
                f"{target.result_id}의 status가 SUCCESS가 아니어서 참조할 수 "
                f"없습니다(status={result.status})."
            )
        if target.item_index >= len(result.items):
            raise ValueError(
                f"{target.result_id}에 item_index {target.item_index}가 없습니다"
                f"(item 개수: {len(result.items)})."
            )
        item = result.items[target.item_index]
        if not isinstance(item, dict) or item.get("type") != "r_table":
            raise ValueError(
                f"{target.result_id}의 item {target.item_index}는 R_TABLE이 아닙니다."
            )
        columns = item.get("columns")
        if not isinstance(columns, list) or column not in columns:
            raise ValueError(
                f"{target.result_id}의 item {target.item_index}에 "
                f"'{column}' 열이 없습니다."
            )
        if row_selector is not None and row_selector.label_column not in columns:
            raise ValueError(
                f"{target.result_id}의 item {target.item_index}에 "
                f"label_column '{row_selector.label_column}'이 없습니다."
            )
        resolved.append(item)
    return resolved


def _next_derived_ids(state: AgentState) -> tuple[str, str, int]:
    """계산 결과에 쓸 plan_id/result_id를 next_plan_seq에서 직접 만듭니다.

    검색 계획이 아니므로 Plan.from_plan_draft()는 사용하지 않습니다.
    """

    next_plan_seq = state.get("next_plan_seq", 1)
    plan_id = f"plan_{next_plan_seq}"
    return f"derived:{plan_id}", plan_id, next_plan_seq


def _normalized_id(value: Any) -> str | None:
    """문자열이면 strip한 뒤 비어 있지 않을 때만 반환하고, 아니면 None을 반환합니다."""

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_reference(reference: Any) -> dict[str, str] | None:
    """disclosure/section/evidence id가 담긴 dict를 검증된 인용 정보로 정규화합니다.

    id는 모두 strip 후 비어 있지 않아야 유효한 것으로 인정합니다.
    evidence_id가 있는데 section_id가 없으면(Citation의 계층 규칙 위반)
    유효하지 않은 것으로 처리합니다 — _extract_citations가 이런 조합을
    조용히 버리기 때문에, 여기서 미리 걸러내지 않으면 source_references에는
    들어있지만 실제 답변에는 인용되지 않는 결과가 생깁니다. R_TABLE
    item의 metadata와 derived item의 source_references 원소 모두
    이 함수 하나로 검증합니다.
    """

    if not isinstance(reference, dict):
        return None
    disclosure_id = _normalized_id(reference.get("disclosure_id"))
    if disclosure_id is None:
        return None
    section_id = _normalized_id(reference.get("section_id"))
    evidence_id = _normalized_id(reference.get("evidence_id"))
    if evidence_id is not None and section_id is None:
        return None
    normalized: dict[str, str] = {"disclosure_id": disclosure_id}
    if section_id is not None:
        normalized["section_id"] = section_id
    if evidence_id is not None:
        normalized["evidence_id"] = evidence_id
    return normalized


def _extract_source_reference(item: dict[str, Any]) -> dict[str, str] | None:
    """R_TABLE item의 metadata에서 인용용 disclosure/section/evidence id를 뽑습니다."""

    return _normalize_reference(item.get("metadata"))


def _collect_source_references(
    items: list[dict[str, Any]],
) -> list[dict[str, str]] | None:
    """모든 item에 유효한 인용 정보가 있는지 확인하고 중복 없이 모읍니다.

    items 중 단 하나라도 유효한 인용을 뽑을 수 없으면 전체를 신뢰할 수
    없는 것으로 보고 None을 반환합니다(일부 chunk만 인용이 있다고 해서
    나머지 chunk의 값까지 인용된 것처럼 보이면 안 되기 때문입니다).
    """

    seen: set[tuple[str, str | None, str | None]] = set()
    references: list[dict[str, str]] = []
    for item in items:
        reference = _extract_source_reference(item)
        if reference is None:
            return None
        key = (
            reference["disclosure_id"],
            reference.get("section_id"),
            reference.get("evidence_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        references.append(reference)
    return references


def _merge_derived_source_references(
    items: list[dict[str, Any]],
) -> list[dict[str, str]] | None:
    """derived record item들이 이미 갖고 있는 source_references를 병합합니다.

    R_TABLE item과 달리 derived item(계산/조합 결과)은 인용 정보를
    item["metadata"]가 아니라 item["source_references"]에 형제 key로
    직접 담고 있으므로 원소 각각을 _normalize_reference로 검증합니다
    (strip, 빈 문자열 거부, evidence_id에는 section_id 필수인 계층 규칙
    모두 동일하게 적용). calculate_table_statistic의 citation_check와
    같은 원칙으로, item 중 하나라도 유효한 인용이 없으면(형식이
    비정상이어도) 전체를 None으로 거부합니다 — 입력이 정상적인 SUCCESS
    derived 결과라면 항상 통과해야 하며, 이 조건은 조작되었거나 손상된
    state에 대한 방어입니다.
    """

    seen: set[tuple[str, str | None, str | None]] = set()
    references: list[dict[str, str]] = []
    for item in items:
        raw_references = item.get("source_references")
        if not isinstance(raw_references, list) or not raw_references:
            return None
        item_references: list[dict[str, str]] = []
        for reference in raw_references:
            normalized = _normalize_reference(reference)
            if normalized is None:
                # 하나라도 무효한 reference가 섞여 있으면 이 item 전체를
                # 신뢰할 수 없는 것으로 보고 부분 인용을 허용하지 않는다.
                return None
            item_references.append(normalized)
        for normalized in item_references:
            key = (
                normalized["disclosure_id"],
                normalized.get("section_id"),
                normalized.get("evidence_id"),
            )
            if key in seen:
                continue
            seen.add(key)
            references.append(normalized)
    return references or None


def _collect_table_units(items: list[dict[str, Any]]) -> list[str]:
    """item들의 table_metadata.units에 선언된 표 단위를 중복 없이 모읍니다.

    셀 자체에 단위가 없을 때(예: 표 전체가 "단위: 백만원"이라고만 표시된
    경우) 계산 결과에 붙일 단위를 보완하는 데 사용합니다. 후보가 여러
    개면 이 열에 어떤 단위가 적용되는지 확정할 수 없으므로, 호출부에서
    후보가 정확히 하나일 때만 unit으로 채택해야 합니다.
    """

    units: list[str] = []
    seen: set[str] = set()
    for item in items:
        table_metadata = item.get("table_metadata")
        if not isinstance(table_metadata, dict):
            continue
        declared = table_metadata.get("units")
        if not isinstance(declared, list):
            continue
        for entry in declared:
            if isinstance(entry, str) and entry and entry not in seen:
                seen.add(entry)
                units.append(entry)
    return units


_RECORD_EXTRACTION_ERROR = object()


def _extract_cell_value(record: Any, column: str) -> str | None | object:
    """record에서 column 값을 안전하게 꺼냅니다.

    RetrievalResult.items는 검증되지 않은 dict이므로, record가 dict가
    아니거나 values가 없거나 dict가 아니거나 column key 자체가 없거나
    셀 값이 str/None이 아니면 예외를 던지지 않고 `_RECORD_EXTRACTION_ERROR`
    sentinel을 반환합니다. 정상적으로 존재하는 값(문자열 또는 진짜
    결측인 None)만 그대로 반환합니다.
    """

    if not isinstance(record, dict):
        return _RECORD_EXTRACTION_ERROR
    values = record.get("values")
    if not isinstance(values, dict) or column not in values:
        return _RECORD_EXTRACTION_ERROR
    cell = values[column]
    if cell is not None and not isinstance(cell, str):
        return _RECORD_EXTRACTION_ERROR
    return cell


def _apply_calculation_operation(
    operation: CalculationOperation,
    values: list[Decimal],
) -> tuple[Decimal | None, str | None]:
    """지정한 연산을 values에 적용합니다.

    계산에 성공하면 (결과값, None)을, 계산할 수 없으면 (None, 실패 사유)를
    반환합니다. mode는 동률이 여러 개면 임의로 하나를 고르지 않고
    계산 불가로 처리합니다. stdev는 표본 표준편차(n-1로 나눔)이며
    값이 2개 미만이면 계산 불가로 처리합니다.
    """

    if operation == "sum":
        return sum(values), None
    if operation == "mean":
        return sum(values) / len(values), None
    if operation == "median":
        return statistics.median(values), None
    if operation == "max":
        return max(values), None
    if operation == "min":
        return min(values), None
    if operation == "mode":
        modes = statistics.multimode(values)
        if len(modes) > 1:
            return None, "최빈값이 여러 개로 동률입니다."
        return modes[0], None
    if operation == "stdev":
        if len(values) < 2:
            return None, "표본 표준편차는 값이 2개 이상 있어야 계산할 수 있습니다."
        return statistics.stdev(values), None
    raise ValueError(f"지원하지 않는 operation입니다: {operation}")


def _build_calculation_invalid_result(
    *,
    result_id: str,
    plan_id: str,
    query: str,
    reason: str,
    failure_stage: str,
    source_result_ids: list[str],
) -> RetrievalResult:
    return RetrievalResult(
        result_id=result_id,
        plan_id=plan_id,
        source="derived",
        status="INVALID_INPUT",
        query=query,
        items=[],
        result_count=0,
        metadata={
            "failure_stage": failure_stage,
            "reason": reason,
            "source_result_ids": source_result_ids,
        },
    )


def _validate_calculate_call(
    variable_name: str,
    column: str,
    targets: list[TableTarget],
    state: AgentState,
    row_selector: TableRowSelector | None = None,
) -> list[dict[str, Any]]:
    """calculate_table_statistic의 구조적 오류를 검증하고 해석된 R_TABLE item들을 반환합니다.

    validate_retriever_tool_call(실행 전 사전 검증)과 calculate_table_statistic
    본문이 모두 이 함수를 호출합니다. 두 경로가 서로 다른 검증을 하면,
    사전 검증은 통과했는데 실제 실행에서 ValueError가 나는 경우
    retriever 노드의 재시도 루프를 벗어나 처리되지 않은 예외로 이어질
    수 있으므로 반드시 동일한 검증을 공유해야 합니다.
    """

    if not targets:
        raise ValueError("targets에는 최소 하나의 TableTarget이 필요합니다.")
    if not variable_name.strip():
        raise ValueError("variable_name은 비어 있을 수 없습니다.")
    if not column.strip():
        raise ValueError("column은 비어 있을 수 없습니다.")
    return _resolve_table_targets(targets, column, state, row_selector)


def _select_labeled_records(
    resolved_items: list[dict[str, Any]],
    row_selector: TableRowSelector,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """resolved_items 전체에서 row_selector.labels와 정확히 일치하는 record를 하나씩 찾습니다.

    각 label은 label_column 값을 공백만 정규화한 뒤(strip) 정확히
    비교합니다. 부분 일치는 하지 않습니다. 반환값은
    (선택된 record 목록, 실패 사유)입니다. label이 하나도 매치되지
    않거나 여러 record와 매치되면 자동으로 무시하거나 합치지 않고
    (None, 사유)를 반환합니다 — chunk가 전부 모인 뒤에만 호출되므로,
    여기서 매치가 없다는 것은 그 label이 이 표에 정말 없다는 뜻입니다.

    label_column 값을 읽을 수 없는(record가 dict가 아니거나 값이 없거나
    문자열이 아니거나 공백뿐인) record가 하나라도 있으면 즉시 실패로
    처리합니다. 그런 record를 조용히 건너뛰면, 그 record가 사실은
    요청한 label과 같은 행이었는지 판별할 수 없어 "정확히 한 행과
    일치"한다는 보장이 깨지기 때문입니다.

    입력 예시:
        resolved_items=[{...columns:["구분","당기금액"], records:[
            {"record_index":0,"values":{"구분":"유동부채","당기금액":"100"}},
            {"record_index":1,"values":{"구분":"비유동부채","당기금액":"200"}},
        ]}]
        row_selector=TableRowSelector(label_column="구분", labels=["유동부채"])

    출력 예시:
        ([{"record_index":0,"values":{"구분":"유동부채","당기금액":"100"}}], None)
    """

    matches: dict[str, list[dict[str, Any]]] = {label: [] for label in row_selector.labels}
    for item in resolved_items:
        for record in item.get("records", []):
            cell = _extract_cell_value(record, row_selector.label_column)
            if not isinstance(cell, str) or not cell.strip():
                return None, (
                    f"label_column '{row_selector.label_column}' 값을 읽을 수 "
                    "없는 행이 있어 라벨을 안전하게 선택할 수 없습니다."
                )
            normalized = cell.strip()
            if normalized in matches:
                matches[normalized].append(record)

    missing = [label for label, records in matches.items() if not records]
    if missing:
        return None, (
            f"label_column '{row_selector.label_column}'에서 다음 label을 "
            f"찾지 못했습니다: {missing}"
        )
    duplicated = [label for label, records in matches.items() if len(records) > 1]
    if duplicated:
        return None, (
            f"label_column '{row_selector.label_column}'에서 다음 label이 "
            f"여러 행과 일치해 자동으로 합산하지 않습니다: {duplicated}"
        )
    return [matches[label][0] for label in row_selector.labels], None


@tool
def calculate_table_statistic(
    variable_name: str,
    operation: CalculationOperation,
    column: str,
    targets: list[TableTarget],
    state: Annotated[AgentState, InjectedState],
    row_selector: TableRowSelector | None = None,
) -> dict:
    """검색된 R_TABLE의 한 열에 통계 연산을 적용하고 결과를 새 RetrievalResult로 state에 저장합니다.

    R_TABLE에만 사용할 수 있습니다. LLM은 숫자를 직접 계산하거나
    전달하지 않습니다 — 어떤 열에 어떤 연산을 적용할지만 지정하면, 실제
    값 추출과 연산은 이 tool이 state에서 직접 수행합니다.

    args:
        variable_name(str): 계산 결과에 붙일 사람이 읽을 이름.
            예: "삼성전자 2024년 매출액 합계". 빈 문자열은 허용하지 않습니다.
        operation(CalculationOperation): 'sum', 'mean', 'median', 'max',
            'min', 'mode', 'stdev'(표본 표준편차, 값 2개 이상 필요) 중 하나
        column(str): 집계할 R_TABLE의 열 이름. 대상 표의 columns에
            정확히 존재해야 합니다. 빈 문자열은 허용하지 않습니다.
        targets(list[TableTarget]): 집계 대상 R_TABLE의 위치 목록(최소
            1개). result_id와 item_index로 정확히 존재하는 R_TABLE
            item을 가리켜야 합니다. 하나의 표가 여러 chunk로 나뉘어
            서로 다른 result_id에 저장된 경우에만 여러 개를 나열하세요.
            서로 다른 표(예: 서로 다른 기업·기간의 표)를 섞지 마세요 —
            표마다 각각 calculate_table_statistic을 호출한 뒤 비교
            전용 tool로 비교하세요.
        state(AgentState): InjectedState로 주입되며 LLM에는 보이지
            않습니다. tool 내부에서만 state["retrieval_results"]를
            조회하는 데 사용합니다.
        row_selector(TableRowSelector | None): 지정하면 표 전체가 아니라
            특정 계정과목(행)만 계산 대상으로 삼습니다. label_column(예:
            "구분") 열의 값이 공백만 정규화한 뒤 정확히 일치하는 행만
            골라 그 값들에 operation을 적용합니다. 부분 일치는 하지
            않습니다("유동부채"로 찾아도 "비유동부채"는 매치되지
            않습니다). labels를 여러 개 지정하면(예: 총부채를 구하려고
            ["유동부채", "비유동부채"]) 각각 정확히 한 행과 매치되어야
            합니다. label_column 값을 읽을 수 없는 행이 하나라도 있으면
            "정확히 한 행과 일치"를 보장할 수 없으므로 계산을 중단합니다.
            선택된 행 중 하나라도 값이 결측이면(전체 열 집계와 달리)
            조용히 제외하지 않고 계산을 중단합니다 — 사용자가 명시한
            행을 빠뜨리면 안 되기 때문입니다. 생략하면(기본값 None)
            기존처럼 표의 모든 행을 계산 대상으로 삼고, 결측 행은
            제외한 채 계산합니다.

    return:
        dict: next_plan_seq와 계산 결과가 담긴 새 RetrievalResult 하나를
            포함한 state update.
            - 계산에 성공하면 status="SUCCESS"이고 items에
              {"type": "record", "fields": {변수명/연산/값/단위/사용한
              값 개수, row_selector 사용 시 row_selection:{label_column,
              labels}}, "source_references": [원본 인용 정보]}가
              담깁니다. 셀 자체에 단위가 없으면 표 전체 단위(정확히
              하나로 확정될 때만)로 보완합니다. 대상 item 중 하나라도
              유효한 인용 정보(disclosure_id, 그리고 evidence_id가
              있다면 그에 대응하는 section_id)가 없으면 성공으로
              처리하지 않습니다.
            - 표가 불완전하거나(chunk 누락, 일부 record 제외 등),
              row_selector의 label이 하나도 없거나 여러 행과 일치하거나
              label_column을 읽을 수 없는 행이 있거나, row_selector로
              선택한 행 중 결측값이 있거나, 값을 계산할 수 없거나
              (비정상 값 포함, 단위 혼재, 표 단위 후보가 여러 개라
              확정할 수 없음, 최빈값 동률, 모든 값이 결측 등), record
              구조 자체가 잘못됐거나, 인용할 원본 정보가 없으면
              status="INVALID_INPUT"이고 items는 빈 목록입니다.
              metadata.failure_stage로 원인 단계
              (completeness_check/citation_check/row_selection/
              record_shape/numeric_parsing/unit_check/calculation)를
              구분합니다.
            - result_id가 존재하지 않거나, 그 RetrievalResult의
              status가 SUCCESS가 아니거나, item_index가 존재하지
              않거나, 가리킨 item이 R_TABLE이 아니거나, column이나
              row_selector.label_column이 없거나, targets가 비어
              있거나, variable_name/column이 비어 있으면 ValueError를
              발생시킵니다(state에 결과를 남기지 않고 tool 호출 자체를
              다시 만들어야 합니다). label이 실제로 몇 개와 일치하는지는
              구조적 오류가 아니므로 여기 포함되지 않습니다(위 INVALID_INPUT
              참고).
    """

    resolved_items = _validate_calculate_call(
        variable_name, column, targets, state, row_selector,
    )
    result_id, plan_id, next_plan_seq = _next_derived_ids(state)
    source_result_ids = [target.result_id for target in targets]
    request_query = json.dumps(
        {
            "operation": operation,
            "column": column,
            "targets": [target.model_dump() for target in targets],
            "row_selector": row_selector.model_dump() if row_selector else None,
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    def invalid(reason: str, failure_stage: str) -> dict:
        result = _build_calculation_invalid_result(
            result_id=result_id,
            plan_id=plan_id,
            query=request_query,
            reason=reason,
            failure_stage=failure_stage,
            source_result_ids=source_result_ids,
        )
        return {"next_plan_seq": next_plan_seq + 1, "retrieval_results": [result]}

    completeness = check_table_completeness(resolved_items, column)
    if not completeness.is_complete:
        return invalid(
            completeness.reason or "표가 완전하지 않습니다.",
            "completeness_check",
        )

    source_references = _collect_source_references(resolved_items)
    if not source_references:
        return invalid(
            "계산 결과에 연결할 원본 공시 인용 정보가 없습니다.",
            "citation_check",
        )

    if row_selector is not None:
        records_to_scan, selection_error = _select_labeled_records(
            resolved_items, row_selector,
        )
        if selection_error is not None:
            return invalid(selection_error, "row_selection")
    else:
        records_to_scan = [
            record
            for item in resolved_items
            for record in item.get("records", [])
        ]

    parsed_values: list[Decimal] = []
    units: set[str] = set()
    for record in records_to_scan:
        cell = _extract_cell_value(record, column)
        if cell is _RECORD_EXTRACTION_ERROR:
            return invalid(
                f"'{column}' 값을 가진 record 구조가 올바르지 않습니다.",
                "record_shape",
            )
        parsed = parse_numeric_cell(cell)
        if parsed.is_missing:
            if row_selector is not None:
                return invalid(
                    f"'{column}' 열에서 선택한 행 중 하나가 결측값입니다: {parsed.raw!r}",
                    "numeric_parsing",
                )
            continue
        if parsed.is_invalid:
            return invalid(
                f"'{column}' 열에 숫자로 해석할 수 없는 값이 있습니다: {parsed.raw!r}",
                "numeric_parsing",
            )
        parsed_values.append(parsed.value)
        if parsed.unit is not None:
            units.add(parsed.unit)

    if len(units) > 1:
        return invalid(
            f"'{column}' 열에 서로 다른 단위가 섞여 있습니다: {sorted(units)}",
            "unit_check",
        )

    table_units = _collect_table_units(resolved_items)
    unit = next(iter(units)) if units else None
    if unit is None:
        if len(table_units) > 1:
            return invalid(
                f"'{column}' 열의 단위를 표에서 하나로 확정할 수 없습니다: {table_units}",
                "unit_check",
            )
        if len(table_units) == 1:
            unit = table_units[0]

    if not parsed_values:
        return invalid(
            f"'{column}' 열에 계산할 수 있는 값이 없습니다(모든 값이 결측).",
            "numeric_parsing",
        )

    value, operation_error = _apply_calculation_operation(operation, parsed_values)
    if operation_error is not None:
        return invalid(operation_error, "calculation")

    fields: dict[str, Any] = {
        "variable_name": variable_name,
        "operation": operation,
        "value": str(value),
        "unit": unit,
        "input_count": len(parsed_values),
    }
    if row_selector is not None:
        fields["row_selection"] = {
            "label_column": row_selector.label_column,
            "labels": row_selector.labels,
        }
    result_item = {
        "type": "record",
        "fields": fields,
        "source_references": source_references,
    }
    result = RetrievalResult(
        result_id=result_id,
        plan_id=plan_id,
        source="derived",
        status="SUCCESS",
        query=request_query,
        items=[result_item],
        result_count=1,
        metadata={
            "source_result_ids": source_result_ids,
            "result_kind": RESULT_KIND_NUMERIC_SCALAR,
        },
    )
    return {"next_plan_seq": next_plan_seq + 1, "retrieval_results": [result]}


CombineOperation = Literal[
    "sum", "mean", "difference", "ratio", "percent_ratio", "percent_change",
    "cagr", "ordering",
]
OrderingDirection = Literal["ascending", "descending"]

# result_kind는 derived RetrievalResult가 어떤 모양의 값을 담고 있는지
# 나타내는 구분값입니다. combine_numeric_results는 numeric_scalar만
# 입력으로 받아, numeric_ordering(순위 목록) 결과를 실수로 다시 더하거나
# 나누는 것을 막습니다.
RESULT_KIND_NUMERIC_SCALAR = "numeric_scalar"
RESULT_KIND_NUMERIC_ORDERING = "numeric_ordering"

# operation별 허용되는 targets 개수 범위. 위쪽 경계가 None이면 상한 없음.
_COMBINE_OPERATION_INPUT_COUNTS: dict[CombineOperation, tuple[int, int | None]] = {
    "sum": (2, None),
    "mean": (2, None),
    "difference": (2, 2),
    "ratio": (2, 2),
    "percent_ratio": (2, 2),
    "percent_change": (2, 2),
    "cagr": (2, 2),
    "ordering": (2, None),
}


class NumericResultTarget(BaseModel):
    """combine_numeric_results가 참조할 numeric_scalar 결과 하나의 주소입니다.

    result_id는 대상 derived RetrievalResult, item_index는 그 안에서
    이 숫자 결과가 위치한 인덱스입니다. numeric_scalar 결과는 현재 항상
    item을 하나만 가지므로 보통 0을 사용합니다.
    """

    result_id: str = Field(
        ...,
        min_length=1,
        description="참조할 derived RetrievalResult의 result_id",
    )
    item_index: int = Field(
        default=0,
        ge=0,
        strict=True,
        description="RetrievalResult.items 안에서 이 숫자 결과의 위치(보통 0)",
    )

    @field_validator("result_id")
    @classmethod
    def validate_result_id(cls, value: str) -> str:
        result_id = value.strip()
        if not result_id:
            raise ValueError("result_id는 비어 있을 수 없습니다.")
        return result_id


def _resolve_combine_operand(
    target: NumericResultTarget,
    state: AgentState,
) -> tuple[Decimal, str | None, dict[str, Any]]:
    """combine_numeric_results가 참조할 하나의 numeric_scalar 결과를 해석합니다.

    calculate_table_statistic이나 combine_numeric_results 자신이 만든
    result_kind="numeric_scalar"인 SUCCESS 상태의 derived RetrievalResult만
    허용합니다. numeric_ordering(순위 목록) 결과는 단일 숫자가 아니므로
    거부합니다. 존재하지 않거나 이 조건에 맞지 않으면 ValueError를
    발생시켜 Retriever LLM이 다른 대상을 고르게 합니다.

    반환값은 (값, 단위, 원본 item)입니다. 원본 item은 source_references를
    다시 모으는 데 사용합니다.
    """

    results_by_id = {
        result.result_id: result
        for result in state.get("retrieval_results", [])
    }
    result = results_by_id.get(target.result_id)
    if result is None:
        raise ValueError(f"RetrievalResult를 찾지 못했습니다: {target.result_id}")
    if result.source != "derived":
        raise ValueError(
            f"{target.result_id}는 derived 계산 결과가 아니어서 참조할 수 없습니다"
            f"(source={result.source})."
        )
    if result.status != "SUCCESS":
        raise ValueError(
            f"{target.result_id}의 status가 SUCCESS가 아니어서 참조할 수 없습니다"
            f"(status={result.status})."
        )
    if result.metadata.get("result_kind") != RESULT_KIND_NUMERIC_SCALAR:
        raise ValueError(
            f"{target.result_id}는 숫자 scalar 결과가 아니어서 참조할 수 없습니다"
            f"(result_kind={result.metadata.get('result_kind')!r})."
        )
    if target.item_index >= len(result.items):
        raise ValueError(
            f"{target.result_id}에 item_index {target.item_index}가 없습니다"
            f"(item 개수: {len(result.items)})."
        )

    item = result.items[target.item_index]
    if not isinstance(item, dict) or item.get("type") != "record":
        raise ValueError(
            f"{target.result_id}의 item {target.item_index}는 계산 결과 형식이 아닙니다."
        )
    fields = item.get("fields")
    if not isinstance(fields, dict):
        raise ValueError(f"{target.result_id}에 fields가 없습니다.")
    value_text = fields.get("value")
    if not isinstance(value_text, str):
        raise ValueError(f"{target.result_id}의 value가 문자열이 아닙니다.")
    try:
        value = Decimal(value_text)
    except InvalidOperation as error:
        raise ValueError(f"{target.result_id}의 value를 숫자로 해석할 수 없습니다.") from error

    unit = fields.get("unit")
    if unit is not None and not isinstance(unit, str):
        raise ValueError(f"{target.result_id}의 unit이 문자열이 아닙니다.")
    return value, unit, item


def _validate_combine_call(
    variable_name: str,
    operation: CombineOperation,
    targets: list[NumericResultTarget],
    state: AgentState,
    periods: int | None = None,
) -> list[tuple[Decimal, str | None, dict[str, Any]]]:
    """combine_numeric_results의 구조적 오류를 검증하고 해석된 operand들을 반환합니다.

    validate_retriever_tool_call(실행 전 사전 검증)과 combine_numeric_results
    본문이 모두 이 함수를 호출합니다. _validate_calculate_call과 같은 이유로
    두 경로는 반드시 같은 검증을 공유해야 합니다.
    """

    if not variable_name.strip():
        raise ValueError("variable_name은 비어 있을 수 없습니다.")
    target_keys = [(target.result_id, target.item_index) for target in targets]
    if len(target_keys) != len(set(target_keys)):
        raise ValueError("targets에는 중복된 (result_id, item_index)를 사용할 수 없습니다.")

    min_count, max_count = _COMBINE_OPERATION_INPUT_COUNTS[operation]
    if len(targets) < min_count or (max_count is not None and len(targets) > max_count):
        expected = (
            f"정확히 {min_count}개" if min_count == max_count else f"최소 {min_count}개"
        )
        raise ValueError(
            f"{operation}에는 targets가 {expected} 필요합니다"
            f"(전달된 개수: {len(targets)})."
        )

    if operation == "cagr":
        if not isinstance(periods, int) or isinstance(periods, bool) or periods < 1:
            raise ValueError("cagr에는 1 이상의 정수 periods가 필요합니다.")

    return [_resolve_combine_operand(target, state) for target in targets]


@tool
def combine_numeric_results(
    variable_name: str,
    operation: CombineOperation,
    targets: list[NumericResultTarget],
    state: Annotated[AgentState, InjectedState],
    direction: OrderingDirection = "ascending",
    periods: Annotated[int | None, Field(strict=True)] = None,
) -> dict:
    """calculate_table_statistic 등이 만든 numeric_scalar 결과 여러 개를 조합합니다.

    서로 다른 표에서 각각 계산한 숫자를 합치거나 비교할 때 사용합니다.
    서로 다른 표를 calculate_table_statistic 하나에 직접 섞을 수 없으므로,
    표마다 각각 계산한 뒤 이 tool로 조합하세요. LLM은 어떤 결과들을
    어떤 연산으로 조합할지만 지정하고, 실제 산술은 이 tool이 수행합니다.

    args:
        variable_name(str): 조합 결과에 붙일 사람이 읽을 이름. 빈
            문자열은 허용하지 않습니다.
        operation(CombineOperation):
            - 'sum': targets 전체를 더함(2개 이상)
            - 'mean': targets 전체의 평균(2개 이상). 기초·기말 평균
              같은 지표에 사용
            - 'difference': 정확히 2개, targets[0] - targets[1]
            - 'ratio': 정확히 2개, targets[0] / targets[1]. 배수로
              표현하는 지표(PER, 회전율 등)에 사용
            - 'percent_ratio': 정확히 2개, targets[0] / targets[1] * 100.
              ROI·ROA·ROE처럼 백분율로 표현하는 비율 지표에 사용
              (예: ROA = 당기순이익 / 자산총계 * 100). 기준값(분자)이
              targets[0], 나누는 값(분모)이 targets[1]입니다.
            - 'percent_change': 정확히 2개,
              (targets[1] - targets[0]) / targets[0] * 100.
              기준은 targets[0](이전 값)입니다. percent_ratio와
              혼동하지 마세요 — percent_change는 "증감률"(예: 전년
              대비 몇 % 늘었는지), percent_ratio는 서로 다른 두
              항목의 "비율"(예: 순이익이 자산의 몇 %인지)입니다.
            - 'cagr': 정확히 2개, (targets[1] / targets[0]) ** (1 / periods)
              - 1을 백분율로 반환. targets[0]이 시작 값, targets[1]이
              끝 값이며 periods(기간 수, 정수)를 반드시 지정해야
              합니다. 시작 값과 끝 값이 모두 양수가 아니면(0, 음수,
              부호 전환 포함) 계산할 수 없습니다.
            - 'ordering': targets를 값 기준으로 정렬해 순위를 매김
              (2개 이상). 정렬 방향은 direction으로 지정합니다.
        targets(list[NumericResultTarget]): 조합할 numeric_scalar
            결과의 위치 목록. calculate_table_statistic 또는
            combine_numeric_results가 만든 result_kind="numeric_scalar",
            status="SUCCESS" 결과만 참조할 수 있습니다(numeric_ordering
            결과는 참조 불가). difference/ratio/percent_ratio/
            percent_change/cagr에서는 순서가 결과에 직접 영향을 주므로
            정확히 지정하세요. 동일한 (result_id, item_index) 조합을
            중복해서 넣을 수 없습니다(이중 계산 방지).
        state(AgentState): InjectedState로 주입되며 LLM에는 보이지
            않습니다. tool 내부에서만 state["retrieval_results"]를
            조회하는 데 사용합니다.
        direction(OrderingDirection): operation='ordering'일 때만
            사용하는 정렬 방향('ascending' 또는 'descending', 기본
            'ascending'). 다른 operation에서는 무시됩니다. 값이 같은
            대상은 입력 순서를 그대로 유지합니다(안정 정렬).
        periods(int | None): operation='cagr'일 때만 사용하는 기간 수
            (예: 2020년부터 2024년까지면 4). 1 이상의 정수여야 합니다.
            다른 operation에서는 무시됩니다.

    return:
        dict: next_plan_seq와 조합 결과가 담긴 새 RetrievalResult 하나를
            포함한 state update.
            - sum/mean/difference/ratio/percent_ratio/percent_change/
              cagr이 성공하면 status="SUCCESS", metadata.result_kind=
              "numeric_scalar"이고 items[0].fields에 변수명/연산/값/
              단위/사용한 값 개수가 담깁니다. sum/mean/difference/ratio/
              percent_ratio/percent_change는 입력 단위가 전부 완전히
              같을 때만(모두 None이거나 모두 같은 문자열) 계산하며,
              calculate_table_statistic의 셀 단위 정책과 달리 단위
              없음과 명시된 단위를 같다고 보지 않습니다. sum·mean·
              difference는 그 공통 단위를 그대로 사용합니다. ratio는
              단위를 None으로, percent_ratio·percent_change·cagr는
              "%"로 반환합니다.
            - ordering이 성공하면 status="SUCCESS",
              metadata.result_kind="numeric_ordering"이고
              items[0].fields는 단일 value 대신 {"direction":...,
              "unit":..., "ordered_results": [{"rank":1, "result_id":...,
              "variable_name":..., "value":...}, ...]} 형태입니다.
              rank는 정렬 순서를 나타내는 1부터 시작하는 순번입니다.
            - 모든 경우 입력 대상 전체의 source_references를
              검증·병합한 결과가 함께 담기며, 하나라도 유효한 인용이
              없으면 성공으로 처리하지 않습니다.
            - 입력 단위가 서로 다르거나(비교 가능한 단위끼리만 허용),
              0으로 나누게 되거나, cagr에서 두 값의 부호가 다르거나,
              인용할 원본 정보가 없으면 status="INVALID_INPUT"이고
              items는 빈 목록입니다. metadata.failure_stage로 원인을
              구분합니다.
            - result_id가 존재하지 않거나, numeric_scalar
              result_kind의 SUCCESS 결과가 아니거나, item_index가
              존재하지 않거나, 계산 결과 형식이 아니거나, operation에
              필요한 개수의 targets가 아니거나, cagr인데 1 이상의
              정수 periods가 없거나, 중복 target이 있거나,
              variable_name이 비어 있으면 ValueError를 발생시킵니다.
    """

    operands = _validate_combine_call(variable_name, operation, targets, state, periods)
    values = [operand[0] for operand in operands]
    units = [operand[1] for operand in operands]
    items = [operand[2] for operand in operands]
    source_result_ids = [target.result_id for target in targets]

    result_id, plan_id, next_plan_seq = _next_derived_ids(state)
    request_query = json.dumps(
        {
            "operation": operation,
            "direction": direction if operation == "ordering" else None,
            "periods": periods if operation == "cagr" else None,
            "targets": [target.model_dump() for target in targets],
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    def invalid(reason: str, failure_stage: str) -> dict:
        result = _build_calculation_invalid_result(
            result_id=result_id,
            plan_id=plan_id,
            query=request_query,
            reason=reason,
            failure_stage=failure_stage,
            source_result_ids=source_result_ids,
        )
        return {"next_plan_seq": next_plan_seq + 1, "retrieval_results": [result]}

    source_references = _merge_derived_source_references(items)
    if not source_references:
        return invalid("조합 결과에 연결할 원본 공시 인용 정보가 없습니다.", "citation_check")

    distinct_units = set(units)
    if len(distinct_units) > 1:
        unit_labels = sorted(
            "(없음)" if unit is None else unit for unit in distinct_units
        )
        return invalid(
            f"입력들의 단위가 서로 달라 조합할 수 없습니다: {unit_labels}",
            "unit_check",
        )
    input_unit = units[0]

    if operation == "sum":
        value = sum(values)
        unit = input_unit
    elif operation == "mean":
        value = sum(values) / len(values)
        unit = input_unit
    elif operation == "difference":
        value = values[0] - values[1]
        unit = input_unit
    elif operation == "ratio":
        if values[1] == 0:
            return invalid("두 번째 값이 0이어서 나눌 수 없습니다.", "calculation")
        value = values[0] / values[1]
        unit = None
    elif operation == "percent_ratio":
        if values[1] == 0:
            return invalid("두 번째 값이 0이어서 나눌 수 없습니다.", "calculation")
        value = values[0] / values[1] * 100
        unit = "%"
    elif operation == "percent_change":
        if values[0] == 0:
            return invalid("기준값(첫 번째 값)이 0이어서 증감률을 계산할 수 없습니다.", "calculation")
        value = (values[1] - values[0]) / values[0] * 100
        unit = "%"
    elif operation == "cagr":
        if values[0] <= 0 or values[1] <= 0:
            return invalid(
                "CAGR은 시작 값과 끝 값이 모두 양수여야 계산할 수 있습니다.",
                "calculation",
            )
        growth_ratio = values[1] / values[0]
        try:
            value = (growth_ratio ** (Decimal(1) / periods) - 1) * 100
        except InvalidOperation:
            return invalid("CAGR 계산 중 값을 확정할 수 없습니다.", "calculation")
        unit = "%"
    elif operation == "ordering":
        ranked = sorted(
            zip(source_result_ids, values, items),
            key=lambda entry: entry[1],
            reverse=(direction == "descending"),
        )
        fields = {
            "variable_name": variable_name,
            "operation": operation,
            "direction": direction,
            "unit": input_unit,
            "input_count": len(values),
            "ordered_results": [
                {
                    "rank": rank,
                    "result_id": ranked_result_id,
                    "variable_name": (ranked_item.get("fields") or {}).get("variable_name"),
                    "value": str(ranked_value),
                }
                for rank, (ranked_result_id, ranked_value, ranked_item) in enumerate(ranked, start=1)
            ],
        }
        result_item = {
            "type": "record",
            "fields": fields,
            "source_references": source_references,
        }
        result = RetrievalResult(
            result_id=result_id,
            plan_id=plan_id,
            source="derived",
            status="SUCCESS",
            query=request_query,
            items=[result_item],
            result_count=1,
            metadata={
                "source_result_ids": source_result_ids,
                "result_kind": RESULT_KIND_NUMERIC_ORDERING,
            },
        )
        return {"next_plan_seq": next_plan_seq + 1, "retrieval_results": [result]}
    else:
        raise ValueError(f"지원하지 않는 operation입니다: {operation}")

    result_item = {
        "type": "record",
        "fields": {
            "variable_name": variable_name,
            "operation": operation,
            "value": str(value),
            "unit": unit,
            "input_count": len(values),
        },
        "source_references": source_references,
    }
    result = RetrievalResult(
        result_id=result_id,
        plan_id=plan_id,
        source="derived",
        status="SUCCESS",
        query=request_query,
        items=[result_item],
        result_count=1,
        metadata={
            "source_result_ids": source_result_ids,
            "result_kind": RESULT_KIND_NUMERIC_SCALAR,
        },
    )
    return {"next_plan_seq": next_plan_seq + 1, "retrieval_results": [result]}

