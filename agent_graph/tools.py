import json
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import Field, ValidationError

from .calculation import check_table_completeness, parse_numeric_cell
from .state import AgentState, Plan, PlanDraft, RetrievalResult
from .utils import (
    CalculationOperation,
    CombineOperation,
    FinishStatus,
    NumericResultTarget,
    OrderingDirection,
    RepeatedNoResultsFilterError,
    RESULT_KIND_NUMERIC_ORDERING,
    RESULT_KIND_NUMERIC_SCALAR,
    TableRowSelector,
    TableTarget,
    _RECORD_EXTRACTION_ERROR,
    _apply_calculation_operation,
    _build_calculation_invalid_result,
    _build_failed_retrieval_result,
    _classify_execution_error,
    _collect_source_references,
    _collect_table_units,
    _execute_retrieval_plan,
    _extract_cell_value,
    _merge_derived_source_references,
    _next_derived_ids,
    _resolve_plan_dependencies,
    _select_labeled_records,
    _validate_calculate_call,
    _validate_combine_call,
    _validate_finish_selection,
    correction_history_executor,
)


@tool
def retrieve_search(
    plan: PlanDraft,
    state: Annotated[AgentState, InjectedState],
    limit: int = 5,
) -> dict:
    """전달받은 단일 plan을 즉시 실행하고 retrieval 결과를 state에 추가합니다.

    args:
        plan(PlanDraft): 즉시 실행할 source, query, purpose, dependencies와
            application이 미리 생성한 scope_id
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
def retrieve_correction_history(
    disclosure_id: Annotated[
        str,
        Field(
            pattern=r"^d\d{14}$",
            description="정정이력을 조회할 최신 공시 ID(d + 숫자 14자리)",
        ),
    ],
    state: Annotated[AgentState, InjectedState],
) -> dict:
    """최신 공시 ID에 연결된 전체 CORRECTS 이력을 조회합니다.

    정정 관계의 날짜, 사유, 정정내용을 최초 정정부터 최신 정정 순서로
    RetrievalResult에 추가합니다. 일반 공시 검색이나 Qdrant는 실행하지 않습니다.
    """

    next_plan_seq = state.get("next_plan_seq", 1)
    plan_id = f"plan_{next_plan_seq}"
    try:
        retrieval_result = correction_history_executor(disclosure_id, plan_id)
    except RuntimeError as error:
        retrieval_result = RetrievalResult(
            result_id=f"retrieval:{plan_id}",
            plan_id=plan_id,
            source="neo4j",
            status=_classify_execution_error(error),
            query="retrieve correction history by disclosure_id",
            items=[],
            result_count=0,
            metadata={
                "failure_stage": "query_executor",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "requested_disclosure_id": disclosure_id,
                "plan_purpose": "공시 정정이력 확인",
            },
        )
    else:
        retrieval_result = retrieval_result.model_copy(update={
            "metadata": {
                **retrieval_result.metadata,
                "plan_purpose": "공시 정정이력 확인",
            }
        })

    return {
        "next_plan_seq": next_plan_seq + 1,
        "retrieval_results": [retrieval_result],
    }


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
