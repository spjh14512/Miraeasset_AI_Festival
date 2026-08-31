"""R_TABLE 셀 문자열을 계산 tool이 사용할 수 있는 숫자로 정규화합니다."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel


_MISSING_MARKERS = {"-", "‐", "–", "—"}
_NUMERIC_SYNTAX_CHARS = "+-.,"
# 쉼표는 반드시 3자리 grouping일 때만 숫자의 일부로 인정한다
# ("1,2,3", "12,34", ",123" 같은 잘못된 grouping은 매칭되지 않고 뒤에 남는다).
_NUMBER_PATTERN = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


class ParsedCellValue(BaseModel):
    """R_TABLE 셀 하나를 정규화한 결과입니다.

    raw는 원본 값을 그대로 보존합니다(None 입력은 None으로 유지).
    is_missing은 빈 값, "-" 같은 결측 표시, 또는 "(-)"/"( )"/"()"처럼
    괄호 안이 비어 있거나 대시뿐인 값을 가리킵니다. value는 결측이 아니고
    정상 파싱됐을 때만 채워지는 Decimal 값이고, unit은 값 뒤에 남은 단위
    문자열("%", "백만원" 등)입니다. is_missing이 False이고 value도 None이면
    숫자로 해석할 수 없는 예상 밖의 값이라는 뜻이며 is_invalid로 구분합니다.

    보수적 파싱 정책: 후행 텍스트에 숫자가 하나라도 있거나 `+-.,` 중
    하나로 시작하면 단위로 인정하지 않고 통째로 invalid 처리합니다. 날짜
    ("2025.02.03", "2025년 10월 22일"), 소수점 오탈자("1.2.3"), 지수
    표기("1e3") 등을 값+단위로 오인해 계산에 섞이는 것을 막기 위함입니다.
    괄호는 전체 문자열을 감쌀 때만 음수로 해석합니다. "(300)"과
    "(300백만원)"은 지원하지만, 괄호 밖에 단위가 붙은 "(300) 백만원"은
    모호한 형식으로 간주해 invalid로 거부합니다.
    """

    raw: str | None
    is_missing: bool = False
    value: Decimal | None = None
    unit: str | None = None

    @property
    def is_invalid(self) -> bool:
        return not self.is_missing and self.value is None


def parse_numeric_cell(raw: str | None) -> ParsedCellValue:
    """R_TABLE 셀 값을 결측/숫자/파싱 불가 중 하나로 분류합니다.

    입력 예시:
        "1,200", "12.5%", "(300)", "(300백만원)", "-", "(-)", "", None

    출력 예시:
        parse_numeric_cell("(300)")
            -> ParsedCellValue(raw="(300)", value=Decimal("-300"), unit=None)
        parse_numeric_cell("12.5%")
            -> ParsedCellValue(raw="12.5%", value=Decimal("12.5"), unit="%")
        parse_numeric_cell("(-)")
            -> ParsedCellValue(raw="(-)", is_missing=True)
        parse_numeric_cell("2025.02.03")
            -> ParsedCellValue(raw="2025.02.03")  # is_invalid == True
    """

    if raw is None:
        return ParsedCellValue(raw=None, is_missing=True)

    stripped = raw.strip()
    if not stripped:
        return ParsedCellValue(raw=raw, is_missing=True)

    is_parenthesized = stripped.startswith("(") and stripped.endswith(")")
    body = stripped[1:-1].strip() if is_parenthesized else stripped

    if not body or body in _MISSING_MARKERS:
        return ParsedCellValue(raw=raw, is_missing=True)

    match = _NUMBER_PATTERN.match(body)
    if not match:
        return ParsedCellValue(raw=raw)

    number_text = match.group(0)
    remainder = body[match.end():].strip()
    if remainder and (
        any(character.isdigit() for character in remainder)
        or remainder[0] in _NUMERIC_SYNTAX_CHARS
    ):
        return ParsedCellValue(raw=raw)

    try:
        value = Decimal(number_text.replace(",", ""))
    except InvalidOperation:
        return ParsedCellValue(raw=raw)

    if is_parenthesized:
        value = -abs(value)

    return ParsedCellValue(raw=raw, value=value, unit=remainder or None)


class TableCompletenessResult(BaseModel):
    """여러 R_TABLE item을 하나로 합쳐 집계해도 되는지 판단한 결과입니다.

    is_complete가 True면 items를 그대로 집계에 사용해도 됩니다. False면
    reason에 사람이 읽을 수 있는 실패 사유가 담기며, 이 경우 집계를
    시도하지 말고 계산을 중단해야 합니다.
    """

    is_complete: bool
    reason: str | None = None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_non_negative_int(value: Any) -> bool:
    return _is_int(value) and value >= 0


def _validate_item_shape(item: dict[str, Any], column: str) -> str | None:
    """item 하나의 구조가 집계 가능한 R_TABLE record 목록인지 확인합니다.

    문제가 없으면 None을, 있으면 사람이 읽을 수 있는 사유 문자열을
    반환합니다. RetrievalResult.items는 검증되지 않은 dict이므로 여기서
    항상 타입과 값 범위를 먼저 확인한 뒤에만 필드를 사용합니다.
    """

    if not isinstance(item, dict):
        return "item이 올바른 형식이 아닙니다."
    if item.get("type") != "r_table":
        return "R_TABLE이 아닌 item이 포함되어 있습니다."

    columns = item.get("columns")
    if not isinstance(columns, list) or column not in columns:
        return f"'{column}' 열이 없는 표가 포함되어 있습니다."

    records = item.get("records")
    if not isinstance(records, list):
        return "R_TABLE에 실제 records가 없습니다(summary만 검색된 item으로 집계할 수 없습니다)."

    available = item.get("available_record_count")
    included = item.get("included_record_count")
    omitted = item.get("omitted_record_count")
    if not (
        _is_non_negative_int(available)
        and _is_non_negative_int(included)
        and _is_non_negative_int(omitted)
    ):
        return "record 개수 필드(available/included/omitted_record_count)가 올바른 정수가 아닙니다."
    if included != len(records):
        return "included_record_count가 실제 records 개수와 다릅니다."
    if available != included + omitted:
        return "available_record_count가 included와 omitted의 합과 다릅니다."
    if omitted > 0:
        return "일부 record가 제외된 표가 포함되어 있습니다(omitted_record_count > 0)."

    return None


def _classify_chunk(item: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """item이 청킹된 chunk인지 청킹 없는 전체 표인지 판별하고 관련 필드를 검증합니다.

    반환값은 (chunk_descriptor, error)입니다. 청킹되지 않은 유효한 전체
    표라면 (None, None)을 반환합니다. chunk라면 검증된 table_id,
    chunk_index, chunk_count, row_start, row_end를 담은 dict를 반환합니다.
    구조가 잘못됐으면 error에 사유가 담깁니다. 예외를 발생시키지 않고
    항상 이 튜플로만 결과를 알립니다.

    chunk로 판정되면 row 구간의 크기(row_end - row_start + 1)가
    available_record_count와 일치하는지도 확인합니다. 일치하지 않으면
    chunk가 선언한 행 범위만큼 실제 record를 확보하지 못한 것이므로
    불완전한 것으로 처리합니다. 이 함수를 호출하기 전에
    _validate_item_shape로 available_record_count가 이미 유효한
    음이 아닌 정수임을 확인해야 합니다.
    """

    scope = item.get("scope")
    if not isinstance(scope, dict) or scope.get("kind") not in {"table", "row_group"}:
        return None, "scope 정보가 올바르지 않습니다."

    raw_metadata = item.get("metadata")
    if raw_metadata is None:
        metadata: dict[str, Any] = {}
    elif isinstance(raw_metadata, dict):
        metadata = raw_metadata
    else:
        return None, "metadata가 올바른 형식이 아닙니다."

    table_id = metadata.get("table_id")
    chunk_index = metadata.get("chunk_index")
    chunk_count = metadata.get("chunk_count")
    meta_row_start = metadata.get("row_start_index")
    meta_row_end = metadata.get("row_end_index")
    chunk_fields = (table_id, chunk_index, chunk_count, meta_row_start, meta_row_end)

    if all(field is None for field in chunk_fields):
        if scope.get("kind") != "table":
            return None, "청킹 정보가 없는데 표 전체(scope.kind == \"table\")가 아닙니다."
        return None, None

    if any(field is None for field in chunk_fields):
        return None, "청킹 metadata가 일부만 존재합니다."

    if not isinstance(table_id, str) or not table_id.strip():
        return None, "table_id가 비어 있거나 문자열이 아닙니다."
    if not _is_non_negative_int(chunk_count) or chunk_count < 2:
        return None, "chunk_count가 2 이상의 정수가 아닙니다."
    if not _is_non_negative_int(chunk_index) or chunk_index >= chunk_count:
        return None, "chunk_index가 0 이상 chunk_count 미만의 정수가 아닙니다."
    if not _is_non_negative_int(meta_row_start) or not _is_non_negative_int(meta_row_end):
        return None, "row_start_index/row_end_index가 올바른 정수가 아닙니다."
    if meta_row_start > meta_row_end:
        return None, "row_start_index가 row_end_index보다 큽니다."

    if scope.get("kind") != "row_group":
        return None, "청킹된 item인데 scope.kind가 \"row_group\"이 아닙니다."
    scope_row_start = scope.get("row_start_index")
    scope_row_end = scope.get("row_end_index")
    if not _is_non_negative_int(scope_row_start) or not _is_non_negative_int(scope_row_end):
        return None, "scope의 행 구간이 올바른 정수가 아닙니다."
    if scope_row_start != meta_row_start or scope_row_end != meta_row_end:
        return None, "scope의 행 구간이 metadata의 행 구간과 일치하지 않습니다."

    expected_record_count = meta_row_end - meta_row_start + 1
    if item.get("available_record_count") != expected_record_count:
        return None, (
            "chunk가 선언한 행 범위 크기와 실제 조회된 record 수가 다릅니다"
            f"(기대 {expected_record_count}개)."
        )

    return {
        "table_id": table_id,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "row_start": meta_row_start,
        "row_end": meta_row_end,
    }, None


def check_table_completeness(
    items: list[dict[str, Any]],
    column: str,
) -> TableCompletenessResult:
    """여러 R_TABLE item이 하나의 표를 빠짐없이 구성하는지 확인합니다.

    표 하나가 여러 Qdrant point(chunk)로 나뉘어 서로 다른 RetrievalResult에
    저장된 경우에만 items를 여러 개 넘길 수 있습니다. 이때 모든 chunk가
    같은 table_id를 가리키는지, chunk_index가 0부터 chunk_count-1까지
    빠짐없이(중복 없이) 모였는지, chunk_index 순서대로 정렬했을 때 row
    구간이 0부터 이어지고 겹치지 않는지, chunk 간 columns 구성이 같은지,
    각 chunk의 scope가 metadata와 일치하는지를 확인합니다. 청킹되지 않은
    표라면 scope가 표 전체(scope.kind == "table")인지 확인합니다. 서로
    다른 표(table_id가 다르거나, 청킹되지 않은 표를 두 개 이상 섞은 경우)를
    하나의 계산 대상으로 합치는 것은 지원하지 않습니다 — 이런 경우는
    표마다 따로 계산한 뒤 compare 전용 tool로 비교해야 합니다. summary만
    가져와 실제 records가 없는 item, Compactor가 일부 record를 제외한
    표(omitted_record_count > 0), 지정한 column이 없는 표, chunk가
    선언한 행 범위 크기(row_end - row_start + 1)와 실제 조회된
    record 수가 다른 표도 완전하지 않은 것으로 취급합니다.

    RetrievalResult.items는 검증되지 않은 dict이므로, 값의 타입이나
    범위가 잘못된 경우 예외를 던지지 않고 항상 is_complete=False와
    사유를 담아 반환합니다.

    입력 예시:
        items=[{"type": "r_table", "columns": ["구분", "금액"],
                "scope": {"kind": "table"}, "records": [...],
                "available_record_count": 1, "included_record_count": 1,
                "omitted_record_count": 0, "metadata": {}}]
        column="금액"

    출력 예시:
        check_table_completeness(items, "금액")
            -> TableCompletenessResult(is_complete=True)
    """

    if not items:
        return TableCompletenessResult(
            is_complete=False,
            reason="집계할 R_TABLE item이 없습니다.",
        )

    for item in items:
        shape_error = _validate_item_shape(item, column)
        if shape_error:
            return TableCompletenessResult(is_complete=False, reason=shape_error)

    descriptors: list[dict[str, Any] | None] = []
    for item in items:
        descriptor, error = _classify_chunk(item)
        if error:
            return TableCompletenessResult(is_complete=False, reason=error)
        descriptors.append(descriptor)

    groups: dict[str, list[dict[str, Any]]] = {}
    ungrouped_count = 0
    for descriptor in descriptors:
        if descriptor is None:
            ungrouped_count += 1
        else:
            groups.setdefault(descriptor["table_id"], []).append(descriptor)

    if len(groups) + ungrouped_count > 1:
        return TableCompletenessResult(
            is_complete=False,
            reason="서로 다른 표를 하나의 계산 대상으로 섞을 수 없습니다. 같은 표의 chunk만 함께 지정하세요.",
        )

    if not groups:
        return TableCompletenessResult(is_complete=True)

    columns_list = [item.get("columns") for item in items]
    if any(columns != columns_list[0] for columns in columns_list[1:]):
        return TableCompletenessResult(
            is_complete=False,
            reason="chunk 간 columns 구성이 다릅니다.",
        )

    ((table_id, group),) = groups.items()

    chunk_counts = {descriptor["chunk_count"] for descriptor in group}
    if len(chunk_counts) != 1:
        return TableCompletenessResult(
            is_complete=False,
            reason=f"표 {table_id}의 chunk_count 값이 chunk마다 다릅니다.",
        )
    chunk_count = chunk_counts.pop()

    chunk_indexes = [descriptor["chunk_index"] for descriptor in group]
    if len(chunk_indexes) != len(set(chunk_indexes)):
        return TableCompletenessResult(
            is_complete=False,
            reason=f"표 {table_id}의 동일 chunk가 중복으로 포함되어 있습니다.",
        )
    if set(chunk_indexes) != set(range(chunk_count)):
        missing = sorted(set(range(chunk_count)) - set(chunk_indexes))
        return TableCompletenessResult(
            is_complete=False,
            reason=f"표 {table_id}의 chunk가 일부 누락되었습니다(누락된 chunk_index: {missing}).",
        )

    ordered = sorted(group, key=lambda descriptor: descriptor["chunk_index"])
    if ordered[0]["row_start"] != 0:
        return TableCompletenessResult(
            is_complete=False,
            reason=f"표 {table_id}의 첫 chunk가 0번 행부터 시작하지 않습니다.",
        )
    for previous, current in zip(ordered, ordered[1:]):
        if current["row_start"] != previous["row_end"] + 1:
            return TableCompletenessResult(
                is_complete=False,
                reason=(
                    f"표 {table_id}의 row 구간이 chunk_index 순서와 맞지 않거나 "
                    "중복·공백이 있습니다."
                ),
            )

    return TableCompletenessResult(is_complete=True)


__all__ = [
    "ParsedCellValue",
    "parse_numeric_cell",
    "TableCompletenessResult",
    "check_table_completeness",
]
