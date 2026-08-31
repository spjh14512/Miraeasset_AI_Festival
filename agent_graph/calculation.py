"""R_TABLE 셀 문자열을 계산 tool이 사용할 수 있는 숫자로 정규화합니다."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

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


__all__ = ["ParsedCellValue", "parse_numeric_cell"]
