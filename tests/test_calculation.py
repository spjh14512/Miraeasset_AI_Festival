from __future__ import annotations

from decimal import Decimal

from agent_graph.calculation import parse_numeric_cell


def test_parses_comma_separated_integer():
    result = parse_numeric_cell("1,200")

    assert result.value == Decimal("1200")
    assert result.unit is None
    assert not result.is_missing
    assert not result.is_invalid


def test_parses_plain_integer_without_comma():
    result = parse_numeric_cell("1200")

    assert result.value == Decimal("1200")


def test_parses_multi_group_comma_integer():
    result = parse_numeric_cell("1,234,567")

    assert result.value == Decimal("1234567")


def test_parses_decimal_value():
    result = parse_numeric_cell("12.5")

    assert result.value == Decimal("12.5")
    assert result.unit is None


def test_parses_comma_grouped_decimal_value():
    result = parse_numeric_cell("1,200.50")

    assert result.value == Decimal("1200.50")


def test_parses_percentage_and_separates_unit():
    result = parse_numeric_cell("12.5%")

    assert result.value == Decimal("12.5")
    assert result.unit == "%"


def test_parses_value_with_trailing_unit_text():
    result = parse_numeric_cell("1,200 백만원")

    assert result.value == Decimal("1200")
    assert result.unit == "백만원"


def test_parenthesized_value_becomes_negative():
    result = parse_numeric_cell("(300)")

    assert result.value == Decimal("-300")


def test_parenthesized_comma_separated_value_becomes_negative():
    result = parse_numeric_cell("(1,200)")

    assert result.value == Decimal("-1200")


def test_parenthesized_value_with_unit_inside_parens():
    result = parse_numeric_cell("(300백만원)")

    assert result.value == Decimal("-300")
    assert result.unit == "백만원"


def test_explicit_minus_sign_without_parens():
    result = parse_numeric_cell("-1,200")

    assert result.value == Decimal("-1200")


def test_explicit_plus_sign():
    result = parse_numeric_cell("+12.5")

    assert result.value == Decimal("12.5")


def test_dash_marker_is_missing():
    result = parse_numeric_cell("-")

    assert result.is_missing
    assert result.value is None
    assert not result.is_invalid


def test_empty_string_is_missing():
    result = parse_numeric_cell("")

    assert result.is_missing
    assert result.raw == ""


def test_whitespace_only_is_missing():
    result = parse_numeric_cell("   ")

    assert result.is_missing


def test_none_is_missing_and_preserves_raw_as_none():
    result = parse_numeric_cell(None)

    assert result.is_missing
    assert result.raw is None


def test_non_numeric_text_is_invalid_not_missing():
    result = parse_numeric_cell("해당없음")

    assert not result.is_missing
    assert result.value is None
    assert result.is_invalid


def test_parenthesized_dash_is_missing():
    result = parse_numeric_cell("(-)")

    assert result.is_missing
    assert not result.is_invalid


def test_parenthesized_blank_with_space_is_missing():
    result = parse_numeric_cell("( )")

    assert result.is_missing


def test_empty_parens_is_missing():
    result = parse_numeric_cell("()")

    assert result.is_missing


def test_malformed_comma_grouping_single_digit_groups_is_invalid():
    result = parse_numeric_cell("1,2,3")

    assert result.is_invalid


def test_malformed_comma_grouping_two_digit_group_is_invalid():
    result = parse_numeric_cell("12,34")

    assert result.is_invalid


def test_leading_comma_is_invalid():
    result = parse_numeric_cell(",123")

    assert result.is_invalid


def test_date_like_value_is_invalid():
    result = parse_numeric_cell("2025.02.03")

    assert result.is_invalid


def test_korean_date_with_digits_in_trailing_text_is_invalid():
    result = parse_numeric_cell("2025년 10월 22일")

    assert result.is_invalid


def test_parenthesized_korean_date_is_invalid_not_negative():
    result = parse_numeric_cell("(2025년 08월 27일)")

    assert result.is_invalid
    assert result.value is None


def test_multiple_decimal_points_is_invalid():
    result = parse_numeric_cell("1.2.3")

    assert result.is_invalid


def test_scientific_notation_is_invalid():
    result = parse_numeric_cell("1e3")

    assert result.is_invalid


def test_paren_number_with_unit_outside_parens_is_unsupported():
    # 정책: 괄호가 전체 문자열을 감쌀 때만 음수로 해석한다.
    # 숫자만 괄호로 감싸고 단위가 밖에 붙은 형식은 모호하므로 invalid로 거부한다.
    result = parse_numeric_cell("(300) 백만원")

    assert result.is_invalid
