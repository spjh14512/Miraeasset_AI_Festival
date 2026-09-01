from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from agent_graph.state import RetrievalResult
from agent_graph.tools import NumericResultTarget, combine_numeric_results


def _derived_result(
    result_id: str,
    *,
    value: str,
    unit: str | None = None,
    variable_name: str = "값",
    disclosure_id: str | None = "d20240101000001",
    source: str = "derived",
    status: str = "SUCCESS",
    result_kind: str | None = "numeric_scalar",
) -> RetrievalResult:
    source_references = (
        [{"disclosure_id": disclosure_id}] if disclosure_id is not None else []
    )
    item = {
        "type": "record",
        "fields": {
            "variable_name": variable_name,
            "operation": "sum",
            "value": value,
            "unit": unit,
            "input_count": 1,
        },
        "source_references": source_references,
    }
    metadata = {} if result_kind is None else {"result_kind": result_kind}
    return RetrievalResult(
        result_id=result_id,
        plan_id=result_id.removeprefix("derived:"),
        source=source,
        status=status,
        query="{}",
        items=[item],
        result_count=1,
        metadata=metadata,
    )


def _state(*results: RetrievalResult, next_plan_seq: int = 1) -> dict[str, Any]:
    return {
        "question_id": "question-1",
        "question_text": "두 값을 비교해줘",
        "next_plan_seq": next_plan_seq,
        "retrieval_results": list(results),
    }


def _target(result_id: str, item_index: int = 0) -> dict[str, Any]:
    return {"result_id": result_id, "item_index": item_index}


def _invoke(**kwargs: Any) -> dict:
    return combine_numeric_results.invoke(kwargs)


def test_numeric_result_target_defaults_item_index_to_zero():
    target = NumericResultTarget(result_id="derived:plan_1")

    assert target.item_index == 0


def test_numeric_result_target_strips_and_rejects_blank_result_id():
    assert NumericResultTarget(result_id="  derived:plan_1  ").result_id == "derived:plan_1"
    with pytest.raises(ValidationError):
        NumericResultTarget(result_id="   ")


def test_sum_of_two_results_succeeds():
    state = _state(
        _derived_result("derived:plan_1", value="100"),
        _derived_result("derived:plan_2", value="200"),
    )

    update = _invoke(
        variable_name="합계",
        operation="sum",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "SUCCESS"
    assert result.source == "derived"
    assert result.metadata["result_kind"] == "numeric_scalar"
    assert result.items[0]["fields"]["value"] == "300"
    assert update["next_plan_seq"] == 2


def test_sum_of_three_results_succeeds():
    state = _state(
        _derived_result("derived:plan_1", value="1"),
        _derived_result("derived:plan_2", value="2"),
        _derived_result("derived:plan_3", value="3"),
    )

    update = _invoke(
        variable_name="합계",
        operation="sum",
        targets=[_target("derived:plan_1"), _target("derived:plan_2"), _target("derived:plan_3")],
        state=state,
    )

    assert update["retrieval_results"][0].items[0]["fields"]["value"] == "6"


def test_mean_of_two_results_succeeds():
    # 기초/기말 평균 같은 지표를 겨냥한 케이스.
    state = _state(
        _derived_result("derived:plan_1", value="100"),
        _derived_result("derived:plan_2", value="200"),
    )

    update = _invoke(
        variable_name="기초기말평균",
        operation="mean",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "SUCCESS"
    assert result.metadata["result_kind"] == "numeric_scalar"
    assert Decimal(result.items[0]["fields"]["value"]) == Decimal("150")


def test_mean_of_three_results_succeeds():
    state = _state(
        _derived_result("derived:plan_1", value="1"),
        _derived_result("derived:plan_2", value="2"),
        _derived_result("derived:plan_3", value="3"),
    )

    update = _invoke(
        variable_name="평균",
        operation="mean",
        targets=[_target("derived:plan_1"), _target("derived:plan_2"), _target("derived:plan_3")],
        state=state,
    )

    assert Decimal(update["retrieval_results"][0].items[0]["fields"]["value"]) == Decimal("2")


def test_mean_requires_at_least_two_results():
    state = _state(_derived_result("derived:plan_1", value="100"))

    with pytest.raises(ValueError, match="mean"):
        _invoke(
            variable_name="평균",
            operation="mean",
            targets=[_target("derived:plan_1")],
            state=state,
        )


def test_mean_uses_common_unit():
    state = _state(
        _derived_result("derived:plan_1", value="100", unit="백만원"),
        _derived_result("derived:plan_2", value="200", unit="백만원"),
    )

    update = _invoke(
        variable_name="평균",
        operation="mean",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    assert update["retrieval_results"][0].items[0]["fields"]["unit"] == "백만원"


def test_mean_rejects_mismatched_units():
    state = _state(
        _derived_result("derived:plan_1", value="100", unit="백만원"),
        _derived_result("derived:plan_2", value="200", unit="주"),
    )

    update = _invoke(
        variable_name="평균",
        operation="mean",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "unit_check"


def test_difference_succeeds_with_correct_order():
    state = _state(
        _derived_result("derived:plan_1", value="500"),
        _derived_result("derived:plan_2", value="200"),
    )

    update = _invoke(
        variable_name="차이",
        operation="difference",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    assert update["retrieval_results"][0].items[0]["fields"]["value"] == "300"


def test_ratio_succeeds():
    state = _state(
        _derived_result("derived:plan_1", value="100"),
        _derived_result("derived:plan_2", value="4"),
    )

    update = _invoke(
        variable_name="비율",
        operation="ratio",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    fields = update["retrieval_results"][0].items[0]["fields"]
    assert fields["value"] == "25"
    assert fields["unit"] is None


def test_percent_ratio_succeeds():
    # ROA = 당기순이익 / 자산총계 * 100 같은 케이스를 겨냥함.
    state = _state(
        _derived_result("derived:plan_1", value="10"),
        _derived_result("derived:plan_2", value="100"),
    )

    update = _invoke(
        variable_name="ROA",
        operation="percent_ratio",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    fields = update["retrieval_results"][0].items[0]["fields"]
    assert Decimal(fields["value"]) == Decimal("10")
    assert fields["unit"] == "%"


def test_percent_ratio_by_zero_makes_invalid_input():
    state = _state(
        _derived_result("derived:plan_1", value="10"),
        _derived_result("derived:plan_2", value="0"),
    )

    update = _invoke(
        variable_name="ROA",
        operation="percent_ratio",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "calculation"


def test_percent_ratio_rejects_three_targets():
    state = _state(
        _derived_result("derived:plan_1", value="10"),
        _derived_result("derived:plan_2", value="100"),
        _derived_result("derived:plan_3", value="1"),
    )

    with pytest.raises(ValueError, match="percent_ratio"):
        _invoke(
            variable_name="ROA",
            operation="percent_ratio",
            targets=[_target("derived:plan_1"), _target("derived:plan_2"), _target("derived:plan_3")],
            state=state,
        )


def test_percent_ratio_differs_from_ratio_and_percent_change():
    # 같은 두 값(10, 100)에 대해 세 연산이 서로 다른 값을 내야 한다
    # (percent_ratio를 ratio나 percent_change와 혼동해서 구현하지
    # 않았는지 확인).
    state = _state(
        _derived_result("derived:plan_1", value="10"),
        _derived_result("derived:plan_2", value="100"),
    )
    targets = [_target("derived:plan_1"), _target("derived:plan_2")]

    ratio_value = Decimal(_invoke(
        variable_name="v", operation="ratio", targets=targets, state=state,
    )["retrieval_results"][0].items[0]["fields"]["value"])
    percent_ratio_value = Decimal(_invoke(
        variable_name="v", operation="percent_ratio", targets=targets, state=state,
    )["retrieval_results"][0].items[0]["fields"]["value"])
    percent_change_value = Decimal(_invoke(
        variable_name="v", operation="percent_change", targets=targets, state=state,
    )["retrieval_results"][0].items[0]["fields"]["value"])

    assert ratio_value == Decimal("0.1")
    assert percent_ratio_value == Decimal("10")
    assert percent_change_value == Decimal("900")
    assert len({ratio_value, percent_ratio_value, percent_change_value}) == 3


def test_cagr_succeeds():
    # 100 -> 200, 5년 -> 2^(1/5) - 1 ≈ 14.8698%
    state = _state(
        _derived_result("derived:plan_1", value="100"),
        _derived_result("derived:plan_2", value="200"),
    )

    update = _invoke(
        variable_name="CAGR",
        operation="cagr",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        periods=5,
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "SUCCESS"
    value = Decimal(result.items[0]["fields"]["value"])
    assert abs(value - Decimal("14.8698354997035006798626947")) < Decimal("0.000001")
    assert result.items[0]["fields"]["unit"] == "%"


def test_cagr_requires_periods():
    state = _state(
        _derived_result("derived:plan_1", value="100"),
        _derived_result("derived:plan_2", value="200"),
    )

    with pytest.raises(ValueError, match="periods"):
        _invoke(
            variable_name="CAGR",
            operation="cagr",
            targets=[_target("derived:plan_1"), _target("derived:plan_2")],
            state=state,
        )


def test_cagr_rejects_non_positive_periods():
    state = _state(
        _derived_result("derived:plan_1", value="100"),
        _derived_result("derived:plan_2", value="200"),
    )

    with pytest.raises(ValueError, match="periods"):
        _invoke(
            variable_name="CAGR",
            operation="cagr",
            targets=[_target("derived:plan_1"), _target("derived:plan_2")],
            periods=0,
            state=state,
        )


def test_cagr_rejects_bool_periods():
    state = _state(
        _derived_result("derived:plan_1", value="100"),
        _derived_result("derived:plan_2", value="200"),
    )

    with pytest.raises(ValueError, match="periods"):
        _invoke(
            variable_name="CAGR",
            operation="cagr",
            targets=[_target("derived:plan_1"), _target("derived:plan_2")],
            periods=True,
            state=state,
        )


def test_cagr_with_zero_start_makes_invalid_input():
    state = _state(
        _derived_result("derived:plan_1", value="0"),
        _derived_result("derived:plan_2", value="200"),
    )

    update = _invoke(
        variable_name="CAGR",
        operation="cagr",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        periods=5,
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "calculation"


def test_cagr_with_sign_change_makes_invalid_input():
    state = _state(
        _derived_result("derived:plan_1", value="100"),
        _derived_result("derived:plan_2", value="-50"),
    )

    update = _invoke(
        variable_name="CAGR",
        operation="cagr",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        periods=5,
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "calculation"


def test_cagr_with_both_negative_makes_invalid_input():
    state = _state(
        _derived_result("derived:plan_1", value="-100"),
        _derived_result("derived:plan_2", value="-50"),
    )

    update = _invoke(
        variable_name="CAGR",
        operation="cagr",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        periods=5,
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "calculation"


def test_cagr_with_zero_end_makes_invalid_input():
    state = _state(
        _derived_result("derived:plan_1", value="200"),
        _derived_result("derived:plan_2", value="0"),
    )

    update = _invoke(
        variable_name="CAGR",
        operation="cagr",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        periods=5,
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "calculation"


def test_cagr_rejects_three_targets():
    state = _state(
        _derived_result("derived:plan_1", value="100"),
        _derived_result("derived:plan_2", value="200"),
        _derived_result("derived:plan_3", value="300"),
    )

    with pytest.raises(ValueError, match="cagr"):
        _invoke(
            variable_name="CAGR",
            operation="cagr",
            targets=[_target("derived:plan_1"), _target("derived:plan_2"), _target("derived:plan_3")],
            periods=5,
            state=state,
        )


def test_ratio_by_zero_makes_invalid_input():
    state = _state(
        _derived_result("derived:plan_1", value="100"),
        _derived_result("derived:plan_2", value="0"),
    )

    update = _invoke(
        variable_name="비율",
        operation="ratio",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "calculation"


def test_percent_change_uses_left_as_baseline():
    state = _state(
        _derived_result("derived:plan_1", value="100"),
        _derived_result("derived:plan_2", value="150"),
    )

    update = _invoke(
        variable_name="증감률",
        operation="percent_change",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    fields = update["retrieval_results"][0].items[0]["fields"]
    assert Decimal(fields["value"]) == Decimal("50")
    assert fields["unit"] == "%"


def test_percent_change_with_zero_baseline_makes_invalid_input():
    state = _state(
        _derived_result("derived:plan_1", value="0"),
        _derived_result("derived:plan_2", value="150"),
    )

    update = _invoke(
        variable_name="증감률",
        operation="percent_change",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "calculation"


def test_ordering_ranks_ascending_by_default():
    state = _state(
        _derived_result("derived:plan_1", value="300", variable_name="A"),
        _derived_result("derived:plan_2", value="100", variable_name="B"),
        _derived_result("derived:plan_3", value="200", variable_name="C"),
    )

    update = _invoke(
        variable_name="순위",
        operation="ordering",
        targets=[_target("derived:plan_1"), _target("derived:plan_2"), _target("derived:plan_3")],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.metadata["result_kind"] == "numeric_ordering"
    fields = result.items[0]["fields"]
    assert fields["direction"] == "ascending"
    assert "value" not in fields
    assert [entry["result_id"] for entry in fields["ordered_results"]] == [
        "derived:plan_2",
        "derived:plan_3",
        "derived:plan_1",
    ]
    assert [entry["rank"] for entry in fields["ordered_results"]] == [1, 2, 3]


def test_ordering_rank_reflects_descending_direction_too():
    state = _state(
        _derived_result("derived:plan_1", value="300"),
        _derived_result("derived:plan_2", value="100"),
    )

    update = _invoke(
        variable_name="순위",
        operation="ordering",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        direction="descending",
        state=state,
    )

    ordered = update["retrieval_results"][0].items[0]["fields"]["ordered_results"]
    assert [(entry["rank"], entry["result_id"]) for entry in ordered] == [
        (1, "derived:plan_1"),
        (2, "derived:plan_2"),
    ]


def test_ordering_descending_direction():
    state = _state(
        _derived_result("derived:plan_1", value="300"),
        _derived_result("derived:plan_2", value="100"),
    )

    update = _invoke(
        variable_name="순위",
        operation="ordering",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
        direction="descending",
    )

    fields = update["retrieval_results"][0].items[0]["fields"]
    assert fields["direction"] == "descending"
    assert [entry["result_id"] for entry in fields["ordered_results"]] == [
        "derived:plan_1",
        "derived:plan_2",
    ]


def test_ordering_ties_preserve_input_order():
    state = _state(
        _derived_result("derived:plan_1", value="100"),
        _derived_result("derived:plan_2", value="100"),
        _derived_result("derived:plan_3", value="50"),
    )

    update = _invoke(
        variable_name="순위",
        operation="ordering",
        targets=[_target("derived:plan_1"), _target("derived:plan_2"), _target("derived:plan_3")],
        state=state,
    )

    fields = update["retrieval_results"][0].items[0]["fields"]
    assert [entry["result_id"] for entry in fields["ordered_results"]] == [
        "derived:plan_3",
        "derived:plan_1",
        "derived:plan_2",
    ]
    assert [entry["rank"] for entry in fields["ordered_results"]] == [1, 2, 3]


def test_ordering_result_cannot_be_reused_as_scalar_input():
    ordering_result = RetrievalResult(
        result_id="derived:plan_order",
        plan_id="plan_order",
        source="derived",
        status="SUCCESS",
        query="{}",
        items=[{
            "type": "record",
            "fields": {
                "variable_name": "순위",
                "operation": "ordering",
                "direction": "ascending",
                "unit": None,
                "input_count": 2,
                "ordered_results": [],
            },
            "source_references": [{"disclosure_id": "d1"}],
        }],
        result_count=1,
        metadata={"result_kind": "numeric_ordering"},
    )
    state = _state(ordering_result, _derived_result("derived:plan_2", value="100"))

    with pytest.raises(ValueError, match="숫자 scalar"):
        _invoke(
            variable_name="합계",
            operation="sum",
            targets=[_target("derived:plan_order"), _target("derived:plan_2")],
            state=state,
        )


def test_mismatched_units_make_invalid_input():
    state = _state(
        _derived_result("derived:plan_1", value="100", unit="백만원"),
        _derived_result("derived:plan_2", value="200", unit="주"),
    )

    update = _invoke(
        variable_name="합계",
        operation="sum",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "unit_check"


def test_none_unit_mixed_with_explicit_unit_makes_invalid_input():
    # combine에서는 calculate와 달리 "단위 없음"과 "명시된 단위"를
    # 같다고 보지 않는다 — 서로 다른 derived 결과이므로 단위가 완전히
    # 같을 때만(둘 다 None이거나 둘 다 같은 문자열) 조합을 허용한다.
    state = _state(
        _derived_result("derived:plan_1", value="100", unit=None),
        _derived_result("derived:plan_2", value="200", unit="백만원"),
    )

    update = _invoke(
        variable_name="합계",
        operation="sum",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "unit_check"


def test_both_none_units_succeed():
    state = _state(
        _derived_result("derived:plan_1", value="100", unit=None),
        _derived_result("derived:plan_2", value="200", unit=None),
    )

    update = _invoke(
        variable_name="합계",
        operation="sum",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "SUCCESS"
    assert result.items[0]["fields"]["unit"] is None


def test_same_explicit_units_succeed():
    state = _state(
        _derived_result("derived:plan_1", value="100", unit="백만원"),
        _derived_result("derived:plan_2", value="200", unit="백만원"),
    )

    update = _invoke(
        variable_name="합계",
        operation="sum",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "SUCCESS"
    assert result.items[0]["fields"]["unit"] == "백만원"


def test_non_string_unit_raises_value_error():
    malformed = RetrievalResult(
        result_id="derived:plan_1",
        plan_id="plan_1",
        source="derived",
        status="SUCCESS",
        query="{}",
        items=[{
            "type": "record",
            "fields": {
                "variable_name": "v",
                "operation": "sum",
                "value": "100",
                "unit": 123,
                "input_count": 1,
            },
            "source_references": [{"disclosure_id": "d1"}],
        }],
        result_count=1,
        metadata={"result_kind": "numeric_scalar"},
    )
    state = _state(malformed, _derived_result("derived:plan_2", value="200"))

    with pytest.raises(ValueError, match="unit"):
        _invoke(
            variable_name="합계",
            operation="sum",
            targets=[_target("derived:plan_1"), _target("derived:plan_2")],
            state=state,
        )


def test_whitespace_only_disclosure_id_in_reference_makes_invalid_input():
    state = _state(
        _derived_result("derived:plan_1", value="100", disclosure_id="   "),
        _derived_result("derived:plan_2", value="200"),
    )

    update = _invoke(
        variable_name="합계",
        operation="sum",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "citation_check"


def test_evidence_id_without_section_id_in_reference_makes_invalid_input():
    incomplete = RetrievalResult(
        result_id="derived:plan_1",
        plan_id="plan_1",
        source="derived",
        status="SUCCESS",
        query="{}",
        items=[{
            "type": "record",
            "fields": {
                "variable_name": "v",
                "operation": "sum",
                "value": "100",
                "unit": None,
                "input_count": 1,
            },
            "source_references": [{"disclosure_id": "d1", "evidence_id": "e1"}],
        }],
        result_count=1,
        metadata={"result_kind": "numeric_scalar"},
    )
    state = _state(incomplete, _derived_result("derived:plan_2", value="200"))

    update = _invoke(
        variable_name="합계",
        operation="sum",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "citation_check"


def test_no_citation_makes_invalid_input():
    state = _state(
        _derived_result("derived:plan_1", value="100", disclosure_id=None),
        _derived_result("derived:plan_2", value="200"),
    )

    update = _invoke(
        variable_name="합계",
        operation="sum",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "citation_check"


def test_item_with_mixed_valid_and_invalid_references_makes_invalid_input():
    # 한 item의 source_references 안에 유효한 것과 무효한 것이 섞여
    # 있으면, 무효한 것만 조용히 건너뛰지 않고 전체를 거부한다.
    mixed = RetrievalResult(
        result_id="derived:plan_1",
        plan_id="plan_1",
        source="derived",
        status="SUCCESS",
        query="{}",
        items=[{
            "type": "record",
            "fields": {
                "variable_name": "v",
                "operation": "sum",
                "value": "100",
                "unit": None,
                "input_count": 1,
            },
            "source_references": [
                {"disclosure_id": "d1"},
                {"disclosure_id": "   "},
            ],
        }],
        result_count=1,
        metadata={"result_kind": "numeric_scalar"},
    )
    state = _state(mixed, _derived_result("derived:plan_2", value="200"))

    update = _invoke(
        variable_name="합계",
        operation="sum",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.status == "INVALID_INPUT"
    assert result.metadata["failure_stage"] == "citation_check"


def test_nonexistent_result_id_raises_value_error():
    state = _state(_derived_result("derived:plan_1", value="100"))

    with pytest.raises(ValueError, match="찾지 못했습니다"):
        _invoke(
            variable_name="합계",
            operation="sum",
            targets=[_target("derived:plan_1"), _target("derived:unknown")],
            state=state,
        )


def test_non_derived_source_raises_value_error():
    raw_result = RetrievalResult(
        result_id="retrieval:plan_1",
        plan_id="plan_1",
        source="qdrant",
        query="q",
        items=[{"type": "r_table"}],
        result_count=1,
    )
    state = _state(raw_result, _derived_result("derived:plan_2", value="100"))

    with pytest.raises(ValueError, match="derived"):
        _invoke(
            variable_name="합계",
            operation="sum",
            targets=[_target("retrieval:plan_1"), _target("derived:plan_2")],
            state=state,
        )


def test_non_success_derived_raises_value_error():
    state = _state(
        _derived_result("derived:plan_1", value="100", status="INVALID_INPUT"),
        _derived_result("derived:plan_2", value="200"),
    )

    with pytest.raises(ValueError, match="SUCCESS"):
        _invoke(
            variable_name="합계",
            operation="sum",
            targets=[_target("derived:plan_1"), _target("derived:plan_2")],
            state=state,
        )


def test_missing_result_kind_raises_value_error():
    state = _state(
        _derived_result("derived:plan_1", value="100", result_kind=None),
        _derived_result("derived:plan_2", value="200"),
    )

    with pytest.raises(ValueError, match="숫자 scalar"):
        _invoke(
            variable_name="합계",
            operation="sum",
            targets=[_target("derived:plan_1"), _target("derived:plan_2")],
            state=state,
        )


def test_out_of_range_item_index_raises_value_error():
    state = _state(_derived_result("derived:plan_1", value="100"), _derived_result("derived:plan_2", value="200"))

    with pytest.raises(ValueError, match="item_index"):
        _invoke(
            variable_name="합계",
            operation="sum",
            targets=[_target("derived:plan_1", item_index=5), _target("derived:plan_2")],
            state=state,
        )


def test_duplicate_targets_raise_value_error():
    state = _state(_derived_result("derived:plan_1", value="100"))

    with pytest.raises(ValueError, match="중복"):
        _invoke(
            variable_name="합계",
            operation="sum",
            targets=[_target("derived:plan_1"), _target("derived:plan_1")],
            state=state,
        )


def test_sum_requires_at_least_two_results():
    state = _state(_derived_result("derived:plan_1", value="100"))

    with pytest.raises(ValueError, match="sum"):
        _invoke(
            variable_name="합계",
            operation="sum",
            targets=[_target("derived:plan_1")],
            state=state,
        )


def test_difference_rejects_three_results():
    state = _state(
        _derived_result("derived:plan_1", value="100"),
        _derived_result("derived:plan_2", value="200"),
        _derived_result("derived:plan_3", value="300"),
    )

    with pytest.raises(ValueError, match="difference"):
        _invoke(
            variable_name="차이",
            operation="difference",
            targets=[_target("derived:plan_1"), _target("derived:plan_2"), _target("derived:plan_3")],
            state=state,
        )


def test_blank_variable_name_raises_value_error():
    state = _state(
        _derived_result("derived:plan_1", value="100"),
        _derived_result("derived:plan_2", value="200"),
    )

    with pytest.raises(ValueError, match="variable_name"):
        _invoke(
            variable_name="   ",
            operation="sum",
            targets=[_target("derived:plan_1"), _target("derived:plan_2")],
            state=state,
        )


def test_non_record_item_raises_value_error():
    malformed = RetrievalResult(
        result_id="derived:plan_1",
        plan_id="plan_1",
        source="derived",
        status="SUCCESS",
        query="{}",
        items=[{"type": "r_table"}],
        result_count=1,
        metadata={"result_kind": "numeric_scalar"},
    )
    state = _state(malformed, _derived_result("derived:plan_2", value="100"))

    with pytest.raises(ValueError, match="형식"):
        _invoke(
            variable_name="합계",
            operation="sum",
            targets=[_target("derived:plan_1"), _target("derived:plan_2")],
            state=state,
        )


def test_next_plan_seq_uses_state_counter():
    state = _state(
        _derived_result("derived:plan_1", value="100"),
        _derived_result("derived:plan_2", value="200"),
        next_plan_seq=9,
    )

    update = _invoke(
        variable_name="합계",
        operation="sum",
        targets=[_target("derived:plan_1"), _target("derived:plan_2")],
        state=state,
    )

    result = update["retrieval_results"][0]
    assert result.plan_id == "plan_9"
    assert result.result_id == "derived:plan_9"
    assert update["next_plan_seq"] == 10
