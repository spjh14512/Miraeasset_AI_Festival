from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from agent_graph.tools import CalculationOperation, TableTarget


def test_table_target_accepts_valid_reference():
    target = TableTarget(result_id="retrieval:plan_1", item_index=0)

    assert target.result_id == "retrieval:plan_1"
    assert target.item_index == 0


def test_table_target_rejects_blank_result_id():
    with pytest.raises(ValidationError):
        TableTarget(result_id="", item_index=0)


def test_table_target_rejects_whitespace_only_result_id():
    with pytest.raises(ValidationError):
        TableTarget(result_id="   ", item_index=0)


def test_table_target_strips_result_id_whitespace():
    target = TableTarget(result_id="  retrieval:plan_1  ", item_index=0)

    assert target.result_id == "retrieval:plan_1"


def test_table_target_rejects_negative_item_index():
    with pytest.raises(ValidationError):
        TableTarget(result_id="retrieval:plan_1", item_index=-1)


def test_table_target_rejects_bool_item_index():
    with pytest.raises(ValidationError):
        TableTarget(result_id="retrieval:plan_1", item_index=True)


def test_table_target_rejects_string_item_index():
    with pytest.raises(ValidationError):
        TableTarget(result_id="retrieval:plan_1", item_index="2")


def test_calculation_operation_accepts_supported_values():
    class _Probe(BaseModel):
        operation: CalculationOperation

    for operation in ("sum", "mean", "median", "max", "min", "mode"):
        assert _Probe(operation=operation).operation == operation


def test_calculation_operation_rejects_unsupported_value():
    class _Probe(BaseModel):
        operation: CalculationOperation

    with pytest.raises(ValidationError):
        _Probe(operation="stddev")
