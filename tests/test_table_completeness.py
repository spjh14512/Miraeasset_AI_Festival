from __future__ import annotations

from typing import Any

from agent_graph.calculation import check_table_completeness

_COLUMNS = ["구분", "금액"]


def _whole_item(
    *,
    columns: list[str] | None = None,
    omitted: int = 0,
    records: list[dict[str, Any]] | None = None,
    include_record_counts: bool = True,
) -> dict[str, Any]:
    resolved_records = [{"record_index": 0, "values": {}}] if records is None else records
    item: dict[str, Any] = {
        "type": "r_table",
        "columns": list(_COLUMNS) if columns is None else columns,
        "scope": {"kind": "table"},
        "records": resolved_records,
        "metadata": {},
    }
    if include_record_counts:
        item["available_record_count"] = len(resolved_records) + omitted
        item["included_record_count"] = len(resolved_records)
        item["omitted_record_count"] = omitted
    return item


def _chunk_item(
    *,
    table_id: str = "table-1",
    chunk_index: int,
    chunk_count: int,
    row_start: int,
    row_end: int,
    columns: list[str] | None = None,
    omitted: int = 0,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    # 기본값은 chunk가 선언한 행 범위 크기(row_end - row_start + 1)와 정확히
    # 맞는 개수의 record를 생성한다. 이 크기를 일부러 어긋나게 하고 싶으면
    # records를 직접 넘긴다.
    expected_count = max(row_end - row_start + 1, 0)
    resolved_records = (
        [{"record_index": i, "values": {}} for i in range(expected_count)]
        if records is None
        else records
    )
    return {
        "type": "r_table",
        "columns": list(_COLUMNS) if columns is None else columns,
        "scope": {
            "kind": "row_group",
            "row_start_index": row_start,
            "row_end_index": row_end,
        },
        "records": resolved_records,
        "available_record_count": len(resolved_records) + omitted,
        "included_record_count": len(resolved_records),
        "omitted_record_count": omitted,
        "metadata": {
            "table_id": table_id,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "row_start_index": row_start,
            "row_end_index": row_end,
        },
    }


def test_single_whole_table_is_complete():
    result = check_table_completeness([_whole_item()], "금액")

    assert result.is_complete
    assert result.reason is None


def test_empty_items_is_incomplete():
    result = check_table_completeness([], "금액")

    assert not result.is_complete


def test_missing_column_is_incomplete():
    result = check_table_completeness([_whole_item()], "존재하지않는열")

    assert not result.is_complete


def test_empty_columns_list_is_incomplete():
    result = check_table_completeness([_whole_item(columns=[])], "금액")

    assert not result.is_complete


def test_non_r_table_item_is_incomplete():
    item = _whole_item()
    item["type"] = "kv_table"

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_omitted_records_makes_incomplete():
    result = check_table_completeness([_whole_item(omitted=3)], "금액")

    assert not result.is_complete


def test_summary_only_item_without_records_is_incomplete():
    item = _whole_item()
    del item["records"]
    del item["available_record_count"]
    del item["included_record_count"]
    del item["omitted_record_count"]

    result = check_table_completeness([item], "금액")

    assert not result.is_complete
    assert "summary" in result.reason


def test_missing_record_count_fields_is_incomplete():
    item = _whole_item(include_record_counts=False)

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_omitted_record_count_none_is_incomplete_not_typeerror():
    item = _whole_item()
    item["omitted_record_count"] = None

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_omitted_record_count_string_is_incomplete():
    item = _whole_item()
    item["omitted_record_count"] = "0"

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_omitted_record_count_negative_is_incomplete():
    item = _whole_item()
    item["omitted_record_count"] = -1
    item["available_record_count"] = 0

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_included_record_count_mismatch_is_incomplete():
    item = _whole_item()
    item["included_record_count"] = 5

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_available_record_count_mismatch_is_incomplete():
    item = _whole_item()
    item["available_record_count"] = 99

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_unchunked_item_with_row_group_scope_is_incomplete():
    item = _whole_item()
    item["scope"] = {"kind": "row_group", "row_start_index": 0, "row_end_index": 9}

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_missing_scope_is_incomplete():
    item = _whole_item()
    del item["scope"]

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_two_chunks_covering_full_table_is_complete():
    items = [
        _chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=49),
        _chunk_item(chunk_index=1, chunk_count=2, row_start=50, row_end=99),
    ]

    result = check_table_completeness(items, "금액")

    assert result.is_complete


def test_missing_chunk_is_incomplete():
    items = [_chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=49)]

    result = check_table_completeness(items, "금액")

    assert not result.is_complete
    assert "1" in result.reason


def test_duplicate_chunk_is_incomplete():
    items = [
        _chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=49),
        _chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=49),
    ]

    result = check_table_completeness(items, "금액")

    assert not result.is_complete


def test_mismatched_columns_between_chunks_is_incomplete():
    items = [
        _chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=49),
        _chunk_item(
            chunk_index=1,
            chunk_count=2,
            row_start=50,
            row_end=99,
            columns=["구분", "다른열"],
        ),
    ]

    result = check_table_completeness(items, "금액")

    assert not result.is_complete


def test_mismatched_chunk_count_between_chunks_is_incomplete():
    items = [
        _chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=49),
        _chunk_item(chunk_index=1, chunk_count=3, row_start=50, row_end=99),
    ]

    result = check_table_completeness(items, "금액")

    assert not result.is_complete


def test_row_gap_is_incomplete():
    items = [
        _chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=49),
        _chunk_item(chunk_index=1, chunk_count=2, row_start=51, row_end=99),
    ]

    result = check_table_completeness(items, "금액")

    assert not result.is_complete


def test_row_overlap_is_incomplete():
    items = [
        _chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=50),
        _chunk_item(chunk_index=1, chunk_count=2, row_start=40, row_end=99),
    ]

    result = check_table_completeness(items, "금액")

    assert not result.is_complete


def test_first_chunk_not_starting_at_zero_is_incomplete():
    items = [
        _chunk_item(chunk_index=0, chunk_count=2, row_start=1, row_end=49),
        _chunk_item(chunk_index=1, chunk_count=2, row_start=50, row_end=99),
    ]

    result = check_table_completeness(items, "금액")

    assert not result.is_complete


def test_chunk_index_order_reversed_relative_to_rows_is_incomplete():
    # chunk_index=0인데 뒤쪽 행 구간을, chunk_index=1인데 앞쪽 행 구간을 가리킴.
    items = [
        _chunk_item(chunk_index=0, chunk_count=2, row_start=50, row_end=99),
        _chunk_item(chunk_index=1, chunk_count=2, row_start=0, row_end=49),
    ]

    result = check_table_completeness(items, "금액")

    assert not result.is_complete


def test_mixing_two_different_table_ids_is_incomplete():
    items = [
        _chunk_item(table_id="table-1", chunk_index=0, chunk_count=2, row_start=0, row_end=49),
        _chunk_item(table_id="table-2", chunk_index=1, chunk_count=2, row_start=50, row_end=99),
    ]

    result = check_table_completeness(items, "금액")

    assert not result.is_complete


def test_mixing_two_whole_tables_is_incomplete():
    result = check_table_completeness([_whole_item(), _whole_item()], "금액")

    assert not result.is_complete


def test_mixing_whole_table_with_chunked_table_is_incomplete():
    items = [
        _whole_item(),
        _chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=49),
    ]

    result = check_table_completeness(items, "금액")

    assert not result.is_complete


def test_scope_kind_table_but_chunk_metadata_present_is_incomplete():
    # scope는 "표 전체"라고 하는데 metadata에는 chunk 정보가 있는 모순된 item.
    item = _chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=49)
    item["scope"] = {"kind": "table"}

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_scope_row_range_mismatched_with_metadata_is_incomplete():
    item = _chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=49)
    item["scope"]["row_end_index"] = 999

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_negative_row_end_index_is_incomplete():
    item = _chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=-1)
    item["scope"]["row_end_index"] = -1

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_row_start_greater_than_row_end_is_incomplete():
    item = _chunk_item(chunk_index=0, chunk_count=2, row_start=50, row_end=10)
    item["scope"]["row_start_index"] = 50
    item["scope"]["row_end_index"] = 10

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_chunk_count_as_string_is_incomplete_not_typeerror():
    item = _chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=49)
    item["metadata"]["chunk_count"] = "2"

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_chunk_index_as_bool_is_incomplete():
    item = _chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=49)
    item["metadata"]["chunk_index"] = False

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_chunk_count_below_minimum_is_incomplete():
    item = _chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=49)
    item["metadata"]["chunk_count"] = 1

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_chunk_index_out_of_range_is_incomplete():
    item = _chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=49)
    item["metadata"]["chunk_index"] = 5

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_partial_chunk_metadata_is_incomplete():
    item = _chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=49)
    del item["metadata"]["chunk_count"]

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_blank_table_id_is_incomplete():
    item = _chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=49)
    item["metadata"]["table_id"] = "   "

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_row_range_size_mismatched_with_actual_record_count_is_incomplete():
    # 행 범위는 0~49(50개)라고 선언했지만 실제 record는 1개뿐인 경우.
    items = [
        _chunk_item(
            chunk_index=0,
            chunk_count=2,
            row_start=0,
            row_end=49,
            records=[{"record_index": 0, "values": {}}],
        ),
        _chunk_item(
            chunk_index=1,
            chunk_count=2,
            row_start=50,
            row_end=99,
            records=[{"record_index": 0, "values": {}}],
        ),
    ]

    result = check_table_completeness(items, "금액")

    assert not result.is_complete


def test_scope_bool_row_index_does_not_coerce_to_matching_int():
    # False == 0이 True로 평가되는 파이썬 특성 때문에 scope의 bool 값이
    # metadata의 정수 0과 같다고 오판하면 안 된다.
    item = _chunk_item(chunk_index=0, chunk_count=2, row_start=0, row_end=49)
    item["scope"]["row_start_index"] = False

    result = check_table_completeness([item], "금액")

    assert not result.is_complete


def test_non_dict_metadata_is_incomplete_not_treated_as_empty():
    item = _whole_item()
    item["metadata"] = "잘못된 값"

    result = check_table_completeness([item], "금액")

    assert not result.is_complete
