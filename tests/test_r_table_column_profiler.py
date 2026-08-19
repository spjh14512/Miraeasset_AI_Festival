from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from vector_db.r_table_column_profiler import (
    RTableColumnRole,
    load_r_table_column_profiling_config,
    profile_r_table_columns,
)


TABLE_ID = "rtable:20250318000123:src0:s12:t0"


def _evidence(headers, record_count):
    return {
        "evidence_id": "evidence:20250318000123:src0:s12:e0",
        "evidence_type": "TABLE",
        "table_type": "R_TABLE",
        "storage_mode": "SECTION_RECORDS",
        "payload": {
            "table_id": TABLE_ID,
            "headers": headers,
            "record_count": record_count,
        },
    }


def _records(rows):
    return [
        {
            "table_id": TABLE_ID,
            "record_index": index,
            "row_type": "DATA",
            "row_context": [],
            "values": row,
        }
        for index, row in enumerate(rows)
    ]


def test_profiles_dimension_and_measure_columns():
    evidence = _evidence(
        [
            ["사업부문"],
            ["품목"],
            ["2024년", "매출액"],
            ["2025년", "매출액"],
            ["비중"],
        ],
        3,
    )
    records = _records(
        [
            ["반도체", "DRAM", "1000", "1500", "35%"],
            ["반도체", "NAND", "800", "950", "22%"],
            ["DX", "TV", "700", "650", "15%"],
        ]
    )

    analysis = profile_r_table_columns(evidence, records)

    assert [column.role for column in analysis.columns] == [
        RTableColumnRole.DIMENSION,
        RTableColumnRole.DIMENSION,
        RTableColumnRole.MEASURE,
        RTableColumnRole.MEASURE,
        RTableColumnRole.MEASURE,
    ]
    assert analysis.columns[0].descriptor_values == ("반도체", "DX")
    assert analysis.columns[1].descriptor_values == ("DRAM", "NAND", "TV")
    assert analysis.columns[2].header_path == ("2024년", "매출액")
    assert analysis.columns[2].header_text == "2024년 > 매출액"
    assert analysis.columns[4].percent_ratio == 1.0
    assert all(not column.descriptor_values for column in analysis.columns[2:])


def test_profiles_identifier_date_and_long_text_columns():
    long_text = (
        "기업의 장기 성장 전략과 주요 사업 추진 현황을 상세하게 설명하고, "
        "향후 시장 변화에 대응하기 위한 구체적인 실행 계획을 포함하는 문장입니다."
    )
    rows = [
        [f"성명-{index}", f"2025-01-{index % 28 + 1:02d}", long_text]
        for index in range(50)
    ]
    evidence = _evidence([["성명"], ["변동일"], ["주요 내용"]], len(rows))

    analysis = profile_r_table_columns(evidence, _records(rows))

    assert [column.role for column in analysis.columns] == [
        RTableColumnRole.IDENTIFIER,
        RTableColumnRole.DATE,
        RTableColumnRole.LONG_TEXT,
    ]
    assert all(not column.descriptor_values for column in analysis.columns)
    assert analysis.identifier_columns == (("성명",),)


def test_empty_cells_are_excluded_and_explicit_values_are_profiled():
    evidence = _evidence([["구분"]], 5)
    records = _records([[""], [None], ["-"], ["0"], ["해당없음"]])

    column = profile_r_table_columns(evidence, records).columns[0]

    assert column.non_empty_count == 3
    assert column.unique_count == 3
    assert column.numeric_ratio == pytest.approx(1 / 3)
    assert column.descriptor_values == ("-", "0", "해당없음")


def test_high_cardinality_dimension_omits_values():
    values = [[f"품목-{index}"] for index in range(55)]
    evidence = _evidence([["품목"]], len(values))

    column = profile_r_table_columns(evidence, _records(values)).columns[0]

    assert column.role is RTableColumnRole.DIMENSION
    assert column.unique_count == 55
    assert column.descriptor_values == ()
    assert column.values_omitted_reason == "high_cardinality"


def test_total_dimension_value_limit_preserves_record_index_order():
    config = replace(
        load_r_table_column_profiling_config(),
        max_distinct_values_per_column=5,
        max_total_distinct_values=3,
    )
    evidence = _evidence([["지역"], ["품목"]], 2)
    records = _records([["한국", "DRAM"], ["미국", "NAND"]])
    records.reverse()

    analysis = profile_r_table_columns(evidence, records, config=config)

    assert analysis.columns[0].descriptor_values == ("한국", "미국")
    assert analysis.columns[1].descriptor_values == ("DRAM",)
    assert analysis.columns[1].values_omitted_reason == "table_value_limit"


def test_config_threshold_changes_column_role():
    evidence = _evidence([["값"]], 2)
    records = _records([["100"], ["미정"]])
    default_config = load_r_table_column_profiling_config()

    dimension = profile_r_table_columns(
        evidence,
        records,
        config=default_config,
    ).columns[0]
    measure = profile_r_table_columns(
        evidence,
        records,
        config=replace(default_config, measure_numeric_ratio_threshold=0.5),
    ).columns[0]

    assert dimension.role is RTableColumnRole.DIMENSION
    assert measure.role is RTableColumnRole.MEASURE


def test_header_keywords_only_adjust_roles_when_values_support_them():
    evidence = _evidence([["변동일"], ["매출액"], ["성명"]], 25)
    rows = [
        [
            f"2025-01-{index % 2 + 1:02d}" if index % 2 == 0 else "미정",
            str(index) if index % 2 == 0 else "미정",
            f"사람-{index}",
        ]
        for index in range(25)
    ]

    analysis = profile_r_table_columns(evidence, _records(rows))

    assert [column.role for column in analysis.columns] == [
        RTableColumnRole.DATE,
        RTableColumnRole.MEASURE,
        RTableColumnRole.IDENTIFIER,
    ]


def test_year_and_numeric_identifier_headers_correct_numeric_profiles():
    rows = [[str(2024 + index % 2), f"001{index:05d}"] for index in range(50)]
    evidence = _evidence([["연도"], ["회사 코드"]], len(rows))

    analysis = profile_r_table_columns(evidence, _records(rows))

    assert analysis.columns[0].role is RTableColumnRole.DATE
    assert analysis.columns[1].role is RTableColumnRole.IDENTIFIER


def test_analysis_to_dict_preserves_multi_level_headers_and_roles():
    evidence = _evidence([["사업부문"], ["당기", "금액"]], 1)
    analysis = profile_r_table_columns(
        evidence,
        _records([["반도체", "100"]]),
    )

    serialized = analysis.to_dict()

    assert serialized["table_id"] == TABLE_ID
    assert serialized["record_count"] == 1
    assert serialized["columns"][0]["role"] == "dimension"
    assert serialized["columns"][1]["header_text"] == "당기 > 금액"
    assert serialized["dimension_columns"] == [["사업부문"]]


def test_loads_default_column_profiling_config():
    config = load_r_table_column_profiling_config()

    assert config.measure_numeric_ratio_threshold == 0.8
    assert config.date_ratio_threshold == 0.8
    assert config.max_distinct_values_per_column == 20
    assert config.max_total_distinct_values == 50


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda evidence, records: records[0].update(table_id="rtable:other"),
            "table_id must match",
        ),
        (
            lambda evidence, records: records[0].update(values=["too short"]),
            "same length",
        ),
        (
            lambda evidence, records: evidence["payload"].update(record_count=2),
            "records length",
        ),
        (
            lambda evidence, records: records.append(deepcopy(records[0])),
            "record_index must be unique",
        ),
    ],
)
def test_rejects_invalid_table_structure(mutation, message):
    evidence = _evidence([["구분"], ["금액"]], 1)
    records = _records([["합계", "100"]])
    mutation(evidence, records)
    if message == "record_index must be unique":
        evidence["payload"]["record_count"] = 2

    with pytest.raises(ValueError, match=message):
        profile_r_table_columns(evidence, records)
