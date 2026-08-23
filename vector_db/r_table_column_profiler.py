"""Profile R_TABLE columns for a future descriptor builder."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from vector_db.r_table_strategy_selector import (
    CONFIG_SCHEMA_VERSION,
    DEFAULT_CONFIG_PATH,
    is_date_cell,
    is_numeric_cell,
)


PATH_SEPARATOR = " > "
_PERCENT_PATTERN = re.compile(
    r"^[\(\[]?[+-△▲▼]?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"\s*(?:%|퍼센트)\s*[\)\]]?$",
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"^(?:19|20|21)\d{2}$")


class RTableColumnRole(StrEnum):
    DIMENSION = "dimension"
    MEASURE = "measure"
    DATE = "date"
    IDENTIFIER = "identifier"
    LONG_TEXT = "long_text"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class RTableColumnProfilingConfig:
    measure_numeric_ratio_threshold: float
    date_ratio_threshold: float
    long_text_avg_length_threshold: int
    identifier_unique_ratio_threshold: float
    identifier_min_unique_count: int
    measure_header_keywords: tuple[str, ...]
    dimension_header_keywords: tuple[str, ...]
    date_header_keywords: tuple[str, ...]
    identifier_header_keywords: tuple[str, ...]
    max_distinct_values_per_column: int
    max_total_distinct_values: int

    def __post_init__(self) -> None:
        for name, value in (
            ("measure_numeric_ratio_threshold", self.measure_numeric_ratio_threshold),
            ("date_ratio_threshold", self.date_ratio_threshold),
            ("identifier_unique_ratio_threshold", self.identifier_unique_ratio_threshold),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0 <= float(value) <= 1
            ):
                raise ValueError(f"{name} must be between 0 and 1")
        for name, value in (
            ("long_text_avg_length_threshold", self.long_text_avg_length_threshold),
            ("identifier_min_unique_count", self.identifier_min_unique_count),
            ("max_distinct_values_per_column", self.max_distinct_values_per_column),
            ("max_total_distinct_values", self.max_total_distinct_values),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, values in (
            ("measure_header_keywords", self.measure_header_keywords),
            ("dimension_header_keywords", self.dimension_header_keywords),
            ("date_header_keywords", self.date_header_keywords),
            ("identifier_header_keywords", self.identifier_header_keywords),
        ):
            if not isinstance(values, tuple) or not values or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                raise ValueError(f"{name} must be a non-empty tuple of strings")


@dataclass(frozen=True, slots=True)
class RTableColumnProfile:
    column_index: int
    header_path: tuple[str, ...]
    role: RTableColumnRole
    non_empty_count: int
    unique_count: int
    unique_ratio: float
    numeric_ratio: float
    date_ratio: float
    percent_ratio: float
    avg_text_length: float
    max_text_length: int
    descriptor_values: tuple[str, ...]
    values_omitted_reason: str | None

    @property
    def header_text(self) -> str:
        return PATH_SEPARATOR.join(self.header_path)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["role"] = self.role.value
        result["header_text"] = self.header_text
        return result


@dataclass(frozen=True, slots=True)
class RTableColumnAnalysis:
    table_id: str
    record_count: int
    columns: tuple[RTableColumnProfile, ...]

    @property
    def dimension_columns(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            column.header_path
            for column in self.columns
            if column.role is RTableColumnRole.DIMENSION
        )

    @property
    def identifier_columns(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            column.header_path
            for column in self.columns
            if column.role is RTableColumnRole.IDENTIFIER
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "record_count": self.record_count,
            "columns": [column.to_dict() for column in self.columns],
            "dimension_columns": [list(path) for path in self.dimension_columns],
            "identifier_columns": [list(path) for path in self.identifier_columns],
        }


@dataclass(frozen=True, slots=True)
class _ColumnDraft:
    column_index: int
    header_path: tuple[str, ...]
    role: RTableColumnRole
    non_empty_count: int
    unique_count: int
    unique_ratio: float
    numeric_ratio: float
    date_ratio: float
    percent_ratio: float
    avg_text_length: float
    max_text_length: int
    distinct_values: tuple[str, ...]


def _string_tuple(values: Any, field: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not values or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise ValueError(f"{field} must be a non-empty list of strings")
    return tuple(value.strip() for value in values)


def load_r_table_column_profiling_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> RTableColumnProfilingConfig:
    """Load column profiling and descriptor-value limits from YAML."""
    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Failed to load R_TABLE profiling config: {config_path}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError("R_TABLE profiling config must be a mapping")
    if loaded.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"R_TABLE profiling config schema_version must be {CONFIG_SCHEMA_VERSION}"
        )

    profiling = loaded.get("column_profiling")
    descriptor = loaded.get("descriptor")
    if not isinstance(profiling, Mapping):
        raise ValueError("column_profiling config must be a mapping")
    if not isinstance(descriptor, Mapping):
        raise ValueError("descriptor config must be a mapping")
    profiling_fields = {
        "measure_numeric_ratio_threshold",
        "date_ratio_threshold",
        "long_text_avg_length_threshold",
        "identifier_unique_ratio_threshold",
        "identifier_min_unique_count",
        "measure_header_keywords",
        "dimension_header_keywords",
        "date_header_keywords",
        "identifier_header_keywords",
    }
    descriptor_fields = {
        "max_distinct_values_per_column",
        "max_total_distinct_values",
    }
    if set(profiling) != profiling_fields:
        raise ValueError(
            "column_profiling config fields must be exactly: "
            + ", ".join(sorted(profiling_fields))
        )
    if set(descriptor) != descriptor_fields:
        raise ValueError(
            "descriptor config fields must be exactly: "
            + ", ".join(sorted(descriptor_fields))
        )
    return RTableColumnProfilingConfig(
        measure_numeric_ratio_threshold=profiling[
            "measure_numeric_ratio_threshold"
        ],
        date_ratio_threshold=profiling["date_ratio_threshold"],
        long_text_avg_length_threshold=profiling[
            "long_text_avg_length_threshold"
        ],
        identifier_unique_ratio_threshold=profiling[
            "identifier_unique_ratio_threshold"
        ],
        identifier_min_unique_count=profiling["identifier_min_unique_count"],
        measure_header_keywords=_string_tuple(
            profiling["measure_header_keywords"], "measure_header_keywords"
        ),
        dimension_header_keywords=_string_tuple(
            profiling["dimension_header_keywords"], "dimension_header_keywords"
        ),
        date_header_keywords=_string_tuple(
            profiling["date_header_keywords"], "date_header_keywords"
        ),
        identifier_header_keywords=_string_tuple(
            profiling["identifier_header_keywords"], "identifier_header_keywords"
        ),
        max_distinct_values_per_column=descriptor[
            "max_distinct_values_per_column"
        ],
        max_total_distinct_values=descriptor["max_total_distinct_values"],
    )


def _contains_keyword(header_text: str, keywords: tuple[str, ...]) -> bool:
    normalized_header = header_text.casefold()
    return any(keyword.casefold() in normalized_header for keyword in keywords)


def _is_percent(value: str) -> bool:
    return _PERCENT_PATTERN.fullmatch(value) is not None


def _classify_role(
    *,
    header_text: str,
    non_empty_count: int,
    unique_count: int,
    unique_ratio: float,
    numeric_ratio: float,
    date_ratio: float,
    year_ratio: float,
    avg_text_length: float,
    config: RTableColumnProfilingConfig,
) -> RTableColumnRole:
    if non_empty_count == 0:
        return RTableColumnRole.OTHER
    identifier_header = _contains_keyword(
        header_text, config.identifier_header_keywords
    )
    if (
        identifier_header
        and unique_ratio >= config.identifier_unique_ratio_threshold / 2
        and unique_count >= max(2, (config.identifier_min_unique_count + 1) // 2)
    ):
        return RTableColumnRole.IDENTIFIER
    date_header = _contains_keyword(header_text, config.date_header_keywords)
    if date_ratio >= config.date_ratio_threshold or (
        date_header
        and max(date_ratio, year_ratio) >= config.date_ratio_threshold / 2
    ):
        return RTableColumnRole.DATE
    measure_header = _contains_keyword(header_text, config.measure_header_keywords)
    if numeric_ratio >= config.measure_numeric_ratio_threshold or (
        measure_header
        and numeric_ratio >= config.measure_numeric_ratio_threshold / 2
    ):
        return RTableColumnRole.MEASURE
    if avg_text_length >= config.long_text_avg_length_threshold:
        return RTableColumnRole.LONG_TEXT
    if _contains_keyword(header_text, config.dimension_header_keywords):
        return RTableColumnRole.DIMENSION
    if (
        unique_ratio >= config.identifier_unique_ratio_threshold
        and unique_count >= config.identifier_min_unique_count
    ):
        return RTableColumnRole.IDENTIFIER
    return RTableColumnRole.DIMENSION


def _validated_columns(
    evidence: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> tuple[str, int, list[tuple[str, ...]], list[list[str]]]:
    if evidence.get("evidence_type") != "TABLE":
        raise ValueError("Expected evidence_type to be TABLE")
    if evidence.get("table_type") != "R_TABLE":
        raise ValueError("Expected table_type to be R_TABLE")
    payload = evidence.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("R_TABLE Evidence payload must be a mapping")
    table_id = payload.get("table_id")
    if not isinstance(table_id, str) or not table_id.strip():
        raise ValueError("R_TABLE payload.table_id must be a non-empty string")
    headers = payload.get("headers")
    if not isinstance(headers, list) or not all(
        isinstance(header, list)
        and all(isinstance(part, str) for part in header)
        for header in headers
    ):
        raise ValueError("R_TABLE payload.headers must be a list of string lists")
    header_paths = [
        tuple(part.strip() for part in header if part.strip()) for header in headers
    ]

    record_count = payload.get("record_count")
    if (
        not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or record_count < 0
    ):
        raise ValueError("R_TABLE payload.record_count must be a non-negative integer")
    if not isinstance(records, (list, tuple)):
        raise ValueError("R_TABLE records must be a sequence")
    if record_count != len(records):
        raise ValueError("R_TABLE payload.record_count must match the records length")

    record_indexes: set[int] = set()
    indexed_values: list[tuple[int, list[str | None]]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Each R_TABLE record must be a mapping")
        if record.get("table_id") != table_id:
            raise ValueError("R_TABLE Evidence and record table_id must match")
        record_index = record.get("record_index")
        if (
            not isinstance(record_index, int)
            or isinstance(record_index, bool)
            or record_index < 0
        ):
            raise ValueError("R_TABLE record_index must be a non-negative integer")
        if record_index in record_indexes:
            raise ValueError("R_TABLE record_index must be unique within a table")
        record_indexes.add(record_index)
        values = record.get("values")
        if not isinstance(values, list) or not all(
            value is None or isinstance(value, str) for value in values
        ):
            raise ValueError("R_TABLE record.values must be a list of strings or nulls")
        if len(values) != len(headers):
            raise ValueError("R_TABLE headers and values must have the same length")
        indexed_values.append((record_index, values))

    columns: list[list[str]] = [[] for _ in headers]
    for _, values in sorted(indexed_values, key=lambda item: item[0]):
        for column, value in zip(columns, values, strict=True):
            if isinstance(value, str) and value.strip():
                column.append(value.strip())
    return table_id.strip(), record_count, header_paths, columns


def _profile_draft(
    column_index: int,
    header_path: tuple[str, ...],
    values: list[str],
    config: RTableColumnProfilingConfig,
) -> _ColumnDraft:
    distinct_values = tuple(dict.fromkeys(values))
    non_empty_count = len(values)
    unique_count = len(distinct_values)
    denominator = non_empty_count or 1
    lengths = [len(value) for value in values]
    numeric_ratio = sum(is_numeric_cell(value) for value in values) / denominator
    date_ratio = sum(is_date_cell(value) for value in values) / denominator
    year_ratio = (
        sum(_YEAR_PATTERN.fullmatch(value) is not None for value in values)
        / denominator
    )
    percent_ratio = sum(_is_percent(value) for value in values) / denominator
    avg_text_length = sum(lengths) / denominator
    max_text_length = max(lengths, default=0)
    header_text = PATH_SEPARATOR.join(header_path)
    return _ColumnDraft(
        column_index=column_index,
        header_path=header_path,
        role=_classify_role(
            header_text=header_text,
            non_empty_count=non_empty_count,
            unique_count=unique_count,
            unique_ratio=unique_count / denominator,
            numeric_ratio=numeric_ratio,
            date_ratio=date_ratio,
            year_ratio=year_ratio,
            avg_text_length=avg_text_length,
            config=config,
        ),
        non_empty_count=non_empty_count,
        unique_count=unique_count,
        unique_ratio=unique_count / denominator,
        numeric_ratio=numeric_ratio,
        date_ratio=date_ratio,
        percent_ratio=percent_ratio,
        avg_text_length=avg_text_length,
        max_text_length=max_text_length,
        distinct_values=distinct_values,
    )


def profile_r_table_columns(
    evidence: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    config: RTableColumnProfilingConfig | None = None,
) -> RTableColumnAnalysis:
    """Classify columns and select bounded values for future descriptors."""
    resolved_config = config or load_r_table_column_profiling_config()
    table_id, record_count, header_paths, columns = _validated_columns(
        evidence, records
    )
    drafts = [
        _profile_draft(index, header_path, values, resolved_config)
        for index, (header_path, values) in enumerate(
            zip(header_paths, columns, strict=True)
        )
    ]

    remaining_values = resolved_config.max_total_distinct_values
    profiles: list[RTableColumnProfile] = []
    for draft in drafts:
        descriptor_values: tuple[str, ...] = ()
        omitted_reason: str | None = None
        if draft.role is RTableColumnRole.DIMENSION:
            if (
                draft.unique_count
                > resolved_config.max_distinct_values_per_column
            ):
                omitted_reason = "high_cardinality"
            elif remaining_values <= 0:
                omitted_reason = "table_value_limit"
            else:
                descriptor_values = draft.distinct_values[:remaining_values]
                remaining_values -= len(descriptor_values)
                if len(descriptor_values) < draft.unique_count:
                    omitted_reason = "table_value_limit"
        profiles.append(
            RTableColumnProfile(
                column_index=draft.column_index,
                header_path=draft.header_path,
                role=draft.role,
                non_empty_count=draft.non_empty_count,
                unique_count=draft.unique_count,
                unique_ratio=draft.unique_ratio,
                numeric_ratio=draft.numeric_ratio,
                date_ratio=draft.date_ratio,
                percent_ratio=draft.percent_ratio,
                avg_text_length=draft.avg_text_length,
                max_text_length=draft.max_text_length,
                descriptor_values=descriptor_values,
                values_omitted_reason=omitted_reason,
            )
        )
    return RTableColumnAnalysis(
        table_id=table_id,
        record_count=record_count,
        columns=tuple(profiles),
    )


__all__ = [
    "RTableColumnAnalysis",
    "RTableColumnProfile",
    "RTableColumnProfilingConfig",
    "RTableColumnRole",
    "load_r_table_column_profiling_config",
    "profile_r_table_columns",
]
