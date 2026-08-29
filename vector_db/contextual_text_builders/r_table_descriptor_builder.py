"""Build retrieval text for a structured oversized R_TABLE."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vector_db.r_table_column_profiler import (
    RTableColumnAnalysis,
    RTableColumnRole,
)
from vector_db.contextual_text_builders.r_table_header_formatter import (
    PATH_SEPARATOR,
    grouped_header_lines,
    normalize_header_paths,
)


BLOCK_SEPARATOR = "\n\n"


def _clean_path(*groups: Sequence[str]) -> list[str]:
    values: list[str] = []
    for group in groups:
        for raw_value in group:
            if not isinstance(raw_value, str):
                raise ValueError("Context path values must be strings")
            value = raw_value.strip()
            if value and (not values or values[-1] != value):
                values.append(value)
    return values


def _optional_text(payload: Mapping[str, Any], field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"R_TABLE payload.{field} must be a string")
    return value.strip() or None


def _optional_text_list(payload: Mapping[str, Any], field: str) -> list[str]:
    value = payload.get(field)
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"R_TABLE payload.{field} must be a sequence of strings")
    return [item.strip() for item in value if item.strip()]


def _validated_payload(
    evidence: Mapping[str, Any],
    analysis: RTableColumnAnalysis,
) -> Mapping[str, Any]:
    if evidence.get("evidence_type") != "TABLE":
        raise ValueError("Expected evidence_type to be TABLE")
    if evidence.get("table_type") != "R_TABLE":
        raise ValueError("Expected table_type to be R_TABLE")
    if not isinstance(analysis, RTableColumnAnalysis):
        raise ValueError("analysis must be an RTableColumnAnalysis")

    payload = evidence.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("R_TABLE Evidence payload must be a mapping")
    table_id = payload.get("table_id")
    if not isinstance(table_id, str) or not table_id.strip():
        raise ValueError("R_TABLE payload.table_id must be a non-empty string")
    if analysis.table_id != table_id.strip():
        raise ValueError("R_TABLE Evidence and column analysis table_id must match")

    record_count = payload.get("record_count")
    if (
        not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or record_count < 0
    ):
        raise ValueError("R_TABLE payload.record_count must be a non-negative integer")
    if analysis.record_count != record_count:
        raise ValueError(
            "R_TABLE Evidence and column analysis record_count must match"
        )

    headers = payload.get("headers")
    if not isinstance(headers, list) or not all(
        isinstance(header, list)
        and all(isinstance(part, str) for part in header)
        for header in headers
    ):
        raise ValueError("R_TABLE payload.headers must be a list of string lists")
    expected_paths = [
        tuple(part.strip() for part in header if part.strip()) for header in headers
    ]
    if len(analysis.columns) != len(expected_paths):
        raise ValueError("R_TABLE Evidence and column analysis column count must match")
    for index, (column, expected_path) in enumerate(
        zip(analysis.columns, expected_paths, strict=True)
    ):
        if column.column_index != index:
            raise ValueError("R_TABLE column analysis indexes must be ordered")
        if column.header_path != expected_path:
            raise ValueError(
                "R_TABLE Evidence and column analysis header paths must match"
            )
    return payload


def build_r_table_descriptor_contextual_text(
    evidence: Mapping[str, Any],
    analysis: RTableColumnAnalysis,
    *,
    corp_name: str,
    report_nm: str,
    section_path: Sequence[str],
) -> str:
    """Render a descriptor without embedding record values or generated prose."""
    if not isinstance(evidence, Mapping):
        raise ValueError("evidence must be a mapping")
    if not isinstance(corp_name, str):
        raise ValueError("corp_name must be a string")
    if not isinstance(report_nm, str):
        raise ValueError("report_nm must be a string")
    if not isinstance(section_path, (list, tuple)):
        raise ValueError("section_path must be a sequence")

    payload = _validated_payload(evidence, analysis)
    heading_path = payload.get("heading_path", [])
    if not isinstance(heading_path, (list, tuple)):
        raise ValueError("R_TABLE payload.heading_path must be a sequence")
    combined_section_path = _clean_path(section_path, heading_path)

    document_lines: list[str] = []
    if corp_name.strip():
        document_lines.append(f"회사 : {corp_name.strip()}")
    if report_nm.strip():
        document_lines.append(f"공시 : {report_nm.strip()}")
    if combined_section_path:
        document_lines.append(f"섹션 : {PATH_SEPARATOR.join(combined_section_path)}")

    table_lines: list[str] = []
    title = _optional_text(payload, "title")
    if title:
        table_lines.append(f"표 제목 : {title}")
    table_lines.extend(
        f"표 설명 : {value}" for value in _optional_text_list(payload, "captions")
    )
    units = _optional_text_list(payload, "units")
    table_lines.extend(f"단위 : {value}" for value in units)
    table_lines.extend(
        f"주석 : {value}" for value in _optional_text_list(payload, "notes")
    )

    normalized_paths = normalize_header_paths(payload.get("headers"))
    header_lines = grouped_header_lines(payload.get("headers"))
    column_lines = ["컬럼 구조 :", *header_lines] if header_lines else []

    dimension_values: dict[str, list[str]] = {}
    for column, path in zip(analysis.columns, normalized_paths, strict=True):
        if column.role is not RTableColumnRole.DIMENSION or not path:
            continue
        key = PATH_SEPARATOR.join(path)
        values = dimension_values.setdefault(key, [])
        for value in column.descriptor_values:
            if value not in values:
                values.append(value)
    dimension_lines = [
        f"{key} : {', '.join(values)}"
        for key, values in dimension_values.items()
        if values
    ]
    item_lines = ["행 항목 :", *dimension_lines] if dimension_lines else []

    search_terms: list[str] = []

    def append_search_term(value: str) -> None:
        cleaned = value.strip()
        if cleaned and cleaned not in search_terms:
            search_terms.append(cleaned)

    if title:
        append_search_term(title)
    for path in normalized_paths:
        for part in path:
            append_search_term(part)
    for values in dimension_values.values():
        for value in values:
            append_search_term(value)
    for unit in units:
        append_search_term(unit)
    search_lines = (
        ["검색어 :", " | ".join(search_terms)] if search_terms else []
    )

    blocks = [
        "\n".join(lines)
        for lines in (
            document_lines,
            table_lines,
            column_lines,
            item_lines,
            search_lines,
        )
        if lines
    ]
    if not blocks:
        raise ValueError("R_TABLE descriptor contextual text cannot be empty")
    return BLOCK_SEPARATOR.join(blocks)


__all__ = ["build_r_table_descriptor_contextual_text"]
