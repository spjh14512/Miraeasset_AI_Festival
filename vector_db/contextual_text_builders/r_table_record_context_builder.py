"""Build embedding text for an R_TABLE Evidence item and its records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vector_db.contextual_text_builders.r_table_header_formatter import (
    grouped_header_lines,
    record_header_keys,
)


PATH_SEPARATOR = " > "
BLOCK_SEPARATOR = "\n\n"
SUPPORTED_ROW_TYPES = {"DATA", "SUBTOTAL", "TOTAL", "UNKNOWN"}


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


def _column_lines(headers: Any, values: Any) -> list[str]:
    keys = record_header_keys(headers)
    if not isinstance(values, list) or not all(
        value is None or isinstance(value, str) for value in values
    ):
        raise ValueError("R_TABLE record.values must be a list of strings or nulls")
    if len(headers) != len(values):
        raise ValueError("R_TABLE headers and values must have the same length")

    lines: list[str] = []
    for key, value in zip(keys, values, strict=True):
        if value is None or value == "":
            continue
        lines.append(f"{key} : {value}")
    return lines


def _validated_payload(evidence: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    if evidence.get("evidence_type") != "TABLE":
        raise ValueError("Expected evidence_type to be TABLE")
    if evidence.get("table_type") != "R_TABLE":
        raise ValueError("Expected table_type to be R_TABLE")

    payload = evidence.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("R_TABLE Evidence payload must be a mapping")
    table_id = payload.get("table_id")
    if not isinstance(table_id, str) or not table_id:
        raise ValueError("R_TABLE payload.table_id must be a non-empty string")
    return payload, table_id


def _context_blocks(
    payload: Mapping[str, Any],
    *,
    corp_name: str,
    report_nm: str,
    section_path: Sequence[str],
    include_headers: bool,
) -> tuple[list[str], list[str]]:
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
    table_lines.extend(
        f"단위 : {value}" for value in _optional_text_list(payload, "units")
    )
    table_lines.extend(
        f"주석 : {value}" for value in _optional_text_list(payload, "notes")
    )
    if include_headers:
        header_lines = grouped_header_lines(payload.get("headers"))
        if header_lines:
            table_lines.extend(["컬럼 구조 :", *header_lines])
    return document_lines, table_lines


def _record_lines(
    payload: Mapping[str, Any],
    table_id: str,
    record: Mapping[str, Any],
) -> list[str]:
    if record.get("table_id") != table_id:
        raise ValueError("R_TABLE Evidence and record table_id must match")

    row_type = record.get("row_type")
    if row_type not in SUPPORTED_ROW_TYPES:
        raise ValueError("R_TABLE record.row_type is not supported")
    row_context = record.get("row_context")
    if not isinstance(row_context, list) or not all(
        isinstance(value, str) for value in row_context
    ):
        raise ValueError("R_TABLE record.row_context must be a list of strings")

    lines: list[str] = []
    if row_type != "DATA":
        lines.append(f"행 유형 : {row_type}")
    cleaned_row_context = _clean_path(row_context)
    if cleaned_row_context:
        lines.append(f"행 문맥 : {PATH_SEPARATOR.join(cleaned_row_context)}")
    lines.extend(_column_lines(payload.get("headers"), record.get("values")))
    return lines


def build_r_table_record_contextual_text(
    evidence: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    corp_name: str,
    report_nm: str,
    section_path: Sequence[str],
) -> str:
    """Render one R_TABLE record as standalone embedding-ready text."""
    payload, table_id = _validated_payload(evidence)
    if not isinstance(record, Mapping):
        raise ValueError("R_TABLE record must be a mapping")
    document_lines, table_lines = _context_blocks(
        payload,
        corp_name=corp_name,
        report_nm=report_nm,
        section_path=section_path,
        include_headers=False,
    )
    row_lines = _record_lines(payload, table_id, record)
    blocks = [
        "\n".join(lines)
        for lines in (document_lines, table_lines, row_lines)
        if lines
    ]
    if not blocks:
        raise ValueError("R_TABLE record contextual text cannot be empty")
    return BLOCK_SEPARATOR.join(blocks)


def build_r_table_contextual_text(
    evidence: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    corp_name: str,
    report_nm: str,
    section_path: Sequence[str],
) -> str:
    """Render one complete R_TABLE as one embedding-ready text."""
    payload, table_id = _validated_payload(evidence)
    if not isinstance(records, (list, tuple)):
        raise ValueError("R_TABLE records must be a sequence")

    indexed_records: list[tuple[int, Mapping[str, Any]]] = []
    record_indexes: set[int] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Each R_TABLE record must be a mapping")
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
        indexed_records.append((record_index, record))

    record_count = payload.get("record_count")
    if (
        not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or record_count < 0
    ):
        raise ValueError("R_TABLE payload.record_count must be a non-negative integer")
    if record_count != len(indexed_records):
        raise ValueError("R_TABLE payload.record_count must match the records length")

    document_lines, table_lines = _context_blocks(
        payload,
        corp_name=corp_name,
        report_nm=report_nm,
        section_path=section_path,
        include_headers=True,
    )
    record_blocks = [
        _record_lines(payload, table_id, record)
        for _, record in sorted(indexed_records, key=lambda item: item[0])
    ]
    blocks = [
        "\n".join(lines)
        for lines in (document_lines, table_lines, *record_blocks)
        if lines
    ]
    if not blocks:
        raise ValueError("R_TABLE contextual text cannot be empty")
    return BLOCK_SEPARATOR.join(blocks)


__all__ = [
    "build_r_table_contextual_text",
    "build_r_table_record_contextual_text",
]
