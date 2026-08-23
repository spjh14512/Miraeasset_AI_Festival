"""Build embedding text for a KV_TABLE Evidence item."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


PATH_SEPARATOR = " > "
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
        raise ValueError(f"KV_TABLE payload.{field} must be a string")
    return value.strip() or None


def _optional_text_list(payload: Mapping[str, Any], field: str) -> list[str]:
    value = payload.get(field)
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"KV_TABLE payload.{field} must be a sequence of strings")
    return [item.strip() for item in value if item.strip()]


def _entry_lines(fields: Any) -> list[str]:
    if not isinstance(fields, list):
        raise ValueError("KV_TABLE payload.fields must be a list")

    lines: list[str] = []
    for field_index, field in enumerate(fields):
        if not isinstance(field, Mapping):
            raise ValueError(f"KV_TABLE payload.fields[{field_index}] must be a mapping")

        key_paths = field.get("key_paths")
        if not isinstance(key_paths, list):
            raise ValueError(
                f"KV_TABLE payload.fields[{field_index}].key_paths must be a list"
            )
        raw_value = field.get("raw_value")
        if not isinstance(raw_value, str):
            raise ValueError(
                f"KV_TABLE payload.fields[{field_index}].raw_value must be a string"
            )

        for path_index, key_path in enumerate(key_paths):
            if not isinstance(key_path, list) or not all(
                isinstance(part, str) for part in key_path
            ):
                raise ValueError(
                    "KV_TABLE payload.fields"
                    f"[{field_index}].key_paths[{path_index}] must be a list of strings"
                )
            key = PATH_SEPARATOR.join(part.strip() for part in key_path)
            lines.append(f"{key} : {raw_value}")
    return lines


def build_kv_contextual_text(
    evidence: Mapping[str, Any],
    *,
    corp_name: str,
    report_nm: str,
    section_path: Sequence[str],
) -> str:
    """Render one complete KV_TABLE Evidence as embedding-ready text."""
    if evidence.get("evidence_type") != "TABLE":
        raise ValueError("Expected evidence_type to be TABLE")
    if evidence.get("table_type") != "KV_TABLE":
        raise ValueError("Expected table_type to be KV_TABLE")

    payload = evidence.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("KV_TABLE Evidence payload must be a mapping")

    heading_path = payload.get("heading_path", [])
    if not isinstance(heading_path, (list, tuple)):
        raise ValueError("KV_TABLE payload.heading_path must be a sequence")
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

    blocks = [
        "\n".join(lines)
        for lines in (document_lines, table_lines, _entry_lines(payload.get("fields")))
        if lines
    ]
    if not blocks:
        raise ValueError("KV_TABLE contextual text cannot be empty")
    return BLOCK_SEPARATOR.join(blocks)


__all__ = ["build_kv_contextual_text"]
