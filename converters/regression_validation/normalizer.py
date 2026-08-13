"""Stable snapshots for converter regression cases."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from typing import Any, Mapping


_VOLATILE_KEYS = {"section_id", "evidence_id", "table_id"}
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,:;!?])")
_ALL_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Normalize insignificant whitespace without changing Korean text content."""
    collapsed = " ".join(value.split())
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", collapsed)


def normalize_value(value: Any, *, semantic_text: bool = False) -> Any:
    """Remove generated identifiers and recursively normalize JSON-compatible data."""
    if isinstance(value, Mapping):
        return {
            key: normalize_value(item, semantic_text=semantic_text)
            for key, item in value.items()
            if key not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [normalize_value(item, semantic_text=semantic_text) for item in value]
    if isinstance(value, str) and semantic_text:
        return normalize_text(value)
    return value


def normalize_duplicate_value(value: Any) -> Any:
    """Normalize source-copy whitespace when comparing src0/srcN duplicates."""
    normalized = normalize_value(value, semantic_text=True)

    def compact(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: compact(child) for key, child in item.items()}
        if isinstance(item, list):
            return [compact(child) for child in item]
        if isinstance(item, str):
            return _ALL_WHITESPACE.sub("", item)
        return item

    return compact(normalized)


def stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fragment_summary(fragment: Mapping[str, Any]) -> dict[str, Any]:
    evidence = list(fragment.get("evidence_list", []))
    records = list(fragment.get("records", []))
    types = Counter(
        (
            str(item.get("evidence_type")),
            str(item.get("table_type")) if item.get("table_type") else None,
        )
        for item in evidence
    )
    return {
        "evidence_count": len(evidence),
        "record_count": len(records),
        "type_counts": [
            {
                "evidence_type": evidence_type,
                "table_type": table_type,
                "count": count,
            }
            for (evidence_type, table_type), count in sorted(
                types.items(), key=lambda item: str(item[0])
            )
        ],
        "normalized_sha256": stable_sha256(normalize_value(fragment)),
        "semantic_sha256": stable_sha256(
            normalize_value(fragment, semantic_text=True)
        ),
    }


def case_snapshot(
    case: Mapping[str, Any],
    fragments: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a compact but sensitive snapshot for one configured case."""
    section_ids = [str(item) for item in case.get("section_ids", [])]
    snapshots: list[dict[str, Any]] = []
    probe_indexes = sorted(
        {
            int(assertion["record_index"])
            for assertion in case.get("assertions", [])
            if "record_index" in assertion
        }
    )
    for section_id in section_ids:
        fragment = fragments[section_id]
        evidence = list(fragment.get("evidence_list", []))
        records = list(fragment.get("records", []))
        indexes = set(probe_indexes)
        if records:
            indexes.update({0, len(records) // 2, len(records) - 1})
        snapshot = {
            "section_id": section_id,
            "summary": fragment_summary(fragment),
            "evidence_list": normalize_value(evidence),
            "record_probes": [
                normalize_value(records[index])
                for index in sorted(indexes)
                if 0 <= index < len(records)
            ],
        }
        if len(records) <= 100:
            snapshot["records"] = normalize_value(records)
        else:
            snapshot["records_sha256"] = stable_sha256(normalize_value(records))
        snapshots.append(snapshot)
    return {
        "schema_version": "converter-regression-golden.v1",
        "case_id": str(case["case_id"]),
        "tags": list(case.get("tags", [])),
        "normalizer_version": "v1",
        "source_sha256": list(case.get("source_sha256", [])),
        "sections": snapshots,
    }


__all__ = [
    "case_snapshot",
    "fragment_summary",
    "normalize_text",
    "normalize_duplicate_value",
    "normalize_value",
    "stable_sha256",
]
