"""Build strategy-routed Qdrant points from one Evidence Fragment."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any
from uuid import UUID, uuid5

from vector_db.contextual_text_builders import (
    build_kv_contextual_text,
    build_r_table_contextual_text,
    build_r_table_descriptor_contextual_text,
    build_r_table_row_groups,
    build_text_contextual_text,
)
from vector_db.r_table_column_profiler import profile_r_table_columns
from vector_db.r_table_strategy_selector import (
    RTableEmbeddingStrategy,
    RTableEmbeddingStrategySelector,
)
from vector_db.text2vector import HybridEmbedding, text_to_hybrid_vector


POINT_ID_NAMESPACE = UUID("9220d409-a029-5497-8ab2-fc39b239ab18")
VECTOR_NAME = "evidence_dense"
SPARSE_VECTOR_NAME = "evidence_sparse"
POINT_KINDS = {"TEXT", "KV_TABLE", "R_TABLE"}
TABLE_METADATA_FIELDS = ("title", "captions", "units", "notes")
PATH_SEPARATOR = " > "

Vectorizer = Callable[[str], HybridEmbedding]


@dataclass(frozen=True, slots=True)
class PointInput:
    """Embedding-independent input used to assemble one Qdrant point."""

    id: str
    contextual_text: str
    embedding_cache_key: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _RTableCanonicalChunk:
    records: tuple[Mapping[str, Any], ...]
    canonical: dict[str, Any]
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _RTablePointPart:
    identity_key: str
    records: tuple[Mapping[str, Any], ...]
    canonical: dict[str, Any]
    contextual_text: str
    row_start_index: int | None
    row_end_index: int | None


def to_neo4j_evidence_id(source_evidence_id: str) -> str:
    """Convert a Canonical Evidence ID to the Neo4j Evidence node ID."""
    if not isinstance(source_evidence_id, str) or not source_evidence_id.strip():
        raise ValueError("source_evidence_id must be a non-empty string")
    source_evidence_id = source_evidence_id.strip()
    prefix = "evidence:"
    if not source_evidence_id.startswith(prefix):
        raise ValueError("source_evidence_id must start with evidence:")
    suffix = source_evidence_id[len(prefix) :]
    if not suffix:
        raise ValueError("source_evidence_id must include an Evidence identifier")
    return f"d{suffix}"


def _evidence_parent_ids(evidence_id: str) -> tuple[str, str]:
    parts = evidence_id.split(":")
    if (
        len(parts) != 4
        or len(parts[0]) != 15
        or not parts[0].startswith("d")
        or not parts[0][1:].isdigit()
        or not parts[1].startswith("src")
        or not parts[1][3:].isdigit()
        or not parts[2].startswith("s")
        or not parts[2][1:].isdigit()
        or not parts[3].startswith("e")
        or not parts[3][1:].isdigit()
    ):
        raise ValueError("evidence_id must use the Neo4j Evidence ID format")
    return parts[0], ":".join(parts[:3])


def _required_text(source: Mapping[str, Any], field: str) -> str:
    value = source.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(source: Mapping[str, Any], field: str) -> str | None:
    value = source.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return value.strip() or None


def _optional_integer(source: Mapping[str, Any], field: str) -> int | None:
    value = source.get(field)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer or null")
    return value


def _section_path(section_context: Mapping[str, Any]) -> list[str]:
    value = section_context.get("section_path")
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(part, str) and part.strip() for part in value
    ):
        raise ValueError("section_path must be a sequence of non-empty strings")
    return [part.strip() for part in value]


def _section_name(
    evidence: Mapping[str, Any],
    section_path: Sequence[str],
) -> str:
    payload = evidence.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Evidence payload must be a mapping")
    heading_path = payload.get("heading_path", [])
    if not isinstance(heading_path, (list, tuple)) or not all(
        isinstance(part, str) for part in heading_path
    ):
        raise ValueError("Evidence payload.heading_path must be a sequence of strings")

    combined: list[str] = []
    for part in (*section_path, *heading_path):
        cleaned = part.strip()
        if cleaned and (not combined or combined[-1] != cleaned):
            combined.append(cleaned)
    if not combined:
        raise ValueError("Combined section and heading path must not be empty")
    return PATH_SEPARATOR.join(combined)


def _table_metadata(payload: Mapping[str, Any], *, table_type: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    title = payload.get("title")
    if title is not None:
        if not isinstance(title, str):
            raise ValueError(f"{table_type} payload.title must be a string")
        if title.strip():
            metadata["title"] = title

    for field in TABLE_METADATA_FIELDS[1:]:
        values = payload.get(field)
        if values is None:
            continue
        if not isinstance(values, (list, tuple)) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(
                f"{table_type} payload.{field} must be a sequence of strings"
            )
        non_empty_values = [value for value in values if value.strip()]
        if non_empty_values:
            metadata[field] = non_empty_values
    return metadata


def _text_canonical(evidence: Mapping[str, Any]) -> dict[str, Any]:
    payload = evidence.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("TEXT Evidence payload must be a mapping")
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("TEXT payload.text must be a string")
    return {"text": text}


def _kv_table_canonical(evidence: Mapping[str, Any]) -> dict[str, Any]:
    payload = evidence.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("KV_TABLE Evidence payload must be a mapping")
    fields = payload.get("fields")
    if not isinstance(fields, list):
        raise ValueError("KV_TABLE payload.fields must be a list")

    entries: list[dict[str, str]] = []
    for field_index, field in enumerate(fields):
        if not isinstance(field, Mapping):
            raise ValueError(
                f"KV_TABLE payload.fields[{field_index}] must be a mapping"
            )
        key_paths = field.get("key_paths")
        if not isinstance(key_paths, list):
            raise ValueError(
                f"KV_TABLE payload.fields[{field_index}].key_paths must be a list"
            )
        value = field.get("raw_value")
        if not isinstance(value, str):
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
            entries.append(
                {
                    "key": PATH_SEPARATOR.join(part.strip() for part in key_path),
                    "value": value,
                }
            )

    canonical: dict[str, Any] = {}
    table_metadata = _table_metadata(payload, table_type="KV_TABLE")
    if table_metadata:
        canonical["table_metadata"] = table_metadata
    canonical["entries"] = entries
    return canonical


def _r_table_canonical(
    evidence: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = evidence.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("R_TABLE Evidence payload must be a mapping")
    headers = payload.get("headers")
    if not isinstance(headers, list) or not all(
        isinstance(header, list)
        and all(isinstance(part, str) for part in header)
        for header in headers
    ):
        raise ValueError("R_TABLE payload.headers must be a list of string lists")

    projected_records: list[tuple[int, dict[str, Any]]] = []
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
        values = record.get("values")
        if not isinstance(values, list) or not all(
            value is None or isinstance(value, str) for value in values
        ):
            raise ValueError("R_TABLE record.values must be a list of strings or nulls")
        projected_records.append(
            (
                record_index,
                {"record_index": record_index, "values": list(values)},
            )
        )

    canonical: dict[str, Any] = {}
    table_metadata = _table_metadata(payload, table_type="R_TABLE")
    if table_metadata:
        canonical["table_metadata"] = table_metadata
    canonical["headers"] = [list(header) for header in headers]
    canonical["records"] = [
        record for _, record in sorted(projected_records, key=lambda item: item[0])
    ]
    return canonical


def _compact_json_size_bytes(value: Mapping[str, Any]) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _r_table_subset_evidence(
    evidence: Mapping[str, Any],
    *,
    record_count: int,
) -> dict[str, Any]:
    payload = evidence.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("R_TABLE Evidence payload must be a mapping")
    return {
        **dict(evidence),
        "payload": {
            **dict(payload),
            "record_count": record_count,
        },
    }


def _minimum_chunk_counts_by_suffix(
    weights: Sequence[int],
    *,
    capacity: int,
) -> list[int]:
    """Return the minimum number of bounded contiguous chunks per suffix."""
    count = len(weights)
    next_indexes = [count] * count
    end = 0
    running_size = 0
    for start in range(count):
        if end < start:
            end = start
            running_size = 0
        while end < count and running_size + weights[end] <= capacity:
            running_size += weights[end]
            end += 1
        next_indexes[start] = end
        running_size -= weights[start]

    minimum_counts = [0] * (count + 1)
    for start in range(count - 1, -1, -1):
        minimum_counts[start] = 1 + minimum_counts[next_indexes[start]]
    return minimum_counts


def _balanced_bounded_groups(
    records: Sequence[Mapping[str, Any]],
    weights: Sequence[int],
    *,
    capacity: int,
) -> list[tuple[Mapping[str, Any], ...]]:
    """Use the minimum chunk count, choosing boundaries nearest remaining averages."""
    if not records:
        return []
    if len(records) != len(weights):
        raise ValueError("R_TABLE canonical record and size counts must match")
    if any(weight <= 0 or weight > capacity for weight in weights):
        raise ValueError("Balanced R_TABLE chunk weights must fit the capacity")

    minimum_counts = _minimum_chunk_counts_by_suffix(
        weights,
        capacity=capacity,
    )
    suffix_totals = [0] * (len(weights) + 1)
    for index in range(len(weights) - 1, -1, -1):
        suffix_totals[index] = suffix_totals[index + 1] + weights[index]
    remaining_groups = minimum_counts[0]
    start = 0
    groups: list[tuple[Mapping[str, Any], ...]] = []

    while remaining_groups > 1:
        remaining_total = suffix_totals[start]
        target_size = remaining_total / remaining_groups
        running_size = 0
        best_cut: int | None = None
        best_score: tuple[float, int] | None = None
        maximum_cut = len(records) - (remaining_groups - 1)

        for cut in range(start + 1, maximum_cut + 1):
            running_size += weights[cut - 1]
            if running_size > capacity:
                break
            if minimum_counts[cut] > remaining_groups - 1:
                continue
            score = (abs(running_size - target_size), -running_size)
            if best_score is None or score < best_score:
                best_score = score
                best_cut = cut

        if best_cut is None:
            raise ValueError("Unable to build balanced R_TABLE canonical chunks")
        groups.append(tuple(records[start:best_cut]))
        start = best_cut
        remaining_groups -= 1

    groups.append(tuple(records[start:]))
    return groups


def _chunk_r_table_records_by_canonical_size(
    evidence: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    max_bytes: int,
    hard_max_bytes: int,
) -> list[_RTableCanonicalChunk]:
    """Split records into balanced compact-canonical UTF-8 byte chunks."""
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("canonical_max_bytes must be a positive integer")
    if (
        not isinstance(hard_max_bytes, int)
        or isinstance(hard_max_bytes, bool)
        or hard_max_bytes < max_bytes
    ):
        raise ValueError(
            "canonical_hard_max_bytes must be an integer greater than or equal "
            "to canonical_max_bytes"
        )

    ordered_records = tuple(
        sorted(records, key=lambda record: int(record["record_index"]))
    )
    full_canonical = _r_table_canonical(evidence, ordered_records)
    full_size = _compact_json_size_bytes(full_canonical)
    if full_size <= max_bytes or not ordered_records:
        return [
            _RTableCanonicalChunk(
                records=ordered_records,
                canonical=full_canonical,
                size_bytes=full_size,
            )
        ]

    empty_canonical = {**full_canonical, "records": []}
    base_size = _compact_json_size_bytes(empty_canonical)
    if base_size > max_bytes:
        # Repeated table metadata and headers alone cannot fit the limit, so row
        # partitioning cannot produce a compliant chunk. Preserve the table
        # boundary only within the explicit indivisible hard limit.
        if full_size > hard_max_bytes:
            raise ValueError(
                "R_TABLE header-only canonical exceeds the hard byte limit"
            )
        return [
            _RTableCanonicalChunk(
                records=ordered_records,
                canonical=full_canonical,
                size_bytes=full_size,
            )
        ]

    projected_records = full_canonical["records"]
    # One comma is charged to every record weight. The fixed portion subtracts
    # one byte, yielding the exact compact JSON size for every non-empty chunk.
    capacity = max_bytes - base_size + 1
    record_weights = [
        _compact_json_size_bytes(record) + 1 for record in projected_records
    ]
    if (
        full_size <= hard_max_bytes
        and record_weights
        and all(weight > capacity for weight in record_weights)
    ):
        # Every record would become a slightly oversized singleton under the
        # target limit. Keep one table Point instead of producing redundant
        # header-heavy Points, while still respecting the hard limit.
        return [
            _RTableCanonicalChunk(
                records=ordered_records,
                canonical=full_canonical,
                size_bytes=full_size,
            )
        ]
    grouped_records: list[tuple[Mapping[str, Any], ...]] = []
    normal_records: list[Mapping[str, Any]] = []
    normal_weights: list[int] = []

    def append_balanced_normal_run() -> None:
        if not normal_records:
            return
        grouped_records.extend(
            _balanced_bounded_groups(
                normal_records,
                normal_weights,
                capacity=capacity,
            )
        )
        normal_records.clear()
        normal_weights.clear()

    for record, record_weight in zip(
        ordered_records,
        record_weights,
        strict=True,
    ):
        if record_weight > capacity:
            append_balanced_normal_run()
            grouped_records.append((record,))
            continue
        normal_records.append(record)
        normal_weights.append(record_weight)
    append_balanced_normal_run()

    chunks: list[_RTableCanonicalChunk] = []
    for grouped in grouped_records:
        canonical = _r_table_canonical(evidence, grouped)
        size_bytes = _compact_json_size_bytes(canonical)
        if size_bytes > hard_max_bytes:
            raise ValueError(
                "R_TABLE indivisible record exceeds the hard byte limit"
            )
        chunks.append(
            _RTableCanonicalChunk(
                records=grouped,
                canonical=canonical,
                size_bytes=size_bytes,
            )
        )
    return chunks


def _record_range(
    records: Sequence[Mapping[str, Any]],
) -> tuple[int | None, int | None]:
    if not records:
        return None, None
    indexes = sorted(int(record["record_index"]) for record in records)
    return indexes[0], indexes[-1]


def _point_payload(
    *,
    point_kind: str,
    evidence_id: str,
    contextual_text: str,
    canonical: Mapping[str, Any],
    document_context: Mapping[str, Any],
    section_name: str,
    table_id: str | None = None,
    row_start_index: int | None = None,
    row_end_index: int | None = None,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> dict[str, Any]:
    if point_kind not in POINT_KINDS:
        raise ValueError(f"Unsupported point_kind: {point_kind}")

    if not isinstance(section_name, str) or not section_name.strip():
        raise ValueError("section_name must be a non-empty string")

    chunk_values = (
        table_id,
        row_start_index,
        row_end_index,
        chunk_index,
        chunk_count,
    )
    has_chunking = any(value is not None for value in chunk_values)
    if has_chunking:
        if point_kind != "R_TABLE":
            raise ValueError("chunking is supported only for R_TABLE Points")
        if not isinstance(table_id, str) or not table_id.strip():
            raise ValueError("chunking.table_id must be a non-empty string")
        if (
            not isinstance(row_start_index, int)
            or isinstance(row_start_index, bool)
            or row_start_index < 0
            or not isinstance(row_end_index, int)
            or isinstance(row_end_index, bool)
            or row_end_index < row_start_index
        ):
            raise ValueError(
                "chunking row indexes must define a valid inclusive range"
            )
        if (
            not isinstance(chunk_index, int)
            or isinstance(chunk_index, bool)
            or chunk_index < 0
            or not isinstance(chunk_count, int)
            or isinstance(chunk_count, bool)
            or chunk_count < 2
            or chunk_index >= chunk_count
        ):
            raise ValueError(
                "chunking indexes must define a valid zero-based position"
            )

    neo4j_evidence_id = to_neo4j_evidence_id(evidence_id)
    disclosure_id, section_id = _evidence_parent_ids(neo4j_evidence_id)
    payload = {
        "point_kind": point_kind,
        "disclosure_id": disclosure_id,
        "section_id": section_id,
        "evidence_id": neo4j_evidence_id,
        "corp_name": _required_text(document_context, "corp_name"),
        "report_name": _required_text(document_context, "report_nm"),
        "section_name": section_name.strip(),
        "rcept_date": _required_text(document_context, "rcept_date"),
        "contextual_text": contextual_text,
        "canonical": dict(canonical),
    }
    if has_chunking:
        payload["chunking"] = {
            "table_id": table_id.strip(),
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "row_start_index": row_start_index,
            "row_end_index": row_end_index,
        }
    return payload


def _point_input(
    *,
    identity_key: str,
    point_kind: str,
    evidence_id: str,
    contextual_text: str,
    canonical: Mapping[str, Any],
    document_context: Mapping[str, Any],
    section_name: str,
    table_id: str | None = None,
    row_start_index: int | None = None,
    row_end_index: int | None = None,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> PointInput:
    return PointInput(
        id=str(uuid5(POINT_ID_NAMESPACE, identity_key)),
        contextual_text=contextual_text,
        embedding_cache_key=sha256(contextual_text.encode("utf-8")).hexdigest(),
        payload=_point_payload(
            point_kind=point_kind,
            evidence_id=evidence_id,
            contextual_text=contextual_text,
            canonical=canonical,
            document_context=document_context,
            section_name=section_name,
            table_id=table_id,
            row_start_index=row_start_index,
            row_end_index=row_end_index,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
        ),
    )


def build_point_inputs(
    fragment: Mapping[str, Any],
    *,
    document_context: Mapping[str, Any],
    section_context: Mapping[str, Any],
    r_table_strategy_selector: RTableEmbeddingStrategySelector | None = None,
) -> list[PointInput]:
    """Build contextual text and payloads without invoking an embedding API."""
    if not isinstance(fragment, Mapping):
        raise ValueError("fragment must be a mapping")
    if not isinstance(document_context, Mapping):
        raise ValueError("document_context must be a mapping")
    if not isinstance(section_context, Mapping):
        raise ValueError("section_context must be a mapping")
    if r_table_strategy_selector is not None and not isinstance(
        r_table_strategy_selector, RTableEmbeddingStrategySelector
    ):
        raise ValueError(
            "r_table_strategy_selector must be an RTableEmbeddingStrategySelector"
        )

    fragment_section_id = _required_text(fragment, "section_id")
    if _required_text(section_context, "section_id") != fragment_section_id:
        raise ValueError("fragment and section_context section_id must match")
    section_path = _section_path(section_context)
    corp_name = _required_text(document_context, "corp_name")
    report_nm = _required_text(document_context, "report_nm")

    evidence_list = fragment.get("evidence_list")
    records = fragment.get("records")
    if not isinstance(evidence_list, list):
        raise ValueError("fragment.evidence_list must be a list")
    if not isinstance(records, list):
        raise ValueError("fragment.records must be a list")

    records_by_table: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"fragment.records[{index}] must be a mapping")
        records_by_table[_required_text(record, "table_id")].append(record)

    point_inputs: list[PointInput] = []
    used_table_ids: set[str] = set()
    point_ids: set[str] = set()
    resolved_selector = r_table_strategy_selector

    def append_point_input(point_input: PointInput) -> None:
        if point_input.id in point_ids:
            raise ValueError(f"Duplicate Qdrant point ID: {point_input.id}")
        point_ids.add(point_input.id)
        point_inputs.append(point_input)

    for index, evidence in enumerate(evidence_list):
        if not isinstance(evidence, Mapping):
            raise ValueError(f"fragment.evidence_list[{index}] must be a mapping")
        evidence_id = _required_text(evidence, "evidence_id")
        evidence_type = evidence.get("evidence_type")
        section_name = _section_name(evidence, section_path)

        if evidence_type == "TEXT":
            contextual_text = build_text_contextual_text(
                evidence,
                corp_name=corp_name,
                report_nm=report_nm,
                section_path=section_path,
            )
            append_point_input(
                _point_input(
                    identity_key=evidence_id,
                    point_kind="TEXT",
                    evidence_id=evidence_id,
                    contextual_text=contextual_text,
                    canonical=_text_canonical(evidence),
                    document_context=document_context,
                    section_name=section_name,
                )
            )
            continue

        if evidence_type != "TABLE":
            raise ValueError(f"Unsupported evidence_type: {evidence_type}")

        table_type = evidence.get("table_type")
        if table_type == "KV_TABLE":
            contextual_text = build_kv_contextual_text(
                evidence,
                corp_name=corp_name,
                report_nm=report_nm,
                section_path=section_path,
            )
            append_point_input(
                _point_input(
                    identity_key=evidence_id,
                    point_kind="KV_TABLE",
                    evidence_id=evidence_id,
                    contextual_text=contextual_text,
                    canonical=_kv_table_canonical(evidence),
                    document_context=document_context,
                    section_name=section_name,
                )
            )
            continue

        if table_type != "R_TABLE":
            raise ValueError(f"Unsupported table_type: {table_type}")

        payload = evidence.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("R_TABLE Evidence payload must be a mapping")
        table_id = _required_text(payload, "table_id")
        used_table_ids.add(table_id)
        table_records = records_by_table.get(table_id, [])
        whole_table_text = build_r_table_contextual_text(
            evidence,
            table_records,
            corp_name=corp_name,
            report_nm=report_nm,
            section_path=section_path,
        )

        if resolved_selector is None:
            resolved_selector = RTableEmbeddingStrategySelector()
        decision = resolved_selector.select(
            evidence,
            table_records,
            whole_table_text,
        )

        point_parts: list[_RTablePointPart] = []

        def append_partition(
            partition_records: Sequence[Mapping[str, Any]],
            *,
            identity_key: str,
            contextual_text: str | None,
            descriptor: bool,
            preserve_row_range: bool,
        ) -> None:
            canonical_chunks = _chunk_r_table_records_by_canonical_size(
                evidence,
                partition_records,
                max_bytes=resolved_selector.config.canonical_max_bytes,
                hard_max_bytes=(
                    resolved_selector.config.canonical_hard_max_bytes
                ),
            )
            partition_was_split = len(canonical_chunks) > 1
            for canonical_chunk_index, canonical_chunk in enumerate(
                canonical_chunks
            ):
                chunk_evidence = _r_table_subset_evidence(
                    evidence,
                    record_count=len(canonical_chunk.records),
                )
                if descriptor:
                    chunk_contextual_text = (
                        build_r_table_descriptor_contextual_text(
                            chunk_evidence,
                            profile_r_table_columns(
                                chunk_evidence,
                                canonical_chunk.records,
                            ),
                            corp_name=corp_name,
                            report_nm=report_nm,
                            section_path=section_path,
                        )
                    )
                elif partition_was_split or contextual_text is None:
                    chunk_contextual_text = build_r_table_contextual_text(
                        chunk_evidence,
                        canonical_chunk.records,
                        corp_name=corp_name,
                        report_nm=report_nm,
                        section_path=section_path,
                    )
                else:
                    chunk_contextual_text = contextual_text

                row_start_index, row_end_index = _record_range(
                    canonical_chunk.records
                )
                chunk_identity_key = identity_key
                if partition_was_split:
                    chunk_identity_key = (
                        f"{identity_key}:canonical_chunk:"
                        f"{canonical_chunk_index}:{row_start_index}:{row_end_index}"
                    )
                point_parts.append(
                    _RTablePointPart(
                        identity_key=chunk_identity_key,
                        records=canonical_chunk.records,
                        canonical=canonical_chunk.canonical,
                        contextual_text=chunk_contextual_text,
                        row_start_index=(
                            row_start_index
                            if preserve_row_range or partition_was_split
                            else None
                        ),
                        row_end_index=(
                            row_end_index
                            if preserve_row_range or partition_was_split
                            else None
                        ),
                    )
                )

        if decision.strategy is RTableEmbeddingStrategy.WHOLE_TABLE:
            append_partition(
                table_records,
                identity_key=evidence_id,
                contextual_text=whole_table_text,
                descriptor=False,
                preserve_row_range=False,
            )
        elif decision.strategy is RTableEmbeddingStrategy.DESCRIPTOR:
            append_partition(
                table_records,
                identity_key=f"{evidence_id}:descriptor",
                contextual_text=None,
                descriptor=True,
                preserve_row_range=False,
            )
        elif decision.strategy is RTableEmbeddingStrategy.ROW_GROUP:
            row_groups = build_r_table_row_groups(
                evidence,
                table_records,
                corp_name=corp_name,
                report_nm=report_nm,
                section_path=section_path,
                max_tokens=resolved_selector.config.row_group_max_tokens,
                token_counter=resolved_selector.token_counter,
            )
            if not row_groups:
                raise ValueError(
                    "R_TABLE row-group strategy requires at least one record"
                )
            for group in row_groups:
                group_records = [
                    record
                    for record in table_records
                    if group.row_start_index
                    <= int(record["record_index"])
                    <= group.row_end_index
                ]
                append_partition(
                    group_records,
                    identity_key=(
                        f"{evidence_id}:row_group:"
                        f"{group.row_start_index}:{group.row_end_index}"
                    ),
                    contextual_text=group.contextual_text,
                    descriptor=False,
                    preserve_row_range=True,
                )
        else:
            raise ValueError(f"Unsupported R_TABLE strategy: {decision.strategy}")

        chunk_count = len(point_parts) if len(point_parts) > 1 else None
        for chunk_index, part in enumerate(point_parts):
            is_chunked = chunk_count is not None
            append_point_input(
                _point_input(
                    identity_key=part.identity_key,
                    point_kind="R_TABLE",
                    evidence_id=evidence_id,
                    contextual_text=part.contextual_text,
                    canonical=part.canonical,
                    document_context=document_context,
                    section_name=section_name,
                    table_id=table_id if is_chunked else None,
                    row_start_index=part.row_start_index if is_chunked else None,
                    row_end_index=part.row_end_index if is_chunked else None,
                    chunk_index=chunk_index if is_chunked else None,
                    chunk_count=chunk_count,
                )
            )

    orphan_table_ids = set(records_by_table) - used_table_ids
    if orphan_table_ids:
        raise ValueError(
            "Fragment contains records without matching R_TABLE Evidence: "
            + ", ".join(sorted(orphan_table_ids))
        )
    return point_inputs


def embed_point_inputs(
    point_inputs: Sequence[PointInput],
    *,
    vectorizer: Vectorizer = text_to_hybrid_vector,
) -> dict[str, HybridEmbedding]:
    """Embed unique contextual texts, keyed for a future persistent cache."""
    embeddings: dict[str, HybridEmbedding] = {}
    cached_texts: dict[str, str] = {}
    for point_input in point_inputs:
        if not isinstance(point_input, PointInput):
            raise ValueError("point_inputs must contain PointInput items")
        cache_key = point_input.embedding_cache_key
        cached_text = cached_texts.get(cache_key)
        if cached_text is not None:
            if cached_text != point_input.contextual_text:
                raise ValueError("Embedding cache key collision detected")
            continue
        embeddings[cache_key] = vectorizer(point_input.contextual_text)
        cached_texts[cache_key] = point_input.contextual_text
    return embeddings


def assemble_qdrant_points(
    point_inputs: Sequence[PointInput],
    embeddings: Mapping[str, HybridEmbedding],
) -> list[dict[str, Any]]:
    """Combine Point inputs and precomputed embeddings into Qdrant dictionaries."""
    points: list[dict[str, Any]] = []
    for point_input in point_inputs:
        if not isinstance(point_input, PointInput):
            raise ValueError("point_inputs must contain PointInput items")
        embedding = embeddings.get(point_input.embedding_cache_key)
        if not isinstance(embedding, HybridEmbedding):
            raise ValueError(
                f"Missing or invalid hybrid embedding for Point input: {point_input.id}"
            )
        dense = embedding.dense
        sparse = embedding.sparse
        if not dense or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in dense
        ):
            raise ValueError(
                f"Invalid dense embedding for Point input: {point_input.id}"
            )
        if (
            not sparse.indices
            or len(sparse.indices) != len(sparse.values)
            or any(index < 0 for index in sparse.indices)
        ):
            raise ValueError(
                f"Invalid sparse embedding for Point input: {point_input.id}"
            )
        points.append(
            {
                "id": point_input.id,
                "vector": {
                    VECTOR_NAME: [float(value) for value in dense],
                    SPARSE_VECTOR_NAME: {
                        "indices": list(sparse.indices),
                        "values": list(sparse.values),
                    },
                },
                "payload": dict(point_input.payload),
            }
        )
    return points


def build_qdrant_points(
    fragment: Mapping[str, Any],
    *,
    document_context: Mapping[str, Any],
    section_context: Mapping[str, Any],
    vectorizer: Vectorizer = text_to_hybrid_vector,
    r_table_strategy_selector: RTableEmbeddingStrategySelector | None = None,
) -> list[dict[str, Any]]:
    """Build, embed, and assemble Qdrant points for one Evidence Fragment."""
    point_inputs = build_point_inputs(
        fragment,
        document_context=document_context,
        section_context=section_context,
        r_table_strategy_selector=r_table_strategy_selector,
    )
    embeddings = embed_point_inputs(point_inputs, vectorizer=vectorizer)
    return assemble_qdrant_points(point_inputs, embeddings)


__all__ = [
    "POINT_ID_NAMESPACE",
    "POINT_KINDS",
    "SPARSE_VECTOR_NAME",
    "VECTOR_NAME",
    "PointInput",
    "assemble_qdrant_points",
    "build_point_inputs",
    "build_qdrant_points",
    "embed_point_inputs",
    "to_neo4j_evidence_id",
]
