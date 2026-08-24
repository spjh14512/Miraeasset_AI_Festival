"""Build strategy-routed Qdrant points from one Evidence Fragment."""

from __future__ import annotations

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
from vector_db.text2vector import text_to_vector


POINT_ID_NAMESPACE = UUID("9220d409-a029-5497-8ab2-fc39b239ab18")
VECTOR_NAME = "evidence_dense"
POINT_KINDS = {"TEXT", "KV_TABLE", "R_TABLE"}
TABLE_METADATA_FIELDS = ("title", "captions", "units", "notes")
PATH_SEPARATOR = " > "

Vectorizer = Callable[[str], list[float]]


@dataclass(frozen=True, slots=True)
class PointInput:
    """Embedding-independent input used to assemble one Qdrant point."""

    id: str
    contextual_text: str
    embedding_cache_key: str
    payload: dict[str, Any]


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


def _point_payload(
    *,
    point_kind: str,
    evidence_id: str,
    contextual_text: str,
    canonical: Mapping[str, Any],
    document_context: Mapping[str, Any],
    table_id: str | None = None,
    row_start_index: int | None = None,
    row_end_index: int | None = None,
) -> dict[str, Any]:
    if point_kind not in POINT_KINDS:
        raise ValueError(f"Unsupported point_kind: {point_kind}")

    base_month = _optional_integer(document_context, "base_month")
    if base_month is not None and not 1 <= base_month <= 12:
        raise ValueError("base_month must be between 1 and 12")

    has_row_range = row_start_index is not None or row_end_index is not None
    if has_row_range:
        if table_id is None:
            raise ValueError("table_id is required for an R_TABLE row-group Point")
        if (
            not isinstance(row_start_index, int)
            or isinstance(row_start_index, bool)
            or row_start_index < 0
            or not isinstance(row_end_index, int)
            or isinstance(row_end_index, bool)
            or row_end_index < row_start_index
        ):
            raise ValueError(
                "row_start_index and row_end_index must define a valid inclusive range"
            )
    if table_id is not None and (not isinstance(table_id, str) or not table_id.strip()):
        raise ValueError("table_id must be a non-empty string or null")

    retrieval_metadata = {
        "point_kind": point_kind,
        "evidence_id": to_neo4j_evidence_id(evidence_id),
        "corp_name": _required_text(document_context, "corp_name"),
        "corp_code": _required_text(document_context, "corp_code"),
        "industry": _optional_text(document_context, "industry"),
        "sector": _optional_text(document_context, "sector"),
        "report_nm": _required_text(document_context, "report_nm"),
        "base_year": _optional_integer(document_context, "base_year"),
        "base_month": base_month,
    }
    if table_id is not None:
        retrieval_metadata["table_id"] = table_id.strip()
    if has_row_range:
        retrieval_metadata["row_start_index"] = row_start_index
        retrieval_metadata["row_end_index"] = row_end_index
    return {
        "retrieval_metadata": retrieval_metadata,
        "contextual_text": contextual_text,
        "canonical": dict(canonical),
    }


def _point_input(
    *,
    identity_key: str,
    point_kind: str,
    evidence_id: str,
    contextual_text: str,
    canonical: Mapping[str, Any],
    document_context: Mapping[str, Any],
    table_id: str | None = None,
    row_start_index: int | None = None,
    row_end_index: int | None = None,
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
            table_id=table_id,
            row_start_index=row_start_index,
            row_end_index=row_end_index,
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

        if decision.strategy is RTableEmbeddingStrategy.WHOLE_TABLE:
            append_point_input(
                _point_input(
                    identity_key=evidence_id,
                    point_kind="R_TABLE",
                    evidence_id=evidence_id,
                    contextual_text=whole_table_text,
                    canonical=_r_table_canonical(evidence, table_records),
                    document_context=document_context,
                    table_id=table_id,
                )
            )
            continue

        if decision.strategy is RTableEmbeddingStrategy.DESCRIPTOR:
            descriptor_text = build_r_table_descriptor_contextual_text(
                evidence,
                profile_r_table_columns(evidence, table_records),
                corp_name=corp_name,
                report_nm=report_nm,
                section_path=section_path,
            )
            append_point_input(
                _point_input(
                    identity_key=f"{evidence_id}:descriptor",
                    point_kind="R_TABLE",
                    evidence_id=evidence_id,
                    contextual_text=descriptor_text,
                    canonical=_r_table_canonical(evidence, table_records),
                    document_context=document_context,
                    table_id=table_id,
                )
            )
            continue

        if decision.strategy is not RTableEmbeddingStrategy.ROW_GROUP:
            raise ValueError(f"Unsupported R_TABLE strategy: {decision.strategy}")

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
            raise ValueError("R_TABLE row-group strategy requires at least one record")
        for group in row_groups:
            group_records = [
                record
                for record in table_records
                if group.row_start_index
                <= int(record["record_index"])
                <= group.row_end_index
            ]
            append_point_input(
                _point_input(
                    identity_key=(
                        f"{evidence_id}:row_group:"
                        f"{group.row_start_index}:{group.row_end_index}"
                    ),
                    point_kind="R_TABLE",
                    evidence_id=evidence_id,
                    contextual_text=group.contextual_text,
                    canonical=_r_table_canonical(evidence, group_records),
                    document_context=document_context,
                    table_id=group.table_id,
                    row_start_index=group.row_start_index,
                    row_end_index=group.row_end_index,
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
    vectorizer: Vectorizer = text_to_vector,
) -> dict[str, list[float]]:
    """Embed unique contextual texts, keyed for a future persistent cache."""
    embeddings: dict[str, list[float]] = {}
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
    embeddings: Mapping[str, Sequence[float]],
) -> list[dict[str, Any]]:
    """Combine Point inputs and precomputed embeddings into Qdrant dictionaries."""
    points: list[dict[str, Any]] = []
    for point_input in point_inputs:
        if not isinstance(point_input, PointInput):
            raise ValueError("point_inputs must contain PointInput items")
        vector = embeddings.get(point_input.embedding_cache_key)
        if not isinstance(vector, (list, tuple)) or not vector or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in vector
        ):
            raise ValueError(
                f"Missing or invalid embedding for Point input: {point_input.id}"
            )
        points.append(
            {
                "id": point_input.id,
                "vector": {VECTOR_NAME: [float(value) for value in vector]},
                "payload": dict(point_input.payload),
            }
        )
    return points


def build_qdrant_points(
    fragment: Mapping[str, Any],
    *,
    document_context: Mapping[str, Any],
    section_context: Mapping[str, Any],
    vectorizer: Vectorizer = text_to_vector,
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
    "VECTOR_NAME",
    "PointInput",
    "assemble_qdrant_points",
    "build_point_inputs",
    "build_qdrant_points",
    "embed_point_inputs",
    "to_neo4j_evidence_id",
]
