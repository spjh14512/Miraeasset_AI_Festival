"""Build non-overlapping token-bounded row groups for one R_TABLE."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from vector_db.bgem3_token_counter import count_bge_m3_tokens
from vector_db.contextual_text_builders.r_table_record_context_builder import (
    build_r_table_contextual_text,
)
from vector_db.r_table_strategy_selector import load_r_table_strategy_config


TokenCounter = Callable[[str], int]


class RowGroupRecordTooLargeError(ValueError):
    """Legacy error retained for callers that imported it before overflow support."""

    def __init__(
        self,
        *,
        table_id: str,
        record_index: int,
        token_count: int,
        max_tokens: int,
    ) -> None:
        self.table_id = table_id
        self.record_index = record_index
        self.token_count = token_count
        self.max_tokens = max_tokens
        super().__init__(
            "One R_TABLE record exceeds the row-group token budget: "
            f"table_id={table_id}, record_index={record_index}, "
            f"token_count={token_count}, max_tokens={max_tokens}"
        )


@dataclass(frozen=True, slots=True)
class RTableRowGroup:
    group_index: int
    table_id: str
    row_start_index: int
    row_end_index: int
    contextual_text: str
    token_count: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _validated_records(
    evidence: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> tuple[str, list[Mapping[str, Any]]]:
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
    table_id = table_id.strip()

    headers = payload.get("headers")
    if not isinstance(headers, list) or not all(
        isinstance(header, list)
        and all(isinstance(part, str) for part in header)
        for header in headers
    ):
        raise ValueError("R_TABLE payload.headers must be a list of string lists")
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

    indexed_records: list[tuple[int, Mapping[str, Any]]] = []
    record_indexes: set[int] = set()
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
        indexed_records.append((record_index, record))

    ordered_indexes = sorted(record_indexes)
    if ordered_indexes != list(range(record_count)):
        raise ValueError(
            "R_TABLE record_index values must be contiguous from 0 to record_count - 1"
        )
    return table_id, [
        record for _, record in sorted(indexed_records, key=lambda item: item[0])
    ]


def _group_evidence(
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


def _render_group(
    evidence: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    corp_name: str,
    report_nm: str,
    section_path: Sequence[str],
    token_counter: TokenCounter,
) -> tuple[str, int]:
    contextual_text = build_r_table_contextual_text(
        _group_evidence(evidence, record_count=len(records)),
        list(records),
        corp_name=corp_name,
        report_nm=report_nm,
        section_path=section_path,
    )
    token_count = token_counter(contextual_text)
    if (
        not isinstance(token_count, int)
        or isinstance(token_count, bool)
        or token_count < 0
    ):
        raise ValueError("token_counter must return a non-negative integer")
    return contextual_text, token_count


def build_r_table_row_groups(
    evidence: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    corp_name: str,
    report_nm: str,
    section_path: Sequence[str],
    max_tokens: int | None = None,
    token_counter: TokenCounter = count_bge_m3_tokens,
) -> list[RTableRowGroup]:
    """Greedily group each record once, allowing an indivisible row to overflow."""
    if not isinstance(evidence, Mapping):
        raise ValueError("evidence must be a mapping")
    if not callable(token_counter):
        raise ValueError("token_counter must be callable")
    resolved_max_tokens = (
        load_r_table_strategy_config().row_group_max_tokens
        if max_tokens is None
        else max_tokens
    )
    if (
        not isinstance(resolved_max_tokens, int)
        or isinstance(resolved_max_tokens, bool)
        or resolved_max_tokens <= 0
    ):
        raise ValueError("max_tokens must be a positive integer")

    table_id, ordered_records = _validated_records(evidence, records)
    if not ordered_records:
        return []

    groups: list[RTableRowGroup] = []
    current_records: list[Mapping[str, Any]] = []
    current_text = ""
    current_token_count = 0

    def append_current_group() -> None:
        if not current_records:
            return
        groups.append(
            RTableRowGroup(
                group_index=len(groups),
                table_id=table_id,
                row_start_index=int(current_records[0]["record_index"]),
                row_end_index=int(current_records[-1]["record_index"]),
                contextual_text=current_text,
                token_count=current_token_count,
            )
        )

    for record in ordered_records:
        candidate_records = [*current_records, record]
        candidate_text, candidate_token_count = _render_group(
            evidence,
            candidate_records,
            corp_name=corp_name,
            report_nm=report_nm,
            section_path=section_path,
            token_counter=token_counter,
        )
        if candidate_token_count <= resolved_max_tokens:
            current_records = candidate_records
            current_text = candidate_text
            current_token_count = candidate_token_count
            continue

        if current_records:
            append_current_group()
            current_records = [record]
            current_text, current_token_count = _render_group(
                evidence,
                current_records,
                corp_name=corp_name,
                report_nm=report_nm,
                section_path=section_path,
                token_counter=token_counter,
            )
        else:
            current_records = [record]
            current_text = candidate_text
            current_token_count = candidate_token_count

        # A record is the smallest canonical retrieval unit. If its rendered text
        # alone exceeds the budget, preserve the row boundary as a singleton
        # group instead of dropping or duplicating part of the record.

    append_current_group()
    return groups


__all__ = [
    "RTableRowGroup",
    "RowGroupRecordTooLargeError",
    "build_r_table_row_groups",
]
