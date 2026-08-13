"""Canonical section-level Evidence Fragment models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


FRAGMENT_SCHEMA_VERSION = "evidence-fragment.v2"


class EvidenceType(StrEnum):
    TEXT = "TEXT"
    TABLE = "TABLE"


class StorageMode(StrEnum):
    SECTION_RECORDS = "SECTION_RECORDS"


@dataclass(frozen=True, slots=True)
class FragmentEvidence:
    evidence_id: str
    evidence_type: EvidenceType
    order: int
    payload: Mapping[str, Any]
    table_type: str | None = None
    storage_mode: StorageMode | None = None
    references: tuple[Mapping[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type.value,
            "order": self.order,
            "payload": dict(self.payload),
        }
        if self.table_type is not None:
            result["table_type"] = self.table_type
        if self.storage_mode is not None:
            result["storage_mode"] = self.storage_mode.value
        if self.references:
            result["references"] = [dict(reference) for reference in self.references]
        return result


@dataclass(frozen=True, slots=True)
class ExternalTableRecord:
    table_id: str
    record_index: int
    row_type: str
    row_context: tuple[str, ...]
    values: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "record_index": self.record_index,
            "row_type": self.row_type,
            "row_context": list(self.row_context),
            "values": list(self.values),
        }


@dataclass(frozen=True, slots=True)
class EvidenceFragment:
    section_id: str
    evidence_list: tuple[FragmentEvidence, ...]
    records: tuple[ExternalTableRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FRAGMENT_SCHEMA_VERSION,
            "section_id": self.section_id,
            "evidence_list": [item.to_dict() for item in self.evidence_list],
            "records": [record.to_dict() for record in self.records],
        }


__all__ = [
    "EvidenceFragment",
    "EvidenceType",
    "ExternalTableRecord",
    "FRAGMENT_SCHEMA_VERSION",
    "FragmentEvidence",
    "StorageMode",
]
