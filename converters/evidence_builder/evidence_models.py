"""Final evidence models exposed to graph ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from converters.common.source_models import SourceRef


class EvidenceType(StrEnum):
    TEXT = "TEXT"
    TABLE = "TABLE"


@dataclass(frozen=True, slots=True)
class FinalEvidenceContext:
    section_path: tuple[str, ...]
    title: str | None = None
    units: tuple[str, ...] = ()
    captions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"section_path": list(self.section_path)}
        if self.title:
            result["title"] = self.title
        if self.units:
            result["units"] = list(self.units)
        if self.captions:
            result["captions"] = list(self.captions)
        if self.notes:
            result["notes"] = list(self.notes)
        return result


@dataclass(frozen=True, slots=True)
class CanonicalEvidence:
    id: str
    evidence_type: EvidenceType
    section_id: str
    order: int
    context: FinalEvidenceContext
    source_refs: tuple[SourceRef, ...]
    markdown: str
    payload: Mapping[str, Any]
    content_sha256: str
    table_type: str | None = None
    parse_status: str | None = None
    issues: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "evidence_type": self.evidence_type.value,
            "section_id": self.section_id,
            "order": self.order,
            "context": self.context.to_dict(),
            "source_refs": [source_ref.to_dict() for source_ref in self.source_refs],
            "markdown": self.markdown,
            "payload": dict(self.payload),
            "content_sha256": self.content_sha256,
        }
        if self.table_type is not None:
            result["table_type"] = self.table_type
        if self.parse_status not in {None, "SUCCESS"}:
            result["parse_status"] = self.parse_status
        if self.issues:
            result["issues"] = [dict(issue) for issue in self.issues]
        return result


__all__ = ["CanonicalEvidence", "EvidenceType", "FinalEvidenceContext"]
