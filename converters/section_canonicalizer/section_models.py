"""Canonical models for document sections and ordered source blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from converters.common.source_models import (
    DocumentContext,
    DocumentSyntax,
    SourceRef,
)


class SectionParseStatus(StrEnum):
    SUCCESS = "SUCCESS"
    RECOVERED = "RECOVERED"
    FAILED = "FAILED"


class SectionBoundaryKind(StrEnum):
    EXPLICIT = "EXPLICIT"
    IMPLICIT = "IMPLICIT"
    SYNTHETIC = "SYNTHETIC"


class SectionBlockType(StrEnum):
    P = "P"
    SPAN_RUN = "SPAN_RUN"
    TABLE = "TABLE"
    TABLE_GROUP = "TABLE_GROUP"
    IMAGE = "IMAGE"


@dataclass(frozen=True, slots=True)
class SectionIssue:
    code: str
    message: str
    severity: str = "WARNING"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class SectionBlockRef:
    block_type: SectionBlockType
    order: int
    source_refs: tuple[SourceRef, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_type": self.block_type.value,
            "order": self.order,
            "source_refs": [source_ref.to_dict() for source_ref in self.source_refs],
        }


@dataclass(frozen=True, slots=True)
class CanonicalSection:
    id: str
    parent_section_id: str | None
    order: int
    level: int
    boundary_kind: SectionBoundaryKind
    title: str | None
    section_path: tuple[str, ...]
    source_ref: SourceRef | None
    title_source_ref: SourceRef | None
    blocks: tuple[SectionBlockRef, ...]
    issues: tuple[SectionIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_section_id": self.parent_section_id,
            "order": self.order,
            "level": self.level,
            "boundary_kind": self.boundary_kind.value,
            "title": self.title,
            "section_path": list(self.section_path),
            "source_ref": (
                self.source_ref.to_dict() if self.source_ref is not None else None
            ),
            "title_source_ref": (
                self.title_source_ref.to_dict()
                if self.title_source_ref is not None
                else None
            ),
            "blocks": [block.to_dict() for block in self.blocks],
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class CanonicalSectionCollection:
    source_document: DocumentContext
    syntax: DocumentSyntax
    parse_status: SectionParseStatus
    sections: tuple[CanonicalSection, ...]
    issues: tuple[SectionIssue, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "section-collection.v1",
            "source_document": self.source_document.to_dict(),
            "syntax": self.syntax.value,
            "parse_status": self.parse_status.value,
            "sections": [section.to_dict() for section in self.sections],
            "issues": [issue.to_dict() for issue in self.issues],
        }


__all__ = [
    "CanonicalSection",
    "CanonicalSectionCollection",
    "SectionBlockRef",
    "SectionBlockType",
    "SectionBoundaryKind",
    "SectionIssue",
    "SectionParseStatus",
]
