"""Canonical models for paragraph evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from converters.common.source_models import (
    DocumentContext,
    DocumentSyntax,
    EvidenceContext,
    SourceRef,
)


class ParagraphParseStatus(StrEnum):
    SUCCESS = "SUCCESS"
    RECOVERED = "RECOVERED"
    FAILED = "FAILED"


class ParagraphSourceKind(StrEnum):
    P = "P"
    SPAN_RUN = "SPAN_RUN"


class FragmentKind(StrEnum):
    TEXT = "TEXT"
    SPAN = "SPAN"


class MarkerKind(StrEnum):
    BULLET = "BULLET"
    NOTE = "NOTE"
    FOOTNOTE = "FOOTNOTE"
    ENUMERATION = "ENUMERATION"


@dataclass(frozen=True, slots=True)
class ParagraphIssue:
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
class LeadingMarker:
    text: str
    kind: MarkerKind

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text, "kind": self.kind.value}


@dataclass(frozen=True, slots=True)
class ParagraphFragment:
    id: str
    kind: FragmentKind
    raw_text: str
    text: str
    source_attributes: Mapping[str, str]
    source_ref: SourceRef

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "raw_text": self.raw_text,
            "text": self.text,
            "source_attributes": dict(self.source_attributes),
            "source_ref": self.source_ref.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CanonicalParagraph:
    source_document: DocumentContext
    source_kind: ParagraphSourceKind
    source_refs: tuple[SourceRef, ...]
    paragraph_index: int
    context: EvidenceContext
    raw_text: str
    text: str
    leading_marker: LeadingMarker | None
    source_attributes: Mapping[str, str]
    fragments: tuple[ParagraphFragment, ...]
    parse_status: ParagraphParseStatus = ParagraphParseStatus.SUCCESS
    issues: tuple[ParagraphIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "paragraph.v2",
            "parse_status": self.parse_status.value,
            "source_document": self.source_document.to_dict(),
            "source_kind": self.source_kind.value,
            "source_refs": [ref.to_dict() for ref in self.source_refs],
            "paragraph_index": self.paragraph_index,
            "context": self.context.to_dict(),
            "raw_text": self.raw_text,
            "text": self.text,
            "leading_marker": (
                self.leading_marker.to_dict()
                if self.leading_marker is not None
                else None
            ),
            "source_attributes": dict(self.source_attributes),
            "fragments": [fragment.to_dict() for fragment in self.fragments],
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class CanonicalParagraphCollection:
    source_document: DocumentContext
    syntax: DocumentSyntax
    parse_status: ParagraphParseStatus
    paragraphs: tuple[CanonicalParagraph, ...]
    issues: tuple[ParagraphIssue, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "paragraph-collection.v2",
            "parse_status": self.parse_status.value,
            "source_document": self.source_document.to_dict(),
            "syntax": self.syntax.value,
            "paragraphs": [paragraph.to_dict() for paragraph in self.paragraphs],
            "issues": [issue.to_dict() for issue in self.issues],
        }


__all__ = [
    "CanonicalParagraph",
    "CanonicalParagraphCollection",
    "FragmentKind",
    "LeadingMarker",
    "MarkerKind",
    "ParagraphFragment",
    "ParagraphIssue",
    "ParagraphParseStatus",
    "ParagraphSourceKind",
]
