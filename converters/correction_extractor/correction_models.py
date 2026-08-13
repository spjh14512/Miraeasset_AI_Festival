"""Data models for non-destructive correction disclosure extraction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from converters.common.source_models import DocumentContext, DocumentSyntax, SourceRef


class CorrectionStatus(StrEnum):
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    RECOVERED = "RECOVERED"
    FAILED = "FAILED"


class CorrectionBlockType(StrEnum):
    TITLE = "TITLE"
    PARAGRAPH = "PARAGRAPH"
    TABLE = "TABLE"


@dataclass(frozen=True, slots=True)
class CorrectionIssue:
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
class CorrectionBlockRef:
    sequence: int
    block_type: CorrectionBlockType
    source_ref: SourceRef

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "block_type": self.block_type.value,
            "source_ref": self.source_ref.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CorrectionMetadata:
    title: str | None
    correction_date: str | None
    target_document_name: str | None
    original_submission_date: str | None
    reason: str | None
    target_rcept_no: str | None
    source_ref: SourceRef

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "correction_date": self.correction_date,
            "target_document_name": self.target_document_name,
            "original_submission_date": self.original_submission_date,
            "reason": self.reason,
            "target_rcept_no": self.target_rcept_no,
            "source_ref": self.source_ref.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CorrectionExtraction:
    source_document: DocumentContext
    syntax: DocumentSyntax
    status: CorrectionStatus
    correction: CorrectionMetadata | None
    correction_blocks: tuple[CorrectionBlockRef, ...] = ()
    excluded_source_refs: tuple[SourceRef, ...] = ()
    issues: tuple[CorrectionIssue, ...] = ()

    @property
    def has_correction(self) -> bool:
        return self.correction is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "correction.v1",
            "source_document": self.source_document.to_dict(),
            "syntax": self.syntax.value,
            "status": self.status.value,
            "has_correction": self.has_correction,
            "correction": (
                self.correction.to_dict() if self.correction is not None else None
            ),
            "correction_blocks": [
                block.to_dict() for block in self.correction_blocks
            ],
            "excluded_source_refs": [
                source_ref.to_dict() for source_ref in self.excluded_source_refs
            ],
            "issues": [issue.to_dict() for issue in self.issues],
        }


__all__ = [
    "CorrectionBlockRef",
    "CorrectionBlockType",
    "CorrectionExtraction",
    "CorrectionIssue",
    "CorrectionMetadata",
    "CorrectionStatus",
]
