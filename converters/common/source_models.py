"""Shared source identity and locator models for converter outputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class DocumentSyntax(StrEnum):
    AUTO = "AUTO"
    DART_XML = "DART_XML"
    HTML = "HTML"


class ContextRole(StrEnum):
    HEADING = "HEADING"
    TITLE = "TITLE"
    UNIT = "UNIT"
    CAPTION = "CAPTION"
    NOTE = "NOTE"


class ContextPosition(StrEnum):
    BEFORE = "BEFORE"
    AFTER = "AFTER"


@dataclass(frozen=True, slots=True)
class DocumentContext:
    """Stable identity supplied by ingestion, not inferred from document text."""

    doc_id: str
    rcept_no: str
    source_path: str | None = None
    doc_group: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "rcept_no": self.rcept_no,
            "source_path": self.source_path,
            "doc_group": self.doc_group,
        }


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Non-destructive path, HTML ID, or table ordinal into the source document."""

    syntax: DocumentSyntax
    element_path: str | None = None
    html_id: str | None = None
    table_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "syntax": self.syntax.value,
            "element_path": self.element_path,
            "html_id": self.html_id,
        }
        if self.table_index is not None:
            result["table_index"] = self.table_index
        return result


@dataclass(frozen=True, slots=True)
class EvidenceContextBlock:
    """A renderable block associated with evidence, with source provenance."""

    role: ContextRole
    position: ContextPosition
    text: str
    source_ref: SourceRef | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "position": self.position.value,
            "text": self.text,
            "source_ref": (
                self.source_ref.to_dict()
                if self.source_ref is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class EvidenceContext:
    """Semantic context supplied by section/block orchestration."""

    section_path: tuple[str, ...] = ()
    blocks: tuple[EvidenceContextBlock, ...] = ()

    def blocks_by_role(self, role: ContextRole) -> tuple[EvidenceContextBlock, ...]:
        return tuple(block for block in self.blocks if block.role == role)

    def _unique_texts(self, role: ContextRole) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(block.text for block in self.blocks_by_role(role))
        )

    @property
    def headings(self) -> tuple[str, ...]:
        return self._unique_texts(ContextRole.HEADING)

    @property
    def block_title(self) -> str | None:
        titles = self.blocks_by_role(ContextRole.TITLE)
        return titles[0].text if titles else None

    @property
    def units(self) -> tuple[str, ...]:
        return self._unique_texts(ContextRole.UNIT)

    @property
    def captions(self) -> tuple[str, ...]:
        return self._unique_texts(ContextRole.CAPTION)

    @property
    def notes(self) -> tuple[str, ...]:
        return self._unique_texts(ContextRole.NOTE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_path": list(self.section_path),
            "blocks": [block.to_dict() for block in self.blocks],
        }


__all__ = [
    "ContextPosition",
    "ContextRole",
    "DocumentContext",
    "DocumentSyntax",
    "EvidenceContext",
    "EvidenceContextBlock",
    "SourceRef",
]
