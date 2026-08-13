"""Data models for structural DART table parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from converters.common.source_models import (
    EvidenceContext,
    EvidenceContextBlock,
    SourceRef,
)


class TableType(StrEnum):
    KV_TABLE = "KV_TABLE"
    R_TABLE = "R_TABLE"
    LAYOUT_TABLE = "LAYOUT_TABLE"
    UNKNOWN = "UNKNOWN"


class ParseStatus(StrEnum):
    SUCCESS = "SUCCESS"
    RECOVERED = "RECOVERED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class CellRole(StrEnum):
    EMPTY = "EMPTY"
    HEADER = "HEADER"
    KEY = "KEY"
    VALUE = "VALUE"
    CONTEXT = "CONTEXT"
    UNKNOWN = "UNKNOWN"


class ContextStatus(StrEnum):
    RESOLVED = "RESOLVED"
    MULTI_CONTEXT = "MULTI_CONTEXT"
    NONE = "NONE"


class RowType(StrEnum):
    DATA = "DATA"
    SUBTOTAL = "SUBTOTAL"
    TOTAL = "TOTAL"
    UNKNOWN = "UNKNOWN"


class LayoutRole(StrEnum):
    TITLE = "TITLE"
    UNIT = "UNIT"
    NOTE = "NOTE"
    NAVIGATION = "NAVIGATION"


class SourceSyntax(StrEnum):
    AUTO = "AUTO"
    DART_XML = "DART_XML"
    HTML = "HTML"


@dataclass(frozen=True, slots=True)
class ParseIssue:
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
class TableContext:
    source_path: str | None = None
    table_index: int | None = None
    table_group_class: str | None = None
    source_ref: SourceRef | None = None


@dataclass(frozen=True, slots=True)
class SourceCell:
    id: str
    tag: str
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    raw_value: str
    text_segments: tuple[str, ...]
    source_attributes: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self, role: CellRole) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": role.value,
            "tag": self.tag,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "col_start": self.col_start,
            "col_end": self.col_end,
            "raw_value": self.raw_value,
            "text_segments": list(self.text_segments),
            "source_attributes": dict(self.source_attributes),
        }


@dataclass(frozen=True, slots=True)
class LogicalGrid:
    rows: tuple[tuple[SourceCell | None, ...], ...]
    cells: tuple[SourceCell, ...]
    issues: tuple[ParseIssue, ...] = ()

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        return len(self.rows[0]) if self.rows else 0


@dataclass(frozen=True, slots=True)
class CanonicalTable:
    table_type: TableType
    parse_status: ParseStatus
    source: Mapping[str, Any]
    dimensions: Mapping[str, int]
    cells: tuple[SourceCell, ...]
    cell_roles: Mapping[str, CellRole]
    content: Mapping[str, Any]
    context: EvidenceContext = field(default_factory=EvidenceContext)
    issues: tuple[ParseIssue, ...] = ()

    @property
    def title(self) -> str | None:
        return self.context.block_title

    @property
    def units(self) -> tuple[str, ...]:
        return self.context.units

    @property
    def captions(self) -> tuple[str, ...]:
        return self.context.captions

    @property
    def notes(self) -> tuple[str, ...]:
        return self.context.notes

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "table.v3",
            "table_type": self.table_type.value,
            "parse_status": self.parse_status.value,
            "source": dict(self.source),
            "dimensions": dict(self.dimensions),
            "context": self.context.to_dict(),
            "cells": [
                cell.to_dict(self.cell_roles.get(cell.id, CellRole.UNKNOWN))
                for cell in self.cells
            ],
            "content": dict(self.content),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class CanonicalTableGroup:
    table_group_class: str | None
    context_blocks: tuple[EvidenceContextBlock, ...]
    tables: tuple[CanonicalTable, ...]
    issues: tuple[ParseIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_group_class": self.table_group_class,
            "context_blocks": [block.to_dict() for block in self.context_blocks],
            "tables": [table.to_dict() for table in self.tables],
            "issues": [issue.to_dict() for issue in self.issues],
        }
