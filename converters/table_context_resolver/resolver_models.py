"""Models for associating nearby text blocks with table evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from converters.common.source_models import SourceRef
from converters.table_parser.table_models import CanonicalTable


@dataclass(frozen=True, slots=True)
class TableEvidenceBundle:
    table_block_index: int
    table: CanonicalTable
    consumed_source_refs: tuple[SourceRef, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_block_index": self.table_block_index,
            "table": self.table.to_dict(),
            "consumed_source_refs": [
                source_ref.to_dict()
                for source_ref in self.consumed_source_refs
            ],
        }


@dataclass(frozen=True, slots=True)
class TableContextResolution:
    tables: tuple[TableEvidenceBundle, ...]
    consumed_source_refs: tuple[SourceRef, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tables": [table.to_dict() for table in self.tables],
            "consumed_source_refs": [
                source_ref.to_dict()
                for source_ref in self.consumed_source_refs
            ],
        }


__all__ = ["TableContextResolution", "TableEvidenceBundle"]
