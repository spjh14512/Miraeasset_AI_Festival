"""Serializable models for Event extraction intermediates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EVENT_SCHEMA_VERSION = "event.v1"


@dataclass(frozen=True, slots=True)
class EventNode:
    id: str
    event_type: str
    event_subtype: str
    event_date: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "event_subtype": self.event_subtype,
            "event_date": self.event_date,
            "content": self.content,
        }


@dataclass(frozen=True, slots=True)
class EventRelation:
    type: str
    source_id: str
    target_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "type": self.type,
            "source_id": self.source_id,
            "target_id": self.target_id,
        }


@dataclass(frozen=True, slots=True)
class EventExtraction:
    source_document: dict[str, Any]
    event: EventNode
    reports: EventRelation
    is_supported_by: tuple[EventRelation, ...]
    parser_name: str
    issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVENT_SCHEMA_VERSION,
            "source_document": dict(self.source_document),
            "event": self.event.to_dict(),
            "relations": {
                "reports": self.reports.to_dict(),
                "is_supported_by": [item.to_dict() for item in self.is_supported_by],
            },
            "parser": self.parser_name,
            "issues": list(self.issues),
        }


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "EventExtraction",
    "EventNode",
    "EventRelation",
]
