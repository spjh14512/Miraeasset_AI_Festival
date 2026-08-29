"""Build one Event and its graph relationships from one latest disclosure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from converters.event_extractor.category_mapping import EventCategory
from converters.event_extractor.content_parsers import (
    collect_evidence_facts,
    content_parser_for,
    neo4j_evidence_id,
)
from converters.event_extractor.event_models import (
    EventExtraction,
    EventNode,
    EventRelation,
)


def extract_event(
    manifest_row: Mapping[str, Any],
    *,
    category: EventCategory,
    fragments: Sequence[Mapping[str, Any]],
) -> EventExtraction:
    """Extract one disclosure-level Event from canonical Evidence fragments."""
    rcept_no = str(manifest_row.get("rcept_no", "")).strip()
    doc_group = str(manifest_row.get("doc_group", "")).strip()
    if not rcept_no or doc_group not in {"major", "exchange"}:
        raise ValueError("Event extraction requires a major/exchange rcept_no")
    receipt_date = str(manifest_row.get("rcept_dt", "")).strip()
    company_name = str(
        manifest_row.get("listed_name") or manifest_row.get("corp_name") or "발행회사"
    ).strip()

    facts = collect_evidence_facts(fragments, doc_group=doc_group)
    parser = content_parser_for(category.event_type, category.event_subtype)
    parsed = parser.parse(
        facts,
        event_type=category.event_type,
        event_subtype=category.event_subtype,
        receipt_date=receipt_date,
        company_name=company_name,
    )
    if not parsed.evidence_ids:
        raise ValueError(f"Event has no supporting Evidence: {rcept_no}")

    event_id = f"event:{rcept_no}"
    disclosure_id = f"d{rcept_no}"
    event = EventNode(
        id=event_id,
        event_type=category.event_type,
        event_subtype=category.event_subtype,
        event_date=parsed.event_date,
        content=parsed.content,
    )
    reports = EventRelation("REPORTS", disclosure_id, event_id)
    supports = tuple(
        EventRelation(
            "IS_SUPPORTED_BY",
            event_id,
            neo4j_evidence_id(evidence_id, rcept_no),
        )
        for evidence_id in parsed.evidence_ids
    )
    issues = (
        ("EVENT_DATE_FALLBACK_TO_RECEIPT_DATE",)
        if parsed.used_receipt_date
        else ()
    )
    return EventExtraction(
        source_document={
            "doc_id": str(manifest_row.get("doc_id", "")),
            "rcept_no": rcept_no,
            "doc_group": doc_group,
            "disclosure_id": disclosure_id,
            "is_latest_version": True,
        },
        event=event,
        reports=reports,
        is_supported_by=supports,
        parser_name=parsed.parser_name,
        issues=issues,
    )


__all__ = ["extract_event"]
