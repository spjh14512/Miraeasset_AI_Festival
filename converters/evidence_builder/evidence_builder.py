"""Assemble final TEXT and TABLE evidence from canonical section block references."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

from converters.common.document_loader import DocumentLoadStatus, load_document
from converters.common.source_models import (
    DocumentContext,
    DocumentSyntax,
    EvidenceContext,
    SourceRef,
)
from converters.common.source_resolver import SourceResolver
from converters.evidence_builder.evidence_models import (
    CanonicalEvidence,
    EvidenceType,
    FinalEvidenceContext,
)
from converters.evidence_builder.markdown_renderer import (
    render_kv,
    render_record,
    render_text,
    render_unknown,
)
from converters.paragraph_parser.paragraph_models import (
    CanonicalParagraph,
    ParagraphParseStatus,
)
from converters.paragraph_parser.paragraph_parser import (
    parse_paragraph_element,
    parse_span_run_elements,
)
from converters.table_context_resolver.table_context_resolver import (
    resolve_table_contexts,
)
from converters.table_parser.table_models import (
    CanonicalTable,
    SourceSyntax,
    TableContext,
    TableType,
)
from converters.table_parser.table_parser import parse_table, parse_table_group


@dataclass(frozen=True, slots=True)
class _ParsedBlock:
    value: CanonicalParagraph | CanonicalTable
    primary_source_refs: tuple[SourceRef, ...]
    source_block_order: int
    table_subindex: int | None = None


def source_ref_from_dict(value: Mapping[str, Any]) -> SourceRef:
    return SourceRef(
        syntax=DocumentSyntax(str(value["syntax"])),
        element_path=value.get("element_path"),
        html_id=value.get("html_id"),
        table_index=value.get("table_index"),
    )


def _unique_refs(refs: Sequence[SourceRef]) -> tuple[SourceRef, ...]:
    return tuple(dict.fromkeys(refs))


def _final_context(table: CanonicalTable) -> FinalEvidenceContext:
    return FinalEvidenceContext(
        section_path=table.context.section_path,
        title=table.title,
        units=table.units,
        captions=table.captions,
        notes=table.notes,
    )


def _content_hash(
    evidence_type: EvidenceType,
    context: FinalEvidenceContext,
    payload: Mapping[str, Any],
    table_type: str | None = None,
) -> str:
    canonical = json.dumps(
        {
            "evidence_type": evidence_type.value,
            "table_type": table_type,
            "context": context.to_dict(),
            "payload": payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _table_refs(
    table: CanonicalTable,
    primary: Sequence[SourceRef],
    consumed: Sequence[SourceRef],
) -> tuple[SourceRef, ...]:
    context_refs = [
        block.source_ref
        for block in table.context.blocks
        if block.source_ref is not None
    ]
    return _unique_refs(tuple(primary) + tuple(consumed) + tuple(context_refs))


def _kv_payload(table: CanonicalTable, parsed: _ParsedBlock) -> dict[str, Any]:
    fields = [
        {
            "raw_value": field.get("raw_value", ""),
            "key_paths": [
                list(path.get("path", [])) for path in field.get("key_paths", [])
            ],
            "source_cell": field.get("value_cell"),
        }
        for field in table.content.get("fields", [])
    ]
    return {
        "source_block_order": parsed.source_block_order,
        "fields": fields,
    }


def _record_payloads(
    table: CanonicalTable,
    parsed: _ParsedBlock,
) -> list[dict[str, Any]]:
    columns = {
        column["id"]: column for column in table.content.get("columns", [])
    }
    cells = {cell.id: cell for cell in table.cells}
    payloads: list[dict[str, Any]] = []
    for record in table.content.get("records", []):
        values: list[dict[str, Any]] = []
        for column_id, cell_id in record.get("cells_by_column", {}).items():
            column = columns.get(column_id, {})
            cell = cells.get(cell_id)
            values.append(
                {
                    "column_id": column_id,
                    "header_path": list(column.get("header_path", [])),
                    "raw_value": cell.raw_value if cell is not None else "",
                    "source_cell": cell_id,
                }
            )
        payloads.append(
            {
                "source_block_order": parsed.source_block_order,
                "record_index": record.get("record_index"),
                "source_row": record.get("source_row"),
                "row_type": record.get("row_type"),
                "row_context": list(record.get("row_context", [])),
                "values": values,
            }
        )
    return payloads


def _unknown_payload(table: CanonicalTable, parsed: _ParsedBlock) -> dict[str, Any]:
    cells = {cell.id: cell for cell in table.cells}
    rows = [
        [cells[cell_id].raw_value if cell_id in cells else "" for cell_id in row]
        for row in table.content.get("rows", [])
    ]
    return {"source_block_order": parsed.source_block_order, "rows": rows}


def _append_evidence(
    evidence: list[CanonicalEvidence],
    *,
    source_index: int,
    section_id: str,
    evidence_type: EvidenceType,
    context: FinalEvidenceContext,
    source_refs: tuple[SourceRef, ...],
    markdown: str,
    payload: Mapping[str, Any],
    table: CanonicalTable | None = None,
) -> None:
    order = len(evidence)
    table_type = table.table_type.value if table is not None else None
    evidence.append(
        CanonicalEvidence(
            id=f"src{source_index}:{section_id}:e{order}",
            evidence_type=evidence_type,
            table_type=table_type,
            section_id=f"src{source_index}:{section_id}",
            order=order,
            context=context,
            source_refs=source_refs,
            markdown=markdown,
            payload=payload,
            content_sha256=_content_hash(
                evidence_type,
                context,
                payload,
                table_type,
            ),
            parse_status=(
                table.parse_status.value
                if table is not None and table.parse_status.value != "SUCCESS"
                else None
            ),
            issues=(
                tuple(issue.to_dict() for issue in table.issues)
                if table is not None
                else ()
            ),
        )
    )


def _first_table_index(element: ET.Element, resolver: SourceResolver) -> int | None:
    for descendant in element.iter():
        if descendant.tag.rsplit("}", 1)[-1].upper() == "TABLE":
            return resolver.table_index(descendant)
    return None


def build_source_evidence(
    source: str | bytes,
    *,
    section_collection: Mapping[str, Any],
    document_context: DocumentContext,
    source_index: int = 0,
) -> dict[str, Any]:
    """Resolve one canonical section collection and assemble final evidence."""
    syntax = DocumentSyntax(str(section_collection["syntax"]))
    loaded = load_document(source, syntax=syntax)
    if loaded.root is None:
        return {
            "status": "FAILED",
            "evidence": [],
            "issues": [issue.to_dict() for issue in loaded.issues],
            "stats": {"TEXT": 0, "TABLE": 0, "IMAGE_SKIPPED": 0},
        }

    resolver = SourceResolver(loaded)
    evidence: list[CanonicalEvidence] = []
    issues: list[dict[str, Any]] = [issue.to_dict() for issue in loaded.issues]
    skipped_images = 0
    unassociated_layouts = 0
    paragraph_status = (
        ParagraphParseStatus.RECOVERED
        if loaded.status == DocumentLoadStatus.RECOVERED
        else ParagraphParseStatus.SUCCESS
    )

    for section in section_collection.get("sections", []):
        section_id = str(section["id"])
        section_path = tuple(str(item) for item in section.get("section_path", []))
        content_context = EvidenceContext(section_path=section_path)
        parsed: list[_ParsedBlock] = []

        for block in sorted(section.get("blocks", []), key=lambda item: item["order"]):
            block_type = str(block["block_type"])
            refs = tuple(source_ref_from_dict(item) for item in block["source_refs"])
            block_order = int(block["order"])
            if block_type == "IMAGE":
                skipped_images += 1
                continue
            try:
                resolved = [resolver.resolve(source_ref) for source_ref in refs]
                if block_type == "P":
                    paragraph = parse_paragraph_element(
                        resolved[0].element,
                        document_context=document_context,
                        source_ref=refs[0],
                        paragraph_index=block_order,
                        content_context=content_context,
                        parse_status=paragraph_status,
                    )
                    parsed.append(_ParsedBlock(paragraph, refs, block_order))
                elif block_type == "SPAN_RUN":
                    paragraph = parse_span_run_elements(
                        tuple(
                            (item.element, source_ref)
                            for item, source_ref in zip(resolved, refs, strict=True)
                        ),
                        document_context=document_context,
                        paragraph_index=block_order,
                        content_context=content_context,
                        parse_status=paragraph_status,
                    )
                    parsed.append(_ParsedBlock(paragraph, refs, block_order))
                elif block_type == "TABLE":
                    table = parse_table(
                        resolved[0].element,
                        context=TableContext(
                            source_path=document_context.source_path,
                            table_index=resolved[0].table_index,
                            source_ref=refs[0],
                        ),
                        content_context=content_context,
                        syntax=SourceSyntax(syntax.value),
                    )
                    parsed.append(_ParsedBlock(table, refs, block_order))
                elif block_type == "TABLE_GROUP":
                    group = parse_table_group(
                        resolved[0].element,
                        context=TableContext(
                            source_path=document_context.source_path,
                            table_index=_first_table_index(
                                resolved[0].element,
                                resolver,
                            ),
                            source_ref=refs[0],
                        ),
                        content_context=content_context,
                        syntax=SourceSyntax(syntax.value),
                    )
                    parsed.extend(
                        _ParsedBlock(table, refs, block_order, table_subindex)
                        for table_subindex, table in enumerate(group.tables)
                    )
                else:
                    issues.append(
                        {
                            "code": "UNSUPPORTED_BLOCK_TYPE",
                            "severity": "WARNING",
                            "message": f"Unsupported section block: {block_type}",
                        }
                    )
            except Exception as error:
                issues.append(
                    {
                        "code": "BLOCK_BUILD_FAILED",
                        "severity": "ERROR",
                        "message": (
                            f"section={section_id} block={block_order} "
                            f"{type(error).__name__}: {error}"
                        ),
                    }
                )

        values = tuple(item.value for item in parsed)
        resolution = resolve_table_contexts(values, section_path=section_path)
        bundles = {bundle.table_block_index: bundle for bundle in resolution.tables}
        consumed_refs = set(resolution.consumed_source_refs)

        for index, item in enumerate(parsed):
            value = item.value
            if isinstance(value, CanonicalParagraph):
                if any(source_ref in consumed_refs for source_ref in value.source_refs):
                    continue
                context = FinalEvidenceContext(section_path=section_path)
                payload: dict[str, Any] = {
                    "source_block_order": item.source_block_order,
                    "source_kind": value.source_kind.value,
                    "text": value.text,
                }
                if value.leading_marker is not None:
                    payload["leading_marker"] = value.leading_marker.to_dict()
                _append_evidence(
                    evidence,
                    source_index=source_index,
                    section_id=section_id,
                    evidence_type=EvidenceType.TEXT,
                    context=context,
                    source_refs=value.source_refs,
                    markdown=render_text(value.text, context),
                    payload=payload,
                )
                continue

            if value.table_type == TableType.LAYOUT_TABLE:
                if not any(
                    source_ref in consumed_refs for source_ref in item.primary_source_refs
                ):
                    unassociated_layouts += 1
                continue

            bundle = bundles.get(index)
            table = bundle.table if bundle is not None else value
            context = _final_context(table)
            source_refs = _table_refs(
                table,
                item.primary_source_refs,
                bundle.consumed_source_refs if bundle is not None else (),
            )
            if table.table_type == TableType.KV_TABLE:
                payload = _kv_payload(table, item)
                _append_evidence(
                    evidence,
                    source_index=source_index,
                    section_id=section_id,
                    evidence_type=EvidenceType.TABLE,
                    context=context,
                    source_refs=source_refs,
                    markdown=render_kv(payload["fields"], context),
                    payload=payload,
                    table=table,
                )
            elif table.table_type == TableType.R_TABLE:
                for payload in _record_payloads(table, item):
                    _append_evidence(
                        evidence,
                        source_index=source_index,
                        section_id=section_id,
                        evidence_type=EvidenceType.TABLE,
                        context=context,
                        source_refs=source_refs,
                        markdown=render_record(payload["values"], context),
                        payload=payload,
                        table=table,
                    )
            elif table.table_type == TableType.UNKNOWN:
                payload = _unknown_payload(table, item)
                _append_evidence(
                    evidence,
                    source_index=source_index,
                    section_id=section_id,
                    evidence_type=EvidenceType.TABLE,
                    context=context,
                    source_refs=source_refs,
                    markdown=render_unknown(payload["rows"], context),
                    payload=payload,
                    table=table,
                )

    text_count = sum(item.evidence_type == EvidenceType.TEXT for item in evidence)
    table_count = len(evidence) - text_count
    if unassociated_layouts:
        issues.append(
            {
                "code": "UNASSOCIATED_LAYOUT_TABLES",
                "severity": "INFO",
                "message": f"{unassociated_layouts} layout tables were not evidence.",
            }
        )
    has_error = any(issue.get("severity") == "ERROR" for issue in issues)
    status = (
        "PARTIAL"
        if has_error
        else "RECOVERED"
        if loaded.status == DocumentLoadStatus.RECOVERED
        else "SUCCESS"
    )
    return {
        "status": status,
        "evidence": [item.to_dict() for item in evidence],
        "issues": issues,
        "stats": {
            "TEXT": text_count,
            "TABLE": table_count,
            "IMAGE_SKIPPED": skipped_images,
            "LAYOUT_TABLE_UNASSOCIATED": unassociated_layouts,
        },
    }


__all__ = ["build_source_evidence", "source_ref_from_dict"]
