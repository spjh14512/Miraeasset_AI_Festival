"""Build section-level Evidence Fragments from chunker block references."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

from converters.common.document_loader import DocumentLoadStatus, load_document
from converters.common.source_models import DocumentContext, EvidenceContext, SourceRef
from converters.common.source_resolver import SourceResolver
from converters.evidence_builder.fragment_models import (
    EvidenceFragment,
    EvidenceType,
    FragmentEvidence,
    StorageMode,
    TableRecord,
)
from converters.evidence_builder.semantic_text_segmenter import (
    SemanticTextRole,
    SemanticTextSegment,
    segment_paragraph,
)
from converters.paragraph_parser.paragraph_models import (
    CanonicalParagraph,
    ParagraphParseStatus,
)
from converters.paragraph_parser.paragraph_parser import (
    parse_paragraph_element,
    parse_span_run_elements,
)
from converters.section_canonicalizer.section_models import (
    CanonicalSection,
    CanonicalSectionCollection,
    SectionBlockType,
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


_NAVIGATION_TEXT = re.compile(
    r"(?:본문\s*(?:위치로\s*)?이동|본문으로|목차로|맨\s*위로)",
    re.IGNORECASE,
)
_LAYOUT_UNKNOWN_TEXT = re.compile(
    r"(?:단위\s*[:：]|기준일|작성기준일|이행현황기준일|"
    r"제\s*\d+\s*기|재무상태표|손익계산서|자본변동표|현금흐름표)",
    re.IGNORECASE,
)
_COVER_UNKNOWN_TEXT = re.compile(
    r"(?:금융위원회|한국거래소\s*귀중|회\s*사\s*명\s*[:：]|"
    r"대\s*표\s*이\s*사\s*[:：]|본\s*점\s*소\s*재\s*지|정정일자)",
    re.IGNORECASE,
)
_SUBTOTAL_TEXT = re.compile(r"^\s*소\s*계(?:\s|\(|$)")
_TOTAL_TEXT = re.compile(r"^\s*(?:총\s*계|합\s*계)(?:\s|\(|$)")
_BULLET_MARKER = re.compile(r"^[-ㆍ·•]$")
_DISCLOSURE_REFNO = re.compile(r"^\d{14}$")


@dataclass(frozen=True, slots=True)
class _ParsedBlock:
    value: CanonicalParagraph | CanonicalTable
    primary_source_refs: tuple[SourceRef, ...]
    source_block_order: int
    table_subindex: int | None = None


@dataclass(frozen=True, slots=True)
class _SemanticBlock:
    parsed_block_index: int
    parsed: _ParsedBlock
    segment: SemanticTextSegment | None = None


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _paragraph_references(
    paragraph: CanonicalParagraph,
) -> tuple[dict[str, str], ...]:
    references: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for fragment in paragraph.fragments:
        refno = fragment.source_attributes.get("REFNO", "").strip()
        text = _clean_text(fragment.text)
        if not _DISCLOSURE_REFNO.fullmatch(refno) or not text:
            continue
        identity = ("disclosure", refno, text)
        if identity in seen:
            continue
        seen.add(identity)
        references.append(
            {"type": "disclosure", "refno": refno, "text": text}
        )
    return tuple(references)


def _unique_references(
    references: Sequence[Mapping[str, str]],
) -> tuple[dict[str, str], ...]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for reference in references:
        identity = (
            str(reference.get("type", "")),
            str(reference.get("refno", "")),
            str(reference.get("text", "")),
        )
        if identity in seen:
            continue
        seen.add(identity)
        result.append(dict(reference))
    return tuple(result)


def _first_table_index(element: ET.Element, resolver: SourceResolver) -> int | None:
    for descendant in element.iter():
        if descendant.tag.rsplit("}", 1)[-1].upper() == "TABLE":
            return resolver.table_index(descendant)
    return None


def _context_payload(table: CanonicalTable) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if table.title:
        result["title"] = table.title
    if table.units:
        result["units"] = list(table.units)
    if table.captions:
        result["captions"] = list(table.captions)
    if table.notes:
        result["notes"] = list(table.notes)
    return result


def _kv_payload(table: CanonicalTable) -> dict[str, Any]:
    payload = _context_payload(table)
    payload["fields"] = [
        {
            "key_paths": [
                list(path.get("path", []))
                for path in field.get("key_paths", [])
            ],
            "raw_value": field.get("raw_value", ""),
        }
        for field in table.content.get("fields", [])
    ]
    return payload


def _record_payloads(table: CanonicalTable) -> list[dict[str, Any]]:
    columns = list(table.content.get("columns", []))
    cells = {cell.id: cell for cell in table.cells}
    payloads: list[dict[str, Any]] = []
    for record in table.content.get("records", []):
        values: list[dict[str, Any]] = []
        cells_by_column = record.get("cells_by_column", {})
        for column in columns:
            column_id = str(column.get("id", ""))
            cell_id = cells_by_column.get(column_id)
            if cell_id is None:
                cell_id = cells_by_column.get(column.get("id"))
            cell = cells.get(str(cell_id)) if cell_id is not None else None
            values.append(
                {
                    "header_path": list(column.get("header_path", [])),
                    "raw_value": cell.raw_value if cell is not None else "",
                }
            )
        payload = _context_payload(table)
        payload.update(
            {
                "row_type": str(record.get("row_type", "UNKNOWN")),
                "row_context": list(record.get("row_context", [])),
                "values": values,
            }
        )
        payloads.append(payload)
    return payloads


def _headers(table: CanonicalTable) -> list[list[str]]:
    return [
        list(column.get("header_path", []))
        for column in table.content.get("columns", [])
    ]


def _unknown_rows(table: CanonicalTable) -> list[list[str]]:
    cells = {cell.id: cell for cell in table.cells}
    return [
        [
            cells[str(cell_id)].raw_value
            if cell_id is not None and str(cell_id) in cells
            else ""
            for cell_id in row
        ]
        for row in table.content.get("rows", [])
    ]


def _looks_like_layout_unknown(rows: Sequence[Sequence[str]]) -> bool:
    nonempty = [_clean_text(value) for row in rows for value in row if _clean_text(value)]
    if not nonempty:
        return True
    combined = " | ".join(nonempty)
    if _COVER_UNKNOWN_TEXT.search(combined):
        return True
    width = max((len(row) for row in rows), default=0)
    body_values = [
        _clean_text(value)
        for row in rows[1:]
        for value in row
        if _clean_text(value)
    ]
    if (
        len(rows) >= 2
        and width >= 3
        and any(re.search(r"\d", value) for value in body_values)
    ):
        return False
    return len(nonempty) <= 8 and _LAYOUT_UNKNOWN_TEXT.search(combined) is not None


def _fallback_row_type(values: Sequence[str]) -> str:
    if any(_SUBTOTAL_TEXT.match(value) for value in values):
        return "SUBTOTAL"
    if any(_TOTAL_TEXT.match(value) for value in values):
        return "TOTAL"
    return "DATA"


def _fallback_unknown(
    table: CanonicalTable,
) -> tuple[str, Any] | None:
    """Preserve non-layout UNKNOWN content using the nearest allowed schema."""
    rows = _unknown_rows(table)
    if _looks_like_layout_unknown(rows):
        return None
    nonempty_rows = [row for row in rows if any(_clean_text(value) for value in row)]
    if not nonempty_rows:
        return None
    width = max(len(row) for row in nonempty_rows)
    if len(nonempty_rows) >= 2 and width >= 2:
        header_row = nonempty_rows[0]
        headers = [
            _clean_text(header_row[index]) if index < len(header_row) else ""
            for index in range(width)
        ]
        records: list[dict[str, Any]] = []
        for row in nonempty_rows[1:]:
            raw_values = [
                _clean_text(row[index]) if index < len(row) else ""
                for index in range(width)
            ]
            if not any(raw_values):
                continue
            payload = _context_payload(table)
            payload.update(
                {
                    "row_type": _fallback_row_type(raw_values),
                    "row_context": [],
                    "values": [
                        {
                            "header_path": [header or f"column_{index + 1}"],
                            "raw_value": raw_value,
                        }
                        for index, (header, raw_value) in enumerate(
                            zip(headers, raw_values, strict=True)
                        )
                    ],
                }
            )
            records.append(payload)
        if records:
            return "R_TABLE", records
    flattened = [_clean_text(value) for row in nonempty_rows for value in row if _clean_text(value)]
    if len(flattened) == 2 and width == 2:
        if _BULLET_MARKER.fullmatch(flattened[0]):
            return "TEXT", {"text": f"{flattened[0]} {flattened[1]}"}
        payload = _context_payload(table)
        payload["fields"] = [
            {"key_paths": [[flattened[0]]], "raw_value": flattened[1]}
        ]
        return "KV_TABLE", payload
    return "TEXT", {"text": "\n".join(flattened)}


def _semantic_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _with_semantic_context(
    payload: Mapping[str, Any],
    *,
    heading_path: Sequence[str],
    leading_captions: Sequence[str] = (),
) -> dict[str, Any]:
    result = dict(payload)
    if heading_path:
        result["heading_path"] = list(heading_path)
    parser_captions = result.pop("captions", ())
    captions: list[str] = []
    seen: set[str] = set()
    for caption in (*leading_captions, *parser_captions):
        if caption in seen:
            continue
        seen.add(caption)
        captions.append(caption)
    if captions:
        result["captions"] = captions
    return result


def _semantic_blocks(
    parsed: Sequence[_ParsedBlock],
    *,
    consumed_refs: set[SourceRef],
) -> list[_SemanticBlock]:
    result: list[_SemanticBlock] = []
    for block_index, item in enumerate(parsed):
        if isinstance(item.value, CanonicalParagraph):
            if any(ref in consumed_refs for ref in item.value.source_refs):
                continue
            result.extend(
                _SemanticBlock(
                    parsed_block_index=block_index,
                    parsed=item,
                    segment=segment,
                )
                for segment in segment_paragraph(item.value)
            )
            continue
        result.append(
            _SemanticBlock(
                parsed_block_index=block_index,
                parsed=item,
            )
        )
    return result


def _captions_by_table(
    blocks: Sequence[_SemanticBlock],
) -> tuple[dict[int, tuple[SemanticTextSegment, ...]], set[int]]:
    captions: dict[int, tuple[SemanticTextSegment, ...]] = {}
    consumed: set[int] = set()
    for index, block in enumerate(blocks):
        value = block.parsed.value
        if (
            isinstance(value, CanonicalParagraph)
            or value.table_type in {TableType.LAYOUT_TABLE, TableType.UNKNOWN}
        ):
            continue
        candidates: list[SemanticTextSegment] = []
        cursor = index - 1
        while cursor >= 0:
            candidate_block = blocks[cursor]
            candidate = candidate_block.segment
            if candidate is None:
                candidate_value = candidate_block.parsed.value
                if (
                    isinstance(candidate_value, CanonicalTable)
                    and (
                        candidate_value.table_type == TableType.LAYOUT_TABLE
                        or (
                            candidate_value.table_type == TableType.UNKNOWN
                            and _fallback_unknown(candidate_value) is None
                        )
                    )
                ):
                    cursor -= 1
                    continue
                break
            if candidate.role != SemanticTextRole.TABLE_CAPTION:
                break
            candidates.append(candidate)
            consumed.add(cursor)
            cursor -= 1
        if candidates:
            resolved = tuple(reversed(candidates))
            captions[index] = resolved
            cursor = index + 1
            while cursor < len(blocks):
                following = blocks[cursor]
                following_value = following.parsed.value
                if following.segment is not None:
                    break
                if not isinstance(following_value, CanonicalTable):
                    break
                if (
                    following_value.table_type == TableType.LAYOUT_TABLE
                    or (
                        following_value.table_type == TableType.UNKNOWN
                        and _fallback_unknown(following_value) is None
                    )
                ):
                    cursor += 1
                    continue
                if following_value.table_type in {
                    TableType.KV_TABLE,
                    TableType.R_TABLE,
                }:
                    captions.setdefault(cursor, resolved)
                    cursor += 1
                    continue
                break
    return captions, consumed


def _parse_blocks(
    section: CanonicalSection,
    *,
    resolver: SourceResolver,
    document_context: DocumentContext,
    syntax: SourceSyntax,
    paragraph_status: ParagraphParseStatus,
    issues: list[dict[str, Any]],
) -> tuple[list[_ParsedBlock], int]:
    parsed: list[_ParsedBlock] = []
    skipped_images = 0
    content_context = EvidenceContext(section_path=section.section_path)

    for block in sorted(section.blocks, key=lambda item: item.order):
        refs = block.source_refs
        if block.block_type == SectionBlockType.IMAGE:
            skipped_images += 1
            continue
        try:
            resolved = [resolver.resolve(source_ref) for source_ref in refs]
            if block.block_type == SectionBlockType.P:
                parsed.append(
                    _ParsedBlock(
                        parse_paragraph_element(
                            resolved[0].element,
                            document_context=document_context,
                            source_ref=refs[0],
                            paragraph_index=block.order,
                            content_context=content_context,
                            parse_status=paragraph_status,
                        ),
                        refs,
                        block.order,
                    )
                )
            elif block.block_type == SectionBlockType.SPAN_RUN:
                parsed.append(
                    _ParsedBlock(
                        parse_span_run_elements(
                            tuple(
                                (item.element, source_ref)
                                for item, source_ref in zip(
                                    resolved,
                                    refs,
                                    strict=True,
                                )
                            ),
                            document_context=document_context,
                            paragraph_index=block.order,
                            content_context=content_context,
                            parse_status=paragraph_status,
                        ),
                        refs,
                        block.order,
                    )
                )
            elif block.block_type == SectionBlockType.TABLE:
                parsed.append(
                    _ParsedBlock(
                        parse_table(
                            resolved[0].element,
                            context=TableContext(
                                source_path=document_context.source_path,
                                table_index=resolved[0].table_index,
                                source_ref=refs[0],
                            ),
                            content_context=content_context,
                            syntax=syntax,
                        ),
                        refs,
                        block.order,
                    )
                )
            elif block.block_type == SectionBlockType.TABLE_GROUP:
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
                    syntax=syntax,
                )
                parsed.extend(
                    _ParsedBlock(table, refs, block.order, subindex)
                    for subindex, table in enumerate(group.tables)
                )
            else:
                issues.append(
                    {
                        "code": "UNSUPPORTED_BLOCK_TYPE",
                        "severity": "WARNING",
                        "section_id": section.id,
                        "block_order": block.order,
                        "message": f"Unsupported block type: {block.block_type.value}",
                    }
                )
        except Exception as error:
            issues.append(
                {
                    "code": "BLOCK_BUILD_FAILED",
                    "severity": "ERROR",
                    "section_id": section.id,
                    "block_order": block.order,
                    "message": f"{type(error).__name__}: {error}",
                }
            )
    return parsed, skipped_images


def build_section_fragment(
    section: CanonicalSection,
    *,
    global_section_id: str,
    rcept_no: str,
    source_index: int,
    resolver: SourceResolver,
    document_context: DocumentContext,
    syntax: SourceSyntax,
    paragraph_status: ParagraphParseStatus,
) -> tuple[EvidenceFragment, list[dict[str, Any]], dict[str, int]]:
    """Build one fragment while keeping operational issues outside the JSON."""
    issues: list[dict[str, Any]] = []
    parsed, skipped_images = _parse_blocks(
        section,
        resolver=resolver,
        document_context=document_context,
        syntax=syntax,
        paragraph_status=paragraph_status,
        issues=issues,
    )
    values = tuple(item.value for item in parsed)
    resolution = resolve_table_contexts(values, section_path=section.section_path)
    bundles = {bundle.table_block_index: bundle for bundle in resolution.tables}
    consumed_refs = set(resolution.consumed_source_refs)
    paragraph_references_by_ref: dict[SourceRef, tuple[dict[str, str], ...]] = {}
    for item in parsed:
        if not isinstance(item.value, CanonicalParagraph):
            continue
        item_references = _paragraph_references(item.value)
        if not item_references:
            continue
        for source_ref in item.value.source_refs:
            paragraph_references_by_ref[source_ref] = item_references

    evidence: list[FragmentEvidence] = []
    records: list[TableRecord] = []
    seen_tables: set[str] = set()
    previous_text: str | None = None
    r_table_index = 0
    stats = {
        "TEXT": 0,
        "TEXT_SEGMENT": 0,
        "TEXT_HEADING_CONTEXT": 0,
        "TEXT_TABLE_CAPTION_CONTEXT": 0,
        "TEXT_REFERENCE_NOTICE": 0,
        "KV_TABLE": 0,
        "R_TABLE": 0,
        "R_TABLE_RECORD": 0,
        "LAYOUT_SKIPPED": 0,
        "UNKNOWN_SKIPPED": 0,
        "UNKNOWN_FALLBACK_TEXT": 0,
        "UNKNOWN_FALLBACK_KV": 0,
        "UNKNOWN_FALLBACK_R": 0,
        "DUPLICATE_SKIPPED": 0,
        "IMAGE_SKIPPED": skipped_images,
    }

    def append_evidence(
        evidence_type: EvidenceType,
        payload: Mapping[str, Any],
        *,
        table_type: str | None = None,
        storage_mode: StorageMode | None = None,
        references: tuple[Mapping[str, str], ...] = (),
    ) -> None:
        order = len(evidence)
        evidence.append(
            FragmentEvidence(
                evidence_id=(
                    f"evidence:{rcept_no}:src{source_index}:{section.id}:e{order}"
                ),
                evidence_type=evidence_type,
                table_type=table_type,
                storage_mode=storage_mode,
                order=order,
                payload=payload,
                references=references,
            )
        )

    semantic_blocks = _semantic_blocks(parsed, consumed_refs=consumed_refs)
    captions_by_table, consumed_caption_indexes = _captions_by_table(semantic_blocks)
    stats["TEXT_SEGMENT"] = sum(
        block.segment is not None for block in semantic_blocks
    )
    heading_stack: list[tuple[int, str]] = []

    def current_heading_path() -> tuple[str, ...]:
        return tuple(text for _, text in heading_stack)

    def update_heading(segment: SemanticTextSegment) -> None:
        level = segment.heading_level or 4
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, segment.text))

    def heading_has_target(index: int) -> bool:
        for later in semantic_blocks[index + 1 :]:
            if later.segment is not None:
                if later.segment.role != SemanticTextRole.HEADING:
                    return True
                continue
            later_value = later.parsed.value
            if (
                isinstance(later_value, CanonicalTable)
                and later_value.table_type != TableType.LAYOUT_TABLE
            ):
                return True
        return False

    for semantic_index, semantic_block in enumerate(semantic_blocks):
        item = semantic_block.parsed
        value = item.value
        segment = semantic_block.segment
        if segment is not None:
            text = _clean_text(segment.text)
            if not text or _NAVIGATION_TEXT.search(text):
                stats["LAYOUT_SKIPPED"] += 1
                continue
            if semantic_index in consumed_caption_indexes:
                stats["TEXT_TABLE_CAPTION_CONTEXT"] += 1
                previous_text = None
                continue
            if (
                segment.role == SemanticTextRole.HEADING
                and heading_has_target(semantic_index)
            ):
                update_heading(segment)
                stats["TEXT_HEADING_CONTEXT"] += 1
                previous_text = None
                continue
            output_role = (
                segment.role
                if segment.role
                in {
                    SemanticTextRole.BODY,
                    SemanticTextRole.NOTE,
                    SemanticTextRole.REFERENCE_NOTICE,
                }
                else SemanticTextRole.BODY
            )
            payload: dict[str, Any] = {
                "text": text,
                "text_role": output_role.value,
            }
            if current_heading_path():
                payload["heading_path"] = list(current_heading_path())
            segment_references = _unique_references(segment.references)
            normalized = _semantic_json(
                {
                    "payload": {
                        **payload,
                        "text": text.casefold(),
                    },
                    "references": segment_references,
                }
            )
            if normalized == previous_text:
                stats["DUPLICATE_SKIPPED"] += 1
                continue
            append_evidence(
                EvidenceType.TEXT,
                payload,
                references=segment_references,
            )
            stats["TEXT"] += 1
            if output_role == SemanticTextRole.REFERENCE_NOTICE:
                stats["TEXT_REFERENCE_NOTICE"] += 1
            previous_text = normalized
            continue

        previous_text = None
        if value.table_type == TableType.LAYOUT_TABLE:
            stats["LAYOUT_SKIPPED"] += 1
            continue

        bundle = bundles.get(semantic_block.parsed_block_index)
        table = bundle.table if bundle is not None else value
        caption_segments = captions_by_table.get(semantic_index, ())
        caption_texts = tuple(segment.text for segment in caption_segments)
        table_references = _unique_references(
            tuple(
                reference
                for source_ref in (
                    bundle.consumed_source_refs if bundle is not None else ()
                )
                for reference in paragraph_references_by_ref.get(source_ref, ())
            )
            + tuple(
                reference
                for caption_segment in caption_segments
                for reference in caption_segment.references
            )
        )
        if table.table_type == TableType.UNKNOWN:
            fallback = _fallback_unknown(table)
            if fallback is None:
                stats["LAYOUT_SKIPPED"] += 1
                continue
            fallback_type, fallback_payload = fallback
            issues.append(
                {
                    "code": f"UNKNOWN_TABLE_FALLBACK_{fallback_type}",
                    "severity": "INFO",
                    "section_id": global_section_id,
                    "block_order": item.source_block_order,
                    "message": "Unclassified non-layout table content was preserved.",
                }
            )
            if fallback_type == "TEXT":
                text = str(fallback_payload["text"])
                fallback_payload = _with_semantic_context(
                    {
                        **fallback_payload,
                        "text_role": SemanticTextRole.BODY.value,
                    },
                    heading_path=current_heading_path(),
                )
                normalized = _semantic_json(
                    {
                        "payload": {
                            **fallback_payload,
                            "text": text.casefold(),
                        },
                        "references": table_references,
                    }
                )
                if normalized == previous_text:
                    stats["DUPLICATE_SKIPPED"] += 1
                else:
                    append_evidence(
                        EvidenceType.TEXT,
                        fallback_payload,
                        references=table_references,
                    )
                    stats["TEXT"] += 1
                    stats["UNKNOWN_FALLBACK_TEXT"] += 1
                    previous_text = normalized
                continue
            previous_text = None
            if fallback_type == "KV_TABLE":
                signature_payload = dict(fallback_payload)
                fallback_payload = _with_semantic_context(
                    fallback_payload,
                    heading_path=current_heading_path(),
                    leading_captions=caption_texts,
                )
                signature = _semantic_json(
                    {
                        "table_type": fallback_type,
                        "payload": signature_payload,
                        "references": table_references,
                    }
                )
                if signature in seen_tables:
                    stats["DUPLICATE_SKIPPED"] += 1
                else:
                    seen_tables.add(signature)
                    append_evidence(
                        EvidenceType.TABLE,
                        fallback_payload,
                        table_type=TableType.KV_TABLE.value,
                        references=table_references,
                    )
                    stats["KV_TABLE"] += 1
                    stats["UNKNOWN_FALLBACK_KV"] += 1
                continue
            fallback_records = list(fallback_payload)
            signature = _semantic_json(
                {
                    "table_type": fallback_type,
                    "records": fallback_records,
                    "references": table_references,
                }
            )
            if signature in seen_tables:
                stats["DUPLICATE_SKIPPED"] += 1
                continue
            seen_tables.add(signature)
            current_table_index = r_table_index
            r_table_index += 1
            table_id = (
                f"rtable:{rcept_no}:src{source_index}:{section.id}:"
                f"t{current_table_index}"
            )
            payload = _with_semantic_context(
                _context_payload(table),
                heading_path=current_heading_path(),
                leading_captions=caption_texts,
            )
            payload.update(
                {
                    "table_id": table_id,
                    "headers": [
                        list(value.get("header_path", []))
                        for value in fallback_records[0].get("values", [])
                    ],
                    "record_count": len(fallback_records),
                }
            )
            append_evidence(
                EvidenceType.TABLE,
                payload,
                table_type=TableType.R_TABLE.value,
                storage_mode=StorageMode.SECTION_RECORDS,
                references=table_references,
            )
            stats["R_TABLE"] += 1
            for record_index, record_payload in enumerate(fallback_records):
                records.append(
                    TableRecord(
                        table_id=table_id,
                        record_index=record_index,
                        row_type=str(record_payload.get("row_type", "UNKNOWN")),
                        row_context=tuple(record_payload.get("row_context", [])),
                        values=tuple(
                            str(value.get("raw_value", ""))
                            for value in record_payload.get("values", [])
                        ),
                    )
                )
            stats["R_TABLE_RECORD"] += len(fallback_records)
            stats["UNKNOWN_FALLBACK_R"] += 1
            continue

        if table.table_type == TableType.KV_TABLE:
            signature_payload = _kv_payload(table)
            payload = _with_semantic_context(
                signature_payload,
                heading_path=current_heading_path(),
                leading_captions=caption_texts,
            )
            signature = _semantic_json(
                {
                    "table_type": TableType.KV_TABLE.value,
                    "payload": signature_payload,
                    "references": table_references,
                }
            )
            if signature in seen_tables:
                stats["DUPLICATE_SKIPPED"] += 1
                continue
            seen_tables.add(signature)
            append_evidence(
                EvidenceType.TABLE,
                payload,
                table_type=TableType.KV_TABLE.value,
                references=table_references,
            )
            stats["KV_TABLE"] += 1
            continue

        if table.table_type != TableType.R_TABLE:
            continue

        record_payloads = _record_payloads(table)
        signature = _semantic_json(
            {
                "table_type": TableType.R_TABLE.value,
                "records": record_payloads,
                "references": table_references,
            }
        )
        if signature in seen_tables:
            stats["DUPLICATE_SKIPPED"] += 1
            continue
        seen_tables.add(signature)
        current_table_index = r_table_index
        r_table_index += 1

        table_id = (
            f"rtable:{rcept_no}:src{source_index}:{section.id}:"
            f"t{current_table_index}"
        )
        payload = _with_semantic_context(
            _context_payload(table),
            heading_path=current_heading_path(),
            leading_captions=caption_texts,
        )
        payload.update(
            {
                "table_id": table_id,
                "headers": _headers(table),
                "record_count": len(record_payloads),
            }
        )
        append_evidence(
            EvidenceType.TABLE,
            payload,
            table_type=TableType.R_TABLE.value,
            storage_mode=StorageMode.SECTION_RECORDS,
            references=table_references,
        )
        stats["R_TABLE"] += 1
        for record_index, record_payload in enumerate(record_payloads):
            records.append(
                TableRecord(
                    table_id=table_id,
                    record_index=record_index,
                    row_type=str(record_payload.get("row_type", "UNKNOWN")),
                    row_context=tuple(record_payload.get("row_context", [])),
                    values=tuple(
                        str(value.get("raw_value", ""))
                        for value in record_payload.get("values", [])
                    ),
                )
            )
        stats["R_TABLE_RECORD"] += len(record_payloads)

    return (
        EvidenceFragment(
            section_id=global_section_id,
            evidence_list=tuple(evidence),
            records=tuple(records),
        ),
        issues,
        stats,
    )


def build_source_fragments(
    source: str | bytes,
    *,
    section_collection: CanonicalSectionCollection,
    kept_section_ids: set[str],
    document_context: DocumentContext,
    source_index: int,
) -> dict[str, Any]:
    """Build all graph-facing section fragments for one source document."""
    loaded = load_document(source, syntax=section_collection.syntax)
    if loaded.root is None:
        return {
            "status": "FAILED",
            "fragments": [],
            "issues": [issue.to_dict() for issue in loaded.issues],
            "stats": {},
        }

    resolver = SourceResolver(loaded)
    paragraph_status = (
        ParagraphParseStatus.RECOVERED
        if loaded.status == DocumentLoadStatus.RECOVERED
        else ParagraphParseStatus.SUCCESS
    )
    syntax = SourceSyntax(section_collection.syntax.value)
    fragments: list[EvidenceFragment] = []
    issues: list[dict[str, Any]] = [issue.to_dict() for issue in loaded.issues]
    aggregate: dict[str, int] = {}

    for section in section_collection.sections:
        if section.id not in kept_section_ids:
            continue
        global_section_id = (
            f"section:{document_context.rcept_no}:src{source_index}:{section.id}"
        )
        fragment, section_issues, stats = build_section_fragment(
            section,
            global_section_id=global_section_id,
            rcept_no=document_context.rcept_no,
            source_index=source_index,
            resolver=resolver,
            document_context=document_context,
            syntax=syntax,
            paragraph_status=paragraph_status,
        )
        fragments.append(fragment)
        issues.extend(section_issues)
        for key, value in stats.items():
            aggregate[key] = aggregate.get(key, 0) + value

    status = (
        "PARTIAL"
        if any(issue.get("severity") == "ERROR" for issue in issues)
        else "RECOVERED"
        if loaded.status == DocumentLoadStatus.RECOVERED
        or any(issue.get("code") == "UNKNOWN_TABLE_SKIPPED" for issue in issues)
        else "SUCCESS"
    )
    return {
        "status": status,
        "fragments": [fragment.to_dict() for fragment in fragments],
        "issues": issues,
        "stats": aggregate,
    }


__all__ = [
    "build_section_fragment",
    "build_source_fragments",
]
