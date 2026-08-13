"""Resolve conservative before/after context for tables in section order."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Sequence, TypeAlias

from converters.common.source_models import (
    ContextPosition,
    ContextRole,
    EvidenceContext,
    EvidenceContextBlock,
    SourceRef,
)
from converters.paragraph_parser.paragraph_models import CanonicalParagraph
from converters.table_context_resolver.resolver_models import (
    TableContextResolution,
    TableEvidenceBundle,
)
from converters.table_parser.table_models import CanonicalTable, TableType


CanonicalSectionBlock: TypeAlias = CanonicalParagraph | CanonicalTable

_UNIT_TEXT = re.compile(r"^\(?\s*단위\s*[:：]", re.IGNORECASE)
_NOTE_TEXT = re.compile(
    r"^(?:※|[*]|주\s*\d+\s*[):.]|[-ㆍ·]\s*상기)",
    re.IGNORECASE,
)
_BRACKETED_TITLE = re.compile(
    r"^(?:\[.*\]|【.*】|〔.*〕|〈.*〉|《.*》|「.*」|『.*』)$"
)
_CAPTION_CLASS = re.compile(r"(?:^|[-_\s])(?:table|tbl)[-_]?caption(?:$|[-_\s])")
_TITLE_CLASS = re.compile(r"(?:^|[-_\s])(?:table|tbl)[-_]?title(?:$|[-_\s])")


def _paragraph_role(paragraph: CanonicalParagraph) -> ContextRole | None:
    text = paragraph.text.strip()
    if _UNIT_TEXT.match(text):
        return ContextRole.UNIT
    if _NOTE_TEXT.match(text):
        return ContextRole.NOTE

    class_name = paragraph.source_attributes.get("CLASS", "").lower()
    if _CAPTION_CLASS.search(class_name):
        return ContextRole.CAPTION
    if _TITLE_CLASS.search(class_name):
        return ContextRole.TITLE
    if len(text) <= 200 and _BRACKETED_TITLE.fullmatch(text):
        return ContextRole.TITLE
    return None


def _context_block(
    paragraph: CanonicalParagraph,
    role: ContextRole,
    position: ContextPosition,
) -> EvidenceContextBlock:
    return EvidenceContextBlock(
        role=role,
        position=position,
        text=paragraph.text,
        source_ref=(paragraph.source_refs[0] if paragraph.source_refs else None),
    )


def _table_source_ref(table: CanonicalTable) -> SourceRef | None:
    value = table.source.get("source_ref")
    if not isinstance(value, dict):
        return None
    try:
        from converters.common.source_models import DocumentSyntax

        return SourceRef(
            syntax=DocumentSyntax(value["syntax"]),
            element_path=value.get("element_path"),
            html_id=value.get("html_id"),
            table_index=value.get("table_index"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _layout_context_block(
    table: CanonicalTable,
    position: ContextPosition,
) -> EvidenceContextBlock | None:
    role_value = table.content.get("layout_role")
    values = table.content.get("values", [])
    if role_value not in {"TITLE", "UNIT", "NOTE"} or not values:
        return None
    return EvidenceContextBlock(
        role=ContextRole(role_value),
        position=position,
        text="\n".join(str(value) for value in values if value),
        source_ref=_table_source_ref(table),
    )


def _merge_context(
    table: CanonicalTable,
    blocks: Sequence[EvidenceContextBlock],
    section_path: tuple[str, ...] | None,
) -> CanonicalTable:
    merged = list(table.context.blocks)
    existing = {
        (block.role, block.position, block.text, block.source_ref)
        for block in merged
    }
    for block in blocks:
        identity = (block.role, block.position, block.text, block.source_ref)
        if identity not in existing:
            existing.add(identity)
            merged.append(block)
    return replace(
        table,
        context=EvidenceContext(
            section_path=(
                section_path
                if section_path is not None
                else table.context.section_path
            ),
            blocks=tuple(merged),
        ),
    )


def _unique_source_refs(
    paragraphs: Sequence[CanonicalParagraph],
) -> tuple[SourceRef, ...]:
    result: list[SourceRef] = []
    seen: set[SourceRef] = set()
    for paragraph in paragraphs:
        for source_ref in paragraph.source_refs:
            if source_ref not in seen:
                seen.add(source_ref)
                result.append(source_ref)
    return tuple(result)


def resolve_table_contexts(
    blocks: Sequence[CanonicalSectionBlock],
    *,
    section_path: tuple[str, ...] | None = None,
) -> TableContextResolution:
    """Bundle tables with only explicitly recognizable adjacent context blocks.

    The caller supplies canonical paragraphs and tables in original section order.
    Paragraphs consumed as table context remain unchanged; their source references
    are returned so the evidence assembler can avoid emitting duplicate evidence.
    """
    consumed_indexes: set[int] = set()
    bundles: list[TableEvidenceBundle] = []

    for table_index, item in enumerate(blocks):
        if (
            not isinstance(item, CanonicalTable)
            or item.table_type == TableType.LAYOUT_TABLE
        ):
            continue

        before: list[tuple[int, EvidenceContextBlock]] = []
        cursor = table_index - 1
        while cursor >= 0:
            if cursor in consumed_indexes:
                break
            candidate = blocks[cursor]
            if isinstance(candidate, CanonicalParagraph):
                role = _paragraph_role(candidate)
                if role not in {
                    ContextRole.TITLE,
                    ContextRole.UNIT,
                    ContextRole.CAPTION,
                    ContextRole.NOTE,
                }:
                    break
                block = _context_block(candidate, role, ContextPosition.BEFORE)
            elif candidate.table_type == TableType.LAYOUT_TABLE:
                block = _layout_context_block(candidate, ContextPosition.BEFORE)
                if block is None or block.role not in {
                    ContextRole.TITLE,
                    ContextRole.UNIT,
                }:
                    break
            else:
                break
            before.append((cursor, block))
            cursor -= 1
        before.reverse()

        after: list[tuple[int, EvidenceContextBlock]] = []
        cursor = table_index + 1
        while cursor < len(blocks):
            if cursor in consumed_indexes:
                break
            candidate = blocks[cursor]
            if isinstance(candidate, CanonicalParagraph):
                role = _paragraph_role(candidate)
                if role not in {ContextRole.CAPTION, ContextRole.NOTE}:
                    break
                block = _context_block(candidate, role, ContextPosition.AFTER)
            elif candidate.table_type == TableType.LAYOUT_TABLE:
                block = _layout_context_block(candidate, ContextPosition.AFTER)
                if block is None or block.role != ContextRole.NOTE:
                    break
            else:
                break
            after.append((cursor, block))
            cursor += 1

        associated = before + after
        for index, _ in associated:
            consumed_indexes.add(index)
        context_blocks = [block for _, block in associated]
        consumed_refs = tuple(
            dict.fromkeys(
                block.source_ref
                for block in context_blocks
                if block.source_ref is not None
            )
        )
        bundles.append(
            TableEvidenceBundle(
                table_block_index=table_index,
                table=_merge_context(item, context_blocks, section_path),
                consumed_source_refs=consumed_refs,
            )
        )

    all_consumed = tuple(
        dict.fromkeys(
            source_ref
            for index in sorted(consumed_indexes)
            for source_ref in (
                blocks[index].source_refs
                if isinstance(blocks[index], CanonicalParagraph)
                else (_table_source_ref(blocks[index]),)
            )
            if source_ref is not None
        )
    )
    return TableContextResolution(
        tables=tuple(bundles),
        consumed_source_refs=all_consumed,
    )


__all__ = ["CanonicalSectionBlock", "resolve_table_contexts"]
