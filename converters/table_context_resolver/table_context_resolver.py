"""Resolve conservative before/after context for tables in section order."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Sequence, TypeAlias

from converters.common.context_text import split_table_context_text
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

_BRACKETED_TITLE = re.compile(
    r"^(?:\[.*\]|【.*】|〔.*〕|〈.*〉|《.*》|「.*」|『.*』)$"
)
_CAPTION_CLASS = re.compile(r"(?:^|[-_\s])(?:table|tbl)[-_]?caption(?:$|[-_\s])")
_TITLE_CLASS = re.compile(r"(?:^|[-_\s])(?:table|tbl)[-_]?title(?:$|[-_\s])")
_PERIOD_CAPTION = re.compile(r"^\d{1,2}\s*[)]")
_BULLET_CELL = re.compile(r"^[-ㆍ·•]$")
_UNIT_TEXT = re.compile(r"^\(?\s*단위\s*[:：]", re.IGNORECASE)
_ANGLE_TITLE = re.compile(r"^<[^<>\n]{1,80}>$")


def _textual_unknown(table: CanonicalTable) -> bool:
    if table.table_type != TableType.UNKNOWN or table.dimensions.get("rows") != 1:
        return False
    values = [cell.raw_value.strip() for cell in table.cells if cell.raw_value.strip()]
    return len(values) == 2 and _BULLET_CELL.fullmatch(values[0]) is not None


def _unknown_layout_context_blocks(
    table: CanonicalTable,
    position: ContextPosition,
) -> tuple[EvidenceContextBlock, ...]:
    """Recover a conservative one-row ``<period> + unit`` layout strip."""
    if table.table_type != TableType.UNKNOWN or table.dimensions.get("rows") != 1:
        return ()
    cells = sorted(
        (cell for cell in table.cells if cell.raw_value.strip()),
        key=lambda cell: cell.col_start,
    )
    if len(cells) != 2 or _UNIT_TEXT.match(cells[1].raw_value.strip()) is None:
        return ()
    title = cells[0].raw_value.strip()
    if _ANGLE_TITLE.fullmatch(title) is None:
        return ()
    source_ref = _table_source_ref(table)
    return (
        EvidenceContextBlock(
            role=ContextRole.TITLE,
            position=position,
            text=title,
            source_ref=source_ref,
        ),
        EvidenceContextBlock(
            role=ContextRole.UNIT,
            position=position,
            text=cells[1].raw_value.strip(),
            source_ref=source_ref,
        ),
    )


def _paragraph_context_blocks(
    paragraph: CanonicalParagraph,
    position: ContextPosition,
) -> tuple[EvidenceContextBlock, ...]:
    text = paragraph.text.strip()
    class_name = paragraph.source_attributes.get("CLASS", "").lower()
    if _CAPTION_CLASS.search(class_name):
        parts = ((ContextRole.CAPTION, text),)
    elif _TITLE_CLASS.search(class_name):
        parts = ((ContextRole.TITLE, text),)
    elif len(text) <= 200 and _BRACKETED_TITLE.fullmatch(text):
        parts = ((ContextRole.TITLE, text),)
    else:
        parts = tuple(
            (part.role, part.text)
            for part in split_table_context_text(
                text,
                bold=paragraph.is_bold,
                fragment_texts=tuple(fragment.text for fragment in paragraph.fragments),
            )
        )
    source_ref = paragraph.source_refs[0] if paragraph.source_refs else None
    return tuple(
        EvidenceContextBlock(
            role=role,
            position=position,
            text=value,
            source_ref=source_ref,
        )
        for role, value in parts
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
    if role_value not in {"TITLE", "UNIT", "CAPTION", "NOTE"} or not values:
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
    additions: list[EvidenceContextBlock] = []
    for block in blocks:
        identity = (block.role, block.position, block.text, block.source_ref)
        if identity not in existing:
            existing.add(identity)
            additions.append(block)
    merged = (
        [block for block in additions if block.position == ContextPosition.BEFORE]
        + merged
        + [block for block in additions if block.position == ContextPosition.AFTER]
    )
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
    last_table_index: int | None = None
    shared_before: tuple[EvidenceContextBlock, ...] = ()

    for table_index, item in enumerate(blocks):
        if (
            not isinstance(item, CanonicalTable)
            or item.table_type == TableType.LAYOUT_TABLE
            or _textual_unknown(item)
            or _unknown_layout_context_blocks(item, ContextPosition.BEFORE)
        ):
            continue

        before_groups: list[tuple[int, tuple[EvidenceContextBlock, ...]]] = []
        cursor = table_index - 1
        while cursor >= 0:
            if cursor in consumed_indexes:
                break
            candidate = blocks[cursor]
            if isinstance(candidate, CanonicalParagraph):
                if not candidate.text.strip():
                    consumed_indexes.add(cursor)
                    cursor -= 1
                    continue
                candidate_blocks = _paragraph_context_blocks(
                    candidate,
                    ContextPosition.BEFORE,
                )
                if not candidate_blocks or any(
                    block.role
                    not in {
                        ContextRole.HEADING,
                        ContextRole.TITLE,
                        ContextRole.UNIT,
                        ContextRole.CAPTION,
                        ContextRole.NOTE,
                    }
                    for block in candidate_blocks
                ):
                    break
            elif candidate.table_type == TableType.LAYOUT_TABLE:
                block = _layout_context_block(candidate, ContextPosition.BEFORE)
                if block is None or block.role not in {
                    ContextRole.TITLE,
                    ContextRole.UNIT,
                    ContextRole.CAPTION,
                }:
                    break
                candidate_blocks = (block,)
            elif candidate.table_type == TableType.UNKNOWN:
                candidate_blocks = _unknown_layout_context_blocks(
                    candidate,
                    ContextPosition.BEFORE,
                )
                if not candidate_blocks:
                    break
            else:
                break
            before_groups.append((cursor, candidate_blocks))
            cursor -= 1
        before = [
            (index, block)
            for index, group in reversed(before_groups)
            for block in group
        ]

        after: list[tuple[int, EvidenceContextBlock]] = []
        cursor = table_index + 1
        while cursor < len(blocks):
            if cursor in consumed_indexes:
                break
            candidate = blocks[cursor]
            if isinstance(candidate, CanonicalParagraph):
                if not candidate.text.strip():
                    consumed_indexes.add(cursor)
                    cursor += 1
                    continue
                candidate_blocks = _paragraph_context_blocks(
                    candidate,
                    ContextPosition.AFTER,
                )
                if not candidate_blocks or any(
                    block.role != ContextRole.NOTE for block in candidate_blocks
                ):
                    break
            elif candidate.table_type == TableType.LAYOUT_TABLE:
                block = _layout_context_block(candidate, ContextPosition.AFTER)
                if block is None or block.role != ContextRole.NOTE:
                    break
                candidate_blocks = (block,)
            else:
                break
            after.extend((cursor, block) for block in candidate_blocks)
            cursor += 1

        associated = before + after
        current_shared = tuple(
            block
            for _, block in before
            if block.role == ContextRole.UNIT
            or (
                block.role == ContextRole.CAPTION
                and _PERIOD_CAPTION.match(block.text) is None
            )
        )
        current_has_general_caption = any(
            block.role == ContextRole.CAPTION
            and _PERIOD_CAPTION.match(block.text) is None
            for _, block in before
        )
        associated_indexes = {index for index, _ in associated}
        inherited: tuple[EvidenceContextBlock, ...] = ()
        if (
            not current_has_general_caption
            and last_table_index is not None
            and all(
                index in consumed_indexes or index in associated_indexes
                for index in range(last_table_index + 1, table_index)
            )
        ):
            inherited = shared_before
        for index, _ in associated:
            consumed_indexes.add(index)
        context_blocks = [*inherited, *(block for _, block in associated)]
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
        shared_before = current_shared or inherited
        last_table_index = table_index

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
