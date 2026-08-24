"""Convert DART XML/HTML TABLE elements into structural canonical tables."""

from __future__ import annotations

from dataclasses import replace
from html.parser import HTMLParser
import re
from typing import Any, Iterable, Mapping
import xml.etree.ElementTree as ET

from converters.common.document_loader import repair_xml_text
from converters.common.source_models import (
    ContextPosition,
    ContextRole,
    DocumentSyntax,
    EvidenceContext,
    EvidenceContextBlock,
    SourceRef,
)
from converters.table_parser.table_models import (
    CanonicalTable,
    CanonicalTableGroup,
    CellRole,
    ContextStatus,
    LayoutRole,
    LogicalGrid,
    ParseIssue,
    ParseStatus,
    RowType,
    SourceCell,
    SourceSyntax,
    TableContext,
    TableType,
)


_CELL_TAGS = {"TD", "TH", "TE", "TU"}
_VALUE_TAGS = {"TE", "TU"}
_SOURCE_ATTRIBUTES = {
    "ACODE",
    "AUNIT",
    "AUNITVALUE",
    "ACONTEXT",
    "ADECIMAL",
    "AUPDATECONT",
    "ENG",
    "ROWSPAN",
    "COLSPAN",
    "XFORMS_INPUT",
}
_UNIT_TEXT = re.compile(r"^\(?\s*단위\s*[:：]", re.IGNORECASE)
_NOTE_TEXT = re.compile(
    r"^(?:※|[*]|주\s*\d+\s*[):.]|[-ㆍ·]\s*상기)",
    re.IGNORECASE,
)
_FOOTNOTE_MARKER = re.compile(
    r"^(?:\(\s*주\s*\d+\s*\)|주\s*\d+\s*[):：.]?)$",
    re.IGNORECASE,
)
_NAVIGATION_TEXT = re.compile(
    r"\s*(?:[☞▶►]\s*)?(?:"
    r"본문\s*(?:(?:위치)?로\s*)?이동|"
    r"목차\s*(?:위치로|로)?\s*이동|"
    r"맨\s*위로(?:\s*이동)?|"
    r"돌아가기|바로가기"
    r")\s*",
    re.IGNORECASE,
)
_SUBTOTAL_TEXT = re.compile(r"^\s*소\s*계(?:\b|\()")
_TOTAL_TEXT = re.compile(r"^\s*(?:총\s*계|합\s*계)(?:\b|\()")
_BASE_DATE_CONTEXT = re.compile(
    r"^\(\s*(?:작성\s*)?기준일\s*[:：]\s*.+\)$",
    re.IGNORECASE,
)
_BASE_DATE_MENTION = re.compile(
    r"(?:배당)?(?:작성\s*)?기준일\s*[:：]",
    re.IGNORECASE,
)


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].upper()


def _positive_span(raw: str | None, name: str, issues: list[ParseIssue]) -> int:
    try:
        value = int(raw or "1")
        if value < 1:
            raise ValueError
        return value
    except ValueError:
        issues.append(
            ParseIssue(
                code="INVALID_SPAN",
                message=f"{name}={raw!r} was replaced with 1.",
            )
        )
        return 1


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _top_level_paragraphs(element: ET.Element) -> list[ET.Element]:
    paragraphs: list[ET.Element] = []

    def visit(node: ET.Element, inside_paragraph: bool = False) -> None:
        for child in list(node):
            child_is_paragraph = _tag(child) == "P"
            if child_is_paragraph and not inside_paragraph:
                paragraphs.append(child)
            elif _tag(child) != "TABLE":
                visit(child, inside_paragraph or child_is_paragraph)

    visit(element)
    return paragraphs


def extract_cell_text(element: ET.Element) -> tuple[str, tuple[str, ...]]:
    """Preserve descendant order and paragraph boundaries, ignoring formatting."""
    paragraphs = _top_level_paragraphs(element)
    if paragraphs:
        segments = tuple(_clean_text("".join(node.itertext())) for node in paragraphs)
        if any(segments):
            return "\n".join(segments), segments

    value = _clean_text("".join(element.itertext()))
    return value, ((value,) if value else ())


def _owned_descendants(root: ET.Element, wanted: str) -> list[ET.Element]:
    found: list[ET.Element] = []

    def visit(node: ET.Element) -> None:
        for child in list(node):
            child_tag = _tag(child)
            if child_tag == "TABLE":
                continue
            if child_tag == wanted:
                found.append(child)
            visit(child)

    visit(root)
    return found


def _table_rows(table: ET.Element) -> list[ET.Element]:
    return _owned_descendants(table, "TR")


def _row_cells(row: ET.Element) -> list[ET.Element]:
    return [child for child in list(row) if _tag(child) in _CELL_TAGS]


def build_logical_grid(table_element: ET.Element) -> LogicalGrid:
    """Expand spans while sharing each origin SourceCell across covered slots."""
    grid: list[list[SourceCell | None]] = []
    cells: list[SourceCell] = []
    issues: list[ParseIssue] = []

    for row_index, row in enumerate(_table_rows(table_element)):
        while len(grid) <= row_index:
            grid.append([])
        column = 0
        for element in _row_cells(row):
            while column < len(grid[row_index]) and grid[row_index][column] is not None:
                column += 1

            rowspan = _positive_span(element.get("ROWSPAN"), "ROWSPAN", issues)
            colspan = _positive_span(element.get("COLSPAN"), "COLSPAN", issues)
            raw_value, segments = extract_cell_text(element)
            attributes = {
                key.upper(): value
                for key, value in element.attrib.items()
                if key.upper() in _SOURCE_ATTRIBUTES
            }
            cell = SourceCell(
                id=f"c{len(cells)}",
                tag=_tag(element),
                row_start=row_index,
                row_end=row_index + rowspan,
                col_start=column,
                col_end=column + colspan,
                raw_value=raw_value,
                text_segments=segments,
                source_attributes=attributes,
            )
            cells.append(cell)

            for covered_row in range(row_index, row_index + rowspan):
                while len(grid) <= covered_row:
                    grid.append([])
                required = column + colspan
                if len(grid[covered_row]) < required:
                    grid[covered_row].extend(
                        [None] * (required - len(grid[covered_row]))
                    )
                for covered_column in range(column, required):
                    existing = grid[covered_row][covered_column]
                    if existing is not None and existing is not cell:
                        issues.append(
                            ParseIssue(
                                code="GRID_OVERLAP",
                                severity="ERROR",
                                message=(
                                    f"{cell.id} overlaps {existing.id} at "
                                    f"({covered_row}, {covered_column})."
                                ),
                            )
                        )
                        continue
                    grid[covered_row][covered_column] = cell
            column += colspan

    width = max((len(row) for row in grid), default=0)
    for row in grid:
        row.extend([None] * (width - len(row)))

    return LogicalGrid(
        rows=tuple(tuple(row) for row in grid),
        cells=tuple(cells),
        issues=tuple(issues),
    )


def _strong_value(cell: SourceCell) -> bool:
    attributes = cell.source_attributes
    return (
        (cell.tag == "TE" and "ACODE" in attributes)
        or (cell.tag == "TU" and "AUNIT" in attributes)
        or attributes.get("XFORMS_INPUT") == "Y"
    )


def _thead_row_indexes(table: ET.Element, rows: list[ET.Element]) -> set[int]:
    header_rows: set[int] = set()
    row_ids = {id(row): index for index, row in enumerate(rows)}
    for thead in _owned_descendants(table, "THEAD"):
        for row in thead.iter():
            if _tag(row) == "TR" and id(row) in row_ids:
                header_rows.add(row_ids[id(row)])
    return header_rows


def _inferred_header_rows(grid: LogicalGrid) -> set[int]:
    """Find a first-row schema when THEAD is absent."""
    if grid.height < 3 or grid.width < 2:
        return set()
    first = grid.rows[0]
    first_cells = _unique_cells(first)
    if len(first_cells) < 2:
        return set()
    if any(_strong_value(cell) or not cell.raw_value for cell in first_cells):
        return set()

    following = grid.rows[1:]
    comparable = sum(
        1
        for row in following
        if sum(cell is not None for cell in row) >= max(2, grid.width - 1)
    )
    return {0} if comparable >= 2 else set()


def _plain_td_kv_value_ids(grid: LogicalGrid) -> set[str]:
    """Recognize conservative multi-row key/value matrices made only of TDs.

    DART's stock-administration tables commonly start with one ``key | value``
    row whose value spans the remaining columns, then mix that shape with
    ``key | value | key | value`` rows.  Without an explicit THEAD or XBRL
    value tags, the generic first-row heuristic otherwise mistakes the first
    key/value row for a record header.
    """
    if grid.height < 3 or grid.width < 4:
        return set()
    if any(_strong_value(cell) for cell in grid.cells):
        return set()

    value_ids: set[str] = set()
    nonempty_row_count = 0
    merged_value_row_count = 0
    multi_pair_row_count = 0
    first_nonempty_is_merged_pair = False

    for row_index, logical_row in enumerate(grid.rows):
        cells = _unique_cells(logical_row)
        if not any(cell.raw_value for cell in cells):
            continue
        nonempty_row_count += 1

        if any(cell.tag != "TD" for cell in cells):
            return set()
        if any(
            cell.row_start != row_index or cell.row_end != row_index + 1
            for cell in cells
        ):
            return set()
        if any(cell is None for cell in logical_row) or len(cells) % 2:
            return set()

        row_has_merged_value = False
        cursor = 0
        for pair_index in range(0, len(cells), 2):
            key = cells[pair_index]
            value = cells[pair_index + 1]
            if (
                not key.raw_value
                or key.col_start != cursor
                or key.col_end != key.col_start + 1
                or value.col_start != key.col_end
            ):
                return set()
            cursor = value.col_end
            if value.col_end - value.col_start > 1:
                row_has_merged_value = True
            value_ids.add(value.id)

        if cursor != grid.width:
            return set()
        if row_has_merged_value:
            merged_value_row_count += 1
        if len(cells) >= 4:
            multi_pair_row_count += 1
        if nonempty_row_count == 1:
            first_nonempty_is_merged_pair = len(cells) == 2 and row_has_merged_value

    if (
        nonempty_row_count >= 3
        and first_nonempty_is_merged_pair
        and merged_value_row_count >= 2
        and multi_pair_row_count >= 1
    ):
        return value_ids
    return set()


def _unique_cells(cells: Iterable[SourceCell | None]) -> list[SourceCell]:
    result: list[SourceCell] = []
    seen: set[str] = set()
    for cell in cells:
        if cell is not None and cell.id not in seen:
            seen.add(cell.id)
            result.append(cell)
    return result


def _layout_role(grid: LogicalGrid) -> LayoutRole | None:
    nonempty = [cell for cell in grid.cells if cell.raw_value]
    if (
        grid.height == 1
        and len(nonempty) == 2
        and all(cell.tag == "TD" for cell in nonempty)
        and _FOOTNOTE_MARKER.fullmatch(nonempty[0].raw_value.strip())
        and not _strong_value(nonempty[1])
    ):
        return LayoutRole.NOTE
    if len(nonempty) != 1:
        return None
    cell = nonempty[0]
    value = cell.raw_value.strip()
    if _NAVIGATION_TEXT.fullmatch(value):
        return LayoutRole.NAVIGATION
    if _UNIT_TEXT.match(value):
        return LayoutRole.UNIT
    if not _strong_value(cell) and _NOTE_TEXT.match(value):
        return LayoutRole.NOTE
    if not _strong_value(cell) and (
        (value.startswith("【") and value.endswith("】"))
        or (grid.height == 1 and len(value) <= 200)
    ):
        return LayoutRole.TITLE
    if not _strong_value(cell) and grid.height <= 2:
        return LayoutRole.NOTE
    return None


def classify_table_type(
    table_element: ET.Element,
    grid: LogicalGrid,
) -> tuple[TableType, set[int], LayoutRole | None]:
    rows = _table_rows(table_element)
    header_rows = _thead_row_indexes(table_element, rows)
    if not header_rows and _plain_td_kv_value_ids(grid):
        return TableType.KV_TABLE, set(), None
    if not header_rows:
        header_rows = _inferred_header_rows(grid)

    body_rows = set(range(grid.height)) - header_rows
    if header_rows and body_rows and grid.width >= 2:
        return TableType.R_TABLE, header_rows, None

    layout_role = _layout_role(grid)
    if layout_role is not None:
        return TableType.LAYOUT_TABLE, set(), layout_role

    if any(_strong_value(cell) for cell in grid.cells):
        return TableType.KV_TABLE, set(), None

    return TableType.UNKNOWN, set(), None


def _scan_left(
    grid: LogicalGrid,
    roles: Mapping[str, CellRole],
    row: int,
    column: int,
) -> tuple[list[str], list[str]]:
    keys: list[SourceCell] = []
    collecting = False
    seen: set[str] = set()
    for current_column in range(column - 1, -1, -1):
        cell = grid.rows[row][current_column]
        if cell is None or cell.id in seen:
            continue
        seen.add(cell.id)
        role = roles.get(cell.id, CellRole.UNKNOWN)
        if role in {CellRole.KEY, CellRole.CONTEXT} and cell.raw_value:
            collecting = True
            keys.append(cell)
            continue
        if role == CellRole.VALUE and collecting:
            break
    keys.reverse()
    return [cell.raw_value for cell in keys], [cell.id for cell in keys]


def _scan_header_up(
    grid: LogicalGrid,
    roles: Mapping[str, CellRole],
    row: int,
    column: int,
) -> tuple[list[str], list[str]]:
    keys: list[SourceCell] = []
    seen: set[str] = set()
    for current_row in range(row - 1, -1, -1):
        cell = grid.rows[current_row][column]
        if cell is None or cell.id in seen:
            continue
        seen.add(cell.id)
        role = roles.get(cell.id, CellRole.UNKNOWN)
        if role == CellRole.VALUE:
            if keys:
                break
            continue
        if role in {CellRole.KEY, CellRole.HEADER} and cell.raw_value:
            keys.append(cell)
    keys.reverse()
    return [cell.raw_value for cell in keys], [cell.id for cell in keys]


def _scan_full_width_section_up(
    grid: LogicalGrid,
    roles: Mapping[str, CellRole],
    value: SourceCell,
) -> tuple[list[str], list[str]]:
    if value.col_start != 0 or value.col_end != grid.width:
        return [], []
    for row in range(value.row_start - 1, -1, -1):
        cell = grid.rows[row][value.col_start]
        if cell is None or not cell.raw_value:
            continue
        if (
            roles.get(cell.id) == CellRole.KEY
            and cell.col_start == value.col_start
            and cell.col_end == value.col_end
        ):
            return [cell.raw_value], [cell.id]
        break
    return [], []


def _deduplicate_paths(
    paths: Iterable[tuple[list[str], list[str]]],
) -> list[dict[str, list[str]]]:
    result: list[dict[str, list[str]]] = []
    seen: set[tuple[str, ...]] = set()
    for texts, cell_ids in paths:
        signature = tuple(texts)
        if not texts or signature in seen:
            continue
        seen.add(signature)
        result.append({"path": texts, "cells": cell_ids})
    return result


def _parse_kv(
    grid: LogicalGrid,
    *,
    inferred_value_ids: set[str] | None = None,
) -> tuple[dict[str, CellRole], dict[str, Any]]:
    roles: dict[str, CellRole] = {}
    for cell in grid.cells:
        if _strong_value(cell) or (
            inferred_value_ids is not None and cell.id in inferred_value_ids
        ):
            roles[cell.id] = CellRole.VALUE
        elif not cell.raw_value:
            roles[cell.id] = CellRole.EMPTY
        else:
            roles[cell.id] = CellRole.KEY

    fields: list[dict[str, Any]] = []
    for value in (cell for cell in grid.cells if roles[cell.id] == CellRole.VALUE):
        row_paths: list[tuple[list[str], list[str]]] = []
        for row in range(value.row_start, value.row_end):
            row_paths.append(_scan_left(grid, roles, row, value.col_start))

        row_value_ids = {
            cell.id
            for cell in _unique_cells(grid.rows[value.row_start])
            if roles.get(cell.id) == CellRole.VALUE
        }
        column_paths: list[tuple[list[str], list[str]]] = []
        if len(row_value_ids) >= 2:
            for column in range(value.col_start, value.col_end):
                column_paths.append(
                    _scan_header_up(grid, roles, value.row_start, column)
                )

        combined: list[tuple[list[str], list[str]]] = []
        nonempty_rows = [path for path in row_paths if path[0]] or [([], [])]
        nonempty_columns = [path for path in column_paths if path[0]] or [([], [])]
        for row_text, row_cells in nonempty_rows:
            for column_text, column_cells in nonempty_columns:
                combined.append(
                    (row_text + column_text, row_cells + column_cells)
                )

        key_paths = _deduplicate_paths(combined)
        if not key_paths:
            key_paths = _deduplicate_paths(
                [_scan_full_width_section_up(grid, roles, value)]
            )
        if len(key_paths) == 1:
            context_status = ContextStatus.RESOLVED
        elif len(key_paths) > 1:
            context_status = ContextStatus.MULTI_CONTEXT
        else:
            context_status = ContextStatus.NONE

        fields.append(
            {
                "value_cell": value.id,
                "raw_value": value.raw_value,
                "key_paths": key_paths,
                "context_status": context_status.value,
            }
        )
    return roles, {"fields": fields}


def _header_paths(
    grid: LogicalGrid,
    header_rows: set[int],
) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = []
    for column in range(grid.width):
        texts: list[str] = []
        cell_ids: list[str] = []
        seen: set[str] = set()
        for row in sorted(header_rows):
            cell = grid.rows[row][column]
            if cell is None or cell.id in seen or not cell.raw_value:
                continue
            seen.add(cell.id)
            texts.append(cell.raw_value)
            cell_ids.append(cell.id)
        columns.append(
            {
                "id": f"col_{column}",
                "index": column,
                "header_path": texts,
                "header_cells": cell_ids,
            }
        )
    return columns


def _classify_row_type(context: Iterable[str]) -> RowType:
    values = [value.strip() for value in context if value.strip()]
    if any(_SUBTOTAL_TEXT.match(value) for value in values):
        return RowType.SUBTOTAL
    if any(_TOTAL_TEXT.match(value) for value in values):
        return RowType.TOTAL
    return RowType.DATA


def _parse_record(
    grid: LogicalGrid,
    header_rows: set[int],
) -> tuple[dict[str, CellRole], dict[str, Any]]:
    roles: dict[str, CellRole] = {}
    header_cell_ids = {
        cell.id
        for row in header_rows
        for cell in _unique_cells(grid.rows[row])
    }
    body_cells = [
        cell
        for row in range(grid.height)
        if row not in header_rows
        for cell in _unique_cells(grid.rows[row])
    ]
    has_coded_values = any(_strong_value(cell) for cell in body_cells)

    for cell in grid.cells:
        if not cell.raw_value:
            roles[cell.id] = CellRole.EMPTY
        elif cell.id in header_cell_ids:
            roles[cell.id] = CellRole.HEADER
        elif has_coded_values:
            roles[cell.id] = CellRole.VALUE if _strong_value(cell) else CellRole.CONTEXT
        else:
            roles[cell.id] = CellRole.VALUE

    columns = _header_paths(grid, header_rows)
    records: list[dict[str, Any]] = []
    first_body_row = max(header_rows) + 1 if header_rows else 0
    for row in range(first_body_row, grid.height):
        logical_cells = grid.rows[row]
        if not any(cell is not None and cell.raw_value for cell in logical_cells):
            continue
        row_context_cells: list[SourceCell] = []
        if has_coded_values:
            for cell in _unique_cells(logical_cells):
                if roles.get(cell.id) == CellRole.VALUE:
                    break
                if roles.get(cell.id) == CellRole.CONTEXT and cell.raw_value:
                    row_context_cells.append(cell)

        cells_by_column = {
            f"col_{column}": cell.id
            for column, cell in enumerate(logical_cells)
            if cell is not None
        }
        context_text = [cell.raw_value for cell in row_context_cells]
        row_text = [
            cell.raw_value
            for cell in _unique_cells(logical_cells)
            if cell.raw_value
        ]
        records.append(
            {
                "record_index": len(records),
                "source_row": row,
                # Totals are semantic row types even when a table has no coded
                # TE/TU values and therefore no separately inferred row context.
                "row_type": _classify_row_type(row_text).value,
                "row_context": context_text,
                "row_context_cells": [cell.id for cell in row_context_cells],
                "cells_by_column": cells_by_column,
            }
        )

    return roles, {"columns": columns, "records": records}


def _parse_layout(
    grid: LogicalGrid,
    layout_role: LayoutRole,
) -> tuple[dict[str, CellRole], dict[str, Any]]:
    roles = {
        cell.id: CellRole.EMPTY if not cell.raw_value else CellRole.CONTEXT
        for cell in grid.cells
    }
    values = [cell.raw_value for cell in grid.cells if cell.raw_value]
    if (
        layout_role == LayoutRole.NOTE
        and len(values) == 2
        and _FOOTNOTE_MARKER.fullmatch(values[0].strip())
    ):
        values = [f"{values[0].strip()} {values[1].strip()}"]
    return roles, {"layout_role": layout_role.value, "values": values}


def _parse_unknown(
    grid: LogicalGrid,
) -> tuple[dict[str, CellRole], dict[str, Any]]:
    roles = {
        cell.id: CellRole.EMPTY if not cell.raw_value else CellRole.UNKNOWN
        for cell in grid.cells
    }
    return roles, {
        "rows": [
            [cell.id if cell is not None else None for cell in row]
            for row in grid.rows
        ]
    }


def parse_table(
    table_element: ET.Element,
    *,
    context: TableContext | None = None,
    content_context: EvidenceContext | None = None,
    syntax: SourceSyntax = SourceSyntax.DART_XML,
) -> CanonicalTable:
    """Parse one TABLE element without mutating it."""
    if _tag(table_element) != "TABLE":
        raise ValueError("parse_table requires a TABLE element")
    context = context or TableContext()
    grid = build_logical_grid(table_element)
    table_type, header_rows, layout_role = classify_table_type(table_element, grid)

    if table_type == TableType.KV_TABLE:
        inferred_value_ids = _plain_td_kv_value_ids(grid)
        roles, content = _parse_kv(
            grid,
            inferred_value_ids=inferred_value_ids or None,
        )
    elif table_type == TableType.R_TABLE:
        roles, content = _parse_record(grid, header_rows)
    elif table_type == TableType.LAYOUT_TABLE and layout_role is not None:
        roles, content = _parse_layout(grid, layout_role)
    else:
        roles, content = _parse_unknown(grid)

    status = (
        ParseStatus.PARTIAL
        if any(issue.severity == "ERROR" for issue in grid.issues)
        else ParseStatus.SUCCESS
    )
    source = {
        "source_path": context.source_path,
        "table_index": context.table_index,
        "table_group_class": context.table_group_class,
        "source_ref": (
            context.source_ref.to_dict() if context.source_ref is not None else None
        ),
        "aclass": table_element.get("ACLASS"),
        "syntax": syntax.value,
    }
    evidence_context = content_context or EvidenceContext()
    caption_blocks: list[EvidenceContextBlock] = []
    for caption in _owned_descendants(table_element, "CAPTION"):
        caption_text = _clean_text("".join(caption.itertext()))
        if not caption_text:
            continue
        caption_blocks.append(
            EvidenceContextBlock(
                role=ContextRole.CAPTION,
                position=ContextPosition.BEFORE,
                text=caption_text,
                source_ref=(
                    SourceRef(
                        syntax=DocumentSyntax(syntax.value),
                        table_index=context.table_index,
                    )
                    if context.table_index is not None
                    else None
                ),
            )
        )
    if caption_blocks:
        evidence_context = EvidenceContext(
            section_path=evidence_context.section_path,
            blocks=evidence_context.blocks + tuple(caption_blocks),
        )

    return CanonicalTable(
        table_type=table_type,
        parse_status=status,
        source=source,
        dimensions={"rows": grid.height, "columns": grid.width},
        cells=grid.cells,
        cell_roles=roles,
        content=content,
        context=evidence_context,
        issues=grid.issues,
    )


def _repair_xml(source: str) -> tuple[str, tuple[ParseIssue, ...]]:
    return repair_xml_text(source), (
        ParseIssue(
            code="XML_RECOVERED",
            message="The fragment required conservative text repair before parsing.",
        ),
    )


class _LooseTableHTMLParser(HTMLParser):
    """Build TABLE elements from DART's non-well-formed exchange HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[ET.Element] = []
        self._table: ET.Element | None = None
        self._row: ET.Element | None = None
        self._cell: ET.Element | None = None
        self._caption: ET.Element | None = None
        self._text_stack: list[ET.Element] = []

    @staticmethod
    def _attributes(attributes: list[tuple[str, str | None]]) -> dict[str, str]:
        return {key.upper(): value or "" for key, value in attributes}

    @staticmethod
    def _append_text(element: ET.Element, value: str) -> None:
        children = list(element)
        if children:
            children[-1].tail = (children[-1].tail or "") + value
        else:
            element.text = (element.text or "") + value

    def handle_starttag(
        self,
        tag: str,
        attributes: list[tuple[str, str | None]],
    ) -> None:
        name = tag.upper()
        attrs = self._attributes(attributes)
        if name == "TABLE":
            self._table = ET.Element("TABLE", attrs)
            self.tables.append(self._table)
            self._row = None
            self._cell = None
            self._caption = None
            self._text_stack.clear()
            return
        if self._table is None:
            return
        if name == "CAPTION":
            self._caption = ET.SubElement(self._table, "CAPTION", attrs)
            self._text_stack = [self._caption]
            return
        if name == "TR":
            self._row = ET.SubElement(self._table, "TR", attrs)
            self._cell = None
            self._text_stack.clear()
            return
        if name in {"TD", "TH"}:
            if self._row is None:
                self._row = ET.SubElement(self._table, "TR")
            self._cell = ET.SubElement(self._row, name, attrs)
            self._text_stack = [self._cell]
            return
        if self._cell is None:
            return
        if name == "SPAN" and "xforms_input" in attrs.get("CLASS", "").split():
            self._cell.set("XFORMS_INPUT", "Y")
        if name in {"P", "SPAN", "BR"}:
            parent = self._text_stack[-1] if self._text_stack else self._cell
            child = ET.SubElement(parent, name, attrs)
            if name != "BR":
                self._text_stack.append(child)

    def handle_endtag(self, tag: str) -> None:
        name = tag.upper()
        if name in {"P", "SPAN"} and len(self._text_stack) > 1:
            self._text_stack.pop()
        elif name == "CAPTION":
            self._caption = None
            self._text_stack.clear()
        elif name in {"TD", "TH"}:
            self._cell = None
            self._text_stack.clear()
        elif name == "TR":
            self._row = None
            self._cell = None
            self._text_stack.clear()
        elif name == "TABLE":
            self._table = None
            self._row = None
            self._cell = None
            self._caption = None
            self._text_stack.clear()

    def handle_data(self, data: str) -> None:
        if self._cell is None and self._caption is None:
            return
        target = (
            self._text_stack[-1]
            if self._text_stack
            else (self._cell if self._cell is not None else self._caption)
        )
        if target is None:
            return
        self._append_text(target, data)


def _first_table(root: ET.Element) -> ET.Element:
    if _tag(root) == "TABLE":
        return root
    try:
        return next(element for element in root.iter() if _tag(element) == "TABLE")
    except StopIteration as exc:
        raise ValueError("fragment does not contain a TABLE") from exc


def _failed_table(
    context: TableContext,
    syntax: SourceSyntax,
    message: str,
    content_context: EvidenceContext | None = None,
) -> CanonicalTable:
    issue = ParseIssue(code="PARSE_FAILED", severity="ERROR", message=message)
    return CanonicalTable(
        table_type=TableType.UNKNOWN,
        parse_status=ParseStatus.FAILED,
        source={
            "source_path": context.source_path,
            "table_index": context.table_index,
            "table_group_class": context.table_group_class,
            "aclass": None,
            "syntax": syntax.value,
        },
        dimensions={"rows": 0, "columns": 0},
        cells=(),
        cell_roles={},
        content={"rows": []},
        context=content_context or EvidenceContext(),
        issues=(issue,),
    )


def parse_table_fragment(
    fragment: str | bytes,
    *,
    syntax: SourceSyntax = SourceSyntax.AUTO,
    context: TableContext | None = None,
    content_context: EvidenceContext | None = None,
) -> CanonicalTable:
    """Parse one XML or exchange-HTML TABLE fragment with traceable recovery."""
    context = context or TableContext()
    source = fragment.decode("utf-8", errors="replace") if isinstance(fragment, bytes) else fragment
    selected_syntax = syntax
    if syntax == SourceSyntax.AUTO:
        selected_syntax = (
            SourceSyntax.HTML
            if re.search(r"<html\b|class\s*=\s*[\"']xforms", source, re.IGNORECASE)
            else SourceSyntax.DART_XML
        )

    if selected_syntax == SourceSyntax.HTML:
        parser = _LooseTableHTMLParser()
        try:
            parser.feed(source)
            if not parser.tables:
                raise ValueError("fragment does not contain a TABLE")
            table = parse_table(
                parser.tables[0],
                context=context,
                content_context=content_context,
                syntax=selected_syntax,
            )
            issues = table.issues
            if len(parser.tables) > 1:
                issues += (
                    ParseIssue(
                        code="MULTIPLE_TABLES",
                        message="Only the first TABLE in the fragment was parsed.",
                    ),
                )
            return replace(table, issues=issues)
        except Exception as exc:
            return _failed_table(
                context,
                selected_syntax,
                f"{type(exc).__name__}: {exc}",
                content_context,
            )

    try:
        root = ET.fromstring(source)
        return parse_table(
            _first_table(root),
            context=context,
            content_context=content_context,
            syntax=selected_syntax,
        )
    except (ET.ParseError, ValueError) as first_error:
        repaired, recovery_issues = _repair_xml(source)
        try:
            root = ET.fromstring(repaired)
            table = parse_table(
                _first_table(root),
                context=context,
                content_context=content_context,
                syntax=selected_syntax,
            )
            return replace(
                table,
                parse_status=ParseStatus.RECOVERED,
                issues=recovery_issues + table.issues,
            )
        except (ET.ParseError, ValueError) as second_error:
            return _failed_table(
                context,
                selected_syntax,
                f"strict={first_error}; recovered={second_error}",
                content_context,
            )


def _direct_group_tables(group_element: ET.Element) -> list[ET.Element]:
    tables: list[ET.Element] = []

    def visit(node: ET.Element) -> None:
        for child in list(node):
            if _tag(child) == "TABLE":
                tables.append(child)
            else:
                visit(child)

    visit(group_element)
    return tables


def _table_source_ref(table: CanonicalTable) -> SourceRef | None:
    syntax = DocumentSyntax(table.source.get("syntax", DocumentSyntax.DART_XML.value))
    table_index = table.source.get("table_index")
    source_ref_value = table.source.get("source_ref")
    if isinstance(source_ref_value, Mapping):
        try:
            return SourceRef(
                syntax=DocumentSyntax(source_ref_value["syntax"]),
                element_path=source_ref_value.get("element_path"),
                html_id=source_ref_value.get("html_id"),
                table_index=source_ref_value.get("table_index"),
            )
        except (KeyError, TypeError, ValueError):
            pass
    return (
        SourceRef(syntax=syntax, table_index=table_index)
        if table_index is not None
        else None
    )


def _group_context_block(
    layout: CanonicalTable,
    position: ContextPosition,
) -> EvidenceContextBlock:
    role = ContextRole(
        layout.content.get("layout_role", LayoutRole.NOTE.value)
    )
    return EvidenceContextBlock(
        role=role,
        position=position,
        text="\n".join(layout.content.get("values", [])),
        source_ref=_table_source_ref(layout),
    )


def _compound_context_blocks(
    table: CanonicalTable,
) -> tuple[EvidenceContextBlock, ...]:
    """Split a one-row base-date/unit strip into table context blocks."""
    if (
        table.table_type not in {TableType.KV_TABLE, TableType.UNKNOWN}
        or table.dimensions.get("rows") != 1
    ):
        return ()

    cells = sorted(
        (cell for cell in table.cells if cell.raw_value.strip()),
        key=lambda cell: cell.col_start,
    )
    unit_indexes = [
        index
        for index, cell in enumerate(cells)
        if _UNIT_TEXT.match(cell.raw_value.strip())
    ]
    if unit_indexes != [len(cells) - 1] or len(cells) < 3:
        return ()

    caption = " ".join(cell.raw_value.strip() for cell in cells[:-1])
    caption = re.sub(r"\(\s+", "(", caption)
    caption = re.sub(r"\s+\)", ")", caption)
    caption = _clean_text(caption)
    if _BASE_DATE_CONTEXT.fullmatch(caption) is None:
        return ()

    source_ref = _table_source_ref(table)

    return (
        EvidenceContextBlock(
            role=ContextRole.CAPTION,
            position=ContextPosition.BEFORE,
            text=caption,
            source_ref=source_ref,
        ),
        EvidenceContextBlock(
            role=ContextRole.UNIT,
            position=ContextPosition.BEFORE,
            text=cells[-1].raw_value.strip(),
            source_ref=source_ref,
        ),
    )


def _split_context_blocks(
    date_table: CanonicalTable,
    unit_table: CanonicalTable,
) -> tuple[EvidenceContextBlock, ...]:
    """Resolve XBRL context split across a date-caption and a unit table."""
    date_cells = [cell for cell in date_table.cells if cell.raw_value.strip()]
    if (
        date_table.dimensions.get("rows") != 1
        or len(date_cells) != 1
        or _BASE_DATE_MENTION.search(date_cells[0].raw_value) is None
    ):
        return ()

    unit_cells = sorted(
        (cell for cell in unit_table.cells if cell.raw_value.strip()),
        key=lambda cell: (cell.row_start, cell.col_start),
    )
    if unit_table.dimensions.get("rows") not in {1, 2}:
        return ()
    units = [cell for cell in unit_cells if _UNIT_TEXT.match(cell.raw_value.strip())]
    if len(units) != 1:
        return ()
    unit_cell = units[0]
    other_cells = [cell for cell in unit_cells if cell is not unit_cell]
    if (
        len(other_cells) not in {1, 2}
        or any(_strong_value(cell) for cell in other_cells)
        or any(len(cell.raw_value.strip()) > 200 for cell in other_cells)
    ):
        return ()

    title = None
    period_cells = other_cells
    if unit_table.dimensions.get("rows") == 2:
        first_row = [cell for cell in other_cells if cell.row_start == 0]
        second_row = [cell for cell in other_cells if cell.row_start == 1]
        if len(first_row) != 1 or len(second_row) != 1:
            return ()
        title = first_row[0]
        period_cells = second_row
    elif len(other_cells) != 1:
        return ()

    blocks = [
        EvidenceContextBlock(
            role=ContextRole.CAPTION,
            position=ContextPosition.BEFORE,
            text=date_cells[0].raw_value.strip(),
            source_ref=_table_source_ref(date_table),
        )
    ]
    if title is not None:
        blocks.append(
            EvidenceContextBlock(
                role=ContextRole.TITLE,
                position=ContextPosition.BEFORE,
                text=title.raw_value.strip(),
                source_ref=_table_source_ref(unit_table),
            )
        )
    blocks.extend(
        EvidenceContextBlock(
            role=ContextRole.CAPTION,
            position=ContextPosition.BEFORE,
            text=cell.raw_value.strip(),
            source_ref=_table_source_ref(unit_table),
        )
        for cell in period_cells
    )
    blocks.append(
        EvidenceContextBlock(
            role=ContextRole.UNIT,
            position=ContextPosition.BEFORE,
            text=unit_cell.raw_value.strip(),
            source_ref=_table_source_ref(unit_table),
        )
    )
    return tuple(blocks)


def _append_context_blocks(
    table: CanonicalTable,
    blocks: Iterable[EvidenceContextBlock],
) -> CanonicalTable:
    additions = tuple(blocks)
    if not additions:
        return table
    return replace(
        table,
        context=EvidenceContext(
            section_path=table.context.section_path,
            blocks=table.context.blocks + additions,
        ),
    )


def parse_table_group(
    group_element: ET.Element,
    *,
    context: TableContext | None = None,
    content_context: EvidenceContext | None = None,
    syntax: SourceSyntax = SourceSyntax.DART_XML,
) -> CanonicalTableGroup:
    """Parse a TABLE-GROUP and attach leading title/unit tables to its data table."""
    if _tag(group_element) != "TABLE-GROUP":
        raise ValueError("parse_table_group requires a TABLE-GROUP element")
    base_context = context or TableContext()
    group_class = group_element.get("ACLASS") or base_context.table_group_class
    parsed: list[CanonicalTable] = []
    for index, table_element in enumerate(_direct_group_tables(group_element)):
        table_context = TableContext(
            source_path=base_context.source_path,
            table_index=(base_context.table_index or 0) + index,
            table_group_class=group_class,
            source_ref=base_context.source_ref,
        )
        parsed.append(
            parse_table(
                table_element,
                context=table_context,
                content_context=content_context,
                syntax=syntax,
            )
        )

    context_blocks: list[EvidenceContextBlock] = []
    semantic_tables: list[CanonicalTable] = []
    pending: list[EvidenceContextBlock] = []
    consumed_context_indexes: set[int] = set()
    for table_index, table in enumerate(parsed):
        if table_index in consumed_context_indexes:
            continue
        split_blocks = (
            _split_context_blocks(table, parsed[table_index + 1])
            if table_index + 2 < len(parsed)
            and parsed[table_index + 2].table_type == TableType.R_TABLE
            else ()
        )
        if split_blocks:
            pending.extend(split_blocks)
            consumed_context_indexes.add(table_index + 1)
            continue
        compound_blocks = (
            _compound_context_blocks(table)
            if table_index + 1 < len(parsed)
            and parsed[table_index + 1].table_type == TableType.R_TABLE
            else ()
        )
        if compound_blocks:
            pending.extend(compound_blocks)
            continue
        if table.table_type == TableType.LAYOUT_TABLE:
            if (
                table.content.get("layout_role")
                == LayoutRole.NAVIGATION.value
            ):
                continue
            pending.append(_group_context_block(table, ContextPosition.BEFORE))
            continue

        semantic_tables.append(table)
        current_index = len(semantic_tables) - 1
        if pending:
            if current_index == 0:
                before = [
                    replace(block, position=ContextPosition.BEFORE)
                    for block in pending
                ]
                semantic_tables[current_index] = _append_context_blocks(
                    semantic_tables[current_index],
                    before,
                )
                context_blocks.extend(before)
            else:
                after_previous: list[EvidenceContextBlock] = []
                before_current: list[EvidenceContextBlock] = []
                ordered_context: list[EvidenceContextBlock] = []
                for pending_block in pending:
                    if pending_block.role == ContextRole.NOTE:
                        block = replace(
                            pending_block,
                            position=ContextPosition.AFTER,
                        )
                        after_previous.append(block)
                    else:
                        block = replace(
                            pending_block,
                            position=ContextPosition.BEFORE,
                        )
                        before_current.append(block)
                    ordered_context.append(block)
                semantic_tables[current_index - 1] = _append_context_blocks(
                    semantic_tables[current_index - 1],
                    after_previous,
                )
                semantic_tables[current_index] = _append_context_blocks(
                    semantic_tables[current_index],
                    before_current,
                )
                context_blocks.extend(ordered_context)
            pending.clear()

    if pending:
        trailing = [
            replace(block, position=ContextPosition.AFTER)
            for block in pending
        ]
        context_blocks.extend(trailing)
        if semantic_tables:
            semantic_tables[-1] = _append_context_blocks(
                semantic_tables[-1],
                trailing,
            )

    return CanonicalTableGroup(
        table_group_class=group_class,
        context_blocks=tuple(context_blocks),
        tables=tuple(semantic_tables),
    )


__all__ = [
    "build_logical_grid",
    "classify_table_type",
    "extract_cell_text",
    "parse_table",
    "parse_table_fragment",
    "parse_table_group",
]
