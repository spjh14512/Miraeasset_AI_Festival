"""DART holding XML tables to loss-aware canonical dictionaries.

The module intentionally uses only the standard library.  Table parsing,
normalisation and the holding registry live together for now, while the public
API accepts a registry argument so another disclosure family can extend it
without changing the parser.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
from typing import Any, Mapping
import xml.etree.ElementTree as ET


FIELD_DEFINITIONS: dict[str, dict[str, Any]] = {
    "CRP_NM": {"semantic_name": "issuer_name", "data_type": "string", "unit": None},
    "FLT_CRP_RLT": {"semantic_name": "issuer_relationship", "data_type": "string", "unit": None},
    "RPT_DST1": {
        "semantic_name": "report_type",
        "data_type": "enum",
        "unit": None,
        "enum": {"1": "NEW", "2": "CHANGE", "3": "MODIFICATION", "4": "CHANGE_AND_MODIFICATION"},
    },
    "SUM_BMT_CNT": {"semantic_name": "previous_holding_count", "data_type": "integer", "unit": "share"},
    "SUM_BMT_RT": {"semantic_name": "previous_holding_ratio", "data_type": "decimal", "unit": "percent"},
    "SUM_TMT_CNT": {"semantic_name": "current_holding_count", "data_type": "integer", "unit": "share"},
    "SUM_TMT_RT": {"semantic_name": "current_holding_ratio", "data_type": "decimal", "unit": "percent"},
    "SUM_CHN_RWN": {"semantic_name": "report_reason", "data_type": "string", "unit": None},
}

_VALUE_TAGS = {"TE", "TU"}
_CELL_TAGS = {"TD", "TE", "TU"}
_BARE_AMPERSAND = re.compile(r"&(?!#\d+;|#x[0-9A-Fa-f]+;|[A-Za-z][A-Za-z0-9]+;)")
_NON_XML_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].upper()


def _text(element: ET.Element) -> str:
    """Keep authored line breaks, removing XML indentation only."""
    raw = "".join(element.itertext()).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in raw.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _clean_segment(value: str) -> str:
    """Collapse layout whitespace without crossing a semantic paragraph boundary."""
    return re.sub(r"\s+", " ", value).strip()


def _inline_text(element: ET.Element) -> str:
    """Read mixed inline text while ignoring XML indentation between tags."""
    parts: list[str] = []
    if element.text and element.text.strip():
        parts.append(element.text)
    for child in list(element):
        parts.append("".join(child.itertext()))
        if child.tail and child.tail.strip():
            parts.append(child.tail)
    return "".join(parts)


def _text_segments(element: ET.Element) -> tuple[list[str], bool]:
    """Flatten P/SPAN markup into ordered logical text segments.

    P boundaries always create segments.  When there are no paragraphs, each
    direct SPAN creates a segment and its tail stays attached to that segment.
    Nested SPAN elements are flattened into their nearest outer SPAN.
    """
    if _tag(element) == "P":
        return [_clean_segment(_inline_text(element))], True

    direct_paragraphs = [child for child in list(element) if _tag(child) == "P"]
    if direct_paragraphs:
        segments = [
            _clean_segment(_inline_text(paragraph))
            for paragraph in direct_paragraphs
        ]
        return segments, True

    direct_spans = [child for child in list(element) if _tag(child) == "SPAN"]
    if direct_spans:
        segments: list[str] = []
        prefix = _clean_segment(element.text or "")
        if prefix:
            segments.append(prefix)
        for span in direct_spans:
            value = "".join(span.itertext())
            if span.tail and span.tail.strip():
                value += span.tail
            cleaned = _clean_segment(value)
            if cleaned:
                segments.append(cleaned)
        return segments, len(segments) > 1

    value = _text(element)
    return ([value] if value else []), False


def _positive_span(raw: str | None) -> int:
    try:
        return max(1, int(raw or "1"))
    except ValueError:
        return 1


def _as_root(raw: str | bytes | Path | ET.Element | ET.ElementTree) -> ET.Element:
    if isinstance(raw, ET.Element):
        return deepcopy(raw)
    if isinstance(raw, ET.ElementTree):
        return deepcopy(raw.getroot())
    if isinstance(raw, Path):
        return ET.parse(raw).getroot()
    if isinstance(raw, bytes):
        return ET.fromstring(raw)
    if isinstance(raw, str):
        stripped = raw.lstrip()
        if stripped.startswith("<"):
            return ET.fromstring(raw)
        path = Path(raw)
        if path.is_file():
            return ET.parse(path).getroot()
        return ET.fromstring(raw)
    raise TypeError(f"unsupported XML input: {type(raw).__name__}")


def _source_text(raw: str | bytes | Path) -> str:
    if isinstance(raw, Path):
        return raw.read_text(encoding="utf-8", errors="replace")
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, str) and not raw.lstrip().startswith("<") and Path(raw).is_file():
        return Path(raw).read_text(encoding="utf-8", errors="replace")
    return raw if isinstance(raw, str) else str(raw)


def _repair_xml(source: str) -> str:
    """Repair common DART HTML-like text defects without altering source files."""
    repaired = _NON_XML_CONTROL.sub("", source)
    repaired = _BARE_AMPERSAND.sub("&amp;", repaired)
    # DART element names are ASCII; expressions such as ``<표1>`` are text.
    repaired = re.sub(r"<(?=[^A-Za-z/!?])", "&lt;", repaired)
    return repaired


def _root_with_recovery(raw: str | bytes | Path | ET.Element | ET.ElementTree) -> tuple[ET.Element, str, str | None]:
    try:
        return _as_root(raw), "success", None
    except ET.ParseError as first_error:
        if isinstance(raw, (ET.Element, ET.ElementTree)):
            raise
        root = ET.fromstring(_repair_xml(_source_text(raw)))
        return root, "recovered", str(first_error)


def _rows(table: ET.Element) -> list[ET.Element]:
    return [node for node in table.iter() if _tag(node) == "TR"]


def _logical_grid(table: ET.Element) -> tuple[list[list[dict[str, Any] | None]], list[dict[str, Any]]]:
    """Expand HTML-style row/column spans into a rectangular logical grid."""
    grid: list[list[dict[str, Any] | None]] = []
    origins: list[dict[str, Any]] = []
    for row_index, row in enumerate(_rows(table)):
        while len(grid) <= row_index:
            grid.append([])
        column = 0
        cells = [child for child in list(row) if _tag(child) in _CELL_TAGS]
        for cell in cells:
            while column < len(grid[row_index]) and grid[row_index][column] is not None:
                column += 1
            rowspan = _positive_span(cell.get("ROWSPAN"))
            colspan = _positive_span(cell.get("COLSPAN"))
            segments, structured_text = _text_segments(cell)
            item = {
                "element": cell,
                "tag": _tag(cell),
                "text": _text(cell),
                "segments": segments,
                "structured_text": structured_text,
                "row": row_index,
                "column": column,
                "rowspan": rowspan,
                "colspan": colspan,
            }
            origins.append(item)
            for r in range(row_index, row_index + rowspan):
                while len(grid) <= r:
                    grid.append([])
                needed = column + colspan
                if len(grid[r]) < needed:
                    grid[r].extend([None] * (needed - len(grid[r])))
                for c in range(column, needed):
                    if grid[r][c] is None:
                        grid[r][c] = item
            column += colspan
    width = max((len(row) for row in grid), default=0)
    for row in grid:
        row.extend([None] * (width - len(row)))
    return grid, origins


def _normalise(raw_value: str, raw_code: str | None, definition: Mapping[str, Any]) -> tuple[Any, str, str | None]:
    data_type = definition.get("data_type", "string")
    source = raw_code if data_type in {"date", "boolean", "enum"} and raw_code is not None else raw_value
    try:
        if data_type == "integer":
            return int(source.replace(",", "")), "success", None
        if data_type == "decimal":
            return float(Decimal(source.replace(",", ""))), "success", None
        if data_type == "date":
            return datetime.strptime(source, "%Y%m%d").date().isoformat(), "success", None
        if data_type in {"boolean", "enum"}:
            mapping = definition.get("enum", {})
            if source not in mapping:
                raise ValueError(f"unmapped {data_type} value")
            return mapping[source], "success", None
        return raw_value, "not_applicable", None
    except (ValueError, InvalidOperation) as exc:
        return raw_value, "failed", str(exc)


def _nearest_current_td(row_cells: list[dict[str, Any]], column: int) -> dict[str, Any] | None:
    candidates = [c for c in row_cells if c["tag"] == "TD" and c["column"] < column and c["text"]]
    return max(candidates, key=lambda c: c["column"], default=None)


def _column_headers(grid: list[list[dict[str, Any] | None]]) -> list[list[dict[str, Any] | None]]:
    """Cache the latest TD above every grid position in one linear pass."""
    width = len(grid[0]) if grid else 0
    latest: list[dict[str, Any] | None] = [None] * width
    headers: list[list[dict[str, Any] | None]] = []
    for row_index, row in enumerate(grid):
        headers.append(latest.copy())
        seen: set[int] = set()
        for cell in row:
            if cell is None or id(cell) in seen or cell["row"] != row_index:
                continue
            seen.add(id(cell))
            if cell["tag"] == "TD" and cell["text"] and cell["colspan"] < width:
                for column in range(cell["column"], min(width, cell["column"] + cell["colspan"])):
                    latest[column] = cell
    return headers


def _contexts(
    grid: list[list[dict[str, Any] | None]],
    row_cells: list[dict[str, Any]],
    column_td: dict[str, Any] | None,
    value: dict[str, Any],
) -> tuple[str | None, str | None, str | None, str | None]:
    row, column = value["row"], value["column"]
    current_td = _nearest_current_td(row_cells, column)
    values_in_row = [c for c in row_cells if c["tag"] in _VALUE_TAGS]
    preceding_tds = [_nearest_current_td(row_cells, c["column"]) for c in values_in_row]
    row_tds = {td["column"] for td in preceding_tds if td is not None}

    active_cells = {id(c): c for c in grid[row][:column] if c is not None}.values()
    group_candidates = [c for c in active_cells if c["tag"] == "TD" and c["row"] < row and c["text"]]
    group = min(group_candidates, key=lambda c: c["column"], default=None)

    is_repeating_row = len(values_in_row) > 1 and len(row_tds) == 1 and current_td is not None and column_td is not None
    row_label = current_td["text"] if is_repeating_row else None
    column_label = column_td["text"] if is_repeating_row else None
    label = column_label or (current_td["text"] if current_td else None)
    return label, (group["text"] if group else None), row_label, column_label


_COVER_LABELS = ("회사명", "대표이사", "본점소재지", "작성책임자")


def _compact_label(value: str) -> str:
    return re.sub(r"[\s:：]", "", value)


def _cover_label(value: str) -> str | None:
    compact = _compact_label(value)
    if compact.startswith("대표이사"):
        return "대표이사"
    return compact if compact in _COVER_LABELS else None


def _cover_text(element: ET.Element) -> str:
    value = _clean_segment(_inline_text(element))
    for pattern, replacement in (
        (r"\(\s*전\s*화\s*\)", "(전화)"),
        (r"\(\s*홈\s*페\s*이\s*지\s*\)", "(홈페이지)"),
        (r"\(\s*직\s*책\s*\)", "(직책)"),
        (r"\(\s*성\s*명\s*\)", "(성명)"),
    ):
        value = re.sub(pattern, replacement, value)
    value = re.sub(r"(https?://)\s+", r"\1", value)
    value = re.sub(
        r"^(\((?:전화|홈페이지|직책|성명)\))\s*",
        r"\1 ",
        value,
    )
    return value


def _major_cover_nodes(table: ET.Element) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return recipient text and the fixed four-field major cover table."""
    grid, origins = _logical_grid(table)
    td_cells = [cell for cell in origins if cell["tag"] == "TD"]
    detected_labels = {
        label
        for cell in td_cells
        if cell["column"] == 0
        and (label := _cover_label(_cover_text(cell["element"]))) is not None
    }
    recipient_cells = [
        cell
        for cell in td_cells
        if "금융위원회" in _cover_text(cell["element"])
    ]
    if not recipient_cells or not set(_COVER_LABELS).issubset(detected_labels):
        return None

    rows: dict[int, list[dict[str, Any]]] = {}
    for cell in td_cells:
        rows.setdefault(cell["row"], []).append(cell)
    for cells in rows.values():
        cells.sort(key=lambda item: item["column"])

    recipient_cell = recipient_cells[0]
    recipient_row = rows[recipient_cell["row"]]
    recipient_values = [
        _cover_text(cell["element"])
        for cell in recipient_row
        if _cover_text(cell["element"])
    ]
    recipient_node = {
        "node_type": "text",
        "text_role": "cover_header",
        "data_type": "text",
        "value": recipient_values,
        "raw_value": "\n".join(recipient_values),
    }

    anchors: list[tuple[int, str, dict[str, Any]]] = []
    for cell in td_cells:
        if cell["column"] != 0:
            continue
        label = _cover_label(_cover_text(cell["element"]))
        if label is not None:
            anchors.append((cell["row"], label, cell))
    anchors.sort(key=lambda item: item[0])

    fields: list[dict[str, Any]] = []
    max_row = max(rows, default=-1)
    for anchor_index, (start_row, label, anchor) in enumerate(anchors):
        end_row = anchors[anchor_index + 1][0] if anchor_index + 1 < len(anchors) else max_row + 1
        values: list[str] = []
        for row_index in range(start_row, end_row):
            for cell in rows.get(row_index, []):
                if row_index == start_row and cell is anchor:
                    continue
                if cell["column"] <= anchor["column"]:
                    continue
                value = _cover_text(cell["element"])
                if value:
                    values.append(value)
        is_text = len(values) > 1
        raw_value = "\n".join(values)
        fields.append(
            {
                "label": label,
                "code": None,
                "code_type": None,
                "raw_value": raw_value,
                "value": values if is_text else (values[0] if values else ""),
                "data_type": "text" if is_text else "string",
                "unit": None,
                "known_field": True,
                "group_label": None,
                "row_label": None,
                "column_label": None,
                "source": {
                    "row": anchor["row"],
                    "column": anchor["column"],
                    "rowspan": anchor["rowspan"],
                    "colspan": anchor["colspan"],
                },
                "normalization_status": "not_applicable",
            }
        )

    cover_table = {
        "parse_status": "success",
        "table_class": table.get("ACLASS"),
        "title": "표지정보",
        "dimensions": {"rows": len(grid), "columns": len(grid[0]) if grid else 0},
        "fields": fields,
    }
    return recipient_node, cover_table


def canonicalize_table(
    raw_table_fragment: str | bytes | Path | ET.Element | ET.ElementTree,
    *,
    field_definitions: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert one TABLE fragment to a canonical, JSON-serialisable mapping."""
    registry = field_definitions if field_definitions is not None else FIELD_DEFINITIONS
    try:
        root, parse_status, parse_warning = _root_with_recovery(raw_table_fragment)
        table = root if _tag(root) == "TABLE" else next(node for node in root.iter() if _tag(node) == "TABLE")
        grid, origins = _logical_grid(table)
    except (ET.ParseError, StopIteration, OSError) as exc:
        source = raw_table_fragment if isinstance(raw_table_fragment, (str, bytes)) else str(raw_table_fragment)
        return {"parse_status": "failed", "parse_error": str(exc), "raw_fragment": source, "fields": []}

    title = None
    width = len(grid[0]) if grid else 0
    for cell in origins:
        if cell["tag"] == "TD" and cell["colspan"] >= width and cell["text"]:
            title = cell["text"]
            break

    fields: list[dict[str, Any]] = []
    column_headers = _column_headers(grid)
    cells_by_row: dict[int, list[dict[str, Any]]] = {}
    for origin in origins:
        cells_by_row.setdefault(origin["row"], []).append(origin)
    for cell in origins:
        if cell["tag"] not in _VALUE_TAGS:
            continue
        element = cell["element"]
        code_type = "ACODE" if cell["tag"] == "TE" else "AUNIT"
        code = element.get(code_type)
        segments = cell["segments"]
        raw_value = "\n".join(segments)
        raw_code = element.get("AUNITVALUE") if cell["tag"] == "TU" else None
        definition = registry.get(code or "")
        effective = definition or {"data_type": "string", "unit": None}
        if cell["structured_text"]:
            effective = {"data_type": "text", "unit": None}
            normalised, status, error = segments, "not_applicable", None
        else:
            normalised, status, error = _normalise(raw_value, raw_code, effective)
        label, group_label, row_label, column_label = _contexts(
            grid,
            cells_by_row[cell["row"]],
            column_headers[cell["row"]][cell["column"]],
            cell,
        )
        field: dict[str, Any] = {
            "label": label,
            "code": code,
            "code_type": code_type,
            "raw_value": raw_value,
            "value": normalised,
            "data_type": effective.get("data_type", "string"),
            "unit": effective.get("unit"),
            "known_field": definition is not None,
            "group_label": group_label,
            "row_label": row_label,
            "column_label": column_label,
            "source": {"row": cell["row"], "column": cell["column"], "rowspan": cell["rowspan"], "colspan": cell["colspan"]},
            "normalization_status": status,
        }
        if raw_code is not None:
            field["raw_code_value"] = raw_code
            field["display_value"] = raw_value
        if definition and definition.get("semantic_name"):
            field["semantic_name"] = definition["semantic_name"]
        if error:
            field["normalization_error"] = error
        fields.append(field)

    return {
        "parse_status": parse_status,
        **({"parse_warning": parse_warning} if parse_warning else {}),
        "table_class": table.get("ACLASS"),
        "title": title,
        "dimensions": {"rows": len(grid), "columns": width},
        "fields": fields,
    }


def canonicalize_document(
    raw_document: str | bytes | Path | ET.Element | ET.ElementTree,
    *,
    field_definitions: Mapping[str, Mapping[str, Any]] | None = None,
    document_type: str | None = None,
) -> dict[str, Any]:
    """Canonicalise every table in source order; malformed input is retained."""
    try:
        root, parse_status, parse_warning = _root_with_recovery(raw_document)
    except (ET.ParseError, OSError) as exc:
        source = raw_document if isinstance(raw_document, (str, bytes)) else str(raw_document)
        return {"parse_status": "failed", "parse_error": str(exc), "raw_document": source, "child": []}

    parent = {child: node for node in root.iter() for child in node}
    document_order = {node: index for index, node in enumerate(root.iter())}
    section_elements = [
        node
        for node in root.iter()
        if _tag(node) == "COVER" or re.fullmatch(r"SECTION(?:-\d+)?", _tag(node))
    ]
    section_by_element: dict[ET.Element, dict[str, Any]] = {}
    section_nodes: list[dict[str, Any]] = []
    for source_index, section_element in enumerate(section_elements):
        source_tag = _tag(section_element)
        if source_tag == "COVER":
            section_type = "COVER"
            level = 0
            title_tag = "COVER-TITLE"
        else:
            section_type = "SECTION"
            match = re.fullmatch(r"SECTION-(\d+)", source_tag)
            if match:
                level = int(match.group(1))
            else:
                level = 1
                ancestor = parent.get(section_element)
                while ancestor is not None:
                    if re.fullmatch(r"SECTION(?:-\d+)?", _tag(ancestor)):
                        level += 1
                    ancestor = parent.get(ancestor)
            title_tag = "TITLE"

        title_element = next(
            (child for child in list(section_element) if _tag(child) == title_tag),
            None,
        )
        title = _text(title_element) if title_element is not None else None
        section = {
            "node_type": "section",
            "section_type": section_type,
            "source_tag": source_tag,
            "level": level,
            "title": title,
            "section_class": section_element.get("ACLASS"),
            "source_index": source_index,
            "child": [],
        }
        if title_element is not None:
            section["title_metadata"] = {
                key.lower(): value
                for key, value in title_element.attrib.items()
                if key.upper() not in {"WIDTH", "HEIGHT", "ALIGN", "VALIGN"}
            }
        section_by_element[section_element] = section
        section_nodes.append(section)

    ordered_children: dict[int, list[tuple[int, dict[str, Any]]]] = {id(section): [] for section in section_nodes}
    root_children: list[tuple[int, dict[str, Any]]] = []
    for section_element, section in section_by_element.items():
        ancestor = parent.get(section_element)
        while ancestor is not None and ancestor not in section_by_element:
            ancestor = parent.get(ancestor)
        destination = ordered_children[id(section_by_element[ancestor])] if ancestor is not None else root_children
        destination.append((document_order[section_element], section))

    for index, table in enumerate(node for node in root.iter() if _tag(node) == "TABLE"):
        major_cover = _major_cover_nodes(table) if document_type == "major" else None
        if major_cover is not None:
            recipient_node, canonical = major_cover
        else:
            recipient_node = None
            canonical = canonicalize_table(table, field_definitions=field_definitions)
        canonical["node_type"] = "table"
        canonical["source_index"] = index
        ancestor = parent.get(table)
        table_group = None
        leaf_section_element = None
        while ancestor is not None:
            if table_group is None and _tag(ancestor) == "TABLE-GROUP":
                table_group = ancestor
            if leaf_section_element is None and ancestor in section_by_element:
                leaf_section_element = ancestor
            ancestor = parent.get(ancestor)
        canonical["table_group_class"] = table_group.get("ACLASS") if table_group is not None else None
        destination = ordered_children[id(section_by_element[leaf_section_element])] if leaf_section_element is not None else root_children
        if recipient_node is not None:
            destination.append((document_order[table], recipient_node))
        destination.append((document_order[table], canonical))

    # Preserve narrative content outside tables.  A P is one paragraph; a
    # top-level SPAN is used only when it is not already represented by a P.
    for text_element in root.iter():
        tag = _tag(text_element)
        if tag not in {"P", "SPAN"}:
            continue
        ancestor = parent.get(text_element)
        inside_table = False
        nested_text = False
        leaf_section_element = None
        while ancestor is not None:
            ancestor_tag = _tag(ancestor)
            if ancestor_tag == "TABLE":
                inside_table = True
                break
            if tag == "SPAN" and ancestor_tag in {"P", "SPAN"}:
                nested_text = True
            if leaf_section_element is None and ancestor in section_by_element:
                leaf_section_element = ancestor
            ancestor = parent.get(ancestor)
        if inside_table or nested_text:
            continue

        segments, _ = _text_segments(text_element)
        text_node = {
            "node_type": "text",
            "data_type": "text",
            "value": segments,
            "raw_value": "\n".join(segments),
        }
        destination = ordered_children[id(section_by_element[leaf_section_element])] if leaf_section_element is not None else root_children
        destination.append((document_order[text_element], text_node))

    def merge_text_nodes(items: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for _, node in sorted(items, key=lambda item: item[0]):
            if (
                node["node_type"] == "text"
                and not node.get("text_role")
                and merged
                and merged[-1]["node_type"] == "text"
                and not merged[-1].get("text_role")
            ):
                previous = merged[-1]
                previous["value"].extend(node["value"])
                previous["raw_value"] = "\n".join(previous["value"])
            else:
                merged.append(node)
        return merged

    for section in section_nodes:
        section["child"] = merge_text_nodes(ordered_children[id(section)])
    child = merge_text_nodes(root_children)

    metadata: dict[str, Any] = {}
    for node in root.iter():
        name = _tag(node)
        if name in {"DOCUMENT-NAME", "COMPANY-NAME", "FORMULA-VERSION"} and name not in metadata:
            metadata[name.lower().replace("-", "_")] = _text(node)
    return {
        "parse_status": parse_status,
        **({"parse_warning": parse_warning} if parse_warning else {}),
        "metadata": metadata,
        "child": child,
    }


def canonicalize_holding_tree(
    input_root: str | Path,
    output_root: str | Path,
    *,
    field_definitions: Mapping[str, Mapping[str, Any]] | None = None,
    progress: bool = False,
    document_type: str | None = None,
) -> dict[str, int]:
    """Canonicalise every holding XML while mirroring its relative directory."""
    source_root = Path(input_root)
    destination_root = Path(output_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"holding input directory not found: {source_root}")

    counts = {"total": 0, "success": 0, "recovered": 0, "failed": 0}
    for source_path in sorted(source_root.rglob("*.xml")):
        counts["total"] += 1
        try:
            canonical = canonicalize_document(
                source_path,
                field_definitions=field_definitions,
                document_type=document_type,
            )
        except Exception as exc:  # Keep a batch moving after an unexpected file error.
            canonical = {
                "parse_status": "failed",
                "parse_error": f"{type(exc).__name__}: {exc}",
                "raw_document": str(source_path),
                "child": [],
            }

        status = canonical.get("parse_status", "failed")
        status_key = status if status in {"success", "recovered"} else "failed"
        counts[status_key] += 1
        canonical["source_path"] = source_path.relative_to(source_root).as_posix()

        relative_output = source_path.relative_to(source_root).with_suffix(".json")
        output_path = destination_root / relative_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(canonical, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if progress and (counts["total"] == 1 or counts["total"] % 25 == 0 or status_key == "failed"):
            print(
                f"processed={counts['total']} status={status_key} "
                f"source={canonical['source_path']}",
                flush=True,
            )
    return counts


def main(argv: list[str] | None = None) -> int:
    """Run holding and major XML-to-canonical JSON batch conversion."""
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Canonicalize DART holding/major XML into mirrored JSON files."
    )
    parser.add_argument(
        "--category",
        action="append",
        choices=("holding", "major"),
        help="category to process; repeat as needed (default: holding and major)",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="override input root; requires exactly one --category",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="override output root; requires exactly one --category",
    )
    args = parser.parse_args(argv)

    categories = args.category or ["holding", "major"]
    if (args.input_root is not None or args.output_root is not None) and len(categories) != 1:
        parser.error("--input-root/--output-root require exactly one --category")

    counts = {"total": 0, "success": 0, "recovered": 0, "failed": 0}
    for category in categories:
        input_root = args.input_root or repository_root / "data" / "raw" / category
        output_root = args.output_root or repository_root / "data" / "canonical" / category
        print(f"category={category} input={input_root} output={output_root}", flush=True)
        try:
            category_counts = canonicalize_holding_tree(
                input_root,
                output_root,
                progress=True,
                document_type=category,
            )
        except FileNotFoundError as exc:
            parser.error(str(exc))
        for key in counts:
            counts[key] += category_counts[key]

    print(
        "canonicalization complete: "
        f"total={counts['total']} "
        f"success={counts['success']} "
        f"recovered={counts['recovered']} "
        f"failed={counts['failed']}"
    )
    return 1 if counts["failed"] else 0


__all__ = [
    "FIELD_DEFINITIONS",
    "canonicalize_document",
    "canonicalize_holding_tree",
    "canonicalize_table",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
