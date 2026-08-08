"""Standalone DART XML/HTML-like disclosure canonicaliser.

Produces one loss-aware Canonical JSON schema for holding, major, periodic and
exchange disclosures. It intentionally preserves source structure and does
not apply business normalization or Markdown rendering.

Usage:
    python converters/canonicalizer2.py periodic data/raw/periodic data/canonical/periodic
    python converters/canonicalizer2.py exchange data/raw/exchange data/canonical/exchange
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any, Iterable
import xml.etree.ElementTree as ET


_CELL_TAGS = {"TD", "TH", "TE", "TU"}
_VALUE_TAGS = {"TE", "TU"}
_SECTION = re.compile(r"SECTION(?:-\d+)?$")
_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")
_AMP = re.compile(r"&(?!#\d+;|#x[0-9A-Fa-f]+;|[A-Za-z][A-Za-z0-9]+;)")
_NON_XML_TAG_START = re.compile(r"<(?=[^A-Za-z/!?])")
_PRESENTATION_ATTRIBUTES = {
    "WIDTH", "HEIGHT", "ALIGN", "VALIGN", "CLASS", "STYLE",
    "BORDER", "BORDERCOLOR", "CELLPADDING", "CELLSPACING",
}


def _tag(node: ET.Element) -> str:
    return node.tag.rsplit("}", 1)[-1].upper()


def _text(node: ET.Element) -> str:
    raw = "".join(node.itertext()).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.strip() for line in raw.split("\n") if line.strip()).strip()


def _inline_text(node: ET.Element) -> str:
    parts = [node.text or ""]
    for child in node:
        parts.append("".join(child.itertext()))
        parts.append(child.tail or "")
    return "".join(parts)


def _segments(node: ET.Element) -> list[str]:
    paragraphs = [child for child in node if _tag(child) == "P"]
    values = [_inline_text(item) for item in paragraphs] if paragraphs else [_inline_text(node)]
    return [re.sub(r"\s+", " ", value).strip() for value in values if value.strip()]


class _TolerantHtmlTree(HTMLParser):
    """Recover DART's HTML-like documents when strict XML parsing fails."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = ET.Element("DOCUMENT")
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = ET.SubElement(self.stack[-1], tag.upper(), {key.upper(): value or "" for key, value in attrs})
        if tag.lower() not in {"br", "meta", "link", "img", "hr", "input", "col"}:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in {"br", "meta", "link", "img", "hr", "input", "col"}:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        wanted = tag.upper()
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == wanted:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if not data:
            return
        node = self.stack[-1]
        if len(node):
            node[-1].tail = (node[-1].tail or "") + data
        else:
            node.text = (node.text or "") + data


def _source(raw: str | bytes | Path) -> str:
    if isinstance(raw, Path):
        return raw.read_text(encoding="utf-8", errors="replace")
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    try:
        path = Path(raw)
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else raw
    except OSError:
        return raw


def _root(raw: str | bytes | Path) -> tuple[ET.Element, str | None]:
    source = _source(raw)
    repaired = _NON_XML_TAG_START.sub("&lt;", _AMP.sub("&amp;", _CONTROL.sub("", source)))
    try:
        return ET.fromstring(repaired), None
    except ET.ParseError as error:
        parser = _TolerantHtmlTree()
        parser.feed(repaired)
        parser.close()
        return parser.root, str(error)


def _span(value: str | None) -> int:
    try:
        return max(1, int(value or "1"))
    except ValueError:
        return 1


def _grid(table: ET.Element) -> tuple[list[list[dict[str, Any] | None]], list[dict[str, Any]]]:
    grid: list[list[dict[str, Any] | None]] = []
    origins: list[dict[str, Any]] = []
    for row_index, row in enumerate(node for node in table.iter() if _tag(node) == "TR"):
        while len(grid) <= row_index:
            grid.append([])
        column = 0
        for cell in (item for item in row if _tag(item) in _CELL_TAGS):
            while column < len(grid[row_index]) and grid[row_index][column] is not None:
                column += 1
            item = {
                "element": cell, "tag": _tag(cell), "text": _text(cell), "segments": _segments(cell),
                "row": row_index, "column": column, "rowspan": _span(cell.get("ROWSPAN")),
                "colspan": _span(cell.get("COLSPAN")),
            }
            origins.append(item)
            for target_row in range(row_index, row_index + item["rowspan"]):
                while len(grid) <= target_row:
                    grid.append([])
                required = column + item["colspan"]
                grid[target_row].extend([None] * max(0, required - len(grid[target_row])))
                for target_column in range(column, required):
                    if grid[target_row][target_column] is None:
                        grid[target_row][target_column] = item
            column += item["colspan"]
    width = max((len(row) for row in grid), default=0)
    for row in grid:
        row.extend([None] * (width - len(row)))
    return grid, origins


def _context(
    grid: list[list[dict[str, Any] | None]],
    rows: dict[int, list[dict[str, Any]]],
    value: dict[str, Any],
) -> tuple[str | None, str | None, str | None, str | None]:
    """Return the same label context keys used by canonicalizer.py.

    This is structural context only: it does not infer a business meaning.
    """
    row, column = value["row"], value["column"]
    preceding = [cell for cell in rows[row] if cell["tag"] in {"TD", "TH"} and cell["column"] < column and cell["text"]]
    row_header = max(preceding, key=lambda cell: cell["column"], default=None)
    groups = [cell for cell in grid[row][:column] if cell and cell["tag"] in {"TD", "TH"} and cell["row"] < row and cell["text"]]
    group = min(groups, key=lambda cell: cell["column"], default=None)
    values_in_row = [cell for cell in rows[row] if cell["tag"] in _VALUE_TAGS]
    repeating_row = len(values_in_row) > 1 and row_header is not None
    row_label = row_header["text"] if repeating_row else None
    # Column labels are deliberately left blank until a later normalisation
    # stage. Guessing them from arbitrary table layouts loses information.
    column_label = None
    label = column_label or (row_header["text"] if row_header else None)
    return label, (group["text"] if group else None), row_label, column_label


def _title(grid: list[list[dict[str, Any] | None]], origins: list[dict[str, Any]]) -> str | None:
    width = len(grid[0]) if grid else 0
    for item in origins:
        if item["tag"] in {"TD", "TH"} and item["colspan"] >= width and item["text"]:
            return item["text"]
    return None


def _table(table: ET.Element, source_index: int, category: str) -> dict[str, Any]:
    grid, origins = _grid(table)
    by_row: dict[int, list[dict[str, Any]]] = {}
    for item in origins:
        by_row.setdefault(item["row"], []).append(item)

    fields: list[dict[str, Any]] = []
    values = [item for item in origins if item["tag"] in _VALUE_TAGS]
    for item in values:
        element = item["element"]
        code_type = "ACODE" if item["tag"] == "TE" else "AUNIT"
        label, group_label, row_label, column_label = _context(grid, by_row, item)
        raw_value = "\n".join(item["segments"])
        multi_segment = len(item["segments"]) > 1
        field: dict[str, Any] = {
            "label": label,
            "code": element.get(code_type),
            "code_type": code_type,
            "raw_value": raw_value,
            "value": item["segments"] if multi_segment else raw_value,
            "data_type": "text" if multi_segment else "string",
            "unit": None,
            "known_field": False,
            "group_label": group_label,
            "row_label": row_label,
            "column_label": column_label,
            "source": {key: item[key] for key in ("row", "column", "rowspan", "colspan")},
            "normalization_status": "not_applicable",
        }
        if item["tag"] == "TU" and element.get("AUNITVALUE") is not None:
            field["raw_code_value"] = element.get("AUNITVALUE")
            field["display_value"] = raw_value
        fields.append(field)

    # Exchange forms are label/value TD rows with no TE/TU field codes.
    if category == "exchange" and not values:
        for row_cells in by_row.values():
            cells = sorted((cell for cell in row_cells if cell["text"]), key=lambda cell: cell["column"])
            if len(cells) < 2:
                continue
            value = cells[-1]
            _, group_label, _, _ = _context(grid, by_row, value)
            raw_value = "\n".join(value["segments"])
            multi_segment = len(value["segments"]) > 1
            fields.append({
                "label": " / ".join(cell["text"] for cell in cells[:-1]),
                "code": None,
                "code_type": None,
                "raw_value": raw_value,
                "value": value["segments"] if multi_segment else raw_value,
                "data_type": "text" if multi_segment else "string",
                "unit": None,
                "known_field": False,
                "group_label": group_label,
                "row_label": None,
                "column_label": None,
                "source": {key: value[key] for key in ("row", "column", "rowspan", "colspan")},
                "normalization_status": "not_applicable",
            })

    return {
        "node_type": "table", "source_index": source_index,
        "table_class": table.get("ACLASS") or table.get("CLASS"),
        "title": _title(grid, origins),
        "dimensions": {"rows": len(grid), "columns": len(grid[0]) if grid else 0},
        "fields": fields,
    }


def _ancestors(node: ET.Element, parent: dict[ET.Element, ET.Element]) -> Iterable[ET.Element]:
    current = parent.get(node)
    while current is not None:
        yield current
        current = parent.get(current)


def canonicalize_document(raw: str | bytes | Path, *, category: str) -> dict[str, Any]:
    """Create pre-normalisation Canonical JSON for one DART disclosure."""
    if category not in {"holding", "major", "periodic", "exchange"}:
        raise ValueError("category must be holding, major, periodic, or exchange")
    root, warning = _root(raw)
    parent = {child: node for node in root.iter() for child in node}
    order = {node: index for index, node in enumerate(root.iter())}
    section_elements = [node for node in root.iter() if _SECTION.fullmatch(_tag(node)) or _tag(node) == "COVER"]
    sections: dict[ET.Element, dict[str, Any]] = {}
    for node in section_elements:
        is_cover = _tag(node) == "COVER"
        title_tag = "COVER-TITLE" if is_cover else "TITLE"
        title_node = next((child for child in node if _tag(child) == title_tag), None)
        sections[node] = {
            "node_type": "section", "section_type": "COVER" if is_cover else "SECTION",
            "source_tag": _tag(node), "source_index": order[node],
            "level": 0 if is_cover else sum(1 for ancestor in _ancestors(node, parent) if _SECTION.fullmatch(_tag(ancestor))) + 1,
            "title": _text(title_node) if title_node is not None else None,
            "section_class": node.get("ACLASS") or node.get("CLASS"),
            "child": [],
        }
        if title_node is not None:
            metadata = {
                key.lower(): value
                for key, value in title_node.attrib.items()
                if key.upper() not in _PRESENTATION_ATTRIBUTES
            }
            if metadata:
                sections[node]["title_metadata"] = metadata
    top: list[tuple[int, dict[str, Any]]] = []
    children: dict[int, list[tuple[int, dict[str, Any]]]] = {id(section): [] for section in sections.values()}

    def destination(node: ET.Element) -> list[tuple[int, dict[str, Any]]]:
        for ancestor in _ancestors(node, parent):
            if ancestor in sections:
                return children[id(sections[ancestor])]
        return top

    for table in (node for node in root.iter() if _tag(node) == "TABLE"):
        destination(table).append((order[table], _table(table, order[table], category)))
    for node in root.iter():
        if _tag(node) != "P" or any(_tag(ancestor) == "TABLE" for ancestor in _ancestors(node, parent)):
            continue
        segments = _segments(node)
        if segments:
            destination(node).append((order[node], {"node_type": "text", "value": segments, "raw_value": "\n".join(segments)}))

    def depth(node: ET.Element) -> int:
        return sum(1 for ancestor in _ancestors(node, parent) if ancestor in sections)

    for node in sorted(sections, key=depth, reverse=True):
        section = sections[node]
        destination(node).append((section["source_index"], section))
    for section in sections.values():
        section["child"] = [value for _, value in sorted(children[id(section)], key=lambda item: item[0])]

    metadata: dict[str, str] = {}
    for node in root.iter():
        name = _tag(node)
        if name in {"DOCUMENT-NAME", "COMPANY-NAME", "FORMULA-VERSION"} and name not in metadata:
            metadata[name.lower().replace("-", "_")] = _text(node)
    result: dict[str, Any] = {
        "schema_version": "1.0", "category": category,
        "parse_status": "recovered" if warning else "success", "metadata": metadata,
        "child": [value for _, value in sorted(top, key=lambda item: item[0])],
    }
    if warning:
        result["parse_warning"] = warning
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create Canonical JSON from DART disclosures.")
    parser.add_argument("category", choices=("holding", "major", "periodic", "exchange"))
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--progress", action="store_true", help="print progress every 25 files")
    args = parser.parse_args(argv)
    if not args.input_root.is_dir():
        parser.error(f"input directory not found: {args.input_root}")

    counts = {"total": 0, "success": 0, "recovered": 0, "failed": 0}
    for source in sorted(args.input_root.rglob("*.xml")):
        counts["total"] += 1
        try:
            result = canonicalize_document(source, category=args.category)
        except Exception as error:  # Keep the batch moving and retain the failure reason.
            result = {"schema_version": "1.0", "category": args.category, "parse_status": "failed", "parse_error": f"{type(error).__name__}: {error}", "child": []}
        status = result["parse_status"]
        counts[status if status in {"success", "recovered"} else "failed"] += 1
        result["source_path"] = source.relative_to(args.input_root).as_posix()
        target = args.output_root / source.relative_to(args.input_root).with_suffix(".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.progress and (counts["total"] == 1 or counts["total"] % 25 == 0 or status == "failed"):
            print(f"processed={counts['total']} status={status} source={result['source_path']}", flush=True)
    print("complete: " + " ".join(f"{key}={value}" for key, value in counts.items()), flush=True)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
