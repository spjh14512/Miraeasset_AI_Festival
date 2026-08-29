"""Load and match the curated Event category workbook without Excel dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile


_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_LEADING_ANNOTATION = re.compile(r"^(?:\s*\[[^]]*(?:정정|첨부추가)[^]]*\]\s*)+")
_MAJOR_REPORT = re.compile(r"^주요사항보고서\s*\((?P<name>.*)\)\s*$")


def _compact(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣%]", "", value).casefold()


def _exact(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def cleaned_report_name(report_name: str) -> str:
    """Remove filing annotations and the generic major-report wrapper."""
    cleaned = _LEADING_ANNOTATION.sub("", report_name).strip()
    match = _MAJOR_REPORT.fullmatch(cleaned)
    return match.group("name").strip() if match else cleaned


@dataclass(frozen=True, slots=True)
class EventCategory:
    event_subtype: str
    event_type: str
    row_number: int


@dataclass(frozen=True, slots=True)
class EventCategoryCatalog:
    entries: tuple[EventCategory, ...]

    def match(self, report_name: str) -> EventCategory | None:
        candidate = cleaned_report_name(report_name)
        exact_candidate = _exact(candidate)
        compact_candidate = _compact(candidate)

        exact_matches = [
            entry
            for entry in self.entries
            if exact_candidate.startswith(_exact(entry.event_subtype))
        ]
        if exact_matches:
            return max(
                exact_matches,
                key=lambda entry: (len(_exact(entry.event_subtype)), -entry.row_number),
            )

        compact_matches = [
            entry
            for entry in self.entries
            if compact_candidate.startswith(_compact(entry.event_subtype))
        ]
        if not compact_matches:
            return None
        return max(
            compact_matches,
            key=lambda entry: (len(_compact(entry.event_subtype)), -entry.row_number),
        )


def _shared_strings(archive: zipfile.ZipFile) -> tuple[str, ...]:
    name = "xl/sharedStrings.xml"
    if name not in archive.namelist():
        return ()
    root = ET.fromstring(archive.read(name))
    namespace = {"m": _MAIN_NS}
    return tuple("".join(item.itertext()) for item in root.findall("m:si", namespace))


def _first_sheet_path(archive: zipfile.ZipFile) -> str:
    namespace = {"m": _MAIN_NS, "r": _REL_NS}
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheet = workbook.find("m:sheets/m:sheet", namespace)
    if sheet is None:
        raise ValueError("Event category workbook contains no worksheet")
    relationship_id = sheet.attrib[f"{{{_REL_NS}}}id"]

    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    target = next(
        (
            item.attrib["Target"]
            for item in relationships.findall(f"{{{_PACKAGE_REL_NS}}}Relationship")
            if item.attrib.get("Id") == relationship_id
        ),
        None,
    )
    if target is None:
        raise ValueError("Cannot resolve the first worksheet in Event category workbook")
    if target.startswith("/"):
        return target.lstrip("/")
    return target if target.startswith("xl/") else f"xl/{target}"


def _cell_value(
    cell: ET.Element,
    shared_strings: tuple[str, ...],
    namespace: dict[str, str],
) -> str:
    inline = cell.find("m:is", namespace)
    if inline is not None:
        return "".join(inline.itertext()).strip()
    value = cell.find("m:v", namespace)
    if value is None or value.text is None:
        return ""
    if cell.attrib.get("t") == "s":
        return shared_strings[int(value.text)].strip()
    return value.text.strip()


def load_event_categories(path: Path) -> EventCategoryCatalog:
    """Read columns A/B from the first worksheet of the curated XLSX file."""
    if not path.is_file():
        raise FileNotFoundError(f"Event category workbook is missing: {path}")
    namespace = {"m": _MAIN_NS}
    entries: list[EventCategory] = []
    with zipfile.ZipFile(path) as archive:
        shared_strings = _shared_strings(archive)
        root = ET.fromstring(archive.read(_first_sheet_path(archive)))
        for row in root.findall(".//m:sheetData/m:row", namespace):
            row_number = int(row.attrib.get("r", "0"))
            values: dict[str, str] = {}
            for cell in row.findall("m:c", namespace):
                reference = cell.attrib.get("r", "")
                column = re.sub(r"\d", "", reference)
                if column in {"A", "B"}:
                    values[column] = _cell_value(cell, shared_strings, namespace)
            subtype = values.get("A", "").strip()
            event_type = values.get("B", "").strip()
            if row_number == 1 and subtype == "문서명":
                continue
            if subtype and event_type:
                entries.append(EventCategory(subtype, event_type, row_number))
    if not entries:
        raise ValueError("Event category workbook has no A/B category rows")
    return EventCategoryCatalog(tuple(entries))


__all__ = [
    "EventCategory",
    "EventCategoryCatalog",
    "cleaned_report_name",
    "load_event_categories",
]
