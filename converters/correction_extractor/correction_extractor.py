"""Extract correction metadata and source boundaries without mutating input."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import re
from typing import Iterable
import xml.etree.ElementTree as ET

from converters.common.document_loader import repair_xml_text
from converters.common.source_models import DocumentContext, DocumentSyntax, SourceRef
from converters.correction_extractor.correction_models import (
    CorrectionBlockRef,
    CorrectionBlockType,
    CorrectionExtraction,
    CorrectionIssue,
    CorrectionMetadata,
    CorrectionStatus,
)
from converters.table_parser.table_parser import build_logical_grid


_CORRECTION_FRAGMENT = re.compile(
    r"<CORRECTION\b[^>]*>.*?</CORRECTION\s*>",
    re.IGNORECASE | re.DOTALL,
)
_KOREAN_DATE = re.compile(
    r"(?P<year>20\d{2})\s*년\s*(?P<month>\d{1,2})\s*월\s*"
    r"(?P<day>\d{1,2})\s*일"
)
_ISO_DATE = re.compile(
    r"(?P<year>20\d{2})\s*[-./]\s*(?P<month>\d{1,2})\s*"
    r"[-./]\s*(?P<day>\d{1,2})"
)
_TARGET_DOCUMENT = re.compile(
    r"(?:1\s*[.]\s*)?정정대상\s*공시서류\s*[:：]\s*(?P<value>.+)"
)
_ORIGINAL_SUBMISSION_DATE = re.compile(
    r"(?:2\s*[.]\s*)?정정대상\s*공시서류의\s*최초제출일\s*"
    r"[:：]\s*(?P<value>.+)"
)
_PARAGRAPH_REASON = re.compile(
    r"(?:3\s*[.]\s*)?정정사유\s*[:：]\s*(?P<value>.+)"
)
_HTML_MARKERS = re.compile(
    r"<html\b|LIB_LC000|XFormD8_",
    re.IGNORECASE,
)
_CORRECTION_ROOT_PATH = "//CORRECTION[1]"


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].upper()


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _element_text(element: ET.Element) -> str:
    return _clean_text("".join(element.itertext()))


def _decode_source(source: str | bytes) -> str:
    if isinstance(source, str):
        return source
    try:
        return source.decode("utf-8")
    except UnicodeDecodeError:
        return source.decode("euc-kr", errors="replace")


def _normalize_date(value: str | None) -> str | None:
    if value is None:
        return None
    for pattern in (_KOREAN_DATE, _ISO_DATE):
        match = pattern.search(value)
        if match:
            return (
                f"{match.group('year')}-"
                f"{int(match.group('month')):02d}-"
                f"{int(match.group('day')):02d}"
            )
    return None


def _first_matching_text(
    elements: Iterable[ET.Element],
    pattern: re.Pattern[str],
) -> str | None:
    for element in elements:
        text = _element_text(element)
        match = pattern.search(text)
        if match:
            return _clean_text(match.group("value"))
    return None


def _value_after_paragraph_label(
    elements: list[ET.Element],
    label: str,
) -> str | None:
    compact_label = _compact_text(label)
    for index, element in enumerate(elements):
        text = _element_text(element)
        if compact_label not in _compact_text(text):
            continue
        parts = re.split(r"[:：]", text, maxsplit=1)
        if len(parts) == 2 and _clean_text(parts[1]):
            return _clean_text(parts[1])
        for following in elements[index + 1 :]:
            following_text = _element_text(following)
            if not following_text:
                continue
            if re.match(r"\d+\s*[.]", following_text):
                return None
            return following_text
    return None


def _reason_from_tables(correction: ET.Element) -> str | None:
    for table in (element for element in correction.iter() if _tag(element) == "TABLE"):
        grid = build_logical_grid(table)
        for header in grid.cells:
            if _compact_text(header.raw_value) != "정정사유":
                continue
            for row in range(header.row_end, grid.height):
                cell = grid.rows[row][header.col_start]
                if cell is None or cell.id == header.id or not cell.raw_value:
                    continue
                if _compact_text(cell.raw_value) == "정정사유":
                    continue
                return cell.raw_value
    return None


def _value_after_table_label(
    correction: ET.Element,
    label: str,
) -> str | None:
    compact_label = _compact_text(label)
    for table in (element for element in correction.iter() if _tag(element) == "TABLE"):
        grid = build_logical_grid(table)
        for label_cell in grid.cells:
            if compact_label not in _compact_text(label_cell.raw_value):
                continue
            for row in range(label_cell.row_start, label_cell.row_end):
                seen: set[str] = set()
                for column in range(label_cell.col_end, grid.width):
                    cell = grid.rows[row][column]
                    if (
                        cell is None
                        or cell.id == label_cell.id
                        or cell.id in seen
                        or not cell.raw_value
                    ):
                        continue
                    seen.add(cell.id)
                    return cell.raw_value
    return None


def _xml_metadata(
    correction: ET.Element,
    source_ref: SourceRef,
) -> tuple[CorrectionMetadata, tuple[CorrectionIssue, ...]]:
    issues: list[CorrectionIssue] = []
    title_element = next(
        (element for element in correction.iter() if _tag(element) == "TITLE"),
        None,
    )
    title = _element_text(title_element) if title_element is not None else None
    paragraphs = [
        element for element in correction.iter() if _tag(element) == "P"
    ]
    target_document_name = _first_matching_text(paragraphs, _TARGET_DOCUMENT)
    if target_document_name is None:
        target_document_name = _value_after_paragraph_label(
            paragraphs,
            "정정대상 공시서류",
        )
    if target_document_name is None:
        target_document_name = _value_after_table_label(
            correction,
            "정정대상 공시서류",
        )
    original_date_text = _first_matching_text(
        paragraphs,
        _ORIGINAL_SUBMISSION_DATE,
    )
    if original_date_text is None:
        original_date_text = _value_after_table_label(
            correction,
            "정정대상 공시서류의 최초제출일",
        )
    reason = _first_matching_text(paragraphs, _PARAGRAPH_REASON)
    if reason is None:
        reason = _reason_from_tables(correction)

    full_text = _element_text(correction)
    correction_date = _normalize_date(full_text)
    original_submission_date = _normalize_date(original_date_text)

    for code, value, label in (
        ("MISSING_TITLE", title, "correction title"),
        ("MISSING_TARGET_DOCUMENT", target_document_name, "target document"),
        (
            "MISSING_ORIGINAL_SUBMISSION_DATE",
            original_submission_date,
            "original submission date",
        ),
        ("MISSING_REASON", reason, "correction reason"),
    ):
        if value is None:
            issues.append(
                CorrectionIssue(
                    code=code,
                    message=f"Could not extract {label} from the correction block.",
                )
            )

    return (
        CorrectionMetadata(
            title=title,
            correction_date=correction_date,
            target_document_name=target_document_name,
            original_submission_date=original_submission_date,
            reason=reason,
            target_rcept_no=None,
            source_ref=source_ref,
        ),
        tuple(issues),
    )


def _xml_blocks(correction: ET.Element) -> tuple[CorrectionBlockRef, ...]:
    blocks: list[CorrectionBlockRef] = []

    def visit(node: ET.Element, path: str) -> None:
        counts: dict[str, int] = {}
        for child in list(node):
            child_tag = _tag(child)
            counts[child_tag] = counts.get(child_tag, 0) + 1
            child_path = f"{path}/{child_tag}[{counts[child_tag]}]"
            if child_tag == "TITLE":
                block_type = CorrectionBlockType.TITLE
            elif child_tag == "P":
                if not _element_text(child):
                    continue
                block_type = CorrectionBlockType.PARAGRAPH
            elif child_tag == "TABLE":
                block_type = CorrectionBlockType.TABLE
            else:
                visit(child, child_path)
                continue

            blocks.append(
                CorrectionBlockRef(
                    sequence=len(blocks),
                    block_type=block_type,
                    source_ref=SourceRef(
                        syntax=DocumentSyntax.DART_XML,
                        element_path=child_path,
                    ),
                )
            )

    visit(correction, _CORRECTION_ROOT_PATH)
    return tuple(blocks)


def _find_correction_element(root: ET.Element) -> ET.Element | None:
    if _tag(root) == "CORRECTION":
        return root
    return next(
        (element for element in root.iter() if _tag(element) == "CORRECTION"),
        None,
    )


def _not_found(
    context: DocumentContext,
    syntax: DocumentSyntax,
) -> CorrectionExtraction:
    return CorrectionExtraction(
        source_document=context,
        syntax=syntax,
        status=CorrectionStatus.NOT_FOUND,
        correction=None,
    )


def _failed(
    context: DocumentContext,
    syntax: DocumentSyntax,
    error: Exception | str,
) -> CorrectionExtraction:
    message = str(error) if isinstance(error, str) else f"{type(error).__name__}: {error}"
    return CorrectionExtraction(
        source_document=context,
        syntax=syntax,
        status=CorrectionStatus.FAILED,
        correction=None,
        issues=(
            CorrectionIssue(
                code="CORRECTION_EXTRACTION_FAILED",
                severity="ERROR",
                message=message,
            ),
        ),
    )


def _extract_xml_element(
    root: ET.Element,
    context: DocumentContext,
) -> CorrectionExtraction:
    correction = _find_correction_element(root)
    if correction is None:
        return _not_found(context, DocumentSyntax.DART_XML)

    source_ref = SourceRef(
        syntax=DocumentSyntax.DART_XML,
        element_path=_CORRECTION_ROOT_PATH,
    )
    metadata, issues = _xml_metadata(correction, source_ref)
    return CorrectionExtraction(
        source_document=context,
        syntax=DocumentSyntax.DART_XML,
        status=CorrectionStatus.FOUND,
        correction=metadata,
        correction_blocks=_xml_blocks(correction),
        excluded_source_refs=(source_ref,),
        issues=issues,
    )


def _extract_xml_source(
    source: str,
    context: DocumentContext,
) -> CorrectionExtraction:
    match = _CORRECTION_FRAGMENT.search(source)
    if match is None:
        return _not_found(context, DocumentSyntax.DART_XML)

    fragment = match.group(0)
    recovered = False
    try:
        correction = ET.fromstring(fragment)
    except ET.ParseError:
        try:
            correction = ET.fromstring(repair_xml_text(fragment))
            recovered = True
        except ET.ParseError as error:
            return _failed(context, DocumentSyntax.DART_XML, error)

    source_ref = SourceRef(
        syntax=DocumentSyntax.DART_XML,
        element_path=_CORRECTION_ROOT_PATH,
    )
    metadata, metadata_issues = _xml_metadata(correction, source_ref)
    issues = list(metadata_issues)
    if recovered:
        issues.insert(
            0,
            CorrectionIssue(
                code="XML_RECOVERED",
                message="The correction block required conservative XML text repair.",
            ),
        )

    return CorrectionExtraction(
        source_document=context,
        syntax=DocumentSyntax.DART_XML,
        status=(CorrectionStatus.RECOVERED if recovered else CorrectionStatus.FOUND),
        correction=metadata,
        correction_blocks=_xml_blocks(correction),
        excluded_source_refs=(source_ref,),
        issues=tuple(issues),
    )


@dataclass(slots=True)
class _HTMLTable:
    html_id: str | None
    inside_container: bool
    rows: list[list[str]]


class _CorrectionHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.found_container = False
        self.container_text: list[str] = []
        self.tables: list[_HTMLTable] = []
        self._container_depth = 0
        self._current_table: _HTMLTable | None = None
        self._current_row: list[str] | None = None
        self._current_cell: int | None = None

    @property
    def inside_container(self) -> bool:
        return self._container_depth > 0

    def handle_starttag(
        self,
        tag: str,
        attributes: list[tuple[str, str | None]],
    ) -> None:
        name = tag.lower()
        attrs = {key.lower(): value or "" for key, value in attributes}
        if name == "div":
            if self.inside_container:
                self._container_depth += 1
            elif attrs.get("id") == "LIB_LC000":
                self.found_container = True
                self._container_depth = 1

        if name == "table":
            self._current_table = _HTMLTable(
                html_id=attrs.get("id") or None,
                inside_container=self.inside_container,
                rows=[],
            )
            self.tables.append(self._current_table)
            self._current_row = None
            self._current_cell = None
        elif name == "tr" and self._current_table is not None:
            self._current_row = []
            self._current_table.rows.append(self._current_row)
            self._current_cell = None
        elif name in {"td", "th"} and self._current_table is not None:
            if self._current_row is None:
                self._current_row = []
                self._current_table.rows.append(self._current_row)
            self._current_row.append("")
            self._current_cell = len(self._current_row) - 1

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name == "table":
            self._current_table = None
            self._current_row = None
            self._current_cell = None
        elif name == "tr":
            self._current_row = None
            self._current_cell = None
        elif name in {"td", "th"}:
            self._current_cell = None

        if name == "div" and self.inside_container:
            self._container_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.inside_container:
            self.container_text.append(data)
        if (
            self._current_row is not None
            and self._current_cell is not None
        ):
            self._current_row[self._current_cell] += data


def _html_label_value(
    tables: Iterable[_HTMLTable],
    label: str,
) -> str | None:
    compact_label = _compact_text(label)
    for table in tables:
        for row in table.rows:
            cells = [_clean_text(cell) for cell in row]
            for index, cell in enumerate(cells):
                if compact_label not in _compact_text(cell):
                    continue
                for value in cells[index + 1 :]:
                    if value:
                        return value
    return None


def _extract_html_source(
    source: str,
    context: DocumentContext,
) -> CorrectionExtraction:
    parser = _CorrectionHTMLParser()
    try:
        parser.feed(source)
        parser.close()
    except Exception as error:
        return _failed(context, DocumentSyntax.HTML, error)

    d8_tables = [
        table
        for table in parser.tables
        if (table.html_id or "").startswith("XFormD8_")
    ]
    correction_tables = (
        [table for table in parser.tables if table.inside_container]
        if parser.found_container
        else d8_tables
    )
    if not parser.found_container and not d8_tables:
        return _not_found(context, DocumentSyntax.HTML)

    issues: list[CorrectionIssue] = []
    title_text = _clean_text(" ".join(parser.container_text))
    title = "정정신고(보고)" if "정정신고(보고)" in _compact_text(title_text) else None
    correction_date = _normalize_date(
        _html_label_value(correction_tables, "정정일자")
    )
    target_document_name = _html_label_value(
        correction_tables,
        "정정관련 공시서류",
    )
    original_submission_date = _normalize_date(
        _html_label_value(correction_tables, "정정관련 공시서류제출일")
    )
    reason = _html_label_value(correction_tables, "정정사유")

    for code, value, label in (
        ("MISSING_TITLE", title, "correction title"),
        ("MISSING_CORRECTION_DATE", correction_date, "correction date"),
        ("MISSING_TARGET_DOCUMENT", target_document_name, "target document"),
        (
            "MISSING_ORIGINAL_SUBMISSION_DATE",
            original_submission_date,
            "original submission date",
        ),
        ("MISSING_REASON", reason, "correction reason"),
    ):
        if value is None:
            issues.append(
                CorrectionIssue(
                    code=code,
                    message=f"Could not extract {label} from the correction block.",
                )
            )

    root_ref = SourceRef(
        syntax=DocumentSyntax.HTML,
        html_id=(
            "LIB_LC000"
            if parser.found_container
            else d8_tables[0].html_id
        ),
    )
    blocks: list[CorrectionBlockRef] = []
    for table in correction_tables:
        if table.html_id is None:
            issues.append(
                CorrectionIssue(
                    code="MISSING_HTML_ID",
                    message="A correction table has no HTML id and cannot be addressed directly.",
                )
            )
            continue
        blocks.append(
            CorrectionBlockRef(
                sequence=len(blocks),
                block_type=CorrectionBlockType.TABLE,
                source_ref=SourceRef(
                    syntax=DocumentSyntax.HTML,
                    html_id=table.html_id,
                ),
            )
        )

    excluded_refs = (
        (root_ref,)
        if parser.found_container
        else tuple(block.source_ref for block in blocks)
    )
    return CorrectionExtraction(
        source_document=context,
        syntax=DocumentSyntax.HTML,
        status=CorrectionStatus.FOUND,
        correction=CorrectionMetadata(
            title=title,
            correction_date=correction_date,
            target_document_name=target_document_name,
            original_submission_date=original_submission_date,
            reason=reason,
            target_rcept_no=None,
            source_ref=root_ref,
        ),
        correction_blocks=tuple(blocks),
        excluded_source_refs=excluded_refs,
        issues=tuple(issues),
    )


def extract_correction(
    document: ET.Element | str | bytes,
    *,
    context: DocumentContext,
    syntax: DocumentSyntax = DocumentSyntax.AUTO,
) -> CorrectionExtraction:
    """Extract correction metadata and exclusion references from one document."""
    if isinstance(document, ET.Element):
        if syntax == DocumentSyntax.HTML:
            return _failed(
                context,
                DocumentSyntax.HTML,
                "HTML extraction requires the original str or bytes source.",
            )
        try:
            return _extract_xml_element(document, context)
        except Exception as error:
            return _failed(context, DocumentSyntax.DART_XML, error)

    source = _decode_source(document)
    selected_syntax = syntax
    if syntax == DocumentSyntax.AUTO:
        selected_syntax = (
            DocumentSyntax.HTML
            if _HTML_MARKERS.search(source)
            else DocumentSyntax.DART_XML
        )

    if selected_syntax == DocumentSyntax.HTML:
        return _extract_html_source(source, context)
    if selected_syntax == DocumentSyntax.DART_XML:
        try:
            return _extract_xml_source(source, context)
        except Exception as error:
            return _failed(context, DocumentSyntax.DART_XML, error)
    return _failed(context, selected_syntax, f"Unsupported syntax: {selected_syntax}")


__all__ = ["extract_correction"]
