"""Load DART XML or exchange HTML into a shared, recoverable element tree."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from html.parser import HTMLParser
import re
from typing import Any
import xml.etree.ElementTree as ET

from converters.common.source_models import DocumentSyntax


_BARE_AMPERSAND = re.compile(
    r"&(?!#\d+;|#x[0-9A-Fa-f]+;|[A-Za-z][A-Za-z0-9]+;)"
)
_NON_XML_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")
_NON_TAG_LT = re.compile(r"<(?=[^A-Za-z/!?])")
_PSEUDO_TAG = re.compile(r"<([A-Za-z][^<>=]*\s[^<>=]*)>")
_HTML_MARKERS = re.compile(
    r"<html\b|class\s*=\s*[\"']xforms|LIB_LC000|XFormD\d+_",
    re.IGNORECASE,
)
_VOID_HTML_TAGS = {
    "AREA",
    "BASE",
    "BR",
    "COL",
    "EMBED",
    "HR",
    "IMG",
    "INPUT",
    "LINK",
    "META",
    "PARAM",
    "SOURCE",
    "TRACK",
    "WBR",
}


class DocumentLoadStatus(StrEnum):
    SUCCESS = "SUCCESS"
    RECOVERED = "RECOVERED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class DocumentLoadIssue:
    code: str
    message: str
    severity: str = "WARNING"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    root: ET.Element | None
    syntax: DocumentSyntax
    status: DocumentLoadStatus
    issues: tuple[DocumentLoadIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "syntax": self.syntax.value,
            "status": self.status.value,
            "root_tag": self.root.tag if self.root is not None else None,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def repair_xml_text(source: str) -> str:
    """Apply conservative repairs without changing valid XML entities or tags."""
    repaired = _NON_XML_CONTROL.sub("", source)
    repaired = _BARE_AMPERSAND.sub("&amp;", repaired)
    repaired = _NON_TAG_LT.sub("&lt;", repaired)
    return _PSEUDO_TAG.sub(lambda match: "&lt;" + match.group(1) + ">", repaired)


def _decode_source(source: str | bytes) -> str:
    if isinstance(source, str):
        return source
    try:
        return source.decode("utf-8")
    except UnicodeDecodeError:
        return source.decode("euc-kr", errors="replace")


class _LooseHTMLTreeBuilder(HTMLParser):
    """Create a usable DOM from exchange HTML with omitted closing tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = ET.Element("HTML_DOCUMENT")
        self._stack: list[ET.Element] = [self.root]

    @staticmethod
    def _tag(element: ET.Element) -> str:
        return element.tag.rsplit("}", 1)[-1].upper()

    def _close_open(self, names: set[str]) -> None:
        for index in range(len(self._stack) - 1, 0, -1):
            if self._tag(self._stack[index]) in names:
                self._stack = self._stack[:index]
                return

    def handle_starttag(
        self,
        tag: str,
        attributes: list[tuple[str, str | None]],
    ) -> None:
        name = tag.upper()
        if name in {"TD", "TH"}:
            self._close_open({"TD", "TH"})
        elif name == "TR":
            self._close_open({"TR"})
        elif name == "P":
            self._close_open({"P"})
        elif name == "LI":
            self._close_open({"LI"})

        attrs = {key.upper(): value or "" for key, value in attributes}
        element = ET.SubElement(self._stack[-1], name, attrs)
        if name not in _VOID_HTML_TAGS:
            self._stack.append(element)

    def handle_startendtag(
        self,
        tag: str,
        attributes: list[tuple[str, str | None]],
    ) -> None:
        attrs = {key.upper(): value or "" for key, value in attributes}
        ET.SubElement(self._stack[-1], tag.upper(), attrs)

    def handle_endtag(self, tag: str) -> None:
        name = tag.upper()
        for index in range(len(self._stack) - 1, 0, -1):
            if self._tag(self._stack[index]) == name:
                self._stack = self._stack[:index]
                return

    def handle_data(self, data: str) -> None:
        parent = self._stack[-1]
        children = list(parent)
        if children:
            children[-1].tail = (children[-1].tail or "") + data
        else:
            parent.text = (parent.text or "") + data


def load_document(
    source: ET.Element | str | bytes,
    *,
    syntax: DocumentSyntax = DocumentSyntax.AUTO,
) -> LoadedDocument:
    """Load a full source document while recording any recovery."""
    if isinstance(source, ET.Element):
        return LoadedDocument(
            root=source,
            syntax=(
                DocumentSyntax.DART_XML
                if syntax == DocumentSyntax.AUTO
                else syntax
            ),
            status=DocumentLoadStatus.SUCCESS,
        )

    text = _decode_source(source)
    selected_syntax = syntax
    if syntax == DocumentSyntax.AUTO:
        selected_syntax = (
            DocumentSyntax.HTML
            if _HTML_MARKERS.search(text)
            else DocumentSyntax.DART_XML
        )

    if selected_syntax == DocumentSyntax.HTML:
        try:
            parser = _LooseHTMLTreeBuilder()
            parser.feed(text)
            parser.close()
            return LoadedDocument(
                root=parser.root,
                syntax=DocumentSyntax.HTML,
                status=DocumentLoadStatus.SUCCESS,
            )
        except Exception as error:
            return LoadedDocument(
                root=None,
                syntax=DocumentSyntax.HTML,
                status=DocumentLoadStatus.FAILED,
                issues=(
                    DocumentLoadIssue(
                        code="HTML_LOAD_FAILED",
                        severity="ERROR",
                        message=f"{type(error).__name__}: {error}",
                    ),
                ),
            )

    try:
        root = ET.fromstring(text)
        return LoadedDocument(
            root=root,
            syntax=DocumentSyntax.DART_XML,
            status=DocumentLoadStatus.SUCCESS,
        )
    except ET.ParseError as strict_error:
        try:
            root = ET.fromstring(repair_xml_text(text))
            return LoadedDocument(
                root=root,
                syntax=DocumentSyntax.DART_XML,
                status=DocumentLoadStatus.RECOVERED,
                issues=(
                    DocumentLoadIssue(
                        code="XML_RECOVERED",
                        message="The document required conservative XML text repair.",
                    ),
                ),
            )
        except ET.ParseError as recovered_error:
            try:
                parser = _LooseHTMLTreeBuilder()
                parser.feed(repair_xml_text(text))
                parser.close()
                document_root = next(
                    (
                        element
                        for element in parser.root.iter()
                        if element.tag.rsplit("}", 1)[-1].upper() == "DOCUMENT"
                    ),
                    None,
                )
                if document_root is None:
                    raise ValueError("Tolerant recovery did not find DOCUMENT.")
                return LoadedDocument(
                    root=document_root,
                    syntax=DocumentSyntax.DART_XML,
                    status=DocumentLoadStatus.RECOVERED,
                    issues=(
                        DocumentLoadIssue(
                            code="XML_TOLERANT_RECOVERY",
                            message=(
                                "The document required tolerant tree recovery after "
                                "conservative XML repair failed."
                            ),
                        ),
                    ),
                )
            except Exception as tolerant_error:
                return LoadedDocument(
                    root=None,
                    syntax=DocumentSyntax.DART_XML,
                    status=DocumentLoadStatus.FAILED,
                    issues=(
                        DocumentLoadIssue(
                            code="XML_LOAD_FAILED",
                            severity="ERROR",
                            message=(
                                f"strict={strict_error}; recovered={recovered_error}; "
                                f"tolerant={type(tolerant_error).__name__}: "
                                f"{tolerant_error}"
                            ),
                        ),
                    ),
                )


__all__ = [
    "DocumentLoadIssue",
    "DocumentLoadStatus",
    "LoadedDocument",
    "load_document",
    "repair_xml_text",
]
