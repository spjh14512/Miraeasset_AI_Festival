"""Detect and canonicalize paragraph blocks while preserving inline provenance."""

from __future__ import annotations

import re
from typing import Iterable, Mapping
import xml.etree.ElementTree as ET

from converters.common.document_loader import (
    DocumentLoadStatus,
    LoadedDocument,
    load_document,
)
from converters.common.source_models import (
    DocumentContext,
    DocumentSyntax,
    EvidenceContext,
    SourceRef,
)
from converters.paragraph_parser.paragraph_models import (
    CanonicalParagraph,
    CanonicalParagraphCollection,
    FragmentKind,
    LeadingMarker,
    MarkerKind,
    ParagraphFragment,
    ParagraphIssue,
    ParagraphParseStatus,
    ParagraphSourceKind,
)


_SOURCE_ATTRIBUTES = {"USERMARK", "CLASS", "STYLE", "REFNO"}
_SKIP_SUBTREES = {
    "CORRECTION",
    "SCRIPT",
    "STYLE",
    "TABLE",
    "TABLE-GROUP",
    "TITLE",
}
_NOTE_MARKER = re.compile(r"^(※)")
_FOOTNOTE_MARKER = re.compile(r"^(주\s*\d+\s*[)])")
_BULLET_MARKER = re.compile(r"^([-ㆍ·])")
_ENUMERATION_MARKER = re.compile(
    r"^((?:[①-⑳])|(?:\(\s*\d+\s*\))|(?:\d+\s*[.])|(?:[가-하]\s*[.]))"
)


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].upper()


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _source_attributes(element: ET.Element) -> dict[str, str]:
    return {
        key.upper(): value
        for key, value in element.attrib.items()
        if key.upper() in _SOURCE_ATTRIBUTES
    }


def _leading_marker(text: str) -> LeadingMarker | None:
    for pattern, kind in (
        (_NOTE_MARKER, MarkerKind.NOTE),
        (_FOOTNOTE_MARKER, MarkerKind.FOOTNOTE),
        (_BULLET_MARKER, MarkerKind.BULLET),
        (_ENUMERATION_MARKER, MarkerKind.ENUMERATION),
    ):
        match = pattern.match(text)
        if match:
            return LeadingMarker(text=_clean_text(match.group(1)), kind=kind)
    return None


def _child_paths(element: ET.Element, path: str) -> list[tuple[ET.Element, str]]:
    counts: dict[str, int] = {}
    result: list[tuple[ET.Element, str]] = []
    for child in list(element):
        child_tag = _tag(child)
        counts[child_tag] = counts.get(child_tag, 0) + 1
        result.append((child, f"{path}/{child_tag}[{counts[child_tag]}]"))
    return result


def _append_fragment(
    fragments: list[ParagraphFragment],
    *,
    kind: FragmentKind,
    raw_text: str | None,
    source_ref: SourceRef,
    source_attributes: Mapping[str, str] | None = None,
) -> None:
    if raw_text is None or not _clean_text(raw_text):
        return
    fragments.append(
        ParagraphFragment(
            id=f"f{len(fragments)}",
            kind=kind,
            raw_text=raw_text,
            text=_clean_text(raw_text),
            source_attributes=source_attributes or {},
            source_ref=source_ref,
        )
    )


def _raw_text_with_breaks(element: ET.Element) -> str:
    parts = [element.text or ""]
    for child in list(element):
        if _tag(child) == "BR":
            parts.append("\n")
        else:
            parts.append(_raw_text_with_breaks(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _p_fragments(
    element: ET.Element,
    path: str,
    syntax: DocumentSyntax,
) -> tuple[ParagraphFragment, ...]:
    fragments: list[ParagraphFragment] = []
    paragraph_ref = SourceRef(syntax=syntax, element_path=path)
    _append_fragment(
        fragments,
        kind=FragmentKind.TEXT,
        raw_text=element.text,
        source_ref=paragraph_ref,
    )
    for child, child_path in _child_paths(element, path):
        child_ref = SourceRef(syntax=syntax, element_path=child_path)
        if _tag(child) == "SPAN":
            _append_fragment(
                fragments,
                kind=FragmentKind.SPAN,
                raw_text="".join(child.itertext()),
                source_ref=child_ref,
                source_attributes=_source_attributes(child),
            )
        elif _tag(child) == "BR":
            fragments.append(
                ParagraphFragment(
                    id=f"f{len(fragments)}",
                    kind=FragmentKind.TEXT,
                    raw_text="\n",
                    text="\n",
                    source_attributes={},
                    source_ref=child_ref,
                )
            )
        else:
            _append_fragment(
                fragments,
                kind=FragmentKind.TEXT,
                raw_text="".join(child.itertext()),
                source_ref=child_ref,
                source_attributes=(
                    _source_attributes(child) if _tag(child) == "A" else None
                ),
            )
        _append_fragment(
            fragments,
            kind=FragmentKind.TEXT,
            raw_text=child.tail,
            source_ref=paragraph_ref,
        )
    return tuple(fragments)


def parse_paragraph_element(
    element: ET.Element,
    *,
    document_context: DocumentContext,
    source_ref: SourceRef,
    paragraph_index: int = 0,
    content_context: EvidenceContext | None = None,
    parse_status: ParagraphParseStatus = ParagraphParseStatus.SUCCESS,
) -> CanonicalParagraph:
    """Canonicalize one P element without mutating it."""
    if _tag(element) != "P":
        raise ValueError("parse_paragraph_element requires a P element")
    raw_text = _raw_text_with_breaks(element)
    text = _clean_text(raw_text)
    fragments = _p_fragments(
        element,
        source_ref.element_path or "//P[1]",
        source_ref.syntax,
    )
    return CanonicalParagraph(
        source_document=document_context,
        source_kind=ParagraphSourceKind.P,
        source_refs=(source_ref,),
        paragraph_index=paragraph_index,
        context=content_context or EvidenceContext(),
        raw_text=raw_text,
        text=text,
        leading_marker=_leading_marker(text),
        source_attributes=_source_attributes(element),
        fragments=fragments,
        parse_status=parse_status,
    )


def _span_run_paragraph(
    run: list[tuple[ET.Element, str]],
    *,
    document_context: DocumentContext,
    paragraph_index: int,
    content_context: EvidenceContext,
    syntax: DocumentSyntax,
    parse_status: ParagraphParseStatus,
) -> CanonicalParagraph:
    fragments: list[ParagraphFragment] = []
    raw_parts: list[str] = []
    refs: list[SourceRef] = []
    for span, path in run:
        source_ref = SourceRef(syntax=syntax, element_path=path)
        refs.append(source_ref)
        span_text = "".join(span.itertext())
        raw_parts.append(span_text)
        _append_fragment(
            fragments,
            kind=FragmentKind.SPAN,
            raw_text=span_text,
            source_ref=source_ref,
            source_attributes=_source_attributes(span),
        )
        if span.tail:
            raw_parts.append(span.tail)
            _append_fragment(
                fragments,
                kind=FragmentKind.TEXT,
                raw_text=span.tail,
                source_ref=source_ref,
            )
    raw_text = "".join(raw_parts)
    text = _clean_text(raw_text)
    return CanonicalParagraph(
        source_document=document_context,
        source_kind=ParagraphSourceKind.SPAN_RUN,
        source_refs=tuple(refs),
        paragraph_index=paragraph_index,
        context=content_context,
        raw_text=raw_text,
        text=text,
        leading_marker=_leading_marker(text),
        source_attributes={},
        fragments=tuple(fragments),
        parse_status=parse_status,
    )


def parse_span_run_elements(
    elements: Iterable[tuple[ET.Element, SourceRef]],
    *,
    document_context: DocumentContext,
    paragraph_index: int = 0,
    content_context: EvidenceContext | None = None,
    parse_status: ParagraphParseStatus = ParagraphParseStatus.SUCCESS,
) -> CanonicalParagraph:
    """Canonicalize one chunker's ordered SPAN_RUN without rescanning the DOM."""
    run = list(elements)
    if not run:
        raise ValueError("parse_span_run_elements requires at least one SPAN")
    if any(_tag(element) != "SPAN" for element, _ in run):
        raise ValueError("parse_span_run_elements accepts only SPAN elements")
    syntaxes = {source_ref.syntax for _, source_ref in run}
    if len(syntaxes) != 1:
        raise ValueError("All SPAN source references must use the same syntax")
    if any(source_ref.element_path is None for _, source_ref in run):
        raise ValueError("SPAN source references require element_path")
    return _span_run_paragraph(
        [(element, source_ref.element_path or "") for element, source_ref in run],
        document_context=document_context,
        paragraph_index=paragraph_index,
        content_context=content_context or EvidenceContext(),
        syntax=next(iter(syntaxes)),
        parse_status=parse_status,
    )


def _is_layout_title_span(element: ET.Element) -> bool:
    style = element.attrib.get("STYLE", "").replace(" ", "").lower()
    return "font-weight:bold" in style


def _is_excluded(
    element: ET.Element,
    path: str,
    excluded_paths: set[str],
    excluded_html_ids: set[str],
) -> bool:
    if path in excluded_paths:
        return True
    html_id = element.attrib.get("ID") or element.attrib.get("id")
    return html_id in excluded_html_ids


def parse_paragraphs(
    document: LoadedDocument | ET.Element | str | bytes,
    *,
    document_context: DocumentContext,
    content_context: EvidenceContext | None = None,
    excluded_source_refs: Iterable[SourceRef] = (),
    syntax: DocumentSyntax = DocumentSyntax.AUTO,
) -> CanonicalParagraphCollection:
    """Detect P blocks and defensive orphan-SPAN runs in document order."""
    loaded = (
        document
        if isinstance(document, LoadedDocument)
        else load_document(document, syntax=syntax)
    )
    if loaded.root is None:
        issues = tuple(
            ParagraphIssue(
                code=issue.code,
                severity=issue.severity,
                message=issue.message,
            )
            for issue in loaded.issues
        )
        return CanonicalParagraphCollection(
            source_document=document_context,
            syntax=loaded.syntax,
            parse_status=ParagraphParseStatus.FAILED,
            paragraphs=(),
            issues=issues,
        )

    paragraph_status = (
        ParagraphParseStatus.RECOVERED
        if loaded.status == DocumentLoadStatus.RECOVERED
        else ParagraphParseStatus.SUCCESS
    )
    context = content_context or EvidenceContext()
    excluded_paths = {
        ref.element_path for ref in excluded_source_refs if ref.element_path
    }
    excluded_html_ids = {
        ref.html_id for ref in excluded_source_refs if ref.html_id
    }
    paragraphs: list[CanonicalParagraph] = []

    root_path = f"/{_tag(loaded.root)}[1]"

    def visit(parent: ET.Element, parent_path: str) -> None:
        indexed_children = _child_paths(parent, parent_path)
        index = 0
        while index < len(indexed_children):
            child, child_path = indexed_children[index]
            child_tag = _tag(child)
            if _is_excluded(
                child,
                child_path,
                excluded_paths,
                excluded_html_ids,
            ):
                index += 1
                continue
            if child_tag in _SKIP_SUBTREES:
                index += 1
                continue
            if child_tag == "P":
                if _clean_text("".join(child.itertext())):
                    paragraphs.append(
                        parse_paragraph_element(
                            child,
                            document_context=document_context,
                            source_ref=SourceRef(
                                syntax=loaded.syntax,
                                element_path=child_path,
                            ),
                            paragraph_index=len(paragraphs),
                            content_context=context,
                            parse_status=paragraph_status,
                        )
                    )
                index += 1
                continue
            if child_tag == "SPAN":
                if _is_layout_title_span(child):
                    index += 1
                    continue
                run: list[tuple[ET.Element, str]] = []
                while index < len(indexed_children):
                    span, span_path = indexed_children[index]
                    if (
                        _tag(span) != "SPAN"
                        or _is_layout_title_span(span)
                        or not _clean_text("".join(span.itertext()))
                    ):
                        break
                    run.append((span, span_path))
                    index += 1
                if run:
                    paragraphs.append(
                        _span_run_paragraph(
                            run,
                            document_context=document_context,
                            paragraph_index=len(paragraphs),
                            content_context=context,
                            syntax=loaded.syntax,
                            parse_status=paragraph_status,
                        )
                    )
                    continue
            visit(child, child_path)
            index += 1

    visit(loaded.root, root_path)
    collection_issues = tuple(
        ParagraphIssue(
            code=issue.code,
            severity=issue.severity,
            message=issue.message,
        )
        for issue in loaded.issues
    )
    return CanonicalParagraphCollection(
        source_document=document_context,
        syntax=loaded.syntax,
        parse_status=paragraph_status,
        paragraphs=tuple(paragraphs),
        issues=collection_issues,
    )


__all__ = [
    "parse_paragraph_element",
    "parse_paragraphs",
    "parse_span_run_elements",
]
