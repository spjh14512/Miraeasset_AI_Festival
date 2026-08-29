"""Split DART XML and exchange HTML into hierarchical ordered sections."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

from converters.common.document_loader import (
    DocumentLoadStatus,
    LoadedDocument,
    load_document,
)
from converters.common.source_models import (
    DocumentContext,
    DocumentSyntax,
    SourceRef,
)
from converters.section_canonicalizer.section_models import (
    CanonicalSection,
    CanonicalSectionCollection,
    SectionBlockRef,
    SectionBlockType,
    SectionBoundaryKind,
    SectionIssue,
    SectionParseStatus,
)


_SECTION_TAG = re.compile(r"^SECTION-(\d+)$")
_UNIT_TEXT = re.compile(r"^\(?\s*단위\s*[:：]", re.IGNORECASE)
_NOTE_TEXT = re.compile(
    r"^(?:※|[*]|주\s*\d+\s*[):.]|[-ㆍ·]\s*상기)",
    re.IGNORECASE,
)
_NUMBERED_HEADINGS: tuple[tuple[str, int, re.Pattern[str]], ...] = (
    ("PART", 1, re.compile(r"^제\s*\d+\s*(?:부|장)\b")),
    ("ROMAN", 1, re.compile(r"^[IVXLC]+[.]\s+", re.IGNORECASE)),
    ("ARABIC", 2, re.compile(r"^\d+[.]\s+")),
    (
        "KOREAN",
        3,
        re.compile(r"^[가나다라마바사아자차카타파하][.]\s+"),
    ),
)
_SKIP_SUBTREES = {"CORRECTION", "SCRIPT", "STYLE"}
_SKIP_ELEMENTS = {"TITLE", "PGBRK", "BR", "META", "LINK"}
_IMAGE_TAGS = {"IMAGE", "IMG"}


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].upper()


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _element_text(element: ET.Element) -> str:
    return _clean_text("".join(element.itertext()))


def _child_paths(element: ET.Element, path: str) -> list[tuple[ET.Element, str]]:
    counts: dict[str, int] = {}
    result: list[tuple[ET.Element, str]] = []
    for child in list(element):
        child_tag = _tag(child)
        counts[child_tag] = counts.get(child_tag, 0) + 1
        result.append((child, f"{path}/{child_tag}[{counts[child_tag]}]"))
    return result


def _source_ref(
    element: ET.Element,
    path: str,
    syntax: DocumentSyntax,
) -> SourceRef:
    html_id = element.attrib.get("ID") or element.attrib.get("id")
    return SourceRef(
        syntax=syntax,
        element_path=path,
        html_id=(html_id if syntax == DocumentSyntax.HTML else None),
    )


@dataclass(frozen=True, slots=True)
class _Exclusions:
    paths: frozenset[str]
    html_ids: frozenset[str]

    def contains(self, element: ET.Element, path: str) -> bool:
        if path in self.paths:
            return True
        html_id = element.attrib.get("ID") or element.attrib.get("id")
        return html_id in self.html_ids


def _exclusions(source_refs: Iterable[SourceRef]) -> _Exclusions:
    return _Exclusions(
        paths=frozenset(
            source_ref.element_path
            for source_ref in source_refs
            if source_ref.element_path
        ),
        html_ids=frozenset(
            source_ref.html_id for source_ref in source_refs if source_ref.html_id
        ),
    )


def _is_section(element: ET.Element) -> bool:
    return _SECTION_TAG.fullmatch(_tag(element)) is not None


def _section_level(element: ET.Element) -> int:
    match = _SECTION_TAG.fullmatch(_tag(element))
    return int(match.group(1)) if match else 0


def _find_section_title(
    section: ET.Element,
    section_path: str,
    syntax: DocumentSyntax,
    exclusions: _Exclusions,
) -> tuple[str | None, SourceRef | None]:
    direct = _child_paths(section, section_path)
    for child, child_path in direct:
        if _tag(child) == "TITLE" and not exclusions.contains(child, child_path):
            title = _element_text(child)
            if title:
                return title, _source_ref(child, child_path, syntax)

    def visit(node: ET.Element, path: str) -> tuple[str | None, SourceRef | None]:
        for child, child_path in _child_paths(node, path):
            child_tag = _tag(child)
            if exclusions.contains(child, child_path) or child_tag in _SKIP_SUBTREES:
                continue
            if _is_section(child) or child_tag in {"TABLE", "TABLE-GROUP"}:
                continue
            if child_tag == "TITLE":
                title = _element_text(child)
                if title:
                    return title, _source_ref(child, child_path, syntax)
            found = visit(child, child_path)
            if found[0] is not None:
                return found
        return None, None

    return visit(section, section_path)


def _table_group_section_title(
    group: ET.Element,
    group_path: str,
    syntax: DocumentSyntax,
    exclusions: _Exclusions,
) -> tuple[str | None, SourceRef | None]:
    """Return a TOC-backed XBRL note title that is a real viewer boundary."""
    group_class = group.attrib.get("ACLASS", "")
    if re.search(r"_DX801000$", group_class.upper()) is None:
        return None, None
    for child, child_path in _child_paths(group, group_path):
        if _tag(child) != "TITLE" or exclusions.contains(child, child_path):
            continue
        if child.attrib.get("ATOC", "").upper() != "Y":
            return None, None
        title = _element_text(child)
        return (
            (title, _source_ref(child, child_path, syntax))
            if title
            else (None, None)
        )
def _find_atoc_table_group_title(
    table_group: ET.Element,
    table_group_path: str,
    syntax: DocumentSyntax,
    exclusions: _Exclusions,
) -> tuple[str | None, SourceRef | None]:
    """Return a direct TABLE-GROUP title explicitly exposed in the DART TOC."""
    for child, child_path in _child_paths(table_group, table_group_path):
        if (
            _tag(child) != "TITLE"
            or child.attrib.get("ATOC", "").strip().upper() != "Y"
            or exclusions.contains(child, child_path)
        ):
            continue
        title = _element_text(child)
        if title:
            return title, _source_ref(child, child_path, syntax)
    return None, None


def _scan_blocks(
    root: ET.Element,
    root_path: str,
    syntax: DocumentSyntax,
    exclusions: _Exclusions,
    *,
    heading_paths: frozenset[str] = frozenset(),
    stop_at_sections: bool = True,
) -> tuple[SectionBlockRef, ...]:
    blocks: list[SectionBlockRef] = []

    def append(block_type: SectionBlockType, refs: Sequence[SourceRef]) -> None:
        blocks.append(
            SectionBlockRef(
                block_type=block_type,
                order=len(blocks),
                source_refs=tuple(refs),
            )
        )

    def visit(parent: ET.Element, parent_path: str) -> None:
        children = _child_paths(parent, parent_path)
        index = 0
        while index < len(children):
            child, child_path = children[index]
            child_tag = _tag(child)
            if exclusions.contains(child, child_path):
                index += 1
                continue
            if child_path in heading_paths:
                index += 1
                continue
            if child_tag in _SKIP_SUBTREES:
                index += 1
                continue
            if stop_at_sections and _is_section(child):
                index += 1
                continue
            if child_tag in _SKIP_ELEMENTS:
                index += 1
                continue
            if child_tag == "TABLE-GROUP":
                append(
                    SectionBlockType.TABLE_GROUP,
                    (_source_ref(child, child_path, syntax),),
                )
                index += 1
                continue
            if child_tag == "TABLE":
                append(
                    SectionBlockType.TABLE,
                    (_source_ref(child, child_path, syntax),),
                )
                index += 1
                continue
            if child_tag == "P":
                if _element_text(child):
                    append(
                        SectionBlockType.P,
                        (_source_ref(child, child_path, syntax),),
                    )
                index += 1
                continue
            if child_tag in _IMAGE_TAGS:
                append(
                    SectionBlockType.IMAGE,
                    (_source_ref(child, child_path, syntax),),
                )
                index += 1
                continue
            if child_tag == "SPAN":
                refs: list[SourceRef] = []
                while index < len(children):
                    span, span_path = children[index]
                    if (
                        _tag(span) != "SPAN"
                        or exclusions.contains(span, span_path)
                        or span_path in heading_paths
                        or not _element_text(span)
                    ):
                        break
                    refs.append(_source_ref(span, span_path, syntax))
                    index += 1
                if refs:
                    append(SectionBlockType.SPAN_RUN, refs)
                    continue
            visit(child, child_path)
            index += 1

    visit(root, root_path)
    return tuple(blocks)


@dataclass(frozen=True, slots=True)
class _HeadingCandidate:
    path: str
    parent_path: str
    title: str
    level: int
    scheme: str
    source_ref: SourceRef
    always: bool = False


def _numbered_heading(text: str) -> tuple[str, int] | None:
    for scheme, level, pattern in _NUMBERED_HEADINGS:
        if pattern.match(text):
            return scheme, level
    return None


def _styled_heading(element: ET.Element) -> bool:
    style = element.attrib.get("STYLE", "").replace(" ", "").lower()
    class_name = element.attrib.get("CLASS", "").lower()
    return (
        "font-weight:bold" in style
        or "heading" in class_name
        or "section-title" in class_name
    )


def _single_cell_table_text(table: ET.Element) -> str | None:
    rows = [node for node in table.iter() if _tag(node) == "TR"]
    cells = [
        node
        for node in table.iter()
        if _tag(node) in {"TD", "TH", "TE", "TU"}
    ]
    if len(rows) != 1 or len(cells) != 1:
        return None
    text = _element_text(cells[0])
    return text or None


def _implicit_headings(
    root: ET.Element,
    root_path: str,
    syntax: DocumentSyntax,
    exclusions: _Exclusions,
) -> Mapping[str, _HeadingCandidate]:
    candidates: list[_HeadingCandidate] = []

    def visit(parent: ET.Element, parent_path: str) -> None:
        for child, child_path in _child_paths(parent, parent_path):
            child_tag = _tag(child)
            if exclusions.contains(child, child_path) or child_tag in _SKIP_SUBTREES:
                continue
            if child_tag == "TABLE-GROUP":
                continue
            if child_tag == "TABLE":
                text = _single_cell_table_text(child)
                numbered = _numbered_heading(text or "")
                if text and numbered and len(text) <= 200:
                    candidates.append(
                        _HeadingCandidate(
                            path=child_path,
                            parent_path=parent_path,
                            title=text,
                            level=numbered[1],
                            scheme=numbered[0],
                            source_ref=_source_ref(child, child_path, syntax),
                        )
                    )
                continue
            if re.fullmatch(r"H[1-6]", child_tag):
                title = _element_text(child)
                if title:
                    candidates.append(
                        _HeadingCandidate(
                            path=child_path,
                            parent_path=parent_path,
                            title=title,
                            level=int(child_tag[1]),
                            scheme="HTML_HEADING",
                            source_ref=_source_ref(child, child_path, syntax),
                            always=True,
                        )
                    )
                continue
            if child_tag == "P":
                title = _element_text(child)
                numbered = _numbered_heading(title)
                if (
                    title
                    and len(title) <= 200
                    and not _UNIT_TEXT.match(title)
                    and not _NOTE_TEXT.match(title)
                    and (numbered is not None or _styled_heading(child))
                ):
                    candidates.append(
                        _HeadingCandidate(
                            path=child_path,
                            parent_path=parent_path,
                            title=title,
                            level=(numbered[1] if numbered else 1),
                            scheme=(numbered[0] if numbered else "STYLED"),
                            source_ref=_source_ref(child, child_path, syntax),
                            always=_styled_heading(child),
                        )
                    )
                continue
            visit(child, child_path)

    visit(root, root_path)
    repeated = Counter(
        (candidate.parent_path, candidate.scheme)
        for candidate in candidates
        if not candidate.always
    )
    accepted = [
        candidate
        for candidate in candidates
        if candidate.always
        or repeated[(candidate.parent_path, candidate.scheme)] >= 2
    ]
    numbered_levels = [
        candidate.level
        for candidate in accepted
        if candidate.scheme not in {"HTML_HEADING", "STYLED"}
    ]
    shift = min(numbered_levels) - 1 if numbered_levels else 0
    return {
        candidate.path: _HeadingCandidate(
            path=candidate.path,
            parent_path=candidate.parent_path,
            title=candidate.title,
            level=(
                candidate.level - shift
                if candidate.scheme not in {"HTML_HEADING", "STYLED"}
                else candidate.level
            ),
            scheme=candidate.scheme,
            source_ref=candidate.source_ref,
            always=candidate.always,
        )
        for candidate in accepted
    }


def _document_title(
    root: ET.Element,
    root_path: str,
    syntax: DocumentSyntax,
) -> tuple[str | None, SourceRef | None]:
    located: list[tuple[ET.Element, str]] = [(root, root_path)]
    for element, path in located:
        located.extend(_child_paths(element, path))
    for element, path in located:
        if _tag(element) != "SPAN":
            continue
        style = element.attrib.get("STYLE", "").replace(" ", "").lower()
        class_name = element.attrib.get("CLASS", "").lower()
        text = _element_text(element)
        if text and "font-weight:bold" in style and "noprint" not in class_name:
            return text, _source_ref(element, path, syntax)
    for element, path in located:
        if _tag(element) == "TITLE":
            text = _element_text(element)
            if text:
                return text, _source_ref(element, path, syntax)
    return None, None


@dataclass(slots=True)
class _SectionBuilder:
    id: str
    parent_section_id: str | None
    order: int
    level: int
    boundary_kind: SectionBoundaryKind
    title: str | None
    section_path: tuple[str, ...]
    source_ref: SourceRef | None
    title_source_ref: SourceRef | None
    blocks: list[SectionBlockRef] = field(default_factory=list)
    issues: list[SectionIssue] = field(default_factory=list)

    def freeze(self) -> CanonicalSection:
        return CanonicalSection(
            id=self.id,
            parent_section_id=self.parent_section_id,
            order=self.order,
            level=self.level,
            boundary_kind=self.boundary_kind,
            title=self.title,
            section_path=self.section_path,
            source_ref=self.source_ref,
            title_source_ref=self.title_source_ref,
            blocks=tuple(
                SectionBlockRef(
                    block_type=block.block_type,
                    order=index,
                    source_refs=block.source_refs,
                )
                for index, block in enumerate(self.blocks)
            ),
            issues=tuple(self.issues),
        )


def _explicit_sections(
    root: ET.Element,
    root_path: str,
    syntax: DocumentSyntax,
    exclusions: _Exclusions,
) -> tuple[CanonicalSection, ...]:
    builders: list[_SectionBuilder] = []
    root_blocks = _scan_blocks(root, root_path, syntax, exclusions)
    root_builder: _SectionBuilder | None = None
    if root_blocks:
        root_builder = _SectionBuilder(
            id="s0",
            parent_section_id=None,
            order=0,
            level=0,
            boundary_kind=SectionBoundaryKind.SYNTHETIC,
            title=None,
            section_path=(),
            source_ref=None,
            title_source_ref=None,
            blocks=list(root_blocks),
        )
        builders.append(root_builder)

    def visit(
        node: ET.Element,
        path: str,
        parent_builder: _SectionBuilder | None,
        parent_titles: tuple[str, ...],
    ) -> None:
        for child, child_path in _child_paths(node, path):
            child_tag = _tag(child)
            if exclusions.contains(child, child_path) or child_tag in _SKIP_SUBTREES:
                continue
            if child_tag == "TABLE-GROUP":
                title, title_ref = _table_group_section_title(
                    child,
                    child_path,
                    syntax,
                    exclusions,
                )
                owner = parent_builder or root_builder
                if title is not None and title_ref is not None and owner is not None:
                    group_ref = _source_ref(child, child_path, syntax)
                    owner.blocks = [
                        block
                        for block in owner.blocks
                        if group_ref not in block.source_refs
                    ]
                    current_titles = parent_titles + (title,)
                    builders.append(
                        _SectionBuilder(
                            id=f"s{len(builders)}",
                            parent_section_id=owner.id,
                            order=len(builders),
                            level=owner.level + 1,
                            boundary_kind=SectionBoundaryKind.IMPLICIT,
                            title=title,
                            section_path=current_titles,
                            source_ref=group_ref,
                            title_source_ref=title_ref,
                            blocks=[
                                SectionBlockRef(
                                    block_type=SectionBlockType.TABLE_GROUP,
                                    order=0,
                                    source_refs=(group_ref,),
                                )
                            ],
                        )
                    )
                    continue
            if _is_section(child):
                title, title_ref = _find_section_title(
                    child,
                    child_path,
                    syntax,
                    exclusions,
                )
                current_titles = parent_titles + ((title,) if title else ())
                builder = _SectionBuilder(
                    id=f"s{len(builders)}",
                    parent_section_id=(
                        parent_builder.id
                        if parent_builder is not None
                        else (root_builder.id if root_builder is not None else None)
                    ),
                    order=len(builders),
                    level=_section_level(child),
                    boundary_kind=SectionBoundaryKind.EXPLICIT,
                    title=title,
                    section_path=current_titles,
                    source_ref=_source_ref(child, child_path, syntax),
                    title_source_ref=title_ref,
                    blocks=list(
                        _scan_blocks(child, child_path, syntax, exclusions)
                    ),
                )
                if title is None:
                    builder.issues.append(
                        SectionIssue(
                            code="MISSING_SECTION_TITLE",
                            message=f"No title was found for {child_path}.",
                        )
                    )
                builders.append(builder)
                visit(child, child_path, builder, current_titles)
            elif child_tag == "TABLE-GROUP" and parent_builder is not None:
                title, title_ref = _find_atoc_table_group_title(
                    child,
                    child_path,
                    syntax,
                    exclusions,
                )
                if title is None:
                    visit(child, child_path, parent_builder, parent_titles)
                    continue
                group_ref = _source_ref(child, child_path, syntax)
                parent_builder.blocks = [
                    block
                    for block in parent_builder.blocks
                    if group_ref not in block.source_refs
                ]
                builder = _SectionBuilder(
                    id=f"s{len(builders)}",
                    parent_section_id=parent_builder.id,
                    order=len(builders),
                    level=parent_builder.level + 1,
                    boundary_kind=SectionBoundaryKind.IMPLICIT,
                    title=title,
                    section_path=parent_titles + (title,),
                    source_ref=group_ref,
                    title_source_ref=title_ref,
                    blocks=[
                        SectionBlockRef(
                            block_type=SectionBlockType.TABLE_GROUP,
                            order=0,
                            source_refs=(group_ref,),
                        )
                    ],
                )
                builders.append(builder)
            else:
                visit(child, child_path, parent_builder, parent_titles)

    visit(root, root_path, None, ())
    return tuple(builder.freeze() for builder in builders)


def _implicit_sections(
    root: ET.Element,
    root_path: str,
    syntax: DocumentSyntax,
    exclusions: _Exclusions,
) -> tuple[CanonicalSection, ...]:
    headings = _implicit_headings(root, root_path, syntax, exclusions)
    document_title, document_title_ref = _document_title(root, root_path, syntax)
    all_blocks = _scan_blocks(
        root,
        root_path,
        syntax,
        exclusions,
        heading_paths=frozenset(headings),
        stop_at_sections=False,
    )
    if not headings:
        if document_title_ref is not None:
            all_blocks = tuple(
                SectionBlockRef(
                    block_type=block.block_type,
                    order=index,
                    source_refs=block.source_refs,
                )
                for index, block in enumerate(
                    block
                    for block in all_blocks
                    if document_title_ref not in block.source_refs
                )
            )
        return (
            CanonicalSection(
                id="s0",
                parent_section_id=None,
                order=0,
                level=0,
                boundary_kind=SectionBoundaryKind.SYNTHETIC,
                title=document_title,
                section_path=((document_title,) if document_title else ()),
                source_ref=None,
                title_source_ref=document_title_ref,
                blocks=all_blocks,
            ),
        )

    builders: list[_SectionBuilder] = []
    stack: list[_SectionBuilder] = []
    root_builder: _SectionBuilder | None = None

    def ensure_root() -> _SectionBuilder:
        nonlocal root_builder
        if root_builder is None:
            root_builder = _SectionBuilder(
                id=f"s{len(builders)}",
                parent_section_id=None,
                order=len(builders),
                level=0,
                boundary_kind=SectionBoundaryKind.SYNTHETIC,
                title=document_title,
                section_path=((document_title,) if document_title else ()),
                source_ref=None,
                title_source_ref=document_title_ref,
            )
            builders.append(root_builder)
        return root_builder

    def walk(parent: ET.Element, parent_path: str) -> None:
        children = _child_paths(parent, parent_path)
        index = 0
        while index < len(children):
            child, child_path = children[index]
            child_tag = _tag(child)
            if exclusions.contains(child, child_path) or child_tag in _SKIP_SUBTREES:
                index += 1
                continue
            if (
                document_title_ref is not None
                and child_path == document_title_ref.element_path
            ):
                index += 1
                continue
            heading = headings.get(child_path)
            if heading is not None:
                while stack and stack[-1].level >= heading.level:
                    stack.pop()
                semantic_parent = stack[-1] if stack else None
                parent_id = (
                    semantic_parent.id
                    if semantic_parent is not None
                    else (root_builder.id if root_builder is not None else None)
                )
                parent_path_titles = (
                    semantic_parent.section_path if semantic_parent is not None else ()
                )
                builder = _SectionBuilder(
                    id=f"s{len(builders)}",
                    parent_section_id=parent_id,
                    order=len(builders),
                    level=heading.level,
                    boundary_kind=SectionBoundaryKind.IMPLICIT,
                    title=heading.title,
                    section_path=parent_path_titles + (heading.title,),
                    source_ref=heading.source_ref,
                    title_source_ref=heading.source_ref,
                )
                builders.append(builder)
                stack.append(builder)
                index += 1
                continue
            if child_tag in _SKIP_ELEMENTS:
                index += 1
                continue
            if child_tag == "SPAN":
                refs: list[SourceRef] = []
                while index < len(children):
                    span, span_path = children[index]
                    if (
                        _tag(span) != "SPAN"
                        or exclusions.contains(span, span_path)
                        or span_path in headings
                        or (
                            document_title_ref is not None
                            and span_path == document_title_ref.element_path
                        )
                        or not _element_text(span)
                    ):
                        break
                    refs.append(_source_ref(span, span_path, syntax))
                    index += 1
                if refs:
                    owner = stack[-1] if stack else ensure_root()
                    owner.blocks.append(
                        SectionBlockRef(
                            block_type=SectionBlockType.SPAN_RUN,
                            order=len(owner.blocks),
                            source_refs=tuple(refs),
                        )
                    )
                    continue
            if child_tag == "TABLE-GROUP":
                block_type = SectionBlockType.TABLE_GROUP
            elif child_tag == "TABLE":
                block_type = SectionBlockType.TABLE
            elif child_tag == "P" and _element_text(child):
                block_type = SectionBlockType.P
            elif child_tag in _IMAGE_TAGS:
                block_type = SectionBlockType.IMAGE
            else:
                walk(child, child_path)
                index += 1
                continue
            owner = stack[-1] if stack else ensure_root()
            owner.blocks.append(
                SectionBlockRef(
                    block_type=block_type,
                    order=len(owner.blocks),
                    source_refs=(_source_ref(child, child_path, syntax),),
                )
            )
            index += 1

    walk(root, root_path)
    return tuple(builder.freeze() for builder in builders)


def chunk_sections(
    document: LoadedDocument | ET.Element | str | bytes,
    *,
    document_context: DocumentContext,
    excluded_source_refs: Iterable[SourceRef] = (),
    syntax: DocumentSyntax = DocumentSyntax.AUTO,
) -> CanonicalSectionCollection:
    """Chunk a source document into non-overlapping hierarchical sections."""
    loaded = (
        document
        if isinstance(document, LoadedDocument)
        else load_document(document, syntax=syntax)
    )
    issues = tuple(
        SectionIssue(
            code=issue.code,
            message=issue.message,
            severity=issue.severity,
        )
        for issue in loaded.issues
    )
    if loaded.root is None:
        return CanonicalSectionCollection(
            source_document=document_context,
            syntax=loaded.syntax,
            parse_status=SectionParseStatus.FAILED,
            sections=(),
            issues=issues,
        )

    root_path = f"/{_tag(loaded.root)}[1]"
    exclusions = _exclusions(excluded_source_refs)
    has_explicit_sections = any(_is_section(node) for node in loaded.root.iter())
    sections = (
        _explicit_sections(loaded.root, root_path, loaded.syntax, exclusions)
        if has_explicit_sections
        else _implicit_sections(loaded.root, root_path, loaded.syntax, exclusions)
    )
    status = (
        SectionParseStatus.RECOVERED
        if loaded.status == DocumentLoadStatus.RECOVERED
        else SectionParseStatus.SUCCESS
    )
    return CanonicalSectionCollection(
        source_document=document_context,
        syntax=loaded.syntax,
        parse_status=status,
        sections=sections,
        issues=issues,
    )


__all__ = ["chunk_sections"]
