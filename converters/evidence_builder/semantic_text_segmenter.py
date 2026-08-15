"""Split canonical paragraphs into retrieval-safe semantic text segments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Mapping

from converters.paragraph_parser.paragraph_models import (
    CanonicalParagraph,
    FragmentKind,
    ParagraphFragment,
)


class SemanticTextRole(StrEnum):
    HEADING = "HEADING"
    BODY = "BODY"
    NOTE = "NOTE"
    REFERENCE_NOTICE = "REFERENCE_NOTICE"
    TABLE_CAPTION = "TABLE_CAPTION"


@dataclass(frozen=True, slots=True)
class SemanticTextSegment:
    text: str
    role: SemanticTextRole
    heading_level: int | None = None
    references: tuple[Mapping[str, str], ...] = ()


_DISCLOSURE_REFNO = re.compile(r"^\d{14}$")
_NOTE_TEXT = re.compile(r"^(?:※|주\s*\d+\s*[):.]|[-ㆍ·]\s*상기)", re.IGNORECASE)
_REFERENCE_NOTICE = re.compile(
    r"(?:기재하지\s*않|관련\s*내용|참고하시|참고\s*바랍니다)",
    re.IGNORECASE,
)
_TABLE_CAPTION = re.compile(
    r"(?:다음과|아래와)\s*같습니다[.]?$|"
    r"(?:내역|현황|구성|변동내역|정보)는?\s*(?:다음과|아래와)\s*같습니다[.]?$",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(
    r"(?:습니다|합니다|됩니다|있습니다|없습니다|입니다|바랍니다|참조합니다)[.]?$"
)
_BOLD_TOKEN = re.compile(r"(?:^|\s)B(?:\s|$)", re.IGNORECASE)

_MARKERS: tuple[tuple[str, re.Pattern[str], int], ...] = (
    ("DECIMAL", re.compile(r"^\d{1,2}(?:[.]\d{1,2})+[.]?\s*"), 2),
    ("NUMBER_DOT", re.compile(r"^\d{1,2}[.]\s*"), 1),
    ("PAREN_NUMBER", re.compile(r"^[(]\s*\d{1,2}\s*[)]\s*"), 1),
    ("NUMBER_PAREN", re.compile(r"^\d{1,2}\s*[)]\s*"), 2),
    ("KOREAN", re.compile(r"^(?:[가-하][.]|[(][가-하][)])\s*"), 3),
    ("CIRCLED", re.compile(r"^[①-⑳]\s*"), 3),
    ("BRACKET", re.compile(r"^(?:\[[^]\n]{1,40}\]|【[^】\n]{1,40}】)\s*"), 4),
)

_INLINE_MARKER = re.compile(
    r"(?:\[[^]\n]{1,40}\]|【[^】\n]{1,40}】|[①-⑳]|"
    r"[(]\s*\d{1,2}\s*[)]|[(][가-하][)]|"
    r"(?<![\d.])\d{1,2}\s*[)]|(?<![\d.])\d{1,2}[.](?!\d)|"
    r"(?<![가-힣A-Za-z0-9])[가-하][.])"
)
_CONCATENATED_KOREAN_MARKER = re.compile(
    r"(?:사항|현황|개요|정보|내역|정책)([가-하][.])\s*"
)
_POSSIBLY_ATTACHED_KOREAN_MARKER = re.compile(r"([가-하][.])\s*")
_CONCATENATED_NUMBER_MARKER = re.compile(
    r"(?:습니다|합니다|없습니다)[.](\d{1,2}\s*[)])"
)
_BODY_START = re.compile(
    r"(?:해당사항|당분기|당기|전기|보고기간|작성기준일|연결실체|연결그룹|연결회사|"
    r"지배기업|지배회사|당사(?!항)|회사는|금융자산|건설산업|전력산업|효성중공업)"
)
_CAPTION_SENTENCE_START = re.compile(
    r"(?:당분기|당기|전기|보고기간|작성기준일|현재|연결그룹|연결회사|"
    r"지배기업|지배회사|당사)"
)


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _fragment_references(
    fragments: tuple[ParagraphFragment, ...],
) -> tuple[dict[str, str], ...]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for fragment in fragments:
        refno = fragment.source_attributes.get("REFNO", "").strip()
        text = _clean_text(fragment.text)
        if not _DISCLOSURE_REFNO.fullmatch(refno) or not text:
            continue
        identity = ("disclosure", refno, text)
        if identity in seen:
            continue
        seen.add(identity)
        result.append({"type": "disclosure", "refno": refno, "text": text})
    return tuple(result)


def _references_for_text(
    references: tuple[dict[str, str], ...],
    text: str,
) -> tuple[dict[str, str], ...]:
    return tuple(reference for reference in references if reference["text"] in text)


def _marker(text: str) -> tuple[str, int, int] | None:
    stripped = text.lstrip()
    offset = len(text) - len(stripped)
    for name, pattern, level in _MARKERS:
        match = pattern.match(stripped)
        if match is not None:
            return name, level, offset + match.end()
    return None


def _looks_like_heading(text: str, *, force: bool = False) -> bool:
    cleaned = _clean_text(text)
    marker = _marker(cleaned)
    if marker is None and not force:
        return False
    if len(cleaned) > 100:
        return False
    if _SENTENCE_END.search(cleaned):
        return False
    return marker is not None or not re.search(r"[.!?]", cleaned)


def _is_bold_heading(fragment: ParagraphFragment) -> bool:
    if fragment.kind != FragmentKind.SPAN:
        return False
    usermark = fragment.source_attributes.get("USERMARK", "")
    style = fragment.source_attributes.get("STYLE", "").replace(" ", "").lower()
    is_bold = _BOLD_TOKEN.search(usermark) is not None or "font-weight:bold" in style
    return is_bold and _looks_like_heading(fragment.text, force=True)


def _marker_kind_at(text: str, position: int) -> str | None:
    value = text[position:]
    marker = _marker(value)
    return marker[0] if marker is not None else None


def _inside_quote(text: str, position: int) -> bool:
    prefix = text[:position]
    if prefix.count("'") % 2 or prefix.count('"') % 2:
        return True
    return any(
        prefix.rfind(opening) > prefix.rfind(closing)
        for opening, closing in (("‘", "’"), ("“", "”"))
    )


def _split_marker_chunks(text: str) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    positions = [0]
    last = 0
    candidate_positions = {match.start() for match in _INLINE_MARKER.finditer(cleaned)}
    concatenated_korean_positions = {
        match.start(1) for match in _CONCATENATED_KOREAN_MARKER.finditer(cleaned)
    }
    candidate_positions.update(concatenated_korean_positions)
    candidate_positions.update(
        match.start(1) for match in _POSSIBLY_ATTACHED_KOREAN_MARKER.finditer(cleaned)
    )
    candidate_positions.update(
        match.start(1) for match in _CONCATENATED_NUMBER_MARKER.finditer(cleaned)
    )
    for position in sorted(candidate_positions):
        if position == 0 or _inside_quote(cleaned, position):
            continue
        marker_kind = _marker_kind_at(cleaned, position)
        prefix = cleaned[last:position].strip()
        prefix_marker = _marker(prefix)
        if marker_kind == "KOREAN" and position not in concatenated_korean_positions:
            preceding = cleaned[:position].rstrip()
            suffix = cleaned[position:]
            joined_heading_caption = (
                prefix_marker is not None
                and len(prefix) <= 80
                and _TABLE_CAPTION.search(suffix) is not None
                and _BODY_START.search(suffix) is not None
            )
            if preceding and preceding[-1] not in ".;:" and not joined_heading_caption:
                continue
        previous = cleaned[position - 1]
        strong = marker_kind in {"BRACKET", "CIRCLED"}
        same_enumeration_chain = (
            prefix_marker is not None
            and prefix_marker[0] == marker_kind
            and marker_kind in {"PAREN_NUMBER", "NUMBER_PAREN", "KOREAN"}
        )
        short_heading_chain = (
            prefix_marker is not None
            and len(prefix) <= 120
            and _SENTENCE_END.search(prefix) is None
        )
        if strong or previous in ".;:\n" or short_heading_chain or same_enumeration_chain:
            positions.append(position)
            last = position
    positions.append(len(cleaned))
    return [
        cleaned[start:end].strip()
        for start, end in zip(positions, positions[1:])
        if cleaned[start:end].strip()
    ]


def _split_heading_body(text: str) -> tuple[str, str] | None:
    marker = _marker(text)
    if marker is None:
        return None
    marker_kind, _, marker_end = marker
    if marker_kind == "KOREAN" and _TABLE_CAPTION.search(text):
        return None
    if marker_kind == "BRACKET" and marker_end < len(text):
        return text[:marker_end].strip(), text[marker_end:].strip()
    search_from = min(len(text), marker_end + 2)
    for match in _BODY_START.finditer(text, search_from):
        heading = text[:match.start()].strip()
        body = text[match.start():].strip()
        if 3 <= len(heading) <= 80 and _looks_like_heading(heading):
            return heading, body
    return None


def _split_caption_prefix(text: str) -> tuple[str, str] | None:
    if _TABLE_CAPTION.search(text) is None:
        return None
    candidates = [match.start() for match in _CAPTION_SENTENCE_START.finditer(text)]
    for position in reversed(candidates):
        if position <= 0:
            continue
        prefix = text[:position].rstrip()
        prefix_marker = _marker(prefix)
        if prefix_marker is not None and prefix_marker[2] == len(prefix):
            continue
        if prefix.endswith("."):
            return prefix, text[position:].strip()
    return None


def _classify(
    text: str,
    *,
    force_heading: bool,
    references: tuple[dict[str, str], ...],
) -> tuple[SemanticTextRole, int | None]:
    marker = _marker(text)
    if _looks_like_heading(text, force=force_heading):
        return SemanticTextRole.HEADING, marker[1] if marker is not None else 4
    if references and _REFERENCE_NOTICE.search(text):
        return SemanticTextRole.REFERENCE_NOTICE, None
    if _TABLE_CAPTION.search(text):
        return SemanticTextRole.TABLE_CAPTION, None
    if _NOTE_TEXT.match(text):
        return SemanticTextRole.NOTE, None
    return SemanticTextRole.BODY, None


def _segments_from_text(
    text: str,
    *,
    force_heading: bool,
    references: tuple[dict[str, str], ...],
) -> list[SemanticTextSegment]:
    result: list[SemanticTextSegment] = []
    for chunk in _split_marker_chunks(text):
        pieces: list[tuple[str, bool]] = [(chunk, force_heading)]
        split = _split_heading_body(chunk)
        if split is not None:
            heading, body = split
            pieces = [(heading, True), (body, False)]
        expanded: list[tuple[str, bool]] = []
        for piece, forced in pieces:
            caption_split = None if forced else _split_caption_prefix(piece)
            if caption_split is None:
                expanded.append((piece, forced))
            else:
                expanded.extend(((caption_split[0], False), (caption_split[1], False)))
        for piece, forced in expanded:
            cleaned = _clean_text(piece)
            if not cleaned:
                continue
            piece_references = _references_for_text(references, cleaned)
            role, level = _classify(
                cleaned,
                force_heading=forced,
                references=piece_references,
            )
            result.append(
                SemanticTextSegment(
                    text=cleaned,
                    role=role,
                    heading_level=level,
                    references=piece_references,
                )
            )
    return result


def segment_paragraph(paragraph: CanonicalParagraph) -> tuple[SemanticTextSegment, ...]:
    """Split one paragraph without rewriting or reordering its source text."""
    references = _fragment_references(paragraph.fragments)
    result: list[SemanticTextSegment] = []
    raw_parts: list[str] = []

    def flush(*, force_heading: bool = False) -> None:
        if not raw_parts:
            return
        raw_text = "".join(raw_parts)
        raw_parts.clear()
        result.extend(
            _segments_from_text(
                raw_text,
                force_heading=force_heading,
                references=references,
            )
        )

    for fragment in paragraph.fragments:
        if fragment.raw_text == "\n":
            flush()
            continue
        if _is_bold_heading(fragment):
            flush()
            result.extend(
                _segments_from_text(
                    fragment.raw_text,
                    force_heading=True,
                    references=references,
                )
            )
            continue
        raw_parts.append(fragment.raw_text)
    flush()

    if not result and paragraph.text.strip():
        result.extend(
            _segments_from_text(
                paragraph.text,
                force_heading=False,
                references=references,
            )
        )
    return tuple(result)


__all__ = [
    "SemanticTextRole",
    "SemanticTextSegment",
    "segment_paragraph",
]
