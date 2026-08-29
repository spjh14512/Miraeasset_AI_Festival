"""Shared, conservative text rules for table-adjacent context."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

from converters.common.source_models import ContextRole


@dataclass(frozen=True, slots=True)
class ContextTextPart:
    role: ContextRole
    text: str


_BOLD_TOKEN = re.compile(r"(?:^|\s)B(?:\s|$)", re.IGNORECASE)
_NOT_BOLD_TOKEN = re.compile(r"(?:^|\s)!B(?:\s|$)", re.IGNORECASE)
_UNIT_ONLY = re.compile(
    r"^\(?\s*(?:(?:원화|외화)\s*)?단위\s*[:：]",
    re.IGNORECASE,
)
_UNIT_PAREN = re.compile(
    r"\((?=[^()]{0,120}단위\s*[:：])[^()]{1,160}\)",
    re.IGNORECASE,
)
_NOTE = re.compile(
    r"^(?:※|\(\s*\*+\d*\s*\)|\*|주\s*\d+\s*[):.]|[-ㆍ·]\s*상기)",
    re.IGNORECASE,
)
_BASE_DATE = re.compile(
    r"^\(\s*(?:작성\s*)?기준일\s*[:：]\s*.+\)$",
    re.IGNORECASE,
)
_PERIOD = re.compile(
    r"^\d{1,2}\s*[)]\s*(?:"
    r"20\d{2}(?:[.]\d{1,2}(?:[.]\d{1,2})?|년\s*\d{1,2}\s*분기)|"
    r"당기|전기|당분기|전분기"
    r")\s*$",
    re.IGNORECASE,
)
_PERIOD_SUFFIX = re.compile(
    r"(?P<value>\d{1,2}\s*[)]\s*(?:"
    r"20\d{2}(?:[.]\d{1,2}(?:[.]\d{1,2})?|년\s*\d{1,2}\s*분기)|"
    r"당기|전기|당분기|전분기"
    r"))\s*$",
    re.IGNORECASE,
)
_CAPTION = re.compile(
    r"(?:다음과|아래와)\s*같습니다[.]?$|"
    r"(?:내역|현황|구성|변동내역|정보)는?\s*(?:다음과|아래와)\s*같습니다[.]?$",
    re.IGNORECASE,
)
_HEADING_MARKER = re.compile(
    r"^(?:"
    r"\d{1,2}(?:[.]\d{1,2})+[.]?|\d{1,2}[.]|"
    r"[(]\s*\d{1,2}\s*[)]|\d{1,2}\s*[)]|"
    r"[가나다라마바사아자차카타파하][.]|"
    r"[(][가나다라마바사아자차카타파하][)]|[①-⑳]|ㅇ\s+|"
    r"\[[^]\n]{1,80}\]|【[^】\n]{1,80}】"
    r")"
)
_EMBEDDED_HEADING_MARKER = re.compile(
    r"(?:"
    r"(?<!\d)\d{1,2}[.]\s+|"
    r"[가나다라마바사아자차카타파하][.]|"
    r"[(]\s*\d{1,2}\s*[)]|"
    r"\d{1,2}\s*[)]"
    r")"
)
_SENTENCE_END = re.compile(
    r"(?:습니다|합니다|됩니다|있습니다|없습니다|입니다|바랍니다|참조합니다)[.]?$"
)
_FOLLOWING_NUMBERED_BODY = re.compile(r"[.]\s*[(]\s*\d{1,2}\s*[)]")
_FOLLOWING_KOREAN_HEADING = re.compile(
    r"[.]\s*[가나다라마바사아자차카타파하][.]\s+"
)
_LEADING_SECTION_MARKER = re.compile(
    r"^(?:\d{1,2}[.]|[가나다라마바사아자차카타파하][.])\s*"
)
_CAPTION_LEAD = re.compile(
    r"^(?:당분기|전분기|당기|전기|보고기간|작성기준일|현재|연결그룹|"
    r"연결회사|지배기업|지배회사|당사)"
)


def clean_context_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def source_attributes_are_bold(attributes: Mapping[str, str]) -> bool:
    """Interpret DART USERMARK/STYLE bold flags, including explicit ``!B``."""
    usermark = attributes.get("USERMARK", "")
    if _NOT_BOLD_TOKEN.search(usermark):
        return False
    style = attributes.get("STYLE", "").replace(" ", "").lower()
    return _BOLD_TOKEN.search(usermark) is not None or "font-weight:bold" in style


def looks_like_context_heading(text: str, *, bold: bool = False) -> bool:
    cleaned = clean_context_text(text)
    if not cleaned or len(cleaned) > 160 or _SENTENCE_END.search(cleaned):
        return False
    return bool(_HEADING_MARKER.match(cleaned)) or (
        bold and not re.search(r"[.!?]", cleaned)
    )


def normalize_base_date_parts(values: Sequence[str]) -> str | None:
    """Join a base-date strip whose parentheses/date are split across cells."""
    cleaned = [clean_context_text(value) for value in values if clean_context_text(value)]
    if not cleaned:
        return None
    joined = " ".join(cleaned)
    if re.search(r"단위\s*[:：]", joined, re.IGNORECASE):
        return None
    joined = re.sub(r"\(\s+", "(", joined)
    joined = re.sub(r"\s+\)", ")", joined)
    joined = re.sub(r"\s*([:：])\s*", r" \1 ", joined)
    joined = clean_context_text(joined)
    return joined if _BASE_DATE.fullmatch(joined) else None


def split_table_context_text(
    text: str,
    *,
    bold: bool = False,
    fragment_texts: Sequence[str] = (),
) -> tuple[ContextTextPart, ...]:
    """Split one adjacent paragraph into heading/caption/unit/note parts.

    The function is deliberately narrow: ordinary narrative returns no parts.
    """
    cleaned = clean_context_text(text)
    if not cleaned:
        return ()

    if _UNIT_ONLY.match(cleaned):
        return (ContextTextPart(ContextRole.UNIT, cleaned),)
    if _BASE_DATE.fullmatch(cleaned) or _PERIOD.fullmatch(cleaned):
        return (ContextTextPart(ContextRole.CAPTION, cleaned),)
    if _NOTE.match(cleaned):
        if (
            _FOLLOWING_NUMBERED_BODY.search(cleaned)
            or _FOLLOWING_KOREAN_HEADING.search(cleaned)
        ):
            return ()
        role = ContextRole.HEADING if bold else ContextRole.NOTE
        return (ContextTextPart(role, cleaned),)

    heading: str | None = None
    remaining = cleaned
    first_fragment = next(
        (clean_context_text(value) for value in fragment_texts if clean_context_text(value)),
        "",
    )
    if (
        remaining.startswith(first_fragment)
        and (
            first_fragment.startswith("ㅇ ")
            or looks_like_context_heading(first_fragment, bold=bold)
        )
        and len(remaining) > len(first_fragment)
    ):
        heading = first_fragment
        remaining = remaining[len(first_fragment) :].strip()

    if heading is None:
        # Some DART paragraphs concatenate a numbered heading and a Korean
        # table lead-in without whitespace (``13. 리스가. 당분기...``).  Split
        # only when the suffix immediately starts like a caption sentence;
        # mixed heading/body chains continue to semantic segmentation.
        for match in _EMBEDDED_HEADING_MARKER.finditer(cleaned):
            if match.start() == 0 or not re.fullmatch(
                r"[가나다라마바사아자차카타파하][.]", match.group(0)
            ):
                continue
            prefix = cleaned[: match.start()].strip()
            suffix = cleaned[match.start() :].strip()
            suffix_marker = _LEADING_SECTION_MARKER.match(suffix)
            if suffix_marker is None:
                continue
            after_suffix_marker = suffix[suffix_marker.end() :].lstrip()
            if (
                looks_like_context_heading(prefix)
                and _CAPTION_LEAD.match(after_suffix_marker)
                and _CAPTION.search(suffix)
            ):
                suffix_parts = split_table_context_text(suffix)
                if suffix_parts and all(
                    part.role in {ContextRole.CAPTION, ContextRole.UNIT}
                    for part in suffix_parts
                ):
                    return (
                        ContextTextPart(ContextRole.HEADING, prefix),
                        *suffix_parts,
                    )
            break

    unit_matches = list(_UNIT_PAREN.finditer(remaining))
    units = [match.group(0) for match in unit_matches]
    without_units = _UNIT_PAREN.sub(" ", remaining)
    without_units = clean_context_text(without_units)
    without_units = re.sub(r"\s+([.])", r"\1", without_units)

    period: str | None = None
    period_match = _PERIOD_SUFFIX.search(without_units)
    if period_match is not None:
        period = clean_context_text(period_match.group("value"))
        without_units = clean_context_text(without_units[: period_match.start()])
        stripped_prefix = without_units.rstrip(". ")
        without_units = stripped_prefix + (
            "." if stripped_prefix.endswith("같습니다") else ""
        )

    # A paragraph containing another numbered heading is not table-only
    # context.  Leave it to semantic segmentation so narrative between the
    # headings is retained as TEXT evidence instead of being swallowed by the
    # following table caption.  Inspect after removing a trailing period label
    # (``1) 2026년 1분기``), which is valid table context rather than a heading.
    marker_source = without_units
    embedded_heading = False
    for match in _EMBEDDED_HEADING_MARKER.finditer(marker_source):
        if match.start() == 0:
            continue
        preceding = marker_source[: match.start()].rstrip()
        if match.group(0).startswith("다.") and re.search(
            r"(?:습니|합니|됩니|입니)$", preceding
        ):
            continue
        embedded_heading = True
        break
    if embedded_heading:
        return ()

    caption = without_units if _CAPTION.search(without_units) else None
    leading_section = _LEADING_SECTION_MARKER.match(cleaned)
    if heading is None and caption is not None and leading_section is not None:
        # A compact "heading + lead-in" paragraph (for example,
        # ``8. 기타자산...다음과 같습니다.``) must be split by the semantic
        # segmenter.  Parenthesized numbering such as ``(6) ...다음과`` is a
        # table lead-in and intentionally remains a caption.
        after_marker = cleaned[leading_section.end() :].lstrip()
        if _CAPTION_LEAD.match(after_marker) is None:
            return ()
    if heading is None and caption is None and not units and period is None:
        return (
            (ContextTextPart(ContextRole.HEADING, cleaned),)
            if bold and looks_like_context_heading(cleaned, bold=True)
            else ()
        )

    parts: list[ContextTextPart] = []
    if heading is not None:
        parts.append(ContextTextPart(ContextRole.HEADING, heading))
    if caption is not None:
        parts.append(ContextTextPart(ContextRole.CAPTION, caption))
    parts.extend(ContextTextPart(ContextRole.UNIT, unit) for unit in units)
    if period is not None:
        parts.append(ContextTextPart(ContextRole.CAPTION, period))
    return tuple(parts)


__all__ = [
    "ContextTextPart",
    "clean_context_text",
    "looks_like_context_heading",
    "normalize_base_date_parts",
    "source_attributes_are_bold",
    "split_table_context_text",
]
