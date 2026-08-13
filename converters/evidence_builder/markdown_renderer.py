"""Compact, provenance-neutral Markdown rendering for final evidence."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from converters.evidence_builder.evidence_models import FinalEvidenceContext


def _inline(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _context_lines(context: FinalEvidenceContext) -> list[str]:
    lines: list[str] = []
    if context.section_path:
        lines.append(f"**섹션:** {' > '.join(context.section_path)}")
    if context.title:
        lines.append(f"**표 제목:** {context.title}")
    if context.units:
        lines.append(f"**단위:** {' / '.join(context.units)}")
    if context.captions:
        lines.append(f"**캡션:** {' / '.join(context.captions)}")
    return lines


def _with_notes(lines: list[str], context: FinalEvidenceContext) -> str:
    if context.notes:
        if lines:
            lines.append("")
        lines.append("**주석:**")
        lines.extend(f"- {note}" for note in context.notes)
    return "\n".join(lines).strip()


def render_text(text: str, context: FinalEvidenceContext) -> str:
    lines = _context_lines(context)
    if lines:
        lines.append("")
    lines.append(text)
    return _with_notes(lines, context)


def render_kv(fields: Iterable[Mapping[str, Any]], context: FinalEvidenceContext) -> str:
    lines = _context_lines(context)
    if lines:
        lines.append("")
    for field in fields:
        paths = field.get("key_paths", [])
        key = " / ".join(" > ".join(path) for path in paths) or "값"
        lines.append(f"- {_inline(key)}: {_inline(field.get('raw_value', ''))}")
    return _with_notes(lines, context)


def render_record(values: Iterable[Mapping[str, Any]], context: FinalEvidenceContext) -> str:
    lines = _context_lines(context)
    if lines:
        lines.append("")
    for value in values:
        header = " > ".join(value.get("header_path", [])) or value.get("column_id", "값")
        lines.append(f"- {_inline(header)}: {_inline(value.get('raw_value', ''))}")
    return _with_notes(lines, context)


def render_unknown(rows: Sequence[Sequence[str]], context: FinalEvidenceContext) -> str:
    lines = _context_lines(context)
    if lines:
        lines.append("")
    width = max((len(row) for row in rows), default=0)
    if width:
        normalized = [list(row) + [""] * (width - len(row)) for row in rows]
        lines.append("| " + " | ".join(f"열 {index + 1}" for index in range(width)) + " |")
        lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
        lines.extend("| " + " | ".join(_inline(value) for value in row) + " |" for row in normalized)
    return _with_notes(lines, context)


__all__ = ["render_kv", "render_record", "render_text", "render_unknown"]
