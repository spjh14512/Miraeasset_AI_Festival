"""Build embedding text for a TEXT Evidence item."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


CONTEXT_SEPARATOR = " > "
BODY_SEPARATOR = "\n\n"


def _context_values(*groups: Sequence[str]) -> list[str]:
    values: list[str] = []
    for group in groups:
        for raw_value in group:
            if not isinstance(raw_value, str):
                raise ValueError("Context values must be strings")
            value = raw_value.strip()
            if value and (not values or values[-1] != value):
                values.append(value)
    return values


def build_text_contextual_text(
    evidence: Mapping[str, Any],
    *,
    corp_name: str,
    report_nm: str,
    section_path: Sequence[str],
) -> str:
    """Combine document/section context and the original TEXT Evidence body.

    ``corp_name`` and ``report_nm`` come from the source manifest, while
    ``section_path`` comes from the canonical section matched by ``section_id``.
    Evidence-level ``heading_path`` is appended when present.
    """
    if evidence.get("evidence_type") != "TEXT":
        raise ValueError("Expected evidence_type to be TEXT")

    payload = evidence.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("TEXT Evidence payload must be a mapping")

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("TEXT Evidence payload.text must be a non-empty string")

    heading_path = payload.get("heading_path", [])
    if not isinstance(heading_path, (list, tuple)):
        raise ValueError("TEXT Evidence payload.heading_path must be a sequence")

    section_context = _context_values(section_path, heading_path)
    document_lines: list[str] = []
    if corp_name.strip():
        document_lines.append(f"회사 : {corp_name.strip()}")
    if report_nm.strip():
        document_lines.append(f"공시 : {report_nm.strip()}")
    if section_context:
        document_lines.append(f"섹션 : {CONTEXT_SEPARATOR.join(section_context)}")
    if not document_lines:
        raise ValueError("At least one document or section context value is required")

    header = "\n".join(document_lines)
    return header + BODY_SEPARATOR + text.strip()


__all__ = ["build_text_contextual_text"]
