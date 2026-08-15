"""Streaming validator for section-level Evidence Fragment output."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from converters.evidence_builder.fragment_models import FRAGMENT_SCHEMA_VERSION
from converters.evidence_builder.fragment_pipeline import MANIFEST_SCHEMA_VERSION


_EVIDENCE_ID = re.compile(
    r"^evidence:(?P<rcept>\d+):src(?P<src>\d+):s(?P<section>\d+):e(?P<order>\d+)$"
)
_SECTION_ID = re.compile(
    r"^section:(?P<rcept>\d+):src(?P<src>\d+):s(?P<section>\d+)$"
)
_TABLE_ID = re.compile(
    r"^rtable:(?P<rcept>\d+):src(?P<src>\d+):s(?P<section>\d+):t(?P<table>\d+)$"
)
_DISCLOSURE_REFNO = re.compile(r"^\d{14}$")
_FORBIDDEN_FIELDS = {
    "doc_id",
    "rcept_no",
    "doc_group",
    "source_index",
    "source_path",
    "source_sha256",
    "canonical_section_path",
    "canonical_section_sha256",
    "source_refs",
    "content_sha256",
    "markdown",
    "context",
    "source_block_order",
    "source_row",
    "source_cell",
    "column_id",
}


def _json_lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_forbidden(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in _FORBIDDEN_FIELDS:
                found.append(path)
            found.extend(_find_forbidden(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_forbidden(item, f"{prefix}[{index}]"))
    return found


def validate_fragment(
    fragment: Mapping[str, Any],
    *,
    expected_section_id: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if set(fragment) != {
        "schema_version",
        "section_id",
        "evidence_list",
        "records",
    }:
        errors.append("top_level_fields")
    if fragment.get("schema_version") != FRAGMENT_SCHEMA_VERSION:
        errors.append("schema_version")
    section_id = fragment.get("section_id")
    section_match = _SECTION_ID.fullmatch(str(section_id))
    if section_match is None:
        errors.append("section_id")
    if expected_section_id is not None and section_id != expected_section_id:
        errors.append("section_id_mismatch")

    evidence_list = fragment.get("evidence_list")
    records = fragment.get("records")
    if not isinstance(evidence_list, list):
        errors.append("evidence_list")
        evidence_list = []
    if not isinstance(records, list):
        errors.append("records")
        records = []

    table_counts: dict[str, int] = {}
    table_widths: dict[str, int] = {}
    table_order: list[str] = []
    seen_evidence: set[str] = set()
    r_table_order = 0
    for order, evidence in enumerate(evidence_list):
        if not isinstance(evidence, Mapping):
            errors.append(f"evidence_shape:{order}")
            continue
        evidence_type = evidence.get("evidence_type")
        required = {"evidence_id", "evidence_type", "order", "payload"}
        optional = {"table_type", "storage_mode", "references"}
        if not required.issubset(evidence) or not set(evidence).issubset(required | optional):
            errors.append(f"evidence_fields:{order}")
        evidence_id = str(evidence.get("evidence_id", ""))
        match = _EVIDENCE_ID.fullmatch(evidence_id)
        if match is None:
            errors.append(f"evidence_id:{order}")
        elif section_match is not None and (
            match.group("rcept") != section_match.group("rcept")
            or match.group("src") != section_match.group("src")
            or match.group("section") != section_match.group("section")
            or int(match.group("order")) != order
        ):
            errors.append(f"evidence_id_scope:{order}")
        if evidence_id in seen_evidence:
            errors.append(f"duplicate_evidence_id:{order}")
        seen_evidence.add(evidence_id)
        if evidence.get("order") != order:
            errors.append(f"evidence_order:{order}")
        if evidence_type not in {"TEXT", "TABLE"}:
            errors.append(f"evidence_type:{order}")
        references = evidence.get("references")
        if references is not None:
            if not isinstance(references, list) or not references:
                errors.append(f"references:{order}")
            else:
                for reference_index, reference in enumerate(references):
                    if (
                        not isinstance(reference, Mapping)
                        or set(reference) != {"type", "refno", "text"}
                        or reference.get("type") != "disclosure"
                        or _DISCLOSURE_REFNO.fullmatch(
                            str(reference.get("refno", ""))
                        )
                        is None
                        or not str(reference.get("text", "")).strip()
                    ):
                        errors.append(f"reference:{order}:{reference_index}")
        payload = evidence.get("payload")
        if not isinstance(payload, Mapping) or not payload:
            errors.append(f"payload:{order}")
        if isinstance(payload, Mapping) and "heading_path" in payload:
            heading_path = payload.get("heading_path")
            if (
                not isinstance(heading_path, list)
                or not heading_path
                or not all(
                    isinstance(value, str) and value.strip()
                    for value in heading_path
                )
            ):
                errors.append(f"heading_path:{order}")
        if evidence_type == "TEXT":
            if not set(evidence).issubset(required | {"references"}):
                errors.append(f"text_fields:{order}")
            if not isinstance(payload, Mapping) or not str(payload.get("text", "")).strip():
                errors.append(f"text_payload:{order}")
            elif set(payload) - {"text", "text_role", "heading_path"}:
                errors.append(f"text_payload_fields:{order}")
            if isinstance(payload, Mapping) and payload.get("text_role") not in {
                None,
                "BODY",
                "NOTE",
                "REFERENCE_NOTICE",
            }:
                errors.append(f"text_role:{order}")
        else:
            if isinstance(payload, Mapping):
                caption = payload.get("caption")
                if caption is not None and (
                    not isinstance(caption, str) or not caption.strip()
                ):
                    errors.append(f"caption:{order}")
                if "text_role" in payload:
                    errors.append(f"table_text_role:{order}")
            table_type = evidence.get("table_type")
            if table_type not in {"KV_TABLE", "R_TABLE"}:
                errors.append(f"table_type:{order}")
            if table_type == "KV_TABLE":
                if "storage_mode" in evidence:
                    errors.append(f"kv_storage_mode:{order}")
                if not isinstance(payload, Mapping) or not isinstance(
                    payload.get("fields"), list
                ):
                    errors.append(f"kv_payload:{order}")
                continue
            if evidence.get("storage_mode") != "SECTION_RECORDS":
                errors.append(f"r_table_storage_mode:{order}")
                continue
            if not isinstance(payload, Mapping):
                errors.append(f"r_table_payload:{order}")
                continue
            table_id = str(payload.get("table_id", ""))
            table_match = _TABLE_ID.fullmatch(table_id)
            if table_match is None:
                errors.append(f"table_id:{order}")
            elif section_match is not None and (
                table_match.group("rcept") != section_match.group("rcept")
                or table_match.group("src") != section_match.group("src")
                or table_match.group("section") != section_match.group("section")
            ):
                errors.append(f"table_id_scope:{order}")
            elif int(table_match.group("table")) != r_table_order:
                errors.append(f"table_id_order:{order}")
            r_table_order += 1
            if table_id in table_counts:
                errors.append(f"duplicate_table_id:{order}")
            record_count = payload.get("record_count")
            if not isinstance(record_count, int) or record_count < 0:
                errors.append(f"record_count_value:{order}")
                record_count = -1
            headers = payload.get("headers")
            if not isinstance(headers, list) or not all(
                isinstance(header, list)
                and all(isinstance(part, str) for part in header)
                for header in headers
            ):
                errors.append(f"headers:{order}")
                headers = []
            table_counts[table_id] = record_count
            table_widths[table_id] = len(headers)
            table_order.append(table_id)

    actual_records: Counter[str] = Counter()
    record_indexes: dict[str, list[int]] = {}
    record_table_order: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            errors.append(f"record_shape:{index}")
            continue
        if set(record) != {
            "table_id",
            "record_index",
            "row_type",
            "row_context",
            "values",
        }:
            errors.append(f"record_fields:{index}")
        table_id = str(record.get("table_id", ""))
        if not record_table_order or record_table_order[-1] != table_id:
            record_table_order.append(table_id)
        record_index = record.get("record_index")
        if table_id not in table_counts:
            errors.append(f"orphan_record:{index}")
        if not isinstance(record_index, int) or record_index < 0:
            errors.append(f"record_index:{index}")
        else:
            record_indexes.setdefault(table_id, []).append(record_index)
        if record.get("row_type") not in {"DATA", "SUBTOTAL", "TOTAL", "UNKNOWN"}:
            errors.append(f"row_type:{index}")
        if not isinstance(record.get("row_context"), list) or not all(
            isinstance(value, str) for value in record.get("row_context", [])
        ):
            errors.append(f"row_context:{index}")
        values = record.get("values")
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            errors.append(f"values:{index}")
        elif table_id in table_widths and len(values) != table_widths[table_id]:
            errors.append(f"record_width:{index}")
        actual_records[table_id] += 1

    for table_id, expected in table_counts.items():
        if expected != actual_records[table_id]:
            errors.append(f"record_count:{table_id}")
        if record_indexes.get(table_id, []) != list(range(expected)):
            errors.append(f"record_order:{table_id}")
    expected_record_table_order = [
        table_id for table_id in table_order if table_counts[table_id] > 0
    ]
    if record_table_order != expected_record_table_order:
        errors.append("record_table_order")

    for path in _find_forbidden(fragment):
        errors.append(f"forbidden:{path}")
    return errors


def validate_evidence_fragments(
    *,
    data_root: Path,
    fragment_manifest_path: Path,
    progress_every: int = 500,
    max_errors: int = 100,
) -> dict[str, Any]:
    records = _json_lines(fragment_manifest_path)
    errors: list[str] = []
    evidence_ids: set[str] = set()
    section_ids: set[str] = set()
    status_counts: Counter[str] = Counter()
    type_counts: Counter[tuple[str | None, str | None]] = Counter()
    total_evidence = 0
    total_records = 0

    for document_index, record in enumerate(records, 1):
        doc_id = str(record.get("doc_id"))
        if record.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            errors.append(f"manifest_schema:{doc_id}")
        status_counts[str(record.get("status"))] += 1
        input_path = data_root / Path(str(record.get("input_path")))
        if not input_path.is_file():
            errors.append(f"missing_input:{doc_id}")
            expected_sections: list[str] = []
        else:
            if _sha256(input_path) != record.get("input_sha256"):
                errors.append(f"stale_input:{doc_id}")
            canonical = json.loads(input_path.read_text(encoding="utf-8"))
            expected_sections = [
                str(section["section_id"])
                for section in canonical.get("sections", [])
            ]

        output_paths = record.get("output_paths", [])
        if not isinstance(output_paths, list):
            errors.append(f"output_paths:{doc_id}")
            output_paths = []
        actual_sections: list[str] = []
        document_evidence = 0
        document_records = 0
        for output_index, relative in enumerate(output_paths):
            path = data_root / Path(str(relative))
            if not path.is_file():
                errors.append(f"missing_fragment:{relative}")
                continue
            try:
                fragment = json.loads(path.read_text(encoding="utf-8"))
            except Exception as error:
                errors.append(f"json:{relative}:{type(error).__name__}:{error}")
                continue
            expected = (
                expected_sections[output_index]
                if output_index < len(expected_sections)
                else None
            )
            fragment_errors = validate_fragment(
                fragment,
                expected_section_id=expected,
            )
            errors.extend(f"{relative}:{error}" for error in fragment_errors)
            section_id = str(fragment.get("section_id", ""))
            actual_sections.append(section_id)
            if section_id in section_ids:
                errors.append(f"duplicate_section_id:{section_id}")
            section_ids.add(section_id)
            for evidence in fragment.get("evidence_list", []):
                evidence_id = str(evidence.get("evidence_id", ""))
                if evidence_id in evidence_ids:
                    errors.append(f"duplicate_evidence_id:{evidence_id}")
                evidence_ids.add(evidence_id)
                type_counts[(evidence.get("evidence_type"), evidence.get("table_type"))] += 1
            document_evidence += len(fragment.get("evidence_list", []))
            document_records += len(fragment.get("records", []))
            if len(errors) >= max_errors:
                break

        if actual_sections != expected_sections:
            errors.append(f"section_coverage:{doc_id}")
        if record.get("n_fragments") != len(actual_sections):
            errors.append(f"fragment_count:{doc_id}")
        if record.get("n_evidence") != document_evidence:
            errors.append(f"evidence_count:{doc_id}")
        if record.get("n_records") != document_records:
            errors.append(f"external_record_count:{doc_id}")
        total_evidence += document_evidence
        total_records += document_records
        if len(errors) >= max_errors:
            break
        if progress_every > 0 and document_index % progress_every == 0:
            print(
                f"validated={document_index}/{len(records)} "
                f"fragments={len(section_ids)} evidence={total_evidence}",
                flush=True,
            )

    return {
        "documents": len(records),
        "fragments": len(section_ids),
        "total_evidence": total_evidence,
        "total_external_records": total_records,
        "status_counts": dict(status_counts),
        "type_counts": [
            {
                "evidence_type": kind[0],
                "table_type": kind[1],
                "count": count,
            }
            for kind, count in sorted(type_counts.items(), key=lambda item: str(item[0]))
        ],
        "errors_count": len(errors),
        "errors": errors[:max_errors],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/evidence_fragment/manifest.jsonl"),
    )
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()
    result = validate_evidence_fragments(
        data_root=args.data_root,
        fragment_manifest_path=args.manifest,
        progress_every=args.progress_every,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["errors_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = ["validate_evidence_fragments", "validate_fragment"]
