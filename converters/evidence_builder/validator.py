"""Validate one-file-per-evidence canonical output without loading the corpus at once."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


OUTPUT_SCHEMA_VERSION = "canonical-evidence.v1"
MANIFEST_SCHEMA_VERSION = "canonical-evidence-manifest.v1"


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


def _validate_evidence(
    evidence: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    output_path: str,
) -> list[str]:
    errors: list[str] = []
    evidence_id = evidence.get("id")

    if evidence.get("schema_version") != OUTPUT_SCHEMA_VERSION:
        errors.append(f"schema:{output_path}")
    for field in ("doc_id", "rcept_no", "doc_group"):
        if str(evidence.get(field, "")) != str(record.get(field, "")):
            errors.append(f"{field}:{output_path}")
    if evidence.get("canonical_section_path") != record.get("input_path"):
        errors.append(f"input_path:{output_path}")
    if evidence.get("canonical_section_sha256") != record.get("input_sha256"):
        errors.append(f"input_hash:{output_path}")
    if not isinstance(evidence.get("source_index"), int):
        errors.append(f"source_index:{output_path}")
    if not evidence.get("source_path"):
        errors.append(f"source_path:{output_path}")
    source_hash = evidence.get("source_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        errors.append(f"source_hash:{output_path}")

    evidence_type = evidence.get("evidence_type")
    if evidence_type not in {"TEXT", "TABLE"}:
        errors.append(f"type:{output_path}:{evidence_id}")
    if not isinstance(evidence.get("section_id"), str):
        errors.append(f"section_id:{output_path}:{evidence_id}")
    if not isinstance(evidence.get("order"), int):
        errors.append(f"order:{output_path}:{evidence_id}")

    context = evidence.get("context")
    if not isinstance(context, dict) or not isinstance(
        context.get("section_path"), list
    ):
        errors.append(f"context:{output_path}:{evidence_id}")
    elif "blocks" in context:
        errors.append(f"context_blocks:{output_path}:{evidence_id}")
    if not evidence.get("source_refs"):
        errors.append(f"source_refs:{output_path}:{evidence_id}")

    markdown = evidence.get("markdown")
    if not isinstance(markdown, str) or not markdown.strip():
        errors.append(f"markdown:{output_path}:{evidence_id}")
    payload = evidence.get("payload")
    if not isinstance(payload, dict) or not payload:
        errors.append(f"payload:{output_path}:{evidence_id}")

    content_hash = evidence.get("content_sha256")
    if not isinstance(content_hash, str) or len(content_hash) != 64:
        errors.append(f"content_hash_shape:{output_path}:{evidence_id}")
    return errors


def validate_canonical_evidence(
    *,
    data_root: Path,
    manifest_path: Path,
    progress_every: int = 500,
    max_errors: int = 100,
) -> dict[str, Any]:
    records = _json_lines(manifest_path)
    errors: list[str] = []
    status_counts: collections.Counter[str] = collections.Counter()
    type_counts: collections.Counter[tuple[str | None, str | None]] = (
        collections.Counter()
    )
    evidence_ids: set[str] = set()
    output_paths_seen: set[str] = set()
    total_sources = 0
    total_evidence = 0
    total_images = 0
    largest_evidence_count: tuple[int, str | None] = (0, None)

    for index, record in enumerate(records, 1):
        if record.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            errors.append(f"manifest_schema:{record.get('doc_id')}")
        status_counts[str(record.get("status"))] += 1
        total_sources += int(record.get("n_sources", 0))
        total_images += int(record.get("n_images_skipped", 0))

        input_path = data_root / Path(str(record["input_path"]))
        if not input_path.is_file():
            errors.append(f"missing_input:{record['input_path']}")
        elif _sha256(input_path) != record.get("input_sha256"):
            errors.append(f"stale_input:{record['input_path']}")

        output_paths = record.get("output_paths")
        if not isinstance(output_paths, list):
            errors.append(f"output_paths:{record.get('doc_id')}")
            output_paths = []
        document_count = 0
        text_count = 0
        table_count = 0
        for output_path_value in output_paths:
            output_path = str(output_path_value)
            if output_path in output_paths_seen:
                errors.append(f"duplicate_output_path:{output_path}")
            output_paths_seen.add(output_path)
            path = data_root / Path(output_path)
            if not path.is_file():
                errors.append(f"missing:{output_path}")
                continue
            try:
                evidence = json.loads(path.read_text(encoding="utf-8"))
            except Exception as error:
                errors.append(f"json:{output_path}:{type(error).__name__}:{error}")
                continue

            document_count += 1
            evidence_id = str(evidence.get("id", ""))
            if not evidence_id:
                errors.append(f"id:{output_path}")
            elif evidence_id in evidence_ids:
                errors.append(f"duplicate_id:{output_path}:{evidence_id}")
            evidence_ids.add(evidence_id)

            errors.extend(
                _validate_evidence(
                    evidence,
                    record=record,
                    output_path=output_path,
                )
            )
            kind = (evidence.get("evidence_type"), evidence.get("table_type"))
            type_counts[kind] += 1
            if kind[0] == "TEXT":
                text_count += 1
            elif kind[0] == "TABLE":
                table_count += 1
            if len(errors) >= max_errors:
                break

        total_evidence += document_count
        if document_count > largest_evidence_count[0]:
            largest_evidence_count = (document_count, str(record.get("doc_id")))
        if record.get("n_evidence") != document_count:
            errors.append(f"manifest_count:{record.get('doc_id')}")
        if record.get("n_text") != text_count:
            errors.append(f"manifest_text_count:{record.get('doc_id')}")
        if record.get("n_table") != table_count:
            errors.append(f"manifest_table_count:{record.get('doc_id')}")
        if len(errors) >= max_errors:
            break
        if progress_every > 0 and index % progress_every == 0:
            print(
                f"validated={index}/{len(records)} evidence={total_evidence}",
                flush=True,
            )

    return {
        "documents": len(records),
        "sources": total_sources,
        "status_counts": dict(status_counts),
        "type_counts": [
            {
                "evidence_type": kind[0],
                "table_type": kind[1],
                "count": count,
            }
            for kind, count in sorted(
                type_counts.items(),
                key=lambda item: str(item[0]),
            )
        ],
        "total_evidence": total_evidence,
        "images_skipped": total_images,
        "largest_evidence_count": largest_evidence_count,
        "errors_count": len(errors),
        "errors": errors[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/canonical_evidence/manifest.jsonl"),
    )
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()
    result = validate_canonical_evidence(
        data_root=args.data_root,
        manifest_path=args.manifest,
        progress_every=args.progress_every,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["errors_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = ["validate_canonical_evidence"]
