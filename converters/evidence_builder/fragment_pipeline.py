"""Materialize one JSON Evidence Fragment per canonical section."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from converters.common.document_loader import load_document
from converters.common.source_models import DocumentContext
from converters.evidence_builder.fragment_assembler import build_source_fragments
from converters.section_canonicalizer.section_canonicalizer import chunk_sections
from converters.section_canonicalizer.section_models import (
    CanonicalSection,
    SectionBoundaryKind,
)


MANIFEST_SCHEMA_VERSION = "evidence-fragment-manifest.v3"
EVIDENCE_BUILDER_VERSION = "evidence-builder.v1"
_SECTION_ID = re.compile(r"^section:(?P<rcept>\d+):src(?P<src>\d+):s(?P<section>\d+)$")


def _json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
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


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _combined_source_hash(source_files: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for source_path, source_hash in source_files:
        digest.update(source_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _kept_sections(
    sections: Iterable[CanonicalSection],
) -> tuple[CanonicalSection, ...]:
    source_sections = tuple(sections)
    has_non_synthetic = any(
        section.boundary_kind != SectionBoundaryKind.SYNTHETIC
        for section in source_sections
    )
    return tuple(
        section
        for section in source_sections
        if not (
            has_non_synthetic
            and section.boundary_kind == SectionBoundaryKind.SYNTHETIC
        )
    )


def project_graph_sections(
    sections: Iterable[CanonicalSection],
    *,
    rcept_no: str,
    source_index: int,
    order_start: int = 0,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Project parser sections with the exact canonical-section v2 rules."""
    source_sections = tuple(sections)
    kept = _kept_sections(source_sections)
    kept_ids = {section.id for section in kept}
    by_id = {section.id: section for section in source_sections}
    prefix = f"section:{rcept_no}:src{source_index}:"

    def kept_parent(section: CanonicalSection) -> str | None:
        parent_id = section.parent_section_id
        while parent_id is not None and parent_id not in kept_ids:
            parent = by_id.get(parent_id)
            parent_id = parent.parent_section_id if parent is not None else None
        return f"{prefix}{parent_id}" if parent_id is not None else None

    projected = [
        {
            "section_id": f"{prefix}{section.id}",
            "parent_section_id": kept_parent(section),
            "order": order_start + index,
            "title": section.title,
            "section_path": list(section.section_path),
            "element_path": (
                section.source_ref.element_path
                if section.source_ref is not None
                else None
            ),
        }
        for index, section in enumerate(kept)
    ]
    return projected, kept_ids


def _fragment_filename(section_id: str) -> str:
    match = _SECTION_ID.fullmatch(section_id)
    if match is None:
        raise ValueError(f"Unsupported section ID: {section_id}")
    return f"src{match.group('src')}__s{match.group('section')}.json"


def _overall_status(statuses: Iterable[str]) -> str:
    values = tuple(statuses)
    if not values or all(value == "FAILED" for value in values):
        return "FAILED"
    if any(value in {"FAILED", "PARTIAL"} for value in values):
        return "PARTIAL"
    if any(value == "RECOVERED" for value in values):
        return "RECOVERED"
    return "SUCCESS"


def _failed_record(
    row: Mapping[str, Any],
    *,
    input_path: str,
    message: str,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "builder_version": EVIDENCE_BUILDER_VERSION,
        "doc_id": row.get("doc_id"),
        "rcept_no": row.get("rcept_no"),
        "doc_group": row.get("doc_group"),
        "input_path": input_path,
        "input_sha256": None,
        "source_sha256": None,
        "status": "FAILED",
        "output_paths": [],
        "n_fragments": 0,
        "n_evidence": 0,
        "n_records": 0,
        "stats": {},
        "issues": [],
        "error": message,
    }


def _build_one(
    index: int,
    row: dict[str, Any],
    data_root_text: str,
    output_root_text: str,
    previous: dict[str, Any] | None,
    force: bool,
) -> tuple[int, dict[str, Any], str]:
    data_root = Path(data_root_text)
    output_root = Path(output_root_text)
    input_relative = str(row["output_path"])
    input_path = data_root / Path(input_relative)
    output_directory = (
        output_root / str(row["doc_group"]) / str(row["rcept_no"])
    )
    try:
        if not input_path.is_file():
            raise FileNotFoundError(f"Canonical section file not found: {input_path}")
        input_hash = _sha256(input_path)
        source_descriptors = []
        for relative in row.get("source_files", []):
            path = data_root / Path(str(relative))
            if not path.is_file():
                raise FileNotFoundError(f"Source file not found: {path}")
            source_descriptors.append((path, str(relative), _sha256(path)))
        source_hash = _combined_source_hash(
            (relative, digest)
            for _, relative, digest in source_descriptors
        )
        if source_hash != row.get("source_sha256"):
            raise ValueError(
                "Raw source hash does not match canonical_section manifest."
            )

        if (
            not force
            and previous is not None
            and previous.get("schema_version") == MANIFEST_SCHEMA_VERSION
            and previous.get("builder_version") == EVIDENCE_BUILDER_VERSION
            and previous.get("input_sha256") == input_hash
            and previous.get("source_sha256") == source_hash
            and previous.get("status") in {"SUCCESS", "RECOVERED", "PARTIAL"}
            and all(
                (data_root / Path(str(path))).is_file()
                for path in previous.get("output_paths", [])
            )
        ):
            return index, previous, "reused"

        canonical = json.loads(input_path.read_text(encoding="utf-8"))
        expected_sections = canonical.get("sections", [])
        projected_sections: list[dict[str, Any]] = []
        source_work: list[tuple[bytes, Any, set[str], DocumentContext, int]] = []

        for source_index, (path, relative, _) in enumerate(source_descriptors):
            source = path.read_bytes()
            context = DocumentContext(
                doc_id=str(row["doc_id"]),
                rcept_no=str(row["rcept_no"]),
                source_path=relative,
                doc_group=str(row["doc_group"]),
            )
            loaded = load_document(source)
            if loaded.root is None:
                raise ValueError(f"Could not load source: {relative}")
            collection = chunk_sections(
                loaded.root,
                syntax=loaded.syntax,
                document_context=context,
            )
            projection, kept_ids = project_graph_sections(
                collection.sections,
                rcept_no=context.rcept_no,
                source_index=source_index,
                order_start=len(projected_sections),
            )
            projected_sections.extend(projection)
            source_work.append(
                (source, collection, kept_ids, context, source_index)
            )

        if projected_sections != expected_sections:
            raise ValueError(
                "Re-chunked graph projection does not match canonical_section."
            )

        fragments: list[dict[str, Any]] = []
        statuses: list[str] = []
        issues: list[dict[str, Any]] = []
        stats: Counter[str] = Counter()
        for source, collection, kept_ids, context, source_index in source_work:
            result = build_source_fragments(
                source,
                section_collection=collection,
                kept_section_ids=kept_ids,
                document_context=context,
                source_index=source_index,
            )
            statuses.append(str(result["status"]))
            fragments.extend(result["fragments"])
            issues.extend(result["issues"])
            stats.update(result["stats"])

        expected_ids = [str(item["section_id"]) for item in expected_sections]
        actual_ids = [str(item["section_id"]) for item in fragments]
        if actual_ids != expected_ids:
            raise ValueError("Fragment section order does not match canonical_section.")

        output_paths: list[str] = []
        for fragment in fragments:
            output_path = output_directory / _fragment_filename(fragment["section_id"])
            output_relative = output_path.relative_to(data_root).as_posix()
            _atomic_write(
                output_path,
                json.dumps(fragment, ensure_ascii=False, separators=(",", ":"))
                + "\n",
            )
            output_paths.append(output_relative)

        current_names = {Path(path).name for path in output_paths}
        if output_directory.is_dir():
            for stale in output_directory.glob("*.json"):
                if stale.name not in current_names:
                    stale.unlink()

        status = _overall_status(statuses)
        record = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "builder_version": EVIDENCE_BUILDER_VERSION,
            "doc_id": row["doc_id"],
            "rcept_no": row["rcept_no"],
            "doc_group": row["doc_group"],
            "input_path": input_relative,
            "input_sha256": input_hash,
            "source_sha256": source_hash,
            "status": status,
            "output_paths": output_paths,
            "n_fragments": len(fragments),
            "n_evidence": sum(len(item["evidence_list"]) for item in fragments),
            "n_records": sum(len(item["records"]) for item in fragments),
            "stats": dict(stats),
            "issues": issues[:100],
            "issues_truncated": max(0, len(issues) - 100),
            "error": None,
        }
        return index, record, "generated"
    except Exception as error:
        return (
            index,
            _failed_record(
                row,
                input_path=input_relative,
                message=f"{type(error).__name__}: {error}",
            ),
            "failed",
        )


def build_evidence_fragments(
    *,
    section_manifest_path: Path,
    data_root: Path,
    output_root: Path,
    workers: int = 1,
    force: bool = False,
    limit: int | None = None,
    doc_ids: set[str] | None = None,
    progress_every: int = 50,
) -> dict[str, int]:
    rows = _json_lines(section_manifest_path)
    if doc_ids is not None:
        rows = [row for row in rows if str(row.get("doc_id")) in doc_ids]
    if limit is not None:
        rows = rows[:limit]
    manifest_path = output_root / "manifest.jsonl"
    previous = {
        str(record.get("doc_id")): record
        for record in _json_lines(manifest_path)
    }
    results: list[dict[str, Any] | None] = [None] * len(rows)
    summary = {"generated": 0, "reused": 0, "failed": 0}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                _build_one,
                index,
                row,
                str(data_root),
                str(output_root),
                previous.get(str(row.get("doc_id"))),
                force,
            )
            for index, row in enumerate(rows)
        ]
        completed = 0
        for future in as_completed(futures):
            index, record, action = future.result()
            results[index] = record
            summary[action] += 1
            completed += 1
            if progress_every > 0 and (
                completed % progress_every == 0 or completed == len(rows)
            ):
                print(
                    f"[{completed}/{len(rows)}] generated={summary['generated']} "
                    f"reused={summary['reused']} failed={summary['failed']}",
                    flush=True,
                )
    records = [record for record in results if record is not None]
    _atomic_write(
        manifest_path,
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
    )
    summary["total"] = len(records)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section-manifest",
        type=Path,
        default=Path("data/canonical_section/manifest.jsonl"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/evidence_fragment"),
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--doc-id",
        action="append",
        dest="doc_ids",
        help="Build only this document ID; may be supplied multiple times.",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    summary = build_evidence_fragments(
        section_manifest_path=args.section_manifest,
        data_root=args.data_root,
        output_root=args.output_root,
        workers=args.workers,
        force=args.force,
        limit=args.limit,
        doc_ids=(set(args.doc_ids) if args.doc_ids else None),
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "EVIDENCE_BUILDER_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "build_evidence_fragments",
    "main",
    "project_graph_sections",
]
