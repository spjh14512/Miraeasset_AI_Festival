"""Materialize section canonicalizer output for data/manifest.jsonl."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from converters.common.source_models import DocumentContext
from converters.section_canonicalizer.section_canonicalizer import chunk_sections
from converters.section_canonicalizer.section_models import SectionBoundaryKind


OUTPUT_SCHEMA_VERSION = "canonical-section-document.v2"
PROCESSABLE_SUFFIXES = {".xml", ".html", ".htm"}


def _json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(
    source_directory: Path,
    rcept_no: str,
) -> tuple[list[Path], list[Path]]:
    files = sorted(path for path in source_directory.iterdir() if path.is_file())
    processable = [
        path for path in files if path.suffix.lower() in PROCESSABLE_SUFFIXES
    ]
    processable.sort(
        key=lambda path: (
            0 if path.stem == rcept_no else 1,
            path.name.lower(),
        )
    )
    skipped = [path for path in files if path not in processable]
    return processable, skipped


def _combined_source_hash(
    source_files: Iterable[tuple[str, str]],
) -> str:
    digest = hashlib.sha256()
    for source_path, source_hash in source_files:
        digest.update(source_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _overall_status(statuses: list[str]) -> str:
    if not statuses or all(status == "FAILED" for status in statuses):
        return "FAILED"
    if any(status == "FAILED" for status in statuses):
        return "PARTIAL"
    if any(status == "RECOVERED" for status in statuses):
        return "RECOVERED"
    return "SUCCESS"


def _append_section_outputs(
    section_outputs: list[dict[str, Any]],
    sections: Iterable[Any],
    *,
    rcept_no: str,
    source_index: int,
) -> None:
    """Append graph-facing sections, omitting parser-only synthetic roots."""
    source_sections = list(sections)
    has_non_synthetic = any(
        section.boundary_kind != SectionBoundaryKind.SYNTHETIC
        for section in source_sections
    )
    kept_sections = [
        section
        for section in source_sections
        if not (
            has_non_synthetic
            and section.boundary_kind == SectionBoundaryKind.SYNTHETIC
        )
    ]
    kept_ids = {section.id for section in kept_sections}
    sections_by_id = {section.id: section for section in source_sections}
    id_prefix = f"section:{rcept_no}:src{source_index}:"

    def kept_parent_id(section: Any) -> str | None:
        parent_id = section.parent_section_id
        while parent_id is not None and parent_id not in kept_ids:
            parent = sections_by_id.get(parent_id)
            parent_id = parent.parent_section_id if parent is not None else None
        return f"{id_prefix}{parent_id}" if parent_id is not None else None

    for section in kept_sections:
        section_outputs.append(
            {
                "section_id": f"{id_prefix}{section.id}",
                "parent_section_id": kept_parent_id(section),
                "order": len(section_outputs),
                "title": section.title,
                "section_path": list(section.section_path),
                "element_path": (
                    section.source_ref.element_path
                    if section.source_ref is not None
                    else None
                ),
            }
        )


def _failed_record(
    row: Mapping[str, Any],
    output_relative: str,
    message: str,
) -> dict[str, Any]:
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "doc_id": row["doc_id"],
        "rcept_no": row["rcept_no"],
        "doc_group": row["doc_group"],
        "source_path": row["file_path"],
        "source_sha256": None,
        "source_files": [],
        "skipped_files": [],
        "output_path": output_relative,
        "status": "FAILED",
        "n_sources": 0,
        "n_sections": 0,
        "n_blocks": 0,
        "error": message,
    }


def _completed_record(
    *,
    row: Mapping[str, Any],
    output_relative: str,
    combined_hash: str,
    source_descriptors: list[tuple[Path, str, str]],
    skipped: list[Path],
    data_root: Path,
    statuses: list[str],
    n_sections: int,
    n_blocks: int,
) -> dict[str, Any]:
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "doc_id": str(row["doc_id"]),
        "rcept_no": str(row["rcept_no"]),
        "doc_group": str(row["doc_group"]),
        "source_path": str(row["file_path"]),
        "source_sha256": combined_hash,
        "source_files": [
            relative_path for _, relative_path, _ in source_descriptors
        ],
        "skipped_files": [
            path.relative_to(data_root).as_posix() for path in skipped
        ],
        "output_path": output_relative,
        "status": _overall_status(statuses),
        "n_sources": len(source_descriptors),
        "n_sections": n_sections,
        "n_blocks": n_blocks,
        "error": None,
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
    rcept_no = str(row["rcept_no"])
    output_path = (
        output_root
        / str(row["doc_group"])
        / f"sections_{rcept_no}.json"
    )
    output_relative = output_path.relative_to(data_root).as_posix()
    source_directory = data_root / Path(str(row["file_path"]))

    try:
        if not source_directory.is_dir():
            return index, _failed_record(
                row,
                output_relative,
                f"Source directory does not exist: {source_directory}",
            ), "failed"

        processable, skipped = _source_files(source_directory, rcept_no)
        if not processable:
            return index, _failed_record(
                row,
                output_relative,
                "No XML or HTML source file was found.",
            ), "failed"

        source_descriptors = [
            (
                path,
                path.relative_to(data_root).as_posix(),
                _sha256(path),
            )
            for path in processable
        ]
        combined_hash = _combined_source_hash(
            (relative_path, source_hash)
            for _, relative_path, source_hash in source_descriptors
        )
        if (
            not force
            and previous is not None
            and previous.get("schema_version") == OUTPUT_SCHEMA_VERSION
            and previous.get("source_sha256") == combined_hash
            and previous.get("status") in {"SUCCESS", "RECOVERED"}
            and output_path.is_file()
        ):
            return index, previous, "reused"
        section_outputs: list[dict[str, Any]] = []
        statuses: list[str] = []
        n_sections = 0
        n_blocks = 0
        for source_index, (path, relative_path, source_hash) in enumerate(
            source_descriptors
        ):
            context = DocumentContext(
                doc_id=str(row["doc_id"]),
                rcept_no=rcept_no,
                source_path=relative_path,
                doc_group=str(row["doc_group"]),
            )
            collection = chunk_sections(
                path.read_bytes(),
                document_context=context,
            )
            statuses.append(collection.parse_status.value)
            n_sections += len(collection.sections)
            n_blocks += sum(len(section.blocks) for section in collection.sections)
            _append_section_outputs(
                section_outputs,
                collection.sections,
                rcept_no=rcept_no,
                source_index=source_index,
            )

        document_output = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "source_doc_id": str(row["doc_id"]),
            "sections": section_outputs,
        }
        _atomic_write_text(
            output_path,
            json.dumps(document_output, ensure_ascii=False, indent=2) + "\n",
        )
        record = _completed_record(
            row=row,
            output_relative=output_relative,
            combined_hash=combined_hash,
            source_descriptors=source_descriptors,
            skipped=skipped,
            data_root=data_root,
            statuses=statuses,
            n_sections=n_sections,
            n_blocks=n_blocks,
        )
        return index, record, "generated"
    except Exception as error:
        return index, _failed_record(
            row,
            output_relative,
            f"{type(error).__name__}: {error}",
        ), "failed"


def build_canonical_sections(
    *,
    manifest_path: Path,
    data_root: Path,
    output_root: Path,
    workers: int = 1,
    force: bool = False,
    limit: int | None = None,
    progress_every: int = 50,
) -> dict[str, int]:
    rows = _json_lines(manifest_path)
    if limit is not None:
        rows = rows[:limit]
    output_manifest = output_root / "manifest.jsonl"
    previous_records = {
        str(record["doc_id"]): record
        for record in _json_lines(output_manifest)
        if "doc_id" in record
    }

    results: list[dict[str, Any] | None] = [None] * len(rows)
    summary = {"generated": 0, "reused": 0, "failed": 0}
    worker_count = max(1, workers)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _build_one,
                index,
                row,
                str(data_root),
                str(output_root),
                previous_records.get(str(row["doc_id"])),
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
                    f"[{completed}/{len(rows)}] "
                    f"generated={summary['generated']} "
                    f"reused={summary['reused']} failed={summary['failed']}",
                    flush=True,
                )

    records = [record for record in results if record is not None]
    _atomic_write_text(
        output_manifest,
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
    )
    summary["total"] = len(records)
    return summary


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/canonical_section"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    summary = build_canonical_sections(
        manifest_path=args.manifest,
        data_root=args.data_root,
        output_root=args.output_root,
        workers=args.workers,
        force=args.force,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
