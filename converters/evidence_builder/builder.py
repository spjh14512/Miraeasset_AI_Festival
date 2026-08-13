"""Materialize final evidence from data/canonical_section into canonical_evidence."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from converters.common.source_models import DocumentContext
from converters.evidence_builder.evidence_builder import build_source_evidence


OUTPUT_SCHEMA_VERSION = "canonical-evidence.v1"
MANIFEST_SCHEMA_VERSION = "canonical-evidence-manifest.v1"


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
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    """Atomically stream JSONL without materializing the full corpus in memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            )
            target.write("\n")
    temporary.replace(path)


def _overall_status(statuses: list[str]) -> str:
    if not statuses or all(status == "FAILED" for status in statuses):
        return "FAILED"
    if any(status in {"FAILED", "PARTIAL"} for status in statuses):
        return "PARTIAL"
    if any(status == "RECOVERED" for status in statuses):
        return "RECOVERED"
    return "SUCCESS"


def _safe_path_part(value: str) -> str:
    """Return a stable Windows-safe path component."""
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return sanitized or "unknown"


def _evidence_filename(
    evidence: Mapping[str, Any],
    *,
    source_index: int,
) -> str:
    section_id = str(evidence["section_id"])
    source_prefix = f"src{source_index}:"
    if section_id.startswith(source_prefix):
        section_id = section_id[len(source_prefix) :]
    return (
        f"src{source_index}__{_safe_path_part(section_id)}"
        f"__e{int(evidence['order'])}.json"
    )


def _evidence_output(
    evidence: Mapping[str, Any],
    *,
    canonical: Mapping[str, Any],
    input_relative: str,
    input_sha256: str,
    source_index: int,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    source_document = canonical["source_document"]
    result = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "doc_id": str(source_document["doc_id"]),
        "rcept_no": str(source_document["rcept_no"]),
        "doc_group": str(source_document.get("doc_group") or ""),
        "source_index": source_index,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "canonical_section_path": input_relative,
        "canonical_section_sha256": input_sha256,
        **dict(evidence),
    }
    result["id"] = f"{source_document['rcept_no']}:{evidence['id']}"
    return result


def _remove_stale_outputs(
    *,
    data_root: Path,
    output_root: Path,
    previous: Mapping[str, Any] | None,
    current_paths: set[str],
) -> None:
    if previous is None:
        return
    output_root_resolved = output_root.resolve()
    for relative in previous.get("output_paths", []):
        relative_text = str(relative)
        if relative_text in current_paths:
            continue
        path = (data_root / Path(relative_text)).resolve()
        if output_root_resolved not in path.parents:
            continue
        if path.is_file():
            path.unlink()


def _record(
    row: Mapping[str, Any],
    *,
    input_relative: str,
    input_sha256: str | None,
    output_paths: list[str],
    status: str,
    stats: Mapping[str, int] | None = None,
    error: str | None = None,
    issues: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    counts = stats or {}
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "doc_id": str(row.get("doc_id", "")),
        "rcept_no": str(row.get("rcept_no", "")),
        "doc_group": str(row.get("doc_group", "")),
        "input_path": input_relative,
        "input_sha256": input_sha256,
        "output_paths": output_paths,
        "status": status,
        "n_sources": counts.get("n_sources", 0),
        "n_evidence": counts.get("n_evidence", 0),
        "n_text": counts.get("n_text", 0),
        "n_table": counts.get("n_table", 0),
        "n_images_skipped": counts.get("n_images_skipped", 0),
        "error": error,
        "issues": [dict(issue) for issue in issues or []],
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
            return index, _record(
                row,
                input_relative=input_relative,
                input_sha256=None,
                output_paths=[],
                status="FAILED",
                error=f"Canonical section input does not exist: {input_path}",
            ), "failed"
        input_sha256 = _sha256(input_path)
        if (
            not force
            and previous is not None
            and previous.get("schema_version") == MANIFEST_SCHEMA_VERSION
            and previous.get("input_sha256") == input_sha256
            and previous.get("status") in {"SUCCESS", "RECOVERED"}
            and all(
                (data_root / Path(str(path))).is_file()
                for path in previous.get("output_paths", [])
            )
            and len(previous.get("output_paths", []))
            == int(previous.get("n_evidence", -1))
        ):
            return index, previous, "reused"

        canonical = json.loads(input_path.read_text(encoding="utf-8"))
        if canonical.get("schema_version") != "canonical-section-document.v1":
            raise ValueError("Unsupported canonical section schema")

        source_outputs: list[dict[str, Any]] = []
        statuses: list[str] = []
        aggregate = {
            "n_sources": 0,
            "n_evidence": 0,
            "n_text": 0,
            "n_table": 0,
            "n_images_skipped": 0,
        }
        output_paths: list[str] = []
        for source_index, source_entry in enumerate(canonical.get("sources", [])):
            relative_path = str(source_entry["source_path"])
            raw_path = data_root / Path(relative_path)
            expected_hash = str(source_entry["source_sha256"])
            if not raw_path.is_file():
                result = {
                    "source_path": relative_path,
                    "source_sha256": expected_hash,
                    "status": "FAILED",
                    "evidence": [],
                    "issues": [
                        {
                            "code": "SOURCE_NOT_FOUND",
                            "severity": "ERROR",
                            "message": f"Raw source does not exist: {raw_path}",
                        }
                    ],
                    "stats": {"TEXT": 0, "TABLE": 0, "IMAGE_SKIPPED": 0},
                }
            else:
                actual_hash = _sha256(raw_path)
                if actual_hash != expected_hash:
                    result = {
                        "source_path": relative_path,
                        "source_sha256": expected_hash,
                        "status": "FAILED",
                        "evidence": [],
                        "issues": [
                            {
                                "code": "STALE_INPUT",
                                "severity": "ERROR",
                                "message": (
                                    f"Raw SHA-256 is {actual_hash}; canonical section "
                                    f"expects {expected_hash}."
                                ),
                            }
                        ],
                        "stats": {"TEXT": 0, "TABLE": 0, "IMAGE_SKIPPED": 0},
                    }
                else:
                    source_document = source_entry["section_collection"][
                        "source_document"
                    ]
                    context = DocumentContext(
                        doc_id=str(source_document["doc_id"]),
                        rcept_no=str(source_document["rcept_no"]),
                        source_path=relative_path,
                        doc_group=source_document.get("doc_group"),
                    )
                    result = build_source_evidence(
                        raw_path.read_bytes(),
                        section_collection=source_entry["section_collection"],
                        document_context=context,
                        source_index=source_index,
                    )
                    result = {
                        "source_path": relative_path,
                        "source_sha256": expected_hash,
                        **result,
                    }

            stats = result["stats"]
            source_outputs.append(result)
            statuses.append(str(result["status"]))
            aggregate["n_sources"] += 1
            aggregate["n_text"] += int(stats.get("TEXT", 0))
            aggregate["n_table"] += int(stats.get("TABLE", 0))
            aggregate["n_images_skipped"] += int(stats.get("IMAGE_SKIPPED", 0))

            for item in result["evidence"]:
                output_path = output_directory / _evidence_filename(
                    item,
                    source_index=source_index,
                )
                output_relative = output_path.relative_to(data_root).as_posix()
                output = _evidence_output(
                    item,
                    canonical=canonical,
                    input_relative=input_relative,
                    input_sha256=input_sha256,
                    source_index=source_index,
                    source_path=relative_path,
                    source_sha256=expected_hash,
                )
                _atomic_write(
                    output_path,
                    json.dumps(
                        output,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n",
                )
                output_paths.append(output_relative)
        aggregate["n_evidence"] = aggregate["n_text"] + aggregate["n_table"]
        status = _overall_status(statuses)
        _remove_stale_outputs(
            data_root=data_root,
            output_root=output_root,
            previous=previous,
            current_paths=set(output_paths),
        )
        return index, _record(
            row,
            input_relative=input_relative,
            input_sha256=input_sha256,
            output_paths=output_paths,
            status=status,
            stats=aggregate,
            issues=[
                issue
                for source in source_outputs
                for issue in source.get("issues", [])
            ],
        ), "failed" if status == "FAILED" else "generated"
    except Exception as error:
        return index, _record(
            row,
            input_relative=input_relative,
            input_sha256=None,
            output_paths=[],
            status="FAILED",
            error=f"{type(error).__name__}: {error}",
        ), "failed"


def _recover_one(
    index: int,
    row: dict[str, Any],
    data_root_text: str,
    output_root_text: str,
) -> tuple[int, dict[str, Any], str]:
    """Recover one manifest record from already materialized Evidence files."""
    data_root = Path(data_root_text)
    output_root = Path(output_root_text)
    input_relative = str(row["output_path"])
    input_path = data_root / Path(input_relative)
    output_directory = (
        output_root / str(row["doc_group"]) / str(row["rcept_no"])
    )
    try:
        if not input_path.is_file():
            raise FileNotFoundError(f"Canonical section input does not exist: {input_path}")
        input_sha256 = _sha256(input_path)
        canonical = json.loads(input_path.read_text(encoding="utf-8"))
        files = sorted(output_directory.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No Evidence output files found: {output_directory}")

        output_paths: list[str] = []
        text_count = 0
        table_count = 0
        evidence_ids: set[str] = set()
        recovery_issues: list[dict[str, Any]] = []
        recovered = False
        for path in files:
            evidence = json.loads(path.read_text(encoding="utf-8"))
            output_relative = path.relative_to(data_root).as_posix()
            output_paths.append(output_relative)
            if evidence.get("schema_version") != OUTPUT_SCHEMA_VERSION:
                raise ValueError(f"Unsupported Evidence schema: {output_relative}")
            if evidence.get("canonical_section_path") != input_relative:
                raise ValueError(f"Canonical section path mismatch: {output_relative}")
            if evidence.get("canonical_section_sha256") != input_sha256:
                raise ValueError(f"Canonical section hash mismatch: {output_relative}")
            if str(evidence.get("doc_id", "")) != str(row.get("doc_id", "")):
                raise ValueError(f"Document ID mismatch: {output_relative}")
            evidence_id = str(evidence.get("id", ""))
            if not evidence_id or evidence_id in evidence_ids:
                raise ValueError(f"Missing or duplicate Evidence ID: {output_relative}")
            evidence_ids.add(evidence_id)
            if not isinstance(evidence.get("payload"), dict) or not evidence["payload"]:
                raise ValueError(f"Missing Evidence payload: {output_relative}")
            if not isinstance(evidence.get("markdown"), str) or not evidence["markdown"].strip():
                raise ValueError(f"Missing Evidence markdown: {output_relative}")
            if evidence.get("evidence_type") == "TEXT":
                text_count += 1
            elif evidence.get("evidence_type") == "TABLE":
                table_count += 1
            else:
                raise ValueError(f"Unsupported Evidence type: {output_relative}")
            if evidence.get("parse_status") not in {None, "SUCCESS"}:
                recovered = True
            recovery_issues.extend(
                dict(issue) for issue in evidence.get("issues", [])
            )

        images_skipped = sum(
            str(block.get("block_type")) == "IMAGE"
            for source in canonical.get("sources", [])
            for section in source.get("section_collection", {}).get("sections", [])
            for block in section.get("blocks", [])
        )
        stats = {
            "n_sources": len(canonical.get("sources", [])),
            "n_evidence": len(output_paths),
            "n_text": text_count,
            "n_table": table_count,
            "n_images_skipped": images_skipped,
        }
        return index, _record(
            row,
            input_relative=input_relative,
            input_sha256=input_sha256,
            output_paths=output_paths,
            status="RECOVERED" if recovered else "SUCCESS",
            stats=stats,
            issues=recovery_issues,
        ), "recovered"
    except Exception as error:
        return index, _record(
            row,
            input_relative=input_relative,
            input_sha256=None,
            output_paths=[],
            status="FAILED",
            error=f"{type(error).__name__}: {error}",
        ), "failed"


def recover_canonical_evidence_manifest(
    *,
    section_manifest_path: Path,
    data_root: Path,
    output_root: Path,
    workers: int = 1,
    progress_every: int = 50,
) -> dict[str, int]:
    rows = _json_lines(section_manifest_path)
    results: list[dict[str, Any] | None] = [None] * len(rows)
    summary = {"recovered": 0, "failed": 0}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                _recover_one,
                index,
                row,
                str(data_root),
                str(output_root),
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
                    f"[{completed}/{len(rows)}] recovered={summary['recovered']} "
                    f"failed={summary['failed']}",
                    flush=True,
                )
    records = [record for record in results if record is not None]
    _atomic_write_jsonl(output_root / "manifest.jsonl", records)
    summary["total"] = len(records)
    return summary


def build_canonical_evidence(
    *,
    section_manifest_path: Path,
    data_root: Path,
    output_root: Path,
    workers: int = 1,
    force: bool = False,
    limit: int | None = None,
    progress_every: int = 50,
) -> dict[str, int]:
    rows = _json_lines(section_manifest_path)
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
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                _build_one,
                index,
                row,
                str(data_root),
                str(output_root),
                previous_records.get(str(row.get("doc_id"))),
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
    _atomic_write_jsonl(output_manifest, records)
    summary["total"] = len(records)
    return summary


def _arguments() -> argparse.Namespace:
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
        default=Path("data/canonical_evidence"),
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--recover-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.recover_manifest:
        summary = recover_canonical_evidence_manifest(
            section_manifest_path=args.section_manifest,
            data_root=args.data_root,
            output_root=args.output_root,
            workers=args.workers,
            progress_every=args.progress_every,
        )
    else:
        summary = build_canonical_evidence(
            section_manifest_path=args.section_manifest,
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


__all__ = [
    "build_canonical_evidence",
    "recover_canonical_evidence_manifest",
    "main",
]
