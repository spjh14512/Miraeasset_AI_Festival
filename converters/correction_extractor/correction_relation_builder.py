"""Extract correction metadata and resolve original disclosure receipts in bulk."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
import re
from typing import Any

from converters.common.source_models import DocumentContext
from converters.correction_extractor.correction_extractor import (
    extract_correction,
    resolve_correction_target,
)
from converters.correction_extractor.correction_models import (
    CorrectionExtraction,
    CorrectionStatus,
    CorrectionTargetCandidate,
)


SOURCE_SUFFIXES = (".xml", ".html", ".htm")


def read_manifest_rows(path: Path) -> tuple[dict[str, Any], ...]:
    """Read the source manifest and report malformed JSON with its line number."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid manifest JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row must be an object at {path}:{line_number}")
            rows.append(row)
    return tuple(rows)


def _manifest_date(value: object) -> str | None:
    """Convert the manifest's YYYYMMDD date to the resolver's ISO date format."""
    text = str(value or "").strip()
    if re.fullmatch(r"20\d{6}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", text):
        return text
    return None


def _company_key(row: Mapping[str, Any]) -> str | None:
    """Use filer identity as well as issuer identity for 5% holding reports."""
    corp_code = str(row.get("corp_code", "")).strip()
    if not corp_code:
        return None
    if str(row.get("doc_group", "")) != "holding":
        return corp_code
    filer = re.sub(r"\W+", "", str(row.get("flr_nm", "")).casefold())
    return f"{corp_code}:{filer}" if filer else corp_code


def _target_candidates(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[CorrectionTargetCandidate, ...]:
    candidates: list[CorrectionTargetCandidate] = []
    for row in rows:
        if row.get("is_correction") is True:
            continue
        rcept_no = str(row.get("rcept_no", "")).strip()
        submission_date = _manifest_date(row.get("rcept_dt"))
        if not rcept_no or submission_date is None:
            continue
        report_name = str(row.get("report_nm", "")).strip() or None
        candidates.append(
            CorrectionTargetCandidate(
                rcept_no=rcept_no,
                submission_date=submission_date,
                document_name=report_name,
                company_key=_company_key(row),
                doc_group=str(row.get("doc_group", "")).strip() or None,
            )
        )
    return tuple(candidates)


def _source_file(data_root: Path, row: Mapping[str, Any]) -> Path:
    rcept_no = str(row.get("rcept_no", "")).strip()
    source_path = data_root / str(row.get("file_path", ""))
    if source_path.is_file():
        return source_path
    if not source_path.is_dir():
        raise FileNotFoundError(f"Source path is missing for {rcept_no}: {source_path}")

    for suffix in SOURCE_SUFFIXES:
        exact = source_path / f"{rcept_no}{suffix}"
        if exact.is_file():
            return exact

    source_files = sorted(
        path for path in source_path.iterdir() if path.suffix.lower() in SOURCE_SUFFIXES
    )
    if len(source_files) == 1:
        return source_files[0]
    raise FileNotFoundError(
        f"Cannot choose one source document for {rcept_no}: {source_path}"
    )


def build_correction_results(
    manifest_rows: Iterable[Mapping[str, Any]],
    *,
    data_root: Path,
) -> tuple[CorrectionExtraction, ...]:
    """Run the existing extractor and resolver for each correction disclosure."""
    rows = tuple(manifest_rows)
    candidates = _target_candidates(rows)
    results: list[CorrectionExtraction] = []

    for row in rows:
        if row.get("is_correction") is not True:
            continue
        rcept_no = str(row.get("rcept_no", "")).strip()
        doc_group = str(row.get("doc_group", "")).strip()
        if not rcept_no or not doc_group:
            raise ValueError("Correction manifest rows require rcept_no and doc_group")
        source_file = _source_file(data_root, row)
        try:
            display_path = str(source_file.relative_to(data_root))
        except ValueError:
            display_path = str(source_file)
        context = DocumentContext(
            doc_id=str(row.get("doc_id", "")).strip() or f"{doc_group}_{rcept_no}",
            rcept_no=rcept_no,
            source_path=display_path,
            doc_group=doc_group,
        )
        extraction = extract_correction(source_file.read_bytes(), context=context)
        results.append(
            resolve_correction_target(
                extraction,
                candidates,
                company_key=_company_key(row),
            )
        )
    return tuple(results)


def write_results(results: Iterable[CorrectionExtraction], path: Path) -> None:
    """Write existing correction.v1 records without introducing another model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for result in results:
            output.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    temporary.replace(path)


def materialize_correction_results(
    *,
    manifest_path: Path = Path("data/manifest.jsonl"),
    data_root: Path = Path("data"),
    output_path: Path = Path("data/correction/manifest.jsonl"),
) -> dict[str, int]:
    results = build_correction_results(
        read_manifest_rows(manifest_path),
        data_root=data_root,
    )
    write_results(results, output_path)
    resolved = sum(
        result.correction is not None
        and result.correction.target_rcept_no is not None
        for result in results
    )
    failed = sum(result.status == CorrectionStatus.FAILED for result in results)
    return {
        "total": len(results),
        "resolved": resolved,
        "unresolved": len(results) - resolved - failed,
        "failed": failed,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract corrections and resolve their original disclosure receipts."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/correction/manifest.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    summary = materialize_correction_results(
        manifest_path=args.manifest,
        data_root=args.data_root,
        output_path=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "build_correction_results",
    "materialize_correction_results",
    "read_manifest_rows",
    "write_results",
]
