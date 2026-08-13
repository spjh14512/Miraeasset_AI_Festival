"""Isolated fixed-corpus runner for the DART converter pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from typing import Any, Iterable, Mapping

from converters.evidence_builder.fragment_pipeline import build_evidence_fragments
from converters.evidence_builder.fragment_validator import validate_evidence_fragments
from converters.regression_validation.assertions import evaluate_case
from converters.regression_validation.normalizer import case_snapshot
from scripts.build_canonical_sections import build_canonical_sections


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "converter_pipeline"
PRODUCTION_OUTPUTS = (
    Path("data/canonical_section"),
    Path("data/evidence_fragment"),
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_signature(path: Path) -> tuple[int, int, int]:
    """Cheap guard against accidental writes to production converter outputs."""
    if not path.exists():
        return (0, 0, 0)
    count = total_size = latest_mtime = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        stat = item.stat()
        count += 1
        total_size += stat.st_size
        latest_mtime = max(latest_mtime, stat.st_mtime_ns)
    return count, total_size, latest_mtime


def _selected_cases(
    corpus: Mapping[str, Any],
    *,
    profile: str,
    case_ids: set[str] | None,
) -> list[dict[str, Any]]:
    cases = list(corpus["cases"])
    if case_ids:
        selected = [case for case in cases if case["case_id"] in case_ids]
        missing = sorted(case_ids - {case["case_id"] for case in selected})
        if missing:
            raise ValueError(f"Unknown case IDs: {', '.join(missing)}")
        return selected
    return [case for case in cases if profile in case.get("profiles", ["full"])]


def _hydrate_sources(
    fixture_root: Path,
    data_root: Path,
    documents: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    for document in documents:
        row = dict(document["manifest"])
        destination = data_root / Path(str(row["file_path"]))
        destination.mkdir(parents=True, exist_ok=True)
        for source in document["sources"]:
            archive = fixture_root / str(source["archive"])
            value = gzip.decompress(archive.read_bytes())
            actual_hash = _sha256_bytes(value)
            if actual_hash != source["sha256"]:
                raise ValueError(
                    f"Fixture source hash mismatch: {archive} "
                    f"expected={source['sha256']} actual={actual_hash}"
                )
            (destination / str(source["filename"])).write_bytes(value)
        manifest_rows.append(row)
    return manifest_rows


def _load_fragments(
    data_root: Path,
    section_ids: Iterable[str],
) -> dict[str, dict[str, Any]]:
    requested = set(section_ids)
    found: dict[str, dict[str, Any]] = {}
    for path in (data_root / "evidence_fragment").rglob("*.json"):
        fragment = _read_json(path)
        section_id = str(fragment.get("section_id", ""))
        if section_id in requested:
            found[section_id] = fragment
    missing = sorted(requested - found.keys())
    if missing:
        raise ValueError(f"Missing generated sections: {', '.join(missing)}")
    return found


def _write_candidate(
    candidate_dir: Path,
    cases: Iterable[Mapping[str, Any]],
    fragments: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for case in cases:
        path = candidate_dir / f"{case['case_id']}.json"
        if path.exists():
            raise FileExistsError(
                f"Candidate already exists; refusing to overwrite: {path}"
            )
        path.write_text(
            json.dumps(case_snapshot(case, fragments), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        written.append(str(path))
    return written


def _compare_golden(
    golden_dir: Path,
    cases: Iterable[Mapping[str, Any]],
    fragments: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for case in cases:
        case_id = str(case["case_id"])
        path = golden_dir / f"{case_id}.json"
        if not path.is_file():
            failures.append(
                {
                    "case_id": case_id,
                    "code": "GOLDEN_MISSING",
                    "message": f"Approved golden is missing: {path}",
                }
            )
            continue
        expected = _read_json(path)
        actual = case_snapshot(case, fragments)
        if actual != expected:
            failures.append(
                {
                    "case_id": case_id,
                    "code": "SNAPSHOT_MISMATCH",
                    "message": f"Normalized snapshot differs from {path}",
                }
            )
    return failures


def run_regression(
    *,
    project_root: Path = PROJECT_ROOT,
    fixture_root: Path = DEFAULT_FIXTURE_ROOT,
    profile: str = "full",
    case_ids: set[str] | None = None,
    semantic_only: bool = False,
    candidate_dir: Path | None = None,
    keep_workdir: Path | None = None,
) -> dict[str, Any]:
    """Run fixtures in an isolated data root and return a JSON-ready report."""
    started_at = perf_counter()
    corpus = _read_json(fixture_root / "corpus.json")
    cases = _selected_cases(corpus, profile=profile, case_ids=case_ids)
    selected_doc_ids = {
        str(doc_id) for case in cases for doc_id in case["document_ids"]
    }
    documents = [
        document
        for document in corpus["documents"]
        if document["manifest"]["doc_id"] in selected_doc_ids
    ]
    missing_docs = selected_doc_ids - {
        str(document["manifest"]["doc_id"]) for document in documents
    }
    if missing_docs:
        raise ValueError(f"Missing fixture documents: {', '.join(sorted(missing_docs))}")
    documents_by_id = {
        str(document["manifest"]["doc_id"]): document for document in documents
    }
    cases = [
        {
            **case,
            "source_sha256": [
                str(source["sha256"])
                for doc_id in case["document_ids"]
                for source in documents_by_id[str(doc_id)]["sources"]
            ],
        }
        for case in cases
    ]

    production_before = {
        str(relative): _tree_signature(project_root / relative)
        for relative in PRODUCTION_OUTPUTS
    }
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if keep_workdir is None:
        temporary = tempfile.TemporaryDirectory(prefix="converter-regression-")
        workdir = Path(temporary.name)
    else:
        if keep_workdir.exists():
            raise FileExistsError(
                f"Refusing to reuse or overwrite work directory: {keep_workdir}"
            )
        keep_workdir.mkdir(parents=True)
        workdir = keep_workdir

    try:
        data_root = workdir / "data"
        rows = _hydrate_sources(fixture_root, data_root, documents)
        manifest = data_root / "manifest.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        canonical_summary = build_canonical_sections(
            manifest_path=manifest,
            data_root=data_root,
            output_root=data_root / "canonical_section",
            workers=1,
            force=True,
            progress_every=0,
        )
        fragment_summary = build_evidence_fragments(
            section_manifest_path=data_root / "canonical_section" / "manifest.jsonl",
            data_root=data_root,
            output_root=data_root / "evidence_fragment",
            workers=1,
            force=True,
            progress_every=0,
        )
        structural = validate_evidence_fragments(
            data_root=data_root,
            fragment_manifest_path=data_root / "evidence_fragment" / "manifest.jsonl",
            progress_every=0,
        )
        section_ids = [
            str(section_id) for case in cases for section_id in case["section_ids"]
        ]
        fragments = _load_fragments(data_root, section_ids)
        failures = list(structural["errors"])
        if structural["errors_count"]:
            failures = [
                {
                    "case_id": "STRUCTURAL",
                    "code": "STRUCTURE_VALIDATION_FAILED",
                    "message": str(item),
                }
                for item in structural["errors"]
            ]
        semantic_failed_cases: set[str] = set()
        for case in cases:
            semantic_failures = evaluate_case(case, fragments)
            if semantic_failures:
                semantic_failed_cases.add(str(case["case_id"]))
            failures.extend(failure.to_dict() for failure in semantic_failures)

        candidate_paths: list[str] = []
        if candidate_dir is not None:
            candidate_paths = _write_candidate(candidate_dir, cases, fragments)
        elif not semantic_only:
            failures.extend(
                _compare_golden(
                    fixture_root / "golden",
                    [
                        case
                        for case in cases
                        if str(case["case_id"]) not in semantic_failed_cases
                    ],
                    fragments,
                )
            )

        status_counts = Counter(item["code"] for item in failures)
        report: dict[str, Any] = {
            "schema_version": "converter-regression-report.v1",
            "status": "FAILED" if failures else "PASSED",
            "profile": profile,
            "cases": [case["case_id"] for case in cases],
            "documents": len(documents),
            "workdir": str(workdir) if keep_workdir is not None else None,
            "canonical_summary": canonical_summary,
            "fragment_summary": fragment_summary,
            "structural_summary": {
                key: structural[key]
                for key in (
                    "documents",
                    "fragments",
                    "total_evidence",
                    "total_external_records",
                    "errors_count",
                )
            },
            "failure_counts": dict(status_counts),
            "failures": failures,
            "candidate_paths": candidate_paths,
        }
    finally:
        if temporary is not None:
            temporary.cleanup()

    production_after = {
        str(relative): _tree_signature(project_root / relative)
        for relative in PRODUCTION_OUTPUTS
    }
    if production_before != production_after:
        report["status"] = "FAILED"
        report["failure_counts"]["PRODUCTION_WRITE_DETECTED"] = 1
        report["failures"].append(
            {
                "case_id": "RUNNER",
                "code": "PRODUCTION_WRITE_DETECTED",
                "message": "Production canonical/evidence output changed during regression run.",
            }
        )
    report["elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--semantic-only", action="store_true")
    parser.add_argument("--write-candidate", type=Path)
    parser.add_argument("--keep-workdir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    result = run_regression(
        project_root=args.project_root.resolve(),
        fixture_root=args.fixture_root.resolve(),
        profile=args.profile,
        case_ids=set(args.cases) if args.cases else None,
        semantic_only=args.semantic_only,
        candidate_dir=(
            args.write_candidate.resolve() if args.write_candidate else None
        ),
        keep_workdir=(args.keep_workdir.resolve() if args.keep_workdir else None),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = ["run_regression"]
