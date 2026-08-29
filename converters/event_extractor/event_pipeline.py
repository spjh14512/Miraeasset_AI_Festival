"""Materialize latest-version major/exchange Event intermediates."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Any

from converters.event_extractor.category_mapping import load_event_categories
from converters.event_extractor.event_extractor import extract_event
from converters.event_extractor.event_models import EventExtraction


VALID_STATUSES = {"SUCCESS", "RECOVERED"}
RESOLVABLE_CORRECTION_STATUSES = {"FOUND", "RECOVERED"}


def read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    return tuple(rows)


def latest_receipt_numbers(
    manifest_rows: Iterable[Mapping[str, Any]],
    *,
    correction_rows: Iterable[Mapping[str, Any]],
) -> set[str]:
    """Mirror the persisted is_latest_version state without re-extracting corrections."""
    rows = tuple(manifest_rows)
    explicit = [row.get("is_latest_version") for row in rows]
    if explicit and all(isinstance(value, bool) for value in explicit):
        return {
            str(row.get("rcept_no", ""))
            for row in rows
            if row.get("is_latest_version") is True
        }
    if any(value is not None for value in explicit):
        raise ValueError("Manifest is_latest_version must be present on every row or none")

    corrections_by_root: dict[str, set[str]] = {}
    source_roots: dict[str, str] = {}
    for row in correction_rows:
        if row.get("status") not in RESOLVABLE_CORRECTION_STATUSES:
            continue
        correction = row.get("correction")
        if not isinstance(correction, Mapping):
            continue
        target_value = correction.get("target_rcept_no")
        target = str(target_value).strip() if target_value is not None else ""
        source_document = row.get("source_document")
        source = (
            str(source_document.get("rcept_no", "")).strip()
            if isinstance(source_document, Mapping)
            else ""
        )
        if not target or not source:
            continue
        previous_root = source_roots.get(source)
        if previous_root is not None and previous_root != target:
            raise ValueError(
                f"Correction {source} resolves to conflicting roots: "
                f"{previous_root}, {target}"
            )
        source_roots[source] = target
        corrections_by_root.setdefault(target, set()).add(source)

    superseded: set[str] = set()
    for root, sources in corrections_by_root.items():
        ordered_sources = sorted(sources)
        superseded.add(root)
        superseded.update(ordered_sources[:-1])
    return {
        str(row.get("rcept_no", ""))
        for row in rows
        if str(row.get("rcept_no", "")).strip() not in superseded
    }


def _evidence_manifest_index(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("doc_group", "")), str(row.get("rcept_no", "")))
        if all(key) and row.get("status") in VALID_STATUSES and not row.get("error"):
            result[key] = row
    return result


def _fragments(data_root: Path, manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for relative_path in manifest.get("output_paths", []):
        path = data_root / str(relative_path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Evidence fragment must be an object: {path}")
        result.append(value)
    return tuple(result)


def build_event_results(
    manifest_rows: Iterable[Mapping[str, Any]],
    *,
    correction_rows: Iterable[Mapping[str, Any]],
    evidence_manifest_rows: Iterable[Mapping[str, Any]],
    category_path: Path,
    data_root: Path,
) -> tuple[EventExtraction, ...]:
    rows = tuple(manifest_rows)
    latest = latest_receipt_numbers(rows, correction_rows=correction_rows)
    evidence_index = _evidence_manifest_index(evidence_manifest_rows)
    catalog = load_event_categories(category_path)
    results: list[EventExtraction] = []

    for row in rows:
        doc_group = str(row.get("doc_group", ""))
        rcept_no = str(row.get("rcept_no", ""))
        if doc_group not in {"major", "exchange"} or rcept_no not in latest:
            continue
        category = catalog.match(str(row.get("report_nm", "")))
        if category is None:
            raise ValueError(
                f"Event category workbook has no match for {rcept_no}: "
                f"{row.get('report_nm')!r}"
            )
        evidence_manifest = evidence_index.get((doc_group, rcept_no))
        if evidence_manifest is None:
            raise ValueError(f"Valid Evidence manifest is missing for {doc_group}/{rcept_no}")
        results.append(
            extract_event(
                row,
                category=category,
                fragments=_fragments(data_root, evidence_manifest),
            )
        )
    return tuple(results)


def write_event_results(results: Iterable[EventExtraction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for result in results:
            output.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    temporary.replace(path)


def materialize_events(
    *,
    manifest_path: Path = Path("data/manifest.jsonl"),
    correction_path: Path = Path("data/correction/manifest.jsonl"),
    evidence_manifest_path: Path = Path("data/evidence_fragment/manifest.jsonl"),
    category_path: Path = Path("DOCS/Event_categories.xlsx"),
    data_root: Path = Path("data"),
    output_path: Path = Path("data/event/manifest.jsonl"),
) -> dict[str, Any]:
    results = build_event_results(
        read_jsonl(manifest_path),
        correction_rows=read_jsonl(correction_path),
        evidence_manifest_rows=read_jsonl(evidence_manifest_path),
        category_path=category_path,
        data_root=data_root,
    )
    write_event_results(results, output_path)
    by_group = Counter(str(result.source_document["doc_group"]) for result in results)
    fallback_dates = sum(bool(result.issues) for result in results)
    return {
        "events": len(results),
        "reports": len(results),
        "is_supported_by": sum(len(result.is_supported_by) for result in results),
        "fallback_event_dates": fallback_dates,
        "by_group": dict(sorted(by_group.items())),
        "output": str(output_path),
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract latest major/exchange Events")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.jsonl"))
    parser.add_argument(
        "--corrections",
        type=Path,
        default=Path("data/correction/manifest.jsonl"),
    )
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        default=Path("data/evidence_fragment/manifest.jsonl"),
    )
    parser.add_argument(
        "--categories",
        type=Path,
        default=Path("DOCS/Event_categories.xlsx"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/event/manifest.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    summary = materialize_events(
        manifest_path=args.manifest,
        correction_path=args.corrections,
        evidence_manifest_path=args.evidence_manifest,
        category_path=args.categories,
        data_root=args.data_root,
        output_path=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "build_event_results",
    "latest_receipt_numbers",
    "materialize_events",
    "read_jsonl",
    "write_event_results",
]
