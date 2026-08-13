from __future__ import annotations

import hashlib
import json
from pathlib import Path

from converters.common.source_models import DocumentContext
from converters.evidence_builder.builder import (
    build_canonical_evidence,
    recover_canonical_evidence_manifest,
)
from converters.evidence_builder.validator import validate_canonical_evidence
from converters.section_canonicalizer.section_canonicalizer import chunk_sections


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(data_root: Path) -> tuple[Path, Path]:
    raw = data_root / "raw/major/sample/20250101000001.xml"
    raw.parent.mkdir(parents=True)
    raw.write_text(
        "<DOCUMENT><SECTION-1><TITLE>Main</TITLE><P>Hello.</P></SECTION-1></DOCUMENT>",
        encoding="utf-8",
    )
    context = DocumentContext(
        doc_id="major_20250101000001",
        rcept_no="20250101000001",
        source_path=raw.relative_to(data_root).as_posix(),
        doc_group="major",
    )
    collection = chunk_sections(raw.read_bytes(), document_context=context)
    section_path = data_root / "canonical_section/major/20250101000001.json"
    section_path.parent.mkdir(parents=True)
    section_path.write_text(
        json.dumps(
            {
                "schema_version": "canonical-section-document.v1",
                "source_document": context.to_dict(),
                "sources": [
                    {
                        "source_path": context.source_path,
                        "source_sha256": _sha256(raw),
                        "section_collection": collection.to_dict(),
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = data_root / "canonical_section/manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "canonical-section-document.v1",
                "doc_id": context.doc_id,
                "rcept_no": context.rcept_no,
                "doc_group": context.doc_group,
                "output_path": section_path.relative_to(data_root).as_posix(),
                "status": "SUCCESS",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return raw, manifest


def test_canonical_evidence_builder_writes_and_reuses_output(tmp_path):
    data_root = tmp_path / "data"
    _, manifest = _fixture(data_root)
    output_root = data_root / "canonical_evidence"

    first = build_canonical_evidence(
        section_manifest_path=manifest,
        data_root=data_root,
        output_root=output_root,
        workers=1,
        progress_every=0,
    )
    second = build_canonical_evidence(
        section_manifest_path=manifest,
        data_root=data_root,
        output_root=output_root,
        workers=1,
        progress_every=0,
    )

    assert first == {"generated": 1, "reused": 0, "failed": 0, "total": 1}
    assert second == {"generated": 0, "reused": 1, "failed": 0, "total": 1}
    manifest_record = json.loads(
        (output_root / "manifest.jsonl").read_text(encoding="utf-8")
    )
    assert manifest_record["schema_version"] == "canonical-evidence-manifest.v1"
    assert manifest_record["n_evidence"] == 1
    assert len(manifest_record["output_paths"]) == 1

    output = json.loads(
        (data_root / manifest_record["output_paths"][0]).read_text(encoding="utf-8")
    )
    assert output["schema_version"] == "canonical-evidence.v1"
    assert output["id"] == "20250101000001:src0:s0:e0"
    assert output["rcept_no"] == "20250101000001"
    assert output["source_index"] == 0
    assert output["payload"]["text"] == "Hello."
    assert output["markdown"].endswith("Hello.")

    validation = validate_canonical_evidence(
        data_root=data_root,
        manifest_path=output_root / "manifest.jsonl",
        progress_every=0,
    )
    assert validation["errors_count"] == 0
    assert validation["total_evidence"] == 1


def test_canonical_evidence_builder_rejects_stale_raw_input(tmp_path):
    data_root = tmp_path / "data"
    raw, manifest = _fixture(data_root)
    raw.write_text("<DOCUMENT />", encoding="utf-8")

    result = build_canonical_evidence(
        section_manifest_path=manifest,
        data_root=data_root,
        output_root=data_root / "canonical_evidence",
        workers=1,
        progress_every=0,
    )

    assert result["failed"] == 1
    manifest_record = json.loads(
        (data_root / "canonical_evidence/manifest.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert manifest_record["output_paths"] == []
    assert manifest_record["issues"][0]["code"] == "STALE_INPUT"


def test_canonical_evidence_builder_removes_outputs_no_longer_generated(tmp_path):
    data_root = tmp_path / "data"
    _, manifest = _fixture(data_root)
    output_root = data_root / "canonical_evidence"

    build_canonical_evidence(
        section_manifest_path=manifest,
        data_root=data_root,
        output_root=output_root,
        workers=1,
        progress_every=0,
    )
    first_record = json.loads(
        (output_root / "manifest.jsonl").read_text(encoding="utf-8")
    )
    stale_output = data_root / first_record["output_paths"][0]
    assert stale_output.is_file()

    section_path = data_root / "canonical_section/major/20250101000001.json"
    canonical = json.loads(section_path.read_text(encoding="utf-8"))
    canonical["sources"][0]["section_collection"]["sections"] = []
    section_path.write_text(
        json.dumps(canonical, ensure_ascii=False),
        encoding="utf-8",
    )

    result = build_canonical_evidence(
        section_manifest_path=manifest,
        data_root=data_root,
        output_root=output_root,
        workers=1,
        progress_every=0,
    )

    assert result == {"generated": 1, "reused": 0, "failed": 0, "total": 1}
    assert not stale_output.exists()
    second_record = json.loads(
        (output_root / "manifest.jsonl").read_text(encoding="utf-8")
    )
    assert second_record["output_paths"] == []
    assert second_record["n_evidence"] == 0


def test_canonical_evidence_manifest_can_be_recovered_from_outputs(tmp_path):
    data_root = tmp_path / "data"
    _, manifest = _fixture(data_root)
    output_root = data_root / "canonical_evidence"
    build_canonical_evidence(
        section_manifest_path=manifest,
        data_root=data_root,
        output_root=output_root,
        workers=1,
        progress_every=0,
    )
    (output_root / "manifest.jsonl").unlink()

    result = recover_canonical_evidence_manifest(
        section_manifest_path=manifest,
        data_root=data_root,
        output_root=output_root,
        workers=1,
        progress_every=0,
    )

    assert result == {"recovered": 1, "failed": 0, "total": 1}
    validation = validate_canonical_evidence(
        data_root=data_root,
        manifest_path=output_root / "manifest.jsonl",
        progress_every=0,
    )
    assert validation["errors_count"] == 0
