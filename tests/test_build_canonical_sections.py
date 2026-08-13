from __future__ import annotations

import json
from pathlib import Path

from converters.common.source_models import DocumentContext
from converters.section_canonicalizer.section_canonicalizer import chunk_sections
from scripts.build_canonical_sections import (
    _append_section_outputs,
    build_canonical_sections,
)


def _write_manifest(path: Path) -> None:
    row = {
        "doc_id": "major_20250101000001",
        "rcept_no": "20250101000001",
        "doc_group": "major",
        "file_path": "raw/major/회사/20250101000001",
        "file_format": "xml",
        "n_files": 2,
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def test_builder_flattens_multiple_sources_and_reuses_matching_hash(tmp_path):
    data_root = tmp_path / "data"
    source = data_root / "raw/major/회사/20250101000001"
    source.mkdir(parents=True)
    (source / "20250101000001.xml").write_text(
        "<DOCUMENT><SECTION-1><TITLE>본문</TITLE><P>A</P></SECTION-1></DOCUMENT>",
        encoding="utf-8",
    )
    (source / "20250101000001_00001.xml").write_text(
        "<DOCUMENT><SECTION-1><TITLE>첨부</TITLE><P>B</P></SECTION-1></DOCUMENT>",
        encoding="utf-8",
    )
    manifest = data_root / "manifest.jsonl"
    _write_manifest(manifest)
    output_root = data_root / "canonical_section"

    first = build_canonical_sections(
        manifest_path=manifest,
        data_root=data_root,
        output_root=output_root,
        workers=1,
        progress_every=0,
    )
    second = build_canonical_sections(
        manifest_path=manifest,
        data_root=data_root,
        output_root=output_root,
        workers=1,
        progress_every=0,
    )

    assert first == {"generated": 1, "reused": 0, "failed": 0, "total": 1}
    assert second == {"generated": 0, "reused": 1, "failed": 0, "total": 1}
    output = json.loads(
        (output_root / "major/sections_20250101000001.json").read_text(
            encoding="utf-8"
        )
    )
    assert output == {
        "schema_version": "canonical-section-document.v2",
        "source_doc_id": "major_20250101000001",
        "sections": [
            {
                "section_id": "section:20250101000001:src0:s0",
                "parent_section_id": None,
                "order": 0,
                "title": "본문",
                "section_path": ["본문"],
                "element_path": "/DOCUMENT[1]/SECTION-1[1]",
            },
            {
                "section_id": "section:20250101000001:src1:s0",
                "parent_section_id": None,
                "order": 1,
                "title": "첨부",
                "section_path": ["첨부"],
                "element_path": "/DOCUMENT[1]/SECTION-1[1]",
            },
        ],
    }
    assert [section["title"] for section in output["sections"]] == [
        "본문",
        "첨부",
    ]

    build_manifest = [
        json.loads(line)
        for line in (output_root / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert build_manifest[0]["n_sources"] == 2
    assert build_manifest[0]["n_sections"] == 2
    assert build_manifest[0]["n_blocks"] == 2

    (output_root / "manifest.jsonl").unlink()
    recovered_without_manifest = build_canonical_sections(
        manifest_path=manifest,
        data_root=data_root,
        output_root=output_root,
        workers=1,
        progress_every=0,
    )
    assert recovered_without_manifest == {
        "generated": 1,
        "reused": 0,
        "failed": 0,
        "total": 1,
    }


def test_builder_records_missing_processable_source_as_failure(tmp_path):
    data_root = tmp_path / "data"
    source = data_root / "raw/major/회사/20250101000001"
    source.mkdir(parents=True)
    (source / "document.pdf").write_bytes(b"pdf")
    manifest = data_root / "manifest.jsonl"
    _write_manifest(manifest)

    summary = build_canonical_sections(
        manifest_path=manifest,
        data_root=data_root,
        output_root=data_root / "canonical_section",
        workers=1,
        progress_every=0,
    )

    assert summary["failed"] == 1
    record = json.loads(
        (data_root / "canonical_section/manifest.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert record["status"] == "FAILED"
    assert record["error"] == "No XML or HTML source file was found."


def test_graph_output_omits_structural_root_and_keeps_standalone_synthetic():
    context = DocumentContext(
        doc_id="major_20250101000001",
        rcept_no="20250101000001",
        source_path="sample.xml",
        doc_group="major",
    )
    hierarchical = chunk_sections(
        """<DOCUMENT><P>표지</P><SECTION-1><TITLE>본문</TITLE>
        <P>내용</P></SECTION-1></DOCUMENT>""",
        document_context=context,
    )
    standalone = chunk_sections(
        "<DOCUMENT><TITLE>첨부 제목</TITLE><P>첨부 본문</P></DOCUMENT>",
        document_context=context,
    )
    output: list[dict[str, object]] = []

    _append_section_outputs(
        output,
        hierarchical.sections,
        rcept_no=context.rcept_no,
        source_index=0,
    )
    _append_section_outputs(
        output,
        standalone.sections,
        rcept_no=context.rcept_no,
        source_index=1,
    )

    assert [section["section_id"] for section in output] == [
        "section:20250101000001:src0:s1",
        "section:20250101000001:src1:s0",
    ]
    assert [section["parent_section_id"] for section in output] == [None, None]
    assert [section["order"] for section in output] == [0, 1]
    assert [section["title"] for section in output] == ["본문", "첨부 제목"]
