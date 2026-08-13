from __future__ import annotations

import json
from pathlib import Path

from converters.common.source_models import DocumentContext
from converters.evidence_builder.fragment_assembler import build_source_fragments
from converters.evidence_builder.fragment_pipeline import build_evidence_fragments
from converters.evidence_builder.fragment_validator import (
    validate_evidence_fragments,
    validate_fragment,
)
from converters.section_canonicalizer.section_canonicalizer import chunk_sections
from scripts.build_canonical_sections import build_canonical_sections


def _context() -> DocumentContext:
    return DocumentContext(
        doc_id="periodic_20250101000001",
        rcept_no="20250101000001",
        source_path="raw.xml",
        doc_group="periodic",
    )


def test_section_fragment_has_minimal_top_level_and_r_table_records():
    xml = """<DOCUMENT><SECTION-1><TITLE>1. 현황</TITLE>
    <P>의미 있는 설명</P><P>의미 있는 설명</P>
    <TABLE-GROUP>
      <TABLE><TR><TD>【임직원 현황】</TD></TR></TABLE>
      <TABLE><TR><TD>(단위 : 명)</TD></TR></TABLE>
      <TABLE><THEAD><TR><TH>성명</TH><TH>직위</TH></TR></THEAD><TBODY>
        <TR><TD>김하나</TD><TD>대표이사</TD></TR>
        <TR><TD>이두리</TD><TD>이사</TD></TR>
      </TBODY></TABLE>
    </TABLE-GROUP></SECTION-1></DOCUMENT>"""
    sections = chunk_sections(xml, document_context=_context())

    result = build_source_fragments(
        xml,
        section_collection=sections,
        kept_section_ids={"s0"},
        document_context=_context(),
        source_index=0,
    )

    fragment = result["fragments"][0]
    assert set(fragment) == {
        "schema_version",
        "section_id",
        "evidence_list",
        "records",
    }
    assert fragment["schema_version"] == "evidence-fragment.v2"
    assert fragment["section_id"] == "section:20250101000001:src0:s0"
    assert [item["evidence_id"] for item in fragment["evidence_list"]] == [
        "evidence:20250101000001:src0:s0:e0",
        "evidence:20250101000001:src0:s0:e1",
    ]
    assert fragment["evidence_list"][0]["payload"] == {
        "text": "의미 있는 설명"
    }
    table = fragment["evidence_list"][1]
    assert table["storage_mode"] == "SECTION_RECORDS"
    assert table["payload"]["record_count"] == 2
    assert table["payload"]["title"] == "【임직원 현황】"
    assert table["payload"]["units"] == ["(단위 : 명)"]
    assert len(fragment["records"]) == 2
    assert fragment["records"][0]["table_id"] == table["payload"]["table_id"]
    assert fragment["records"][0]["values"] == [
        "김하나",
        "대표이사",
    ]
    assert validate_fragment(fragment) == []


def test_small_r_table_is_one_evidence_and_preserves_row_types_in_records():
    xml = """<DOCUMENT><SECTION-1><TITLE>1. 자기주식</TITLE><TABLE>
    <THEAD><TR><TH>구분</TH><TH>수량</TH></TR></THEAD><TBODY>
      <TR><TD>장내 취득</TD><TD>8</TD></TR>
      <TR><TD>소계(a)</TD><TD>8</TD></TR>
      <TR><TD>총계(a+b)</TD><TD>8</TD></TR>
    </TBODY></TABLE></SECTION-1></DOCUMENT>"""
    sections = chunk_sections(xml, document_context=_context())

    result = build_source_fragments(
        xml,
        section_collection=sections,
        kept_section_ids={"s0"},
        document_context=_context(),
        source_index=0,
    )

    fragment = result["fragments"][0]
    assert len(fragment["evidence_list"]) == 1
    table = fragment["evidence_list"][0]
    assert table["table_type"] == "R_TABLE"
    assert table["storage_mode"] == "SECTION_RECORDS"
    assert table["payload"]["record_count"] == 3
    assert [
        record["row_type"]
        for record in fragment["records"]
    ] == ["DATA", "SUBTOTAL", "TOTAL"]
    assert [record["values"] for record in fragment["records"]] == [
        ["장내 취득", "8"],
        ["소계(a)", "8"],
        ["총계(a+b)", "8"],
    ]
    assert validate_fragment(fragment) == []


def test_navigation_layout_does_not_become_table_title():
    xml = """<DOCUMENT><SECTION-1><TITLE>1. 상세표</TITLE>
    <TABLE><TR><TD><A REFNO="I. 회사의 개요|ADX-001">☞ 본문 위치로 이동</A></TD></TR></TABLE>
    <TABLE><THEAD><TR><TH>구분</TH><TH>수량</TH></TR></THEAD><TBODY>
      <TR><TD>보통주식</TD><TD>10</TD></TR>
    </TBODY></TABLE></SECTION-1></DOCUMENT>"""
    sections = chunk_sections(xml, document_context=_context())

    result = build_source_fragments(
        xml,
        section_collection=sections,
        kept_section_ids={"s0"},
        document_context=_context(),
        source_index=0,
    )

    fragment = result["fragments"][0]
    assert len(fragment["evidence_list"]) == 1
    table = fragment["evidence_list"][0]
    assert table["table_type"] == "R_TABLE"
    assert "title" not in table["payload"]
    assert table["payload"]["record_count"] == 1
    assert fragment["records"][0]["values"] == ["보통주식", "10"]
    assert validate_fragment(fragment) == []


def test_text_evidence_preserves_disclosure_reference_but_not_internal_link():
    xml = """<DOCUMENT><SECTION-1><TITLE>3. 타법인출자 현황(상세)</TITLE>
    <P>※ 관련 내용은 <A REFNO="20230309000420">2022년도 사업보고서</A>를
    참고하시기 바랍니다. <A REFNO="I. 회사의 개요|ADX-001">본문 위치</A></P>
    </SECTION-1></DOCUMENT>"""
    sections = chunk_sections(xml, document_context=_context())

    result = build_source_fragments(
        xml,
        section_collection=sections,
        kept_section_ids={"s0"},
        document_context=_context(),
        source_index=0,
    )

    evidence = result["fragments"][0]["evidence_list"][0]
    assert evidence["payload"]["text"] == (
        "※ 관련 내용은 2022년도 사업보고서를 참고하시기 바랍니다. 본문 위치"
    )
    assert evidence["references"] == [
        {
            "type": "disclosure",
            "refno": "20230309000420",
            "text": "2022년도 사업보고서",
        }
    ]
    assert validate_fragment(result["fragments"][0]) == []


def test_text_deduplication_keeps_same_text_with_different_disclosure_refs():
    xml = """<DOCUMENT><SECTION-1><TITLE>참조</TITLE>
    <P><A REFNO="20230309000420">사업보고서</A></P>
    <P><A REFNO="20240312000530">사업보고서</A></P>
    <P><A REFNO="20240312000530">사업보고서</A></P>
    </SECTION-1></DOCUMENT>"""
    sections = chunk_sections(xml, document_context=_context())

    result = build_source_fragments(
        xml,
        section_collection=sections,
        kept_section_ids={"s0"},
        document_context=_context(),
        source_index=0,
    )

    evidence = result["fragments"][0]["evidence_list"]
    assert len(evidence) == 2
    assert [item["references"][0]["refno"] for item in evidence] == [
        "20230309000420",
        "20240312000530",
    ]
    assert result["stats"]["DUPLICATE_SKIPPED"] == 1


def test_validator_rejects_embedded_r_table_and_misaligned_record_width():
    fragment = {
        "schema_version": "evidence-fragment.v2",
        "section_id": "section:20250101000001:src0:s0",
        "evidence_list": [
            {
                "evidence_id": "evidence:20250101000001:src0:s0:e0",
                "evidence_type": "TABLE",
                "table_type": "R_TABLE",
                "order": 0,
                "payload": {
                    "table_id": "rtable:20250101000001:src0:s0:t0",
                    "headers": [["name"], ["count"]],
                    "record_count": 1,
                },
            }
        ],
        "records": [
            {
                "table_id": "rtable:20250101000001:src0:s0:t0",
                "record_index": 0,
                "row_type": "DATA",
                "row_context": [],
                "values": ["only one value"],
            }
        ],
    }

    errors = validate_fragment(fragment)

    assert "r_table_storage_mode:0" in errors
    assert "orphan_record:0" in errors

    fragment["evidence_list"][0]["storage_mode"] = "SECTION_RECORDS"
    errors = validate_fragment(fragment)

    assert "record_width:0" in errors


def test_multiple_r_tables_have_stable_table_ids_and_record_groups():
    xml = """<DOCUMENT><SECTION-1><TITLE>Tables</TITLE>
    <TABLE><THEAD><TR><TH>Name</TH><TH>Count</TH></TR></THEAD><TBODY>
      <TR><TD>Alpha</TD><TD>1</TD></TR>
    </TBODY></TABLE>
    <TABLE><THEAD><TR><TH>Category</TH><TH>Amount</TH></TR></THEAD><TBODY>
      <TR><TD>Beta</TD><TD>2</TD></TR>
      <TR><TD>Gamma</TD><TD>3</TD></TR>
    </TBODY></TABLE>
    </SECTION-1></DOCUMENT>"""
    sections = chunk_sections(xml, document_context=_context())

    result = build_source_fragments(
        xml,
        section_collection=sections,
        kept_section_ids={"s0"},
        document_context=_context(),
        source_index=0,
    )

    fragment = result["fragments"][0]
    tables = [
        evidence
        for evidence in fragment["evidence_list"]
        if evidence.get("table_type") == "R_TABLE"
    ]
    assert [table["payload"]["table_id"] for table in tables] == [
        "rtable:20250101000001:src0:s0:t0",
        "rtable:20250101000001:src0:s0:t1",
    ]
    assert [record["table_id"] for record in fragment["records"]] == [
        "rtable:20250101000001:src0:s0:t0",
        "rtable:20250101000001:src0:s0:t1",
        "rtable:20250101000001:src0:s0:t1",
    ]
    assert [record["values"] for record in fragment["records"]] == [
        ["Alpha", "1"],
        ["Beta", "2"],
        ["Gamma", "3"],
    ]
    assert validate_fragment(fragment) == []


def test_kv_payload_omits_provenance_fields():
    xml = """<DOCUMENT><SECTION-1><TITLE>본문</TITLE><TABLE>
    <TR><TD>회사명</TD><TE ACODE="CRP_NM">주식회사 미래</TE>
    <TD>주식수</TD><TE ACODE="CNT">10</TE></TR>
    </TABLE></SECTION-1></DOCUMENT>"""
    sections = chunk_sections(xml, document_context=_context())

    result = build_source_fragments(
        xml,
        section_collection=sections,
        kept_section_ids={"s0"},
        document_context=_context(),
        source_index=0,
    )

    fragment = result["fragments"][0]
    evidence = fragment["evidence_list"][0]
    assert evidence["table_type"] == "KV_TABLE"
    assert evidence["payload"]["fields"][0] == {
        "key_paths": [["회사명"]],
        "raw_value": "주식회사 미래",
    }
    serialized = json.dumps(fragment, ensure_ascii=False)
    for forbidden in (
        "source_refs",
        "source_path",
        "source_cell",
        "column_id",
        "markdown",
        "content_sha256",
    ):
        assert forbidden not in serialized
    assert validate_fragment(fragment) == []


def test_unclassified_non_layout_table_is_preserved_as_text_fallback():
    xml = """<DOCUMENT><SECTION-1><TITLE>주석</TITLE><TABLE>
    <TR><TD>첫 번째 설명</TD></TR>
    <TR><TD>두 번째 설명</TD></TR>
    <TR><TD>세 번째 설명</TD></TR>
    </TABLE></SECTION-1></DOCUMENT>"""
    sections = chunk_sections(xml, document_context=_context())

    result = build_source_fragments(
        xml,
        section_collection=sections,
        kept_section_ids={"s0"},
        document_context=_context(),
        source_index=0,
    )

    fragment = result["fragments"][0]
    assert fragment["evidence_list"] == [
        {
            "evidence_id": "evidence:20250101000001:src0:s0:e0",
            "evidence_type": "TEXT",
            "order": 0,
            "payload": {
                "text": "첫 번째 설명\n두 번째 설명\n세 번째 설명"
            },
        }
    ]
    assert result["stats"]["UNKNOWN_FALLBACK_TEXT"] == 1
    assert validate_fragment(fragment) == []


def test_pipeline_writes_one_fragment_per_graph_section_and_validates(tmp_path: Path):
    data_root = tmp_path / "data"
    source_dir = data_root / "raw/major/회사/20250101000001"
    source_dir.mkdir(parents=True)
    (source_dir / "20250101000001.xml").write_text(
        """<DOCUMENT><P>표지</P><SECTION-1><TITLE>상위</TITLE><P>상위 본문</P>
        <SECTION-2><TITLE>하위</TITLE><P>하위 본문</P></SECTION-2>
        </SECTION-1></DOCUMENT>""",
        encoding="utf-8",
    )
    source_manifest = data_root / "manifest.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "doc_id": "major_20250101000001",
                "rcept_no": "20250101000001",
                "doc_group": "major",
                "file_path": "raw/major/회사/20250101000001",
                "file_format": "xml",
                "n_files": 1,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    canonical_root = data_root / "canonical_section"
    build_canonical_sections(
        manifest_path=source_manifest,
        data_root=data_root,
        output_root=canonical_root,
        workers=1,
        progress_every=0,
    )

    output_root = data_root / "evidence_fragment"
    summary = build_evidence_fragments(
        section_manifest_path=canonical_root / "manifest.jsonl",
        data_root=data_root,
        output_root=output_root,
        workers=1,
        progress_every=0,
    )

    assert summary == {"generated": 1, "reused": 0, "failed": 0, "total": 1}
    files = sorted((output_root / "major/20250101000001").glob("*.json"))
    assert [path.name for path in files] == ["src0__s1.json", "src0__s2.json"]
    fragments = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    assert [fragment["section_id"] for fragment in fragments] == [
        "section:20250101000001:src0:s1",
        "section:20250101000001:src0:s2",
    ]
    assert [fragment["evidence_list"][0]["payload"]["text"] for fragment in fragments] == [
        "상위 본문",
        "하위 본문",
    ]
    validation = validate_evidence_fragments(
        data_root=data_root,
        fragment_manifest_path=output_root / "manifest.jsonl",
        progress_every=0,
    )
    assert validation["errors"] == []
    assert validation["fragments"] == 2
