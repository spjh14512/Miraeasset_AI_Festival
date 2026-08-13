from __future__ import annotations

from converters.common.source_models import DocumentContext
from converters.evidence_builder.evidence_builder import build_source_evidence
from converters.section_canonicalizer.section_canonicalizer import chunk_sections


def _context() -> DocumentContext:
    return DocumentContext(
        doc_id="major_20250101000001",
        rcept_no="20250101000001",
        source_path="raw/major/sample/20250101000001.xml",
        doc_group="major",
    )


def test_evidence_builder_splits_record_tables_and_consumes_table_context():
    source = """<DOCUMENT><SECTION-1><TITLE>Business</TITLE>
    <P>Narrative paragraph.</P>
    <P CLASS="table-title">Ownership status</P>
    <TABLE><THEAD><TR><TD>Name</TD><TD>Amount</TD></TR></THEAD><TBODY>
      <TR><TD>A</TD><TE ACODE="x">10</TE></TR>
      <TR><TD>B</TD><TE ACODE="x">20</TE></TR>
    </TBODY></TABLE>
    <P CLASS="table-caption">Audited figures</P>
    <P>Following paragraph.</P>
    </SECTION-1></DOCUMENT>"""
    sections = chunk_sections(source, document_context=_context()).to_dict()

    result = build_source_evidence(
        source,
        section_collection=sections,
        document_context=_context(),
    )

    assert result["status"] == "SUCCESS"
    assert result["stats"] == {
        "TEXT": 2,
        "TABLE": 2,
        "IMAGE_SKIPPED": 0,
        "LAYOUT_TABLE_UNASSOCIATED": 0,
    }
    evidence = result["evidence"]
    assert [item["evidence_type"] for item in evidence] == [
        "TEXT",
        "TABLE",
        "TABLE",
        "TEXT",
    ]
    assert [item["payload"]["record_index"] for item in evidence[1:3]] == [0, 1]
    assert all(item["table_type"] == "R_TABLE" for item in evidence[1:3])
    assert evidence[1]["context"]["title"] == "Ownership status"
    assert evidence[1]["context"]["captions"] == ["Audited figures"]
    assert "**표 제목:** Ownership status" in evidence[1]["markdown"]
    assert len(evidence[1]["source_refs"]) == 3


def test_evidence_builder_emits_kv_and_unknown_as_whole_tables():
    source = """<DOCUMENT><SECTION-1><TITLE>Tables</TITLE>
    <TABLE><TR><TD>Revenue</TD><TE ACODE="x">100</TE></TR></TABLE>
    <TABLE><TR><TD>A</TD><TD>B</TD></TR><TR><TD>C</TD><TD>D</TD></TR></TABLE>
    </SECTION-1></DOCUMENT>"""
    sections = chunk_sections(source, document_context=_context()).to_dict()

    result = build_source_evidence(
        source,
        section_collection=sections,
        document_context=_context(),
    )

    assert [item["table_type"] for item in result["evidence"]] == [
        "KV_TABLE",
        "UNKNOWN",
    ]
    assert result["evidence"][0]["payload"]["fields"][0]["key_paths"] == [
        ["Revenue"]
    ]
    assert result["evidence"][1]["payload"]["rows"] == [["A", "B"], ["C", "D"]]


def test_standalone_layout_table_is_attached_not_emitted():
    source = """<DOCUMENT><SECTION-1><TITLE>Tables</TITLE>
    <TABLE><TR><TD>Ownership status</TD></TR></TABLE>
    <TABLE><TR><TD>Revenue</TD><TE ACODE="x">100</TE></TR></TABLE>
    </SECTION-1></DOCUMENT>"""
    sections = chunk_sections(source, document_context=_context()).to_dict()

    result = build_source_evidence(
        source,
        section_collection=sections,
        document_context=_context(),
    )

    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["context"]["title"] == "Ownership status"
    assert result["stats"]["LAYOUT_TABLE_UNASSOCIATED"] == 0
