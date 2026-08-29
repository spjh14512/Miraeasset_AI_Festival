from __future__ import annotations

import xml.etree.ElementTree as ET

from converters.common.source_models import (
    DocumentContext,
    DocumentSyntax,
    SourceRef,
)
from converters.section_canonicalizer.section_canonicalizer import chunk_sections
from converters.section_canonicalizer.section_models import (
    SectionBlockType,
    SectionBoundaryKind,
    SectionParseStatus,
)


def _document_context() -> DocumentContext:
    return DocumentContext(
        doc_id="periodic_20230515002335",
        rcept_no="20230515002335",
        source_path="sample.xml",
        doc_group="periodic",
    )


def test_explicit_sections_preserve_hierarchy_paths_and_direct_block_order():
    xml = """<DOCUMENT><BODY>
    <P>표지 설명</P>
    <SECTION-1><TITLE>I. 회사의 개요</TITLE>
      <P>상위 서론</P>
      <LIBRARY>
        <SECTION-2><TITLE>1. 회사의 개요</TITLE>
          <P>하위 본문</P>
          <TABLE-GROUP><TABLE><TR><TD>표</TD></TR></TABLE></TABLE-GROUP>
          <P>하위 결론</P>
        </SECTION-2>
        <P>상위 후속 설명</P>
      </LIBRARY>
    </SECTION-1>
    </BODY></DOCUMENT>"""

    result = chunk_sections(xml, document_context=_document_context())

    assert [section.boundary_kind for section in result.sections] == [
        SectionBoundaryKind.SYNTHETIC,
        SectionBoundaryKind.EXPLICIT,
        SectionBoundaryKind.EXPLICIT,
    ]
    root, parent, child = result.sections
    assert [block.block_type for block in root.blocks] == [SectionBlockType.P]
    assert parent.parent_section_id == root.id
    assert parent.section_path == ("I. 회사의 개요",)
    assert [block.block_type for block in parent.blocks] == [
        SectionBlockType.P,
        SectionBlockType.P,
    ]
    assert child.parent_section_id == parent.id
    assert child.level == 2
    assert child.section_path == ("I. 회사의 개요", "1. 회사의 개요")
    assert [block.block_type for block in child.blocks] == [
        SectionBlockType.P,
        SectionBlockType.TABLE_GROUP,
        SectionBlockType.P,
    ]


def test_atoc_table_group_titles_create_children_of_explicit_section():
    xml = """<DOCUMENT><BODY>
    <SECTION-1><TITLE>III. 재무에 관한 사항</TITLE>
      <SECTION-2><TITLE>3. 연결재무제표 주석</TITLE>
        <P>주석 서론</P>
        <TABLE-GROUP ACLASS="{XBRL}NT_C_DX801000">
          <TITLE ATOC="Y" ATOCID="267">1. 일반사항 (연결)</TITLE>
          <TABLE><TR><TD>첫 번째 주석</TD></TR></TABLE>
        </TABLE-GROUP>
        <TABLE-GROUP>
          <TITLE ATOC="N">표 제목</TITLE>
          <TABLE><TR><TD>표 본문</TD></TR></TABLE>
        </TABLE-GROUP>
        <TABLE-GROUP ACLASS="{XBRL}NT_C_DX841000">
          <TITLE ATOC="Y" ATOCID="268">2. 비연결구조화기업 (연결)</TITLE>
          <TABLE><TR><TD>두 번째 주석</TD></TR></TABLE>
        </TABLE-GROUP>
      </SECTION-2>
    </SECTION-1></BODY></DOCUMENT>"""

    result = chunk_sections(xml, document_context=_document_context())

    assert [section.title for section in result.sections] == [
        "III. 재무에 관한 사항",
        "3. 연결재무제표 주석",
        "1. 일반사항 (연결)",
        "2. 비연결구조화기업 (연결)",
    ]
    parent, first, second = result.sections[1:]
    assert parent.boundary_kind == SectionBoundaryKind.EXPLICIT
    assert [block.block_type for block in parent.blocks] == [
        SectionBlockType.P,
        SectionBlockType.TABLE_GROUP,
    ]
    assert first.parent_section_id == parent.id
    assert second.parent_section_id == parent.id
    assert first.level == parent.level + 1
    assert first.boundary_kind == SectionBoundaryKind.IMPLICIT
    assert first.section_path == (
        "III. 재무에 관한 사항",
        "3. 연결재무제표 주석",
        "1. 일반사항 (연결)",
    )
    assert [block.block_type for block in first.blocks] == [
        SectionBlockType.TABLE_GROUP
    ]
    assert [block.block_type for block in second.blocks] == [
        SectionBlockType.TABLE_GROUP
    ]
    paths = [
        source_ref.element_path
        for section in result.sections
        for block in section.blocks
        for source_ref in block.source_refs
    ]
    assert len(paths) == len(set(paths))


def test_every_source_block_belongs_to_only_one_deepest_section():
    xml = """<DOCUMENT><SECTION-1><TITLE>상위</TITLE><P>상위 본문</P>
    <SECTION-2><TITLE>하위</TITLE><P>하위 본문</P><TABLE><TR><TD>A</TD></TR></TABLE>
    </SECTION-2></SECTION-1></DOCUMENT>"""

    result = chunk_sections(xml, document_context=_document_context())
    paths = [
        source_ref.element_path
        for section in result.sections
        for block in section.blocks
        for source_ref in block.source_refs
    ]

    assert len(paths) == len(set(paths))
    assert len(result.sections[0].blocks) == 1
    assert len(result.sections[1].blocks) == 2


def test_title_is_metadata_and_not_a_paragraph_block():
    result = chunk_sections(
        "<DOCUMENT><SECTION-1><TITLE>제목</TITLE><P>본문</P></SECTION-1></DOCUMENT>",
        document_context=_document_context(),
    )

    section = result.sections[0]
    assert section.title == "제목"
    assert section.title_source_ref is not None
    assert [block.block_type for block in section.blocks] == [SectionBlockType.P]


def test_correction_subtree_is_skipped_without_mutating_input():
    root = ET.fromstring(
        """<DOCUMENT><CORRECTION><P>정정 안내</P><TABLE><TR><TD>정정표</TD></TR></TABLE>
        </CORRECTION><SECTION-1><TITLE>본문</TITLE><P>남길 문단</P></SECTION-1></DOCUMENT>"""
    )
    before = ET.tostring(root, encoding="unicode")

    result = chunk_sections(root, document_context=_document_context())

    assert ET.tostring(root, encoding="unicode") == before
    assert len(result.sections) == 1
    assert [block.block_type for block in result.sections[0].blocks] == [
        SectionBlockType.P
    ]


def test_explicit_html_exclusion_skips_entire_container():
    html = """<html><head><title>문서</title></head><body>
    <div id="LIB_LC000"><p>정정 내용</p><table><tr><td>정정표</table></div>
    <p>본문</p><table><tr><td>본문표</table>
    </body></html>"""

    result = chunk_sections(
        html,
        syntax=DocumentSyntax.HTML,
        document_context=_document_context(),
        excluded_source_refs=(
            SourceRef(syntax=DocumentSyntax.HTML, html_id="LIB_LC000"),
        ),
    )

    assert len(result.sections) == 1
    assert [block.block_type for block in result.sections[0].blocks] == [
        SectionBlockType.P,
        SectionBlockType.TABLE,
    ]


def test_html_headings_create_implicit_hierarchy_and_keep_preamble_root():
    html = """<html><head><title>사업보고서</title></head><body>
    <p>문서 서문</p>
    <h1>I. 회사의 개요</h1><p>회사 설명</p>
    <h2>1. 회사의 개요</h2><table><tr><td>표</td></tr></table>
    <h1>II. 사업의 내용</h1><p>사업 설명</p>
    </body></html>"""

    result = chunk_sections(
        html,
        syntax=DocumentSyntax.HTML,
        document_context=_document_context(),
    )

    root, first, child, second = result.sections
    assert root.boundary_kind == SectionBoundaryKind.SYNTHETIC
    assert root.title == "사업보고서"
    assert first.parent_section_id == root.id
    assert first.boundary_kind == SectionBoundaryKind.IMPLICIT
    assert child.parent_section_id == first.id
    assert child.section_path == ("I. 회사의 개요", "1. 회사의 개요")
    assert [block.block_type for block in child.blocks] == [SectionBlockType.TABLE]
    assert second.parent_section_id == root.id


def test_repeated_numbered_paragraphs_are_implicit_sections():
    xml = """<DOCUMENT><BODY>
    <P>1. 첫 번째 항목</P><P>첫 번째 본문</P>
    <P>2. 두 번째 항목</P><P>두 번째 본문</P>
    </BODY></DOCUMENT>"""

    result = chunk_sections(xml, document_context=_document_context())

    assert [section.title for section in result.sections] == [
        "1. 첫 번째 항목",
        "2. 두 번째 항목",
    ]
    assert all(
        section.boundary_kind == SectionBoundaryKind.IMPLICIT
        for section in result.sections
    )
    assert all(section.level == 1 for section in result.sections)
    assert all(len(section.blocks) == 1 for section in result.sections)


def test_single_numbered_paragraph_does_not_create_implicit_boundary():
    result = chunk_sections(
        "<DOCUMENT><P>1. 목록 항목</P><P>설명</P></DOCUMENT>",
        document_context=_document_context(),
    )

    assert len(result.sections) == 1
    assert result.sections[0].boundary_kind == SectionBoundaryKind.SYNTHETIC
    assert [block.block_type for block in result.sections[0].blocks] == [
        SectionBlockType.P,
        SectionBlockType.P,
    ]


def test_repeated_numbered_single_cell_tables_can_be_implicit_headings():
    xml = """<DOCUMENT>
    <TABLE><TR><TD>1. 첫 번째 항목</TD></TR></TABLE><P>첫 본문</P>
    <TABLE><TR><TD>2. 두 번째 항목</TD></TR></TABLE><P>둘째 본문</P>
    </DOCUMENT>"""

    result = chunk_sections(xml, document_context=_document_context())

    assert [section.title for section in result.sections] == [
        "1. 첫 번째 항목",
        "2. 두 번째 항목",
    ]
    assert all(
        [block.block_type for block in section.blocks] == [SectionBlockType.P]
        for section in result.sections
    )


def test_orphan_spans_and_images_are_preserved_as_ordered_refs():
    html = """<html><body><div><span>첫 조각</span><span>둘째 조각</span></div>
    <img src="chart.png"><p>본문</p></body></html>"""

    result = chunk_sections(
        html,
        syntax=DocumentSyntax.HTML,
        document_context=_document_context(),
    )

    blocks = result.sections[0].blocks
    assert [block.block_type for block in blocks] == [
        SectionBlockType.SPAN_RUN,
        SectionBlockType.IMAGE,
        SectionBlockType.P,
    ]
    assert len(blocks[0].source_refs) == 2
    assert [block.order for block in blocks] == [0, 1, 2]


def test_synthetic_html_title_is_metadata_not_duplicate_span_block():
    html = """<html><body><div>
    <span style="font-weight:bold">단일판매ㆍ공급계약 체결</span>
    <table><tr><td>본문표</td></tr></table>
    </div></body></html>"""

    result = chunk_sections(
        html,
        syntax=DocumentSyntax.HTML,
        document_context=_document_context(),
    )

    section = result.sections[0]
    assert section.title == "단일판매ㆍ공급계약 체결"
    assert section.title_source_ref is not None
    assert [block.block_type for block in section.blocks] == [
        SectionBlockType.TABLE
    ]


def test_recovered_and_failed_document_statuses_are_traceable():
    recovered = chunk_sections(
        "<DOCUMENT><P>A & B</P></DOCUMENT>",
        document_context=_document_context(),
    )
    failed = chunk_sections(
        "<NOT_DOCUMENT><P>본문</NOT_DOCUMENT_EXTRA>",
        document_context=_document_context(),
    )

    assert recovered.parse_status == SectionParseStatus.RECOVERED
    assert recovered.issues[0].code == "XML_RECOVERED"
    assert failed.parse_status == SectionParseStatus.FAILED
    assert failed.sections == ()
