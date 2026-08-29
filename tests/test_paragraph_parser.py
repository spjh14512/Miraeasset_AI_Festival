from __future__ import annotations

import xml.etree.ElementTree as ET

from converters.common.source_models import (
    ContextPosition,
    ContextRole,
    DocumentContext,
    DocumentSyntax,
    EvidenceContext,
    EvidenceContextBlock,
    SourceRef,
)
from converters.paragraph_parser.paragraph_models import (
    FragmentKind,
    MarkerKind,
    ParagraphParseStatus,
    ParagraphSourceKind,
)
from converters.paragraph_parser.paragraph_parser import (
    parse_paragraph_element,
    parse_paragraphs,
)


def _document_context() -> DocumentContext:
    return DocumentContext(
        doc_id="major/회사/20250528000403",
        rcept_no="20250528000403",
        source_path="data/raw/major/회사/20250528000403/document.xml",
        doc_group="major",
    )


def test_inline_spans_form_one_paragraph_and_preserve_order_and_attributes():
    xml = """<DOCUMENT><P USERMARK="F-1">
    - 상기 '1. 취득예정주식(주)'은 <SPAN USERMARK="F-2">취득후 전량 소각</SPAN>할 계획임.
    <SPAN>※ 산출근거</SPAN><SPAN>① 취득신고주식수</SPAN>
    </P></DOCUMENT>"""
    result = parse_paragraphs(xml, document_context=_document_context())

    assert len(result.paragraphs) == 1
    paragraph = result.paragraphs[0]
    assert paragraph.source_kind == ParagraphSourceKind.P
    assert paragraph.leading_marker is not None
    assert paragraph.leading_marker.kind == MarkerKind.BULLET
    assert paragraph.source_attributes == {"USERMARK": "F-1"}
    assert [fragment.kind for fragment in paragraph.fragments] == [
        FragmentKind.TEXT,
        FragmentKind.SPAN,
        FragmentKind.TEXT,
        FragmentKind.SPAN,
        FragmentKind.SPAN,
    ]
    assert paragraph.fragments[1].source_attributes == {"USERMARK": "F-2"}
    assert paragraph.text == (
        "- 상기 '1. 취득예정주식(주)'은 취득후 전량 소각할 계획임. "
        "※ 산출근거① 취득신고주식수"
    )


def test_paragraph_bold_is_inherited_but_inline_not_bold_is_respected():
    result = parse_paragraphs(
        """<DOCUMENT><P USERMARK="B"><SPAN>ㅇ 비씨카드㈜</SPAN>
        <SPAN USERMARK="!B">소송 현황입니다.</SPAN></P></DOCUMENT>""",
        document_context=_document_context(),
    )

    paragraph = result.paragraphs[0]
    assert paragraph.is_bold is True
    assert paragraph.fragments[0].source_attributes["USERMARK"] == "B"
    assert paragraph.fragments[1].source_attributes["USERMARK"] == "!B"


def test_inline_disclosure_anchor_preserves_refno_and_text():
    result = parse_paragraphs(
        """<DOCUMENT><P>관련 내용은
        <A REFNO="20230309000420">2022년도 사업보고서</A>를 참고하십시오.
        </P></DOCUMENT>""",
        document_context=_document_context(),
    )

    paragraph = result.paragraphs[0]
    anchor = paragraph.fragments[1]
    assert anchor.text == "2022년도 사업보고서"
    assert anchor.source_attributes == {"REFNO": "20230309000420"}
    assert anchor.source_ref.element_path == "/DOCUMENT[1]/P[1]/A[1]"


def test_consecutive_p_elements_stay_separate_and_empty_p_is_skipped():
    xml = "<DOCUMENT><P>첫 문단</P><P> </P><P>둘째 문단</P></DOCUMENT>"

    result = parse_paragraphs(xml, document_context=_document_context())

    assert [paragraph.text for paragraph in result.paragraphs] == [
        "첫 문단",
        "둘째 문단",
    ]
    assert [paragraph.paragraph_index for paragraph in result.paragraphs] == [0, 1]


def test_explicit_br_is_preserved_without_inventing_screen_line_breaks():
    result = parse_paragraphs(
        "<DOCUMENT><P>첫 줄<BR/>둘째 줄</P></DOCUMENT>",
        document_context=_document_context(),
    )

    paragraph = result.paragraphs[0]
    assert paragraph.raw_text == "첫 줄\n둘째 줄"
    assert paragraph.text == "첫 줄 둘째 줄"
    assert paragraph.fragments[1].raw_text == "\n"


def test_table_correction_and_title_subtrees_are_not_duplicate_paragraphs():
    xml = """<DOCUMENT>
    <TITLE><P>문서 제목</P></TITLE>
    <CORRECTION><P>정정 내용</P></CORRECTION>
    <TABLE><TR><TD><P>표 내부</P></TD></TR></TABLE>
    <SECTION><P>본문만 남음</P></SECTION>
    </DOCUMENT>"""

    result = parse_paragraphs(xml, document_context=_document_context())

    assert [paragraph.text for paragraph in result.paragraphs] == ["본문만 남음"]


def test_orphan_span_siblings_become_one_span_run_but_bold_title_is_skipped():
    html = """<html><body>
    <span style="font-weight: bold">공시 제목</span>
    <div><span USERMARK="A">첫 조각</span><span>둘째 조각</span></div>
    </body></html>"""

    result = parse_paragraphs(
        html,
        syntax=DocumentSyntax.HTML,
        document_context=_document_context(),
    )

    assert len(result.paragraphs) == 1
    paragraph = result.paragraphs[0]
    assert paragraph.source_kind == ParagraphSourceKind.SPAN_RUN
    assert paragraph.text == "첫 조각둘째 조각"
    assert len(paragraph.source_refs) == 2
    assert paragraph.fragments[0].source_attributes == {"USERMARK": "A"}


def test_evidence_and_document_context_are_preserved_in_output():
    evidence_context = EvidenceContext(
        section_path=("III. 재무에 관한 사항", "1. 요약재무정보"),
        blocks=(
            EvidenceContextBlock(
                role=ContextRole.TITLE,
                position=ContextPosition.BEFORE,
                text="기타 투자판단에 참고할 사항",
            ),
            EvidenceContextBlock(
                role=ContextRole.UNIT,
                position=ContextPosition.BEFORE,
                text="(단위: 원)",
            ),
            EvidenceContextBlock(
                role=ContextRole.NOTE,
                position=ContextPosition.AFTER,
                text="주1) 별도 기준",
            ),
        ),
    )
    result = parse_paragraphs(
        "<DOCUMENT><P>본문</P></DOCUMENT>",
        document_context=_document_context(),
        content_context=evidence_context,
    )

    payload = result.paragraphs[0].to_dict()
    assert payload["schema_version"] == "paragraph.v2"
    assert payload["source_document"]["rcept_no"] == "20250528000403"
    assert payload["context"] == {
        "section_path": ["III. 재무에 관한 사항", "1. 요약재무정보"],
        "blocks": [
            {
                "role": "TITLE",
                "position": "BEFORE",
                "text": "기타 투자판단에 참고할 사항",
                "source_ref": None,
            },
            {
                "role": "UNIT",
                "position": "BEFORE",
                "text": "(단위: 원)",
                "source_ref": None,
            },
            {
                "role": "NOTE",
                "position": "AFTER",
                "text": "주1) 별도 기준",
                "source_ref": None,
            },
        ],
    }


def test_leading_marker_classification():
    xml = """<DOCUMENT>
    <P>※ 참고사항</P><P>주1) 연결 기준</P><P>① 첫째</P><P>1. 항목</P>
    </DOCUMENT>"""

    result = parse_paragraphs(xml, document_context=_document_context())

    assert [paragraph.leading_marker.kind for paragraph in result.paragraphs] == [
        MarkerKind.NOTE,
        MarkerKind.FOOTNOTE,
        MarkerKind.ENUMERATION,
        MarkerKind.ENUMERATION,
    ]


def test_parse_paragraph_element_does_not_mutate_input():
    element = ET.fromstring('<P USERMARK="A">앞<SPAN USERMARK="B">중간</SPAN>뒤</P>')
    before = ET.tostring(element, encoding="unicode")

    parse_paragraph_element(
        element,
        document_context=_document_context(),
        source_ref=SourceRef(
            syntax=DocumentSyntax.DART_XML,
            element_path="/DOCUMENT[1]/P[1]",
        ),
    )

    assert ET.tostring(element, encoding="unicode") == before


def test_recovered_document_marks_collection_and_paragraph():
    result = parse_paragraphs(
        "<DOCUMENT><P>A & B</P></DOCUMENT>",
        document_context=_document_context(),
    )

    assert result.parse_status == ParagraphParseStatus.RECOVERED
    assert result.paragraphs[0].parse_status == ParagraphParseStatus.RECOVERED
    assert result.issues[0].code == "XML_RECOVERED"


def test_explicit_source_ref_excludes_html_subtree():
    html = '<html><body><div id="correction"><p>정정</p></div><p>본문</p></body></html>'

    result = parse_paragraphs(
        html,
        syntax=DocumentSyntax.HTML,
        document_context=_document_context(),
        excluded_source_refs=(
            SourceRef(syntax=DocumentSyntax.HTML, html_id="correction"),
        ),
    )

    assert [paragraph.text for paragraph in result.paragraphs] == ["본문"]
