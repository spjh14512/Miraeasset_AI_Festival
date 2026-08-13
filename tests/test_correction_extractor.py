from __future__ import annotations

import xml.etree.ElementTree as ET

from converters.common.source_models import DocumentContext, DocumentSyntax
from converters.correction_extractor.correction_extractor import extract_correction
from converters.correction_extractor.correction_models import CorrectionStatus


def _context(rcept_no: str = "20241118000171") -> DocumentContext:
    return DocumentContext(
        doc_id=f"major_{rcept_no}",
        rcept_no=rcept_no,
        source_path=f"data/raw/major/example/{rcept_no}.xml",
        doc_group="major",
    )


def _xml_correction(reason: str = "기재 내용 정정") -> str:
    return f"""
    <DOCUMENT><BODY><LIBRARY><CORRECTION>
      <TITLE AASSOCNOTE="CORRECTION">정 정 신 고 (보고)</TITLE>
      <TABLE><TR><TD></TD></TR><TR><TD>2024년 11월 18일</TD></TR></TABLE>
      <P>1. 정정대상 공시서류 : 주요사항보고서(자기주식취득결정)</P>
      <P>2. 정정대상 공시서류의 최초제출일 : 2024년 07월 25일</P>
      <P>3. 정정사항</P>
      <TABLE>
        <THEAD><TR><TH>항목</TH><TH>정정사유</TH><TH>정 정 전</TH><TH>정 정 후</TH></TR></THEAD>
        <TBODY><TR><TD>취득예정금액</TD><TD>{reason}</TD><TD>100</TD><TD>200</TD></TR></TBODY>
      </TABLE>
    </CORRECTION></LIBRARY>
    <SECTION-1><TITLE>자기주식 취득 결정</TITLE>
      <TABLE><TR><TD>본문</TD><TE ACODE="VALUE">200</TE></TR></TABLE>
    </SECTION-1></BODY></DOCUMENT>
    """


def _html_correction() -> str:
    return """
    <html><body>
      <div id="LIB_LC000">
        <span>정정신고(보고)</span>
        <table id="XFormD8_Form0_Table1"><tr>
          <td>정정일자</td><td><span class="xforms_input">2025-07-31</span></td>
        </tr></table>
        <table id="XFormD8_Form0_RepeatTable0">
          <tr><td>1. 정정관련 공시서류</td><td>단일판매ㆍ공급계약 체결</td></tr>
          <tr><td>2. 정정관련 공시서류제출일</td><td>2025-07-28</td></tr>
          <tr><td>3. 정정사유</td><td>계약상대방 공개</td></tr>
          <tr><td colspan="3">4. 정정사항</td></tr>
          <tr><td>정정항목</td><td>정정전</td><td>정정후</td></tr>
        </table>
        <table id="XFormD8_Form0_Table0"><tr><td>-</td></tr></table>
      </div>
      <table id="XFormD1_Form0_Table0"><tr><td>본문</td><td>값</td></tr></table>
    </body></html>
    """


def test_xml_extracts_metadata_and_keeps_current_rcept_no_from_context():
    result = extract_correction(_xml_correction(), context=_context())
    payload = result.to_dict()

    assert result.status == CorrectionStatus.FOUND
    assert payload["source_document"]["rcept_no"] == "20241118000171"
    assert payload["correction"] == {
        "title": "정 정 신 고 (보고)",
        "correction_date": "2024-11-18",
        "target_document_name": "주요사항보고서(자기주식취득결정)",
        "original_submission_date": "2024-07-25",
        "reason": "기재 내용 정정",
        "target_rcept_no": None,
        "source_ref": {
            "syntax": "DART_XML",
            "element_path": "//CORRECTION[1]",
            "html_id": None,
        },
    }
    assert payload["excluded_source_refs"] == [
        {
            "syntax": "DART_XML",
            "element_path": "//CORRECTION[1]",
            "html_id": None,
        }
    ]
    assert [block["block_type"] for block in payload["correction_blocks"]] == [
        "TITLE",
        "TABLE",
        "PARAGRAPH",
        "PARAGRAPH",
        "PARAGRAPH",
        "TABLE",
    ]


def test_current_rcept_no_does_not_depend_on_correction_date():
    context = _context("20240925000218")
    result = extract_correction(_xml_correction(), context=context)

    assert result.source_document.rcept_no == "20240925000218"
    assert result.correction is not None
    assert result.correction.correction_date == "2024-11-18"


def test_xml_element_input_is_not_mutated():
    element = ET.fromstring(_xml_correction())
    before = ET.tostring(element)

    result = extract_correction(element, context=_context())

    assert result.status == CorrectionStatus.FOUND
    assert ET.tostring(element) == before


def test_xml_recovery_preserves_bare_ampersand_reason():
    result = extract_correction(
        _xml_correction("C&T 계약명 정정"),
        context=_context(),
    )

    assert result.status == CorrectionStatus.RECOVERED
    assert result.correction is not None
    assert result.correction.reason == "C&T 계약명 정정"
    assert result.issues[0].code == "XML_RECOVERED"


def test_xml_metadata_can_be_split_across_table_cells():
    document = """
    <DOCUMENT><BODY><LIBRARY><CORRECTION>
      <TITLE AASSOCNOTE="CORRECTION">정 정 신 고 (보고)</TITLE>
      <TABLE><TR><TD>2025년 2월 18일</TD></TR></TABLE>
      <TABLE><TR><TD>1. 정정대상 공시서류 :</TD>
        <TD>주요사항보고서(자기주식취득결정)</TD></TR></TABLE>
      <TABLE><TR><TD>2. 정정대상 공시서류의 최초제출일 :</TD>
        <TD>2025년 2월 10일</TD></TR></TABLE>
      <TABLE><TR><TH>항목</TH><TH>정정사유</TH></TR>
        <TR><TD>취득예정금액</TD><TD>기재 내용 정정</TD></TR></TABLE>
    </CORRECTION></LIBRARY></BODY></DOCUMENT>
    """

    result = extract_correction(document, context=_context())

    assert result.correction is not None
    assert result.correction.target_document_name == "주요사항보고서(자기주식취득결정)"
    assert result.correction.original_submission_date == "2025-02-10"


def test_xml_target_can_follow_label_paragraph_and_date_allows_spaces():
    document = """
    <DOCUMENT><BODY><LIBRARY><CORRECTION>
      <TITLE AASSOCNOTE="CORRECTION">정 정 신 고 (보고)</TITLE>
      <TABLE><TR><TD>2024년 6월 20일</TD></TR></TABLE>
      <P>1. 정정대상 공시서류 :</P>
      <P>주식등의 대량보유상황보고서</P>
      <P>2. 정정대상 공시서류의 최초제출일 : 2024. 6. 19.</P>
      <TABLE><TR><TH>정정사유</TH></TR><TR><TD>단순 기재 정정</TD></TR></TABLE>
    </CORRECTION></LIBRARY></BODY></DOCUMENT>
    """

    result = extract_correction(document, context=_context())

    assert result.correction is not None
    assert result.correction.target_document_name == "주식등의 대량보유상황보고서"
    assert result.correction.original_submission_date == "2024-06-19"


def test_non_correction_phrase_is_not_a_false_positive():
    document = """
    <DOCUMENT><BODY><SECTION-1>
      <P>과거 정정신고(보고)의 내용을 참고하시기 바랍니다.</P>
      <TABLE><TR><TD>본문</TD></TR></TABLE>
    </SECTION-1></BODY></DOCUMENT>
    """

    result = extract_correction(document, context=_context())

    assert result.status == CorrectionStatus.NOT_FOUND
    assert result.has_correction is False
    assert result.excluded_source_refs == ()


def test_exchange_html_uses_container_and_three_d8_tables():
    context = DocumentContext(
        doc_id="exchange_20250731800028",
        rcept_no="20250731800028",
        doc_group="exchange",
    )
    result = extract_correction(_html_correction(), context=context)

    assert result.syntax == DocumentSyntax.HTML
    assert result.status == CorrectionStatus.FOUND
    assert result.correction is not None
    assert result.correction.correction_date == "2025-07-31"
    assert result.correction.target_document_name == "단일판매ㆍ공급계약 체결"
    assert result.correction.original_submission_date == "2025-07-28"
    assert result.correction.reason == "계약상대방 공개"
    assert [block.source_ref.html_id for block in result.correction_blocks] == [
        "XFormD8_Form0_Table1",
        "XFormD8_Form0_RepeatTable0",
        "XFormD8_Form0_Table0",
    ]
    assert [ref.html_id for ref in result.excluded_source_refs] == ["LIB_LC000"]


def test_exchange_html_can_fall_back_to_d8_table_ids():
    html = _html_correction().replace('<div id="LIB_LC000">', "<section>").replace(
        "</div>", "</section>"
    )
    result = extract_correction(html, context=_context(), syntax=DocumentSyntax.HTML)

    assert result.status == CorrectionStatus.FOUND
    assert [ref.html_id for ref in result.excluded_source_refs] == [
        "XFormD8_Form0_Table1",
        "XFormD8_Form0_RepeatTable0",
        "XFormD8_Form0_Table0",
    ]


def test_html_without_correction_markers_is_not_found():
    html = """
    <html><body><table id="XFormD1_Form0_Table0">
      <tr><td>제목</td><td>값</td></tr>
    </table></body></html>
    """

    result = extract_correction(html, context=_context())

    assert result.syntax == DocumentSyntax.HTML
    assert result.status == CorrectionStatus.NOT_FOUND
