from __future__ import annotations

import xml.etree.ElementTree as ET

from converters.common.source_models import (
    ContextPosition,
    ContextRole,
    EvidenceContext,
    EvidenceContextBlock,
)
from converters.table_parser.table_models import SourceSyntax, TableContext
from converters.table_parser.table_parser import (
    build_logical_grid,
    parse_table,
    parse_table_fragment,
    parse_table_group,
)


def _field_by_value(result, raw_value):
    return next(
        field for field in result.content["fields"] if field["raw_value"] == raw_value
    )


def test_table_preserves_shared_evidence_context_in_canonical_output():
    evidence_context = EvidenceContext(
        section_path=("III. 재무에 관한 사항", "1. 요약재무정보"),
        blocks=(
            EvidenceContextBlock(
                role=ContextRole.TITLE,
                position=ContextPosition.BEFORE,
                text="요약재무정보",
            ),
            EvidenceContextBlock(
                role=ContextRole.UNIT,
                position=ContextPosition.BEFORE,
                text="(단위: 원)",
            ),
            EvidenceContextBlock(
                role=ContextRole.NOTE,
                position=ContextPosition.AFTER,
                text="주1) 연결 기준",
            ),
        ),
    )
    result = parse_table_fragment(
        "<TABLE><TR><TD>항목</TD><TE>10</TE></TR></TABLE>",
        content_context=evidence_context,
    )

    payload = result.to_dict()
    assert payload["schema_version"] == "table.v3"
    assert payload["context"] == {
        "section_path": ["III. 재무에 관한 사항", "1. 요약재무정보"],
        "blocks": [
            {
                "role": "TITLE",
                "position": "BEFORE",
                "text": "요약재무정보",
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
                "text": "주1) 연결 기준",
                "source_ref": None,
            },
        ],
    }
    assert result.title == "요약재무정보"
    assert result.units == ("(단위: 원)",)
    assert result.notes == ("주1) 연결 기준",)


def test_logical_grid_shares_merged_cell_identity():
    table = ET.fromstring(
        """<TABLE><TR><TD ROWSPAN="2" COLSPAN="2">A</TD><TE ACODE="X">1</TE></TR>
        <TR><TE ACODE="Y">2</TE></TR></TABLE>"""
    )
    grid = build_logical_grid(table)

    assert (grid.height, grid.width) == (2, 3)
    assert grid.rows[0][0] is grid.rows[0][1]
    assert grid.rows[0][0] is grid.rows[1][0]
    assert grid.rows[0][0] is grid.rows[1][1]
    assert grid.rows[0][0].row_end == 2
    assert grid.rows[0][0].col_end == 2


def test_simple_kv_keeps_only_final_key_paths_and_source_attributes():
    table = ET.fromstring(
        """<TABLE ACLASS="EXTRACTION"><TR>
        <TD AUPDATECONT="N">회사명</TD><TE ACODE="CRP_NM">(주)에스엠엔터테인먼트</TE>
        <TD AUPDATECONT="N">회사코드</TD><TE ACODE="CRP_CD">041510</TE>
        </TR><TR><TD>법인구분</TD>
        <TU AUNIT="CRP_DST" AUNITVALUE="2">코스닥상장법인</TU>
        <TD>발행주식</TD><TE ACODE="FLT_STK_CNT">23,830,901</TE>
        </TR></TABLE>"""
    )
    result = parse_table(table)
    company = _field_by_value(result, "(주)에스엠엔터테인먼트")

    assert result.table_type.value == "KV_TABLE"
    assert company["key_paths"] == [{"path": ["회사명"], "cells": ["c0"]}]
    assert "row_paths" not in company
    assert "column_paths" not in company
    assert "code" not in company
    assert result.cells[1].source_attributes["ACODE"] == "CRP_NM"


def test_hierarchical_kv_context_and_colspan_do_not_leak_previous_labels():
    table = ET.fromstring(
        """<TABLE><TR><TD>발행회사명</TD><TE ACODE="CRP_NM">회사</TE>
        <TD>발행회사와의 관계</TD><TU AUNIT="RLT" AUNITVALUE="02">주요주주</TU></TR>
        <TR><TD>보고구분</TD><TU COLSPAN="3" AUNIT="RPT" AUNITVALUE="2">변동</TU></TR>
        <TR><TD ROWSPAN="3">보유주식등의 수 및 보유비율</TD><TD></TD>
        <TD>보유주식등의 수</TD><TD>보유비율</TD></TR>
        <TR><TD>직전 보고서</TD><TE ACODE="SUM_BMT_CNT">2,098,811</TE>
        <TE ACODE="SUM_BMT_RT">8.81</TE></TR>
        <TR><TD>이번 보고서</TD><TE ACODE="SUM_TMT_CNT">2,967,759</TE>
        <TE ACODE="SUM_TMT_RT">12.45</TE></TR></TABLE>"""
    )
    result = parse_table(table)

    report_type = _field_by_value(result, "변동")
    previous_count = _field_by_value(result, "2,098,811")
    current_count = _field_by_value(result, "2,967,759")
    current_ratio = _field_by_value(result, "12.45")
    assert report_type["key_paths"] == [
        {"path": ["보고구분"], "cells": ["c4"]}
    ]
    assert previous_count["key_paths"][0]["path"] == [
        "보유주식등의 수 및 보유비율",
        "직전 보고서",
        "보유주식등의 수",
    ]
    assert current_count["key_paths"][0]["path"] == [
        "보유주식등의 수 및 보유비율",
        "이번 보고서",
        "보유주식등의 수",
    ]
    assert current_ratio["key_paths"][0]["path"] == [
        "보유주식등의 수 및 보유비율",
        "이번 보고서",
        "보유비율",
    ]


def test_merged_value_retains_multiple_contexts():
    table = ET.fromstring(
        """<TABLE><TR><TD>A</TD><TE ROWSPAN="2" ACODE="X">shared</TE></TR>
        <TR><TD>B</TD></TR></TABLE>"""
    )
    result = parse_table(table)
    field = _field_by_value(result, "shared")

    assert field["context_status"] == "MULTI_CONTEXT"
    assert [item["path"] for item in field["key_paths"]] == [["A"], ["B"]]


def test_nested_p_and_span_keep_paragraph_boundaries():
    table = ET.fromstring(
        """<TABLE><TR><TD>설명</TD><TE ACODE="DETAIL">
        <P>- 우선주</P><P><SPAN>ㆍ</SPAN><SPAN>보통주보다 추가 배당</SPAN></P>
        </TE></TR></TABLE>"""
    )
    result = parse_table(table)
    value_cell = result.cells[1]

    assert value_cell.text_segments == ("- 우선주", "ㆍ보통주보다 추가 배당")
    assert value_cell.raw_value == "- 우선주\nㆍ보통주보다 추가 배당"


def test_aupdatecont_is_metadata_not_a_role_rule():
    table = ET.fromstring(
        """<TABLE><TR><TD>금액</TD>
        <TE ACODE="AMOUNT" AUPDATECONT="N">320,939,749,998</TE>
        </TR></TABLE>"""
    )
    result = parse_table(table)
    cell = result.cells[1]

    assert result.cell_roles[cell.id].value == "VALUE"
    assert cell.source_attributes["AUPDATECONT"] == "N"


def test_tu_preserves_aunit_and_aunitvalue_without_dedicated_code_fields():
    result = parse_table_fragment(
        '<TABLE><TR><TD>법인구분</TD><TU AUNIT="CRP_DST" '
        'AUNITVALUE="2">코스닥상장법인</TU></TR></TABLE>'
    )
    payload = result.to_dict()
    value_cell = payload["cells"][1]

    assert value_cell["source_attributes"] == {
        "AUNIT": "CRP_DST",
        "AUNITVALUE": "2",
    }
    assert "code" not in value_cell
    assert "code_type" not in value_cell


def test_simple_record_table_extracts_columns_and_records():
    table = ET.fromstring(
        """<TABLE><THEAD><TR><TH>연번</TH><TH>성명</TH><TH>구분</TH></TR></THEAD>
        <TBODY><TR><TE ACODE="SEQ">1</TE><TE ACODE="NAME">북일학원</TE>
        <TU AUNIT="TYPE" AUNITVALUE="N">기타단체(국내)</TU></TR>
        <TR><TE ACODE="SEQ">2</TE><TE ACODE="NAME">이구영</TE>
        <TU AUNIT="TYPE" AUNITVALUE="P">개인(국내)</TU></TR></TBODY></TABLE>"""
    )
    result = parse_table(table)

    assert result.table_type.value == "R_TABLE"
    assert [column["header_path"] for column in result.content["columns"]] == [
        ["연번"],
        ["성명"],
        ["구분"],
    ]
    assert len(result.content["records"]) == 2
    assert result.content["records"][0]["cells_by_column"] == {
        "col_0": "c3",
        "col_1": "c4",
        "col_2": "c5",
    }


def test_record_without_thead_is_inferred_from_repeated_rows():
    table = ET.fromstring(
        """<TABLE ACLASS="NORMAL"><TR><TD>구분</TD><TD>내용</TD></TR>
        <TR><TD>발행회사의 경영</TD><TD>의결권을 공동으로 행사</TD></TR>
        <TR><TD>주식처분제한</TD><TD>제3자 처분 제한</TD></TR></TABLE>"""
    )
    result = parse_table(table)

    assert result.table_type.value == "R_TABLE"
    assert [column["header_path"] for column in result.content["columns"]] == [
        ["구분"],
        ["내용"],
    ]
    assert len(result.content["records"]) == 2


def test_plain_td_key_value_matrix_is_not_mistaken_for_record_header():
    table = ET.fromstring(
        """<TABLE><TBODY>
        <TR><TD>정관상신주인수권 내용</TD><TD COLSPAN="3">제9조 신주 발행 및 배정</TD></TR>
        <TR><TD>결산일</TD><TD>12월 31일</TD><TD>정기주주총회개최</TD><TD>3개월 이내</TD></TR>
        <TR><TD>기준일</TD><TD COLSPAN="3">매년 이사회 결의로 정함</TD></TR>
        <TR><TD>주권의 종류</TD><TD COLSPAN="3">전자등록</TD></TR>
        <TR><TD>명의개서대리인</TD><TD COLSPAN="3">KB국민은행</TD></TR>
        <TR><TD>주주의 특전</TD><TD>해당사항 없음</TD><TD>공고방법</TD><TD>회사 홈페이지</TD></TR>
        </TBODY></TABLE>"""
    )
    result = parse_table(table)

    assert result.table_type.value == "KV_TABLE"
    assert [field["raw_value"] for field in result.content["fields"]] == [
        "제9조 신주 발행 및 배정",
        "12월 31일",
        "3개월 이내",
        "매년 이사회 결의로 정함",
        "전자등록",
        "KB국민은행",
        "해당사항 없음",
        "회사 홈페이지",
    ]
    assert [
        field["key_paths"][0]["path"] for field in result.content["fields"]
    ] == [
        ["정관상신주인수권 내용"],
        ["결산일"],
        ["정기주주총회개최"],
        ["기준일"],
        ["주권의 종류"],
        ["명의개서대리인"],
        ["주주의 특전"],
        ["공고방법"],
    ]


def test_multilevel_record_header_row_context_and_totals():
    table = ET.fromstring(
        """<TABLE><THEAD><TR>
        <TH ROWSPAN="2" COLSPAN="3">취득방법</TH><TH ROWSPAN="2">주식의 종류</TH>
        <TH ROWSPAN="2">기초수량</TH><TH COLSPAN="3">변동 수량</TH><TH ROWSPAN="2">기말수량</TH>
        </TR><TR><TH>취득(+)</TH><TH>처분(-)</TH><TH>소각(-)</TH></TR></THEAD>
        <TBODY><TR><TD ROWSPAN="2">배당가능이익범위이내취득</TD><TD ROWSPAN="2">직접취득</TD>
        <TD>장내 직접 취득</TD><TD>보통주식</TD><TE ACODE="BGN">8</TE>
        <TE ACODE="ACQ">-</TE><TE ACODE="TRD">-</TE><TE ACODE="RTM">2</TE><TE ACODE="END">6</TE></TR>
        <TR><TD>소계(a)</TD><TD>보통주식</TD><TE ACODE="SBGN">8</TE>
        <TE ACODE="SACQ">-</TE><TE ACODE="STRD">-</TE><TE ACODE="SRTM">2</TE><TE ACODE="SEND">6</TE></TR>
        <TR><TD COLSPAN="3">총 계(a+b+c)</TD><TD>보통주식</TD><TE ACODE="TBGN">8</TE>
        <TE ACODE="TACQ">-</TE><TE ACODE="TTRD">-</TE><TE ACODE="TRTM">2</TE><TE ACODE="TEND">6</TE></TR>
        </TBODY></TABLE>"""
    )
    result = parse_table(table)
    columns = result.content["columns"]
    records = result.content["records"]

    assert columns[5]["header_path"] == ["변동 수량", "취득(+)"]
    assert records[0]["row_context"] == [
        "배당가능이익범위이내취득",
        "직접취득",
        "장내 직접 취득",
        "보통주식",
    ]
    assert records[1]["row_type"] == "SUBTOTAL"
    assert records[2]["row_type"] == "TOTAL"
    assert records[0]["cells_by_column"]["col_0"] == records[1]["cells_by_column"]["col_0"]


def test_layout_and_table_group_attach_title_and_unit():
    group = ET.fromstring(
        """<TABLE-GROUP ACLASS="TBL_OWN_STK">
        <TABLE ACLASS="NORMAL"><TR><TD>【자기주식 취득 결정 전 자기주식 보유현황】</TD></TR></TABLE>
        <TABLE ACLASS="EXTRACTION"><TR><TU AUNIT="STOCK" AUNITVALUE="1">(단위 : 주)</TU></TR></TABLE>
        <TABLE ACLASS="EXTRACTION"><THEAD><TR><TH>구분</TH><TH>수량</TH></TR></THEAD>
        <TBODY><TR><TD>보통주식</TD><TE ACODE="COUNT">10</TE></TR></TBODY></TABLE>
        <TABLE ACLASS="NORMAL"><TR><TD>- 상기 '기초수량'은 사업연도 개시일 기준임.</TD></TR></TABLE>
        <TABLE ACLASS="NORMAL"><TR><TD>- 상기 '기말수량'은 작성기준일 기준임.</TD></TR></TABLE>
        </TABLE-GROUP>"""
    )
    result = parse_table_group(group, context=TableContext(table_index=4))

    assert [block.role.value for block in result.context_blocks] == [
        "TITLE",
        "UNIT",
        "NOTE",
        "NOTE",
    ]
    assert [block.position.value for block in result.context_blocks] == [
        "BEFORE",
        "BEFORE",
        "AFTER",
        "AFTER",
    ]
    assert len(result.tables) == 1
    assert result.tables[0].title == "【자기주식 취득 결정 전 자기주식 보유현황】"
    assert result.tables[0].units == ("(단위 : 주)",)
    assert result.tables[0].notes == (
        "- 상기 '기초수량'은 사업연도 개시일 기준임.",
        "- 상기 '기말수량'은 작성기준일 기준임.",
    )


def test_table_group_attaches_compound_base_date_and_unit_strip():
    group = ET.fromstring(
        """<TABLE-GROUP ACLASS="SUB_PIS">
        <TABLE ACLASS="EXTRACTION"><TR>
          <TD>(기준일 : </TD>
          <TU AUNIT="BASE_DT" AUNITVALUE="20230331">2023년 03월 31일</TU>
          <TD>)</TD>
          <TU AUNIT="WONPERCENT" AUNITVALUE="3">(단위 : 백만원, %)</TU>
        </TR></TABLE>
        <TABLE ACLASS="EXTRACTION"><THEAD><TR>
          <TH>발행회사</TH><TH>권면총액</TH>
        </TR></THEAD><TBODY><TR>
          <TE ACODE="COMPANY">삼성전자㈜</TE><TE ACODE="AMOUNT">130,380</TE>
        </TR></TBODY></TABLE>
        </TABLE-GROUP>"""
    )

    result = parse_table_group(group, context=TableContext(table_index=10))

    assert len(result.tables) == 1
    assert result.tables[0].table_type.value == "R_TABLE"
    assert result.tables[0].captions == ("(기준일 : 2023년 03월 31일)",)
    assert result.tables[0].units == ("(단위 : 백만원, %)",)
    assert [block.role.value for block in result.context_blocks] == [
        "CAPTION",
        "UNIT",
    ]
    assert [block.position.value for block in result.context_blocks] == [
        "BEFORE",
        "BEFORE",
    ]


def test_table_group_keeps_non_context_single_row_kv_table():
    group = ET.fromstring(
        """<TABLE-GROUP>
        <TABLE><TR>
          <TD>기준일</TD><TU AUNIT="BASE_DT">2023년 03월 31일</TU>
          <TD>단위</TD><TU AUNIT="TEXT">백만원</TU>
        </TR></TABLE>
        <TABLE><THEAD><TR><TH>항목</TH><TH>금액</TH></TR></THEAD>
        <TBODY><TR><TD>A</TD><TE ACODE="AMOUNT">10</TE></TR></TBODY></TABLE>
        </TABLE-GROUP>"""
    )

    result = parse_table_group(group)

    assert [table.table_type.value for table in result.tables] == [
        "KV_TABLE",
        "R_TABLE",
    ]
    assert result.context_blocks == ()


def test_table_group_attaches_split_date_caption_and_unit_strip():
    group = ET.fromstring(
        """<TABLE-GROUP ACLASS="{XBRL}NT_C_U800900">
        <TABLE><TR><TE><P>파생상품의 평가내역(기준일: 당기 2023년 12월 31일, 전기 2022년 12월 31일)</P></TE></TR></TABLE>
        <TABLE><TR><TE><P>당기</P></TE><TE><P>(단위 : 천원)</P></TE></TR></TABLE>
        <TABLE><THEAD><TR><TH>금융상품</TH><TH>금액</TH></TR></THEAD>
        <TBODY><TR><TD>통화선도</TD><TE ACODE="AMOUNT">10</TE></TR></TBODY></TABLE>
        </TABLE-GROUP>"""
    )

    result = parse_table_group(group)

    assert len(result.tables) == 1
    assert result.tables[0].captions == (
        "파생상품의 평가내역(기준일: 당기 2023년 12월 31일, 전기 2022년 12월 31일)",
        "당기",
    )
    assert result.tables[0].units == ("(단위 : 천원)",)
    assert result.tables[0].title is None


def test_table_group_attaches_split_context_title_period_and_unit():
    group = ET.fromstring(
        """<TABLE-GROUP ACLASS="{XBRL}NT_C_D861300">
        <TABLE><TR><TE><P>분기배당(배당기준일: 2024년 3월 31일, 2024년 6월 30일)</P></TE></TR></TABLE>
        <TABLE>
          <TR><TE COLSPAN="2"><P>배당금에 대한 공시</P></TE></TR>
          <TR><TE><P>당반기</P></TE><TE><P>(단위 : 천원)</P></TE></TR>
        </TABLE>
        <TABLE><THEAD><TR><TH>구분</TH><TH>금액</TH></TR></THEAD>
        <TBODY><TR><TD>현금배당</TD><TE ACODE="AMOUNT">10</TE></TR></TBODY></TABLE>
        </TABLE-GROUP>"""
    )

    result = parse_table_group(group)

    assert len(result.tables) == 1
    assert result.tables[0].title == "배당금에 대한 공시"
    assert result.tables[0].captions == (
        "분기배당(배당기준일: 2024년 3월 31일, 2024년 6월 30일)",
        "당반기",
    )
    assert result.tables[0].units == ("(단위 : 천원)",)


def test_table_group_keeps_split_context_candidate_with_real_kv_value():
    group = ET.fromstring(
        """<TABLE-GROUP>
        <TABLE><TR><TE><P>평가기준일: 2024년 6월 30일</P></TE></TR></TABLE>
        <TABLE><TR><TD>보고기간</TD><TE ACODE="PERIOD">당반기</TE><TD>(단위 : 천원)</TD></TR></TABLE>
        <TABLE><THEAD><TR><TH>구분</TH><TH>금액</TH></TR></THEAD>
        <TBODY><TR><TD>A</TD><TE ACODE="AMOUNT">10</TE></TR></TBODY></TABLE>
        </TABLE-GROUP>"""
    )

    result = parse_table_group(group)

    assert [table.table_type.value for table in result.tables] == [
        "KV_TABLE",
        "R_TABLE",
    ]


def test_navigation_layout_is_ignored_but_anchor_title_is_preserved():
    navigation_group = ET.fromstring(
        """<TABLE-GROUP>
        <TABLE><TR><TD><A REFNO="I. 회사의 개요|ADX-001">☞ 본문 위치로 이동</A></TD></TR></TABLE>
        <TABLE><TR><TD>회사명</TD><TE ACODE="CRP_NM">미래</TE></TR></TABLE>
        </TABLE-GROUP>"""
    )

    navigation = parse_table_group(navigation_group)

    assert len(navigation.tables) == 1
    assert navigation.tables[0].title is None
    assert navigation.context_blocks == ()

    linked_title_group = ET.fromstring(
        """<TABLE-GROUP>
        <TABLE><TR><TD><A REFNO="homepage">회사 홈페이지 바로가기</A></TD></TR></TABLE>
        <TABLE><TR><TD>회사명</TD><TE ACODE="CRP_NM">미래</TE></TR></TABLE>
        </TABLE-GROUP>"""
    )

    linked_title = parse_table_group(linked_title_group)

    assert linked_title.tables[0].title == "회사 홈페이지 바로가기"
    assert [block.role.value for block in linked_title.context_blocks] == ["TITLE"]


def test_explicit_html_caption_is_kept_as_context():
    result = parse_table_fragment(
        "<html><body><table><caption>연결재무제표</caption>"
        "<tr><td>항목<td><span class='xforms_input'>10</span>"
        "</table></body></html>",
        context=TableContext(table_index=3),
    )

    assert result.captions == ("연결재무제표",)
    assert result.context.blocks[0].role == ContextRole.CAPTION
    assert result.context.blocks[0].source_ref is not None
    assert result.context.blocks[0].source_ref.table_index == 3


def test_unknown_table_is_not_forced_into_kv_or_record():
    table = ET.fromstring(
        "<TABLE><TR><TD>A</TD></TR><TR><TD>B</TD><TD>C</TD></TR></TABLE>"
    )
    result = parse_table(table)

    assert result.table_type.value == "UNKNOWN"
    assert result.content["rows"]


def test_empty_and_dash_values_are_distinct():
    table = ET.fromstring(
        """<TABLE><TR><TD>빈 값</TD><TE ACODE="EMPTY"></TE>
        <TD>대시</TD><TE ACODE="DASH">-</TE></TR></TABLE>"""
    )
    result = parse_table(table)

    assert _field_by_value(result, "")["raw_value"] == ""
    assert _field_by_value(result, "-")["raw_value"] == "-"


def test_exchange_html_fragment_uses_xforms_input_as_value():
    html = """<html><body><table id="XFormD1_Form0_Table0">
    <tr><td><span>1. 계약명</span><td><span class="xforms_input">LNG선 2척</span>
    <tr><td rowspan="2"><span>2. 계약기간</span><td><span>시작일</span>
    <td><span class="xforms_input">2023-03-18</span>
    <tr><td><span>종료일</span><td><span class="xforms_input">2026-08-31</span>
    <tr><td colspan="3"><span>3. 기타 투자판단과 관련한 중요사항</span>
    <tr><td colspan="3"><span class="xforms_input">참고 내용</span>
    </table></body></html>"""
    result = parse_table_fragment(html, syntax=SourceSyntax.AUTO)

    assert result.source["syntax"] == "HTML"
    assert result.table_type.value == "KV_TABLE"
    assert _field_by_value(result, "LNG선 2척")["key_paths"][0]["path"] == ["1. 계약명"]
    assert _field_by_value(result, "2026-08-31")["key_paths"][0]["path"] == [
        "2. 계약기간",
        "종료일",
    ]
    assert _field_by_value(result, "참고 내용")["key_paths"][0]["path"] == [
        "3. 기타 투자판단과 관련한 중요사항"
    ]


def test_recovery_and_input_non_mutation():
    raw = '<TABLE><TR><TD>회사</TD><TE ACODE="NAME">C&T</TE></TR></TABLE>'
    element = ET.fromstring(
        '<TABLE><TR><TD>회사</TD><TE ACODE="NAME">A</TE></TR></TABLE>'
    )
    before = ET.tostring(element)

    recovered = parse_table_fragment(raw)
    parse_table(element)

    assert recovered.parse_status.value == "RECOVERED"
    assert _field_by_value(recovered, "C&T")["raw_value"] == "C&T"
    assert ET.tostring(element) == before
