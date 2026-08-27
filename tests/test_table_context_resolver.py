from __future__ import annotations

from converters.common.source_models import DocumentContext
from converters.paragraph_parser.paragraph_parser import parse_paragraphs
from converters.table_context_resolver.table_context_resolver import (
    resolve_table_contexts,
)
from converters.table_parser.table_parser import parse_table_fragment


def _document_context() -> DocumentContext:
    return DocumentContext(
        doc_id="major_20250528000403",
        rcept_no="20250528000403",
        source_path="sample.xml",
        doc_group="major",
    )


def test_resolver_bundles_photo_style_title_unit_and_trailing_notes():
    paragraphs = parse_paragraphs(
        """<DOCUMENT>
        <P>【자기주식 취득 결정 전 자기주식 보유현황】</P>
        <P>(단위 : 주)</P>
        <P>- 상기 '기초수량'은 당해 사업연도 개시일 기준 보유 수량임.</P>
        <P>- 상기 '기말수량'은 공시서류작성기준일 기준 보유 수량임.</P>
        <P>이 문장은 다음 본문 설명이다.</P>
        </DOCUMENT>""",
        document_context=_document_context(),
    ).paragraphs
    table = parse_table_fragment(
        "<TABLE><TR><TD>구분</TD><TE>수량</TE></TR></TABLE>"
    )
    ordered = (
        paragraphs[0],
        paragraphs[1],
        table,
        paragraphs[2],
        paragraphs[3],
        paragraphs[4],
    )

    result = resolve_table_contexts(
        ordered,
        section_path=("III. 재무에 관한 사항",),
    )

    assert len(result.tables) == 1
    bundle = result.tables[0]
    assert bundle.table_block_index == 2
    assert bundle.table.context.section_path == ("III. 재무에 관한 사항",)
    assert [block.role.value for block in bundle.table.context.blocks] == [
        "TITLE",
        "UNIT",
        "NOTE",
        "NOTE",
    ]
    assert [block.position.value for block in bundle.table.context.blocks] == [
        "BEFORE",
        "BEFORE",
        "AFTER",
        "AFTER",
    ]
    assert bundle.table.title == "【자기주식 취득 결정 전 자기주식 보유현황】"
    assert bundle.table.units == ("(단위 : 주)",)
    assert len(bundle.consumed_source_refs) == 4
    assert len(result.consumed_source_refs) == 4
    assert paragraphs[4].source_refs[0] not in result.consumed_source_refs


def test_resolver_does_not_attach_ambiguous_narrative_or_cross_table_boundary():
    paragraphs = parse_paragraphs(
        "<DOCUMENT><P>일반 설명</P><P>※ 첫 표 주석</P><P>(단위: 원)</P></DOCUMENT>",
        document_context=_document_context(),
    ).paragraphs
    first = parse_table_fragment(
        "<TABLE><TR><TD>A</TD><TE>1</TE></TR></TABLE>"
    )
    second = parse_table_fragment(
        "<TABLE><TR><TD>B</TD><TE>2</TE></TR></TABLE>"
    )

    result = resolve_table_contexts(
        (paragraphs[0], first, paragraphs[1], paragraphs[2], second)
    )

    assert result.tables[0].table.notes == ("※ 첫 표 주석",)
    assert result.tables[1].table.units == ("(단위: 원)",)
    assert paragraphs[0].source_refs[0] not in result.consumed_source_refs


def test_resolver_splits_compound_caption_unit_period_and_trailing_star_note():
    paragraphs = parse_paragraphs(
        """<DOCUMENT>
        <P>(6) 당분기와 전분기 중 자금거래는 다음과 같습니다(단위: 백만원).1) 2026년 1분기</P>
        <P>(*) 자금차입 거래에는 리스거래가 포함되어 있습니다.</P>
        <P>2) 2025년 1분기</P>
        </DOCUMENT>""",
        document_context=_document_context(),
    ).paragraphs
    first = parse_table_fragment(
        "<TABLE><THEAD><TR><TH>구분</TH><TH>금액</TH></TR></THEAD>"
        "<TBODY><TR><TD>A</TD><TD>1</TD></TR></TBODY></TABLE>"
    )
    second = parse_table_fragment(
        "<TABLE><THEAD><TR><TH>구분</TH><TH>금액</TH></TR></THEAD>"
        "<TBODY><TR><TD>B</TD><TD>2</TD></TR></TBODY></TABLE>"
    )

    result = resolve_table_contexts(
        (paragraphs[0], first, paragraphs[1], paragraphs[2], second)
    )

    first_table, second_table = (bundle.table for bundle in result.tables)
    assert first_table.captions == (
        "(6) 당분기와 전분기 중 자금거래는 다음과 같습니다.",
        "1) 2026년 1분기",
    )
    assert first_table.units == ("(단위: 백만원)",)
    assert first_table.notes == (
        "(*) 자금차입 거래에는 리스거래가 포함되어 있습니다.",
    )
    assert second_table.captions == (
        "(6) 당분기와 전분기 중 자금거래는 다음과 같습니다.",
        "2) 2025년 1분기",
    )
    assert second_table.units == ("(단위: 백만원)",)


def test_bold_note_shaped_heading_starts_next_table_instead_of_previous_notes():
    paragraphs = parse_paragraphs(
        """<DOCUMENT>
        <P>※ 이전 표의 실제 주석입니다.</P>
        <P USERMARK="B">※ 작성기준일 이후 변동사항</P>
        <P>변동된 미등기임원의 현황은 다음과 같습니다.</P>
        </DOCUMENT>""",
        document_context=_document_context(),
    ).paragraphs
    first = parse_table_fragment(
        "<TABLE><THEAD><TR><TH>A</TH><TH>B</TH></TR></THEAD>"
        "<TBODY><TR><TD>x</TD><TD>1</TD></TR></TBODY></TABLE>"
    )
    second = parse_table_fragment(
        "<TABLE><THEAD><TR><TH>A</TH><TH>B</TH></TR></THEAD>"
        "<TBODY><TR><TD>y</TD><TD>2</TD></TR></TBODY></TABLE>"
    )

    result = resolve_table_contexts(
        (first, paragraphs[0], paragraphs[1], paragraphs[2], second)
    )

    assert result.tables[0].table.notes == ("※ 이전 표의 실제 주석입니다.",)
    assert result.tables[1].table.context.headings == (
        "※ 작성기준일 이후 변동사항",
    )
    assert result.tables[1].table.captions == (
        "변동된 미등기임원의 현황은 다음과 같습니다.",
    )


def test_compound_heading_chain_is_left_for_semantic_segmentation():
    paragraph = parse_paragraphs(
        """<DOCUMENT><P USERMARK="B">1. 대주주등에 대한 신용공여 등
        가. 채무보증 현황(1) 국내법인 : 해당사항 없음(2) 해외법인</P>
        </DOCUMENT>""",
        document_context=_document_context(),
    ).paragraphs[0]
    table = parse_table_fragment(
        "<TABLE><THEAD><TR><TH>구분</TH><TH>금액</TH></TR></THEAD>"
        "<TBODY><TR><TD>A</TD><TD>1</TD></TR></TBODY></TABLE>"
    )

    result = resolve_table_contexts((paragraph, table))

    assert paragraph.source_refs[0] not in result.consumed_source_refs
    assert result.tables[0].table.context.headings == ()


def test_mixed_heading_narrative_and_caption_is_not_consumed_as_table_context():
    paragraphs = parse_paragraphs(
        """<DOCUMENT>
        <P><SPAN USERMARK="B">(4) 회사의 상표 및 고객관리 정책</SPAN>
        상표 정책에 관한 본문입니다.
        <SPAN USERMARK="B">(5) 지적재산권 보유 현황</SPAN>
        보유한 지적재산권은 아래와 같습니다.</P>
        <P>8. 기타자산당분기말 및 전기말 현재 기타자산의 내역은 다음과 같습니다.</P>
        </DOCUMENT>""",
        document_context=_document_context(),
    ).paragraphs
    table = parse_table_fragment(
        "<TABLE><THEAD><TR><TH>구분</TH><TH>금액</TH></TR></THEAD>"
        "<TBODY><TR><TD>A</TD><TD>1</TD></TR></TBODY></TABLE>"
    )

    first = resolve_table_contexts((paragraphs[0], table))
    second = resolve_table_contexts((paragraphs[1], table))

    assert paragraphs[0].source_refs[0] not in first.consumed_source_refs
    assert paragraphs[1].source_refs[0] not in second.consumed_source_refs
    assert first.tables[0].table.captions == ()
    assert second.tables[0].table.captions == ()


def test_numbered_narrative_sentence_is_left_as_text_before_the_next_table():
    paragraph = parse_paragraphs(
        """<DOCUMENT><P>나. 상기 유형자산 중 일부 토지와 건물은 연결그룹의
        차입금과 관련하여 담보로 제공되어 있습니다(주석 17 참조).</P></DOCUMENT>""",
        document_context=_document_context(),
    ).paragraphs[0]
    table = parse_table_fragment(
        "<TABLE><THEAD><TR><TH>구분</TH><TH>금액</TH></TR></THEAD>"
        "<TBODY><TR><TD>A</TD><TD>1</TD></TR></TBODY></TABLE>"
    )

    result = resolve_table_contexts((paragraph, table))

    assert paragraph.source_refs[0] not in result.consumed_source_refs
    assert result.tables[0].table.context.headings == ()


def test_unknown_angle_period_and_unit_strip_is_context_for_record_table():
    paragraph = parse_paragraphs(
        "<DOCUMENT><P>13. 리스가. 당분기 및 전분기 중 사용권자산의 "
        "변동내역은 다음과 같습니다.</P></DOCUMENT>",
        document_context=_document_context(),
    ).paragraphs[0]
    period_unit = parse_table_fragment(
        "<TABLE><TR><TD>&lt;당분기&gt;</TD><TD>(단위: 백만원)</TD></TR></TABLE>"
    )
    table = parse_table_fragment(
        "<TABLE><THEAD><TR><TH>구분</TH><TH>금액</TH></TR></THEAD>"
        "<TBODY><TR><TD>A</TD><TD>1</TD></TR></TBODY></TABLE>"
    )

    result = resolve_table_contexts((paragraph, period_unit, table))
    resolved = result.tables[0].table

    assert resolved.context.headings == ("13. 리스",)
    assert resolved.captions == (
        "가. 당분기 및 전분기 중 사용권자산의 변동내역은 다음과 같습니다.",
    )
    assert resolved.title == "<당분기>"
    assert resolved.units == ("(단위: 백만원)",)
