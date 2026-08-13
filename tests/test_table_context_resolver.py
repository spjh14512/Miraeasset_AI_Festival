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
