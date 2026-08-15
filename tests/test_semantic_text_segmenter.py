from __future__ import annotations

from converters.common.source_models import DocumentContext
from converters.evidence_builder.semantic_text_segmenter import (
    SemanticTextRole,
    segment_paragraph,
)
from converters.paragraph_parser.paragraph_parser import parse_paragraphs


def _context() -> DocumentContext:
    return DocumentContext(
        doc_id="periodic_20230512000974",
        rcept_no="20230512000974",
        source_path="sample.xml",
        doc_group="periodic",
    )


def _segments(xml: str):
    paragraph = parse_paragraphs(
        f"<DOCUMENT>{xml}</DOCUMENT>",
        document_context=_context(),
    ).paragraphs[0]
    return segment_paragraph(paragraph)


def test_bold_inline_headings_split_one_p_into_semantic_units():
    segments = _segments(
        """<P><SPAN USERMARK=" B">(4) 회사의 상표 및 고객관리 정책</SPAN>
        당사는 브랜드를 보유하고 있습니다.
        <SPAN USERMARK=" B">(5) 지적재산권 보유 현황</SPAN>
        최근 취득 현황은 아래와 같습니다.</P>"""
    )

    assert [(segment.role, segment.text) for segment in segments] == [
        (SemanticTextRole.HEADING, "(4) 회사의 상표 및 고객관리 정책"),
        (SemanticTextRole.BODY, "당사는 브랜드를 보유하고 있습니다."),
        (SemanticTextRole.HEADING, "(5) 지적재산권 보유 현황"),
        (SemanticTextRole.TABLE_CAPTION, "최근 취득 현황은 아래와 같습니다."),
    ]


def test_concatenated_marker_chain_is_split_without_rewriting_text():
    segments = _segments(
        "<P>1. 일반사항가. 지배기업의 개요</P>"
    )

    assert [(segment.role, segment.text, segment.heading_level) for segment in segments] == [
        (SemanticTextRole.HEADING, "1. 일반사항", 1),
        (SemanticTextRole.HEADING, "가. 지배기업의 개요", 3),
    ]


def test_br_is_a_semantic_boundary_and_reference_stays_with_notice():
    segments = _segments(
        """<P>본문 설명<BR/>※ 관련 내용은
        <A REFNO="20230309000420">2022년도 사업보고서</A>를 참고하시기 바랍니다.</P>"""
    )

    assert [segment.role for segment in segments] == [
        SemanticTextRole.BODY,
        SemanticTextRole.REFERENCE_NOTICE,
    ]
    assert segments[0].references == ()
    assert segments[1].references == (
        {
            "type": "disclosure",
            "refno": "20230309000420",
            "text": "2022년도 사업보고서",
        },
    )


def test_substantive_prefix_is_separated_from_final_table_caption_sentence():
    segments = _segments(
        """<P>지배회사는 선물환계약 883건을 체결하고 있습니다.
        당분기말 보유 중인 파생상품은 다음과 같습니다.</P>"""
    )

    assert [(segment.role, segment.text) for segment in segments] == [
        (SemanticTextRole.BODY, "지배회사는 선물환계약 883건을 체결하고 있습니다."),
        (
            SemanticTextRole.TABLE_CAPTION,
            "당분기말 보유 중인 파생상품은 다음과 같습니다.",
        ),
    ]


def test_sentence_wrapping_and_substring_do_not_create_false_headings():
    segments = _segments(
        "<P>분류합니 다. - 상각후원가로 측정하는 금융자산 해당사항 없습니다.</P>"
    )

    assert [(segment.role, segment.text) for segment in segments] == [
        (
            SemanticTextRole.BODY,
            "분류합니 다. - 상각후원가로 측정하는 금융자산 해당사항 없습니다.",
        )
    ]


def test_numbered_section_name_inside_reference_sentence_is_not_a_heading():
    segments = _segments(
        "<P>[지배회사의 내용] 2023년 사업보고서 'Ⅰ. 회사의 개요 "
        "1. 공시내용 진행 및 변경상황'을 참고하시기 바랍니다.</P>"
    )

    assert segments[0].role == SemanticTextRole.HEADING
    assert segments[0].text == "[지배회사의 내용]"
    assert len(segments) == 2
    assert all(segment.role != SemanticTextRole.HEADING for segment in segments[1:])


def test_concatenated_numeric_heading_and_korean_caption_are_split():
    segments = _segments(
        "<P>13. 리스가. 당분기 및 전분기 중 사용권자산의 "
        "순장부금액 변동내역은 다음과 같습니다.</P>"
    )

    assert [(segment.role, segment.text) for segment in segments] == [
        (SemanticTextRole.HEADING, "13. 리스"),
        (
            SemanticTextRole.TABLE_CAPTION,
            "가. 당분기 및 전분기 중 사용권자산의 순장부금액 변동내역은 "
            "다음과 같습니다.",
        ),
    ]


def test_concatenated_numbered_items_split_after_sentence_end():
    segments = _segments(
        "<P>1) 재무제표 재작성 해당사항 없습니다.2) 합병, 분할</P>"
    )

    assert [(segment.role, segment.text) for segment in segments] == [
        (SemanticTextRole.HEADING, "1) 재무제표 재작성"),
        (SemanticTextRole.BODY, "해당사항 없습니다."),
        (SemanticTextRole.HEADING, "2) 합병, 분할"),
    ]


def test_parenthesized_heading_is_split_from_consolidated_entity_body():
    segments = _segments(
        "<P>(2) 측정 기준 연결실체의 연결재무제표는 역사적원가에 의하여 "
        "작성되었습니다.</P>"
    )

    assert [(segment.role, segment.text) for segment in segments] == [
        (SemanticTextRole.HEADING, "(2) 측정 기준"),
        (
            SemanticTextRole.BODY,
            "연결실체의 연결재무제표는 역사적원가에 의하여 작성되었습니다.",
        ),
    ]
