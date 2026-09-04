from openpyxl import Workbook

from vector_db.insert_knowledge_base import (
    KnowledgeItem,
    build_embedding_text,
    build_point_inputs,
    load_knowledge_items,
)


HEADERS = [
    "index",
    "knowledge_type",
    "canonical_term",
    "aliases",
    "description",
    "layperson_description",
    "dataset",
    "category",
    "is_metric",
]


def _write_workbook(path):
    workbook = Workbook()
    term = workbook.active
    term.title = "Term"
    metric = workbook.create_sheet("Metric")
    for sheet in (term, metric):
        sheet.append(HEADERS)
    term.append(
        [
            None,
            "TERM",
            "시가총액",
            "시가총액 | market capitalization | Market Cap",
            "주가와 발행주식수로 본 시장가치",
            "시장이 평가한 회사 몸값",
            "exchange",
            "거래소공시·공정위공시|재무제표·회계·감사",
            "N",
        ]
    )
    metric_row = [
        None,
        "METRIC",
        "현금및현금등가물",
        "현금및현금등가물 | 현금성자산",
        "현금과 단기 현금성 자산",
        None,
        "periodic",
        "재무제표·회계·감사",
        "Y",
    ]
    metric.append(metric_row)
    metric.append(metric_row)
    workbook.save(path)


def test_load_knowledge_items_normalizes_lists_and_deduplicates(tmp_path):
    workbook_path = tmp_path / "knowledge.xlsx"
    _write_workbook(workbook_path)

    items = load_knowledge_items(workbook_path)

    assert len(items) == 2
    assert items[0] == KnowledgeItem(
        knowledge_type="TERM",
        name="시가총액",
        aliases=("시가총액", "market capitalization", "Market Cap"),
        description="주가와 발행주식수로 본 시장가치",
        datasets=("exchange",),
        categories=("거래소공시·공정위공시", "재무제표·회계·감사"),
    )


def test_build_embedding_text_and_point_id_are_deterministic():
    item = KnowledgeItem(
        knowledge_type="METRIC",
        name="현금",
        aliases=(),
        description="즉시 결제에 쓸 수 있는 돈",
        datasets=("periodic",),
        categories=("재무제표·회계·감사",),
    )

    text = build_embedding_text(item)
    first = build_point_inputs([item])[0]
    second = build_point_inputs([item])[0]

    assert text == (
        "지식 유형 : METRIC\n"
        "이름 : 현금\n"
        "설명 : 즉시 결제에 쓸 수 있는 돈\n"
        "데이터셋 : periodic\n"
        "카테고리 : 재무제표·회계·감사"
    )
    assert first.id == second.id
    assert first.payload == {
        "knowledge_type": "METRIC",
        "name": "현금",
        "aliases": [],
        "description": "즉시 결제에 쓸 수 있는 돈",
        "datasets": ["periodic"],
        "categories": ["재무제표·회계·감사"],
    }
