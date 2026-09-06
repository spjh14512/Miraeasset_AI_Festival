from __future__ import annotations

from agent_graph.state import QuestionAnalysis, SubQuestion
from agent_graph.utils import (
    _issuer_universe_matches,
    load_issuer_universe,
    normalize_question_analysis_entities,
)


def _analysis(mention: str, role: str = "ISSUER") -> QuestionAnalysis:
    return QuestionAnalysis(
        decision="retrieve",
        normalized_question=f"{mention}의 정보를 알려줘",
        decision_reason="검색이 필요합니다.",
        sub_questions=[SubQuestion(
            question=f"{mention}의 정보는 무엇인가?",
            entities=[{
                "mention": mention,
                "roles": [role],
                "match_status": "UNKNOWN",
            }],
            intents=["DETAIL"],
            requested_facts=["기업 정보"],
        )],
    )


def test_issuer_universe_uses_english_headers():
    row = load_issuer_universe()[0]

    assert "corp_code" in row
    assert "stock_code" in row
    assert "corp_name" in row
    assert "listed_name" in row
    assert "corp_eng_name" in row


def test_issuer_universe_matches_company_name_and_stock_code():
    assert _issuer_universe_matches("삼성전자")[0]["corp_name"] == "삼성전자"
    assert _issuer_universe_matches("005930")[0]["corp_name"] == "삼성전자"


def test_normalize_matches_supported_issuer():
    normalized = normalize_question_analysis_entities(_analysis("삼성전자"))
    entity = normalized.sub_questions[0].entities[0]

    assert entity.match_status == "MATCHED"
    assert entity.canonical_name == "삼성전자"


def test_normalize_marks_unsupported_issuer_out_of_universe():
    normalized = normalize_question_analysis_entities(_analysis("존재하지않는기업"))
    entity = normalized.sub_questions[0].entities[0]

    assert entity.match_status == "OUT_OF_UNIVERSE"
    assert entity.canonical_name is None


def test_normalize_keeps_unsupported_non_issuer_unknown():
    normalized = normalize_question_analysis_entities(
        _analysis("외부거래상대방", role="COUNTERPARTY")
    )
    entity = normalized.sub_questions[0].entities[0]

    assert entity.match_status == "UNKNOWN"
    assert entity.canonical_name is None
