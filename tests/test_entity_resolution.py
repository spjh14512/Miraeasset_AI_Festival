from __future__ import annotations

import pytest

from agent_graph.state import QuestionAnalysis, SubQuestion
from agent_graph.utils import (
    load_issuer_universe,
    subquestion_has_out_of_universe_issuer,
    validate_question_analysis_issuers,
)


def _analysis(
    mention: str,
    *,
    role: str = "ISSUER",
    canonical_name: str | None = None,
    corp_code: str | None = None,
    match_status: str = "UNKNOWN",
) -> QuestionAnalysis:
    return QuestionAnalysis(
        decision="retrieve",
        normalized_question=f"{mention}의 정보를 알려줘",
        decision_reason="검색이 필요합니다.",
        sub_questions=[SubQuestion(
            question=f"{mention}의 정보는 무엇인가?",
            entities=[{
                "mention": mention,
                "roles": [role],
                "canonical_name": canonical_name,
                "corp_code": corp_code,
                "match_status": match_status,
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


def test_validator_accepts_llm_selected_issuer_from_one_universe_row():
    analysis = _analysis(
        "디엔디파마텍",
        canonical_name="디앤디파마텍",
        corp_code="01376715",
        match_status="UNKNOWN",
    )

    validated = validate_question_analysis_issuers(analysis)

    entity = validated.sub_questions[0].entities[0]
    assert entity.mention == "디엔디파마텍"
    assert entity.canonical_name == "디앤디파마텍"
    assert entity.corp_code == "01376715"
    assert entity.match_status == "UNKNOWN"


def test_validator_converts_exact_name_and_stock_code_pair_to_corp_code():
    analysis = _analysis(
        "삼성전자",
        canonical_name="삼성전자",
        corp_code="005930",
        match_status="MATCHED",
    )

    validated = validate_question_analysis_issuers(analysis)

    entity = validated.sub_questions[0].entities[0]
    assert entity.canonical_name == "삼성전자"
    assert entity.corp_code == "00126380"


def test_validator_rejects_name_and_code_from_different_universe_rows():
    analysis = _analysis(
        "삼성전자",
        canonical_name="삼성전자",
        corp_code="00164779",
        match_status="MATCHED",
    )

    with pytest.raises(ValueError, match="동일한 행") as exc_info:
        validate_question_analysis_issuers(analysis)

    message = str(exc_info.value)
    assert "canonical_name='삼성전자'" in message
    assert "corp_code='00164779'" in message
    assert "corp_name은 'SK하이닉스'" in message


def test_entity_schema_explains_exact_universe_columns():
    entity_schema = SubQuestion.model_json_schema()["$defs"]["EntityMention"]

    assert "corp_name" in entity_schema["properties"]["canonical_name"]["description"]
    assert "corp_eng_name" in entity_schema["properties"]["canonical_name"]["description"]
    assert "동일한" in entity_schema["properties"]["corp_code"]["description"]
    assert "8자리" in entity_schema["properties"]["corp_code"]["description"]
    assert "stock_code" in entity_schema["properties"]["corp_code"]["description"]


def test_validator_accepts_out_of_universe_without_selected_row():
    analysis = _analysis("존재하지않는기업", match_status="OUT_OF_UNIVERSE")

    validated = validate_question_analysis_issuers(analysis)

    entity = validated.sub_questions[0].entities[0]
    assert entity.match_status == "OUT_OF_UNIVERSE"
    assert entity.canonical_name is None
    assert entity.corp_code is None


def test_missing_issuer_pair_is_unresolved_regardless_of_match_status():
    analysis = _analysis("삼성전자", match_status="MATCHED")

    validated = validate_question_analysis_issuers(analysis)

    assert validated.sub_questions[0].entities[0].match_status == "MATCHED"
    assert subquestion_has_out_of_universe_issuer(
        validated.sub_questions[0]
    ) is True


def test_validator_does_not_match_non_issuer_entities():
    analysis = _analysis("외부거래상대방", role="COUNTERPARTY")

    validated = validate_question_analysis_issuers(analysis)

    entity = validated.sub_questions[0].entities[0]
    assert entity.match_status == "UNKNOWN"
    assert entity.canonical_name is None
    assert entity.corp_code is None
