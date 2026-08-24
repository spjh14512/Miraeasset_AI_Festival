from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_graph.state import AiAnswer


def _citation(evidence_id: str):
    return {
        "evidence_id": evidence_id,
        "section_id": "section-1",
        "disclosure_id": "disclosure-1",
    }


def test_ai_answer_accepts_multiple_citations():
    answer = AiAnswer(
        answer="확인된 답변입니다.",
        citation=[_citation("evidence-1"), _citation("evidence-2")],
    )

    assert [item.evidence_id for item in answer.citation] == [
        "evidence-1",
        "evidence-2",
    ]


def test_ai_answer_rejects_single_citation_object():
    with pytest.raises(ValidationError):
        AiAnswer(
            answer="확인된 답변입니다.",
            citation=_citation("evidence-1"),
        )


@pytest.mark.parametrize(
    "citation",
    [
        {"disclosure_id": "disclosure-1"},
        {
            "disclosure_id": "disclosure-1",
            "section_id": "section-1",
        },
        {
            "disclosure_id": "disclosure-1",
            "section_id": "section-1",
            "evidence_id": "evidence-1",
        },
    ],
)
def test_citation_accepts_valid_id_hierarchy(citation):
    answer = AiAnswer(answer="확인된 답변입니다.", citation=[citation])

    assert answer.citation[0].disclosure_id == "disclosure-1"


def test_citation_rejects_evidence_without_section():
    with pytest.raises(ValidationError, match="section_id"):
        AiAnswer(
            answer="확인된 답변입니다.",
            citation=[{
                "disclosure_id": "disclosure-1",
                "evidence_id": "evidence-1",
            }],
        )


@pytest.mark.parametrize(
    "citation",
    [
        {"section_id": "section-1"},
        {"section_id": "section-1", "evidence_id": "evidence-1"},
        {"evidence_id": "evidence-1"},
    ],
)
def test_citation_rejects_ids_without_disclosure(citation):
    with pytest.raises(ValidationError, match="disclosure_id"):
        AiAnswer(answer="확인된 답변입니다.", citation=[citation])
