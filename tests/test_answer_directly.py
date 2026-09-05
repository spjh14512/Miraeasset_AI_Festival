from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_graph.graph import answer_directly
from agent_graph.state import AiAnswer


class _FakeDirectAnswerLlm:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return SimpleNamespace(content=self.content)


def _state(question: str = "안녕하세요") -> dict:
    return {"question_id": "question-1", "question_text": question}


def test_returns_llm_response_as_ai_answer():
    fake_llm = _FakeDirectAnswerLlm("안녕하세요! 무엇을 도와드릴까요?")

    update = answer_directly(_state(), llm=fake_llm)

    assert update == {
        "ai_answer": AiAnswer(
            answer="안녕하세요! 무엇을 도와드릴까요?",
            citation=[],
        )
    }
    assert fake_llm.calls[0][-1].content == "안녕하세요"


def test_rejects_blank_question_text():
    fake_llm = _FakeDirectAnswerLlm("응답")

    with pytest.raises(ValueError, match="question_text는 비어 있을 수 없습니다"):
        answer_directly(_state("   "), llm=fake_llm)
