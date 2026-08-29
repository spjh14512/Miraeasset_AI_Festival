from agent_graph import system_prompts as sp
from langchain_core.utils.function_calling import convert_to_openai_tool

from agent_graph.state import AnswerGeneratorOutput


def test_answer_generator_prompt_matches_answer_draft_reference_selection():
    prompt = sp.ANSWER_GENERATOR_SYSTEM_PROMPT

    assert "`retrieval_results`" in prompt
    assert "`result_id`" in prompt
    assert "`used_result_ids`" in prompt
    assert "빈 목록" in prompt
    assert "AnswerGeneratorOutput" in prompt
    assert "일반 본문은 `content`" in prompt
    assert "citation_candidates" not in prompt
    assert "selected_retrieval_results" not in prompt


def test_answer_generator_function_description_covers_insufficient_results():
    description = convert_to_openai_tool(AnswerGeneratorOutput)["function"][
        "description"
    ]

    assert "충분하거나 부족한 모든 경우" in description
    assert "근거가 없으면" in description
