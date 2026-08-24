from agent_graph import system_prompts as sp


def test_answer_generator_prompt_matches_answer_draft_reference_selection():
    prompt = sp.ANSWER_GENERATOR_SYSTEM_PROMPT

    assert "selected_retrieval_results" in prompt
    assert "citation_candidates" in prompt
    assert "reference_id" in prompt
    assert "citation_reference_ids" in prompt
    assert "빈 목록" in prompt
    assert "AnswerDraft" in prompt
