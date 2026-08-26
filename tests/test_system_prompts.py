from agent_graph import system_prompts as sp


def test_planner_prompt_matches_current_retrieval_sources():
    prompt = sp.PLANNER_SYSTEM_PROMPT

    assert "2023-01-02 ~ 2026-06-01" in prompt
    assert "Company → Disclosure → Section → Evidence" in prompt
    assert "Entity" not in prompt
    assert "Event" not in prompt
    assert "테마" not in prompt
    assert "결산월" not in prompt
    assert "dependencies: list[str]" in prompt
    assert "`dependencies`는 빈 목록" in prompt


def test_retriever_prompt_matches_current_state_and_qdrant_items():
    prompt = sp.RETRIEVER_SYSTEM_PROMPT

    assert "`user_question`" in prompt
    assert "`Plan`" in prompt
    assert "QueryPlan" not in prompt
    assert "plan_executions" not in prompt
    assert "`COMPLETE`" in prompt
    assert "`INSUFFICIENT`" in prompt
    assert "`metadata`" in prompt
    assert "`content`" in prompt
    assert "`omitted_record_count`" in prompt
    assert "`row_group`" in prompt
    assert "`retrieve_search(plan_id, limit)`" in prompt
    assert "`5`, `10`, `15`" in prompt
    assert "`requested_limit`" in prompt
    assert "`duplicate_point_count`" in prompt
    assert "`dependencies`" in prompt
    assert "`dependency_results`" in prompt


def test_answer_generator_prompt_matches_compacted_qdrant_output():
    prompt = sp.ANSWER_GENERATOR_SYSTEM_PROMPT

    assert "`metadata.retrieval_context`" in prompt
    assert "TEXT는 `content`" in prompt
    assert "`omitted_record_count`" in prompt
    assert "`row_group`" in prompt
    assert "전체 표라고 단정하지 마세요" in prompt


def test_builder_prompts_use_current_names():
    assert sp.CYPHER_BUILDER_SYSTEM_PROMPT
    assert sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT
    assert not hasattr(sp, "CYPHER_SYSTEM_PROMPT")
    assert not hasattr(sp, "QDRANT_QUERY_SYSTEM_PROMPT")
    assert "`limit`을 출력하지 않았는가" in sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT
    assert "`dependency_results`" in sp.CYPHER_BUILDER_SYSTEM_PROMPT
    assert "`dependency_results`" in sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT


def test_compactor_prompt_prefers_recall_when_uncertain():
    prompt = sp.QDRANT_POINT_COMPACTOR_SYSTEM_PROMPT

    assert "필요한 item을 제외하는 것을 더 큰 오류" in prompt
    assert "판단이 확실하지 않은 item도 보존" in prompt
    assert "질문과 명백히 무관한 경우에만" in prompt
