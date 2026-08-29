from pathlib import Path

from agent_graph import system_prompts as sp


QDRANT_QUERY_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "agent_graph"
    / "qdrant_query_schema.yaml"
)


def test_planner_prompt_matches_current_retrieval_sources():
    prompt = sp.PLANNER_SYSTEM_PROMPT

    assert "2023-01-02 ~ 2026-06-01" in prompt
    assert "Company → Disclosure → Section → Evidence" in prompt
    assert "Entity" not in prompt
    assert "Event" not in prompt
    assert "테마" not in prompt
    assert "결산월" not in prompt
    assert "question_analysis: QuestionAnalysis" in prompt
    assert "plans: list[PlanDraft]" not in prompt
    assert "Retriever가 `retrieve_search`를 호출할 때 즉석에서 생성" in prompt


def test_retriever_prompt_matches_current_state_and_qdrant_items():
    prompt = sp.RETRIEVER_SYSTEM_PROMPT

    assert "user_question" in prompt
    assert "retrieve_search(plan, limit)" in prompt
    assert "QueryPlan" not in prompt
    assert "plan_executions" not in prompt
    assert "COMPLETE" in prompt
    assert "INSUFFICIENT" in prompt
    assert "metadata" in prompt
    assert "content" in prompt
    assert "omitted_record_count" in prompt
    assert "row_group" in prompt
    assert "최초 검색은 5" in prompt
    assert "10, 15, 20" in prompt
    assert "requested_limit" in prompt
    assert "duplicate_point_count" in prompt
    assert "dependencies" in prompt
    assert "plan_id는 application이 자동 할당" in prompt
    assert "create_plan" not in prompt
    assert "modify_plan" not in prompt
    assert "delete_plan" not in prompt
    assert "rcept_date" in prompt
    assert "disclosure_id, 공시명, rcept_date" in prompt
    assert "result_id를 dependencies에 넣은 Qdrant Plan" in prompt


def test_answer_generator_prompt_matches_compacted_qdrant_output():
    prompt = sp.ANSWER_GENERATOR_SYSTEM_PROMPT

    assert "`context`" in prompt
    assert "일반 본문은 `content`" in prompt
    assert "`table_info.omitted_record_count`" in prompt
    assert "`table_info.scope.kind`가 `row_group`" in prompt
    assert "일부만 제공된 표로 전체 목록" in prompt


def test_builder_prompts_use_current_names():
    assert sp.CYPHER_BUILDER_SYSTEM_PROMPT
    assert sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT
    assert not hasattr(sp, "CYPHER_SYSTEM_PROMPT")
    assert not hasattr(sp, "QDRANT_QUERY_SYSTEM_PROMPT")
    assert "`limit`을 출력하지 않았는가" in sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT
    assert "`previous_results`" in sp.CYPHER_BUILDER_SYSTEM_PROMPT
    assert "`previous_results`" in sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT
    assert "dependency_results" not in sp.CYPHER_BUILDER_SYSTEM_PROMPT
    assert "dependency_results" not in sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT
    assert "point_kinds" not in sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT
    assert "실패 결과를 자동으로 추가하지 않습니다" in sp.RETRIEVER_SYSTEM_PROMPT
    assert "filter 조합은 절대 다시 사용하지 마세요" in sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT
    assert "filter 조건의 제거 또는 완화를 먼저 시도하세요" in sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT


def test_qdrant_builder_uses_flat_payload_filter_paths():
    prompt = sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT
    schema = QDRANT_QUERY_SCHEMA_PATH.read_text(encoding="utf-8")

    for content in (prompt, schema):
        assert "기준일" in content
        assert "취득일" in content
        assert "query_text" in content

    assert "rcept_date" in schema
    assert "chunking.table_id" in schema
    assert "retrieval_metadata" not in schema
    allowed_keys = schema.split("allowed_keys:", 1)[1].split("condition_rules:", 1)[0]
    assert "base_year" not in allowed_keys
    assert "base_month" not in allowed_keys
    assert "`corp_name`" in prompt


def test_retrieval_prompts_use_rcept_date_with_month_tolerance():
    retriever = sp.RETRIEVER_SYSTEM_PROMPT
    cypher = sp.CYPHER_BUILDER_SYSTEM_PROMPT
    query = sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT

    for prompt in (retriever, cypher, query):
        assert "rcept_date" in prompt
        assert "최소 1개월" in prompt
        assert "base_year" in prompt
        assert "base_month" in prompt

    assert "d.rcept_date >= $start_date" in cypher
    assert "d.rcept_date <= $end_date" in cypher
    assert "단일 `rcept_date` exact-match filter로 변환하지 마세요" in query


def test_compactor_prompt_prefers_recall_when_uncertain():
    prompt = sp.QDRANT_POINT_COMPACTOR_SYSTEM_PROMPT

    assert "필요한 item을 제외하는 것을 더 큰 오류" in prompt
    assert "판단이 확실하지 않은 item도 보존" in prompt
    assert "질문과 명백히 무관한 경우에만" in prompt
