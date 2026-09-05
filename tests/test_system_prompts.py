from pathlib import Path

from agent_graph import system_prompts as sp


QDRANT_QUERY_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "agent_graph"
    / "qdrant_query_schema.yaml"
)


def test_question_analyzer_prompt_matches_current_contract():
    prompt = sp.QUESTION_ANALYZER_SYSTEM_PROMPT

    assert "current_date" in prompt
    assert "sub_questions" in prompt
    assert "entities" in prompt
    assert "events" in prompt
    assert "intents" in prompt
    assert "periods" in prompt
    assert "requested_facts" in prompt
    assert "synthesis_requirement" in prompt
    assert "registry 기반 정규화" in prompt
    assert "QuestionAnalyzerOutput" in prompt
    assert "Plan, PlanDraft, retrieval source" in prompt
    assert "Cypher 또는 Qdrant filter를 생성하지 마세요" in prompt
    assert not hasattr(sp, "PLANNER_SYSTEM_PROMPT")


def test_retriever_prompt_matches_current_state_and_qdrant_items():
    prompt = sp.RETRIEVER_SYSTEM_PROMPT

    assert "user_question" in prompt
    assert "sub_questions" in prompt
    assert "정보 요구 checklist" in prompt
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
    assert "scope_candidates" in prompt
    assert "scope_id" in prompt
    assert "Scope는 검색 범위이고 RetrievalResult는 검색 근거" in prompt
    for level in ("GLOBAL", "COMPANY", "DISCLOSURE", "SECTION"):
        assert level in prompt
    assert "retrieval 시작 전에 이미 생성" in prompt
    assert "Event가 아니라 METRIC" in prompt
    assert "selected_result_ids에는 SUCCESS 상태의 RetrievalResult만 포함" in prompt
    assert "INVALID_INPUT 결과는 선택할 수 없습니다" in prompt
    assert "calculate_table_statistic(variable_name, operation, column, targets, row_selector)" in prompt
    assert "combine_numeric_results(variable_name, operation, targets, direction, periods)" in prompt
    assert "숫자 계산은 암산하지 말고" in prompt
    assert "하나의 호출에 섞지 마세요" in prompt
    assert "numeric_ordering" in prompt
    assert "단위 없음과 명시된 단위는 다른 것으로 취급" in prompt
    assert "targets[0] - targets[1]" in prompt
    assert "최종 계산, 비교, 판단과 답변 생성은 downstream" not in prompt
    assert "그 결과의 status가 SUCCESS가 아니거나" in prompt
    assert "가리킨 item이 R_TABLE이 아니거나" in prompt
    assert "targets에 중복해서 넣을 수 없습니다" in prompt
    assert "나누는 값이 0이거나" in prompt
    assert "ordering 결과(numeric_ordering)는 대상으로 쓸 수 없습니다" in prompt
    assert "operation=mean: targets 전체의 평균(2개 이상)" in prompt
    assert "operation=percent_ratio: 정확히 2개, targets[0] / targets[1] * 100" in prompt
    assert "percent_ratio와 다릅니다" in prompt
    assert "row_selector={label_column, labels}" in prompt
    assert "stdev(표본 표준편차, 값 2개 이상 필요)" in prompt
    assert "operation=cagr: 정확히 2개, (targets[1] / targets[0]) ** (1 / periods) - 1" in prompt
    assert "cagr" in prompt and "두 값이 모두 양수여야 합니다" in prompt
    assert "1부터 시작하는 rank(순위)" in prompt
    assert "numeric_scalar 결과(sum/mean/difference/ratio/percent_ratio/percent_change/cagr)" in prompt
    assert "cagr인데 1 이상의 정수 periods가 없거나" in prompt
    assert "부분 일치는 하지 않습니다" in prompt
    assert "이미 표에 소계·합계 행이 있으면" in prompt
    assert "row_selector의 label이 표에서 하나도 없거나 여러 행과 일치하거나" in prompt


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
    assert "`scope`" in sp.CYPHER_BUILDER_SYSTEM_PROMPT
    assert "`scope`" in sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT
    assert "dependency_results" not in sp.CYPHER_BUILDER_SYSTEM_PROMPT
    assert "dependency_results" not in sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT
    assert "point_kinds" not in sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT
    assert "실패 결과를 자동으로 추가하지 않습니다" in sp.RETRIEVER_SYSTEM_PROMPT
    assert "filter 조합은 절대 다시 사용하지 마세요" in sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT
    assert "filter 조건의 제거 또는 완화를 먼저 시도하세요" in sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT


def test_narrow_scope_prompt_matches_internal_agent_contract():
    disclosure_prompt = sp.NARROW_SCOPE_DISCLOSURE_SELECTION_SYSTEM_PROMPT
    section_prompt = sp.NARROW_SCOPE_SECTION_SELECTION_SYSTEM_PROMPT

    assert "knowledge_hints" in disclosure_prompt
    assert "disclosure_candidates" in disclosure_prompt
    assert "REPORTING_PERIOD" in disclosure_prompt
    assert "rcept_date" in disclosure_prompt
    assert "knowledge_hints" in section_prompt
    assert "section_candidates" in section_prompt
    assert "DISCLOSURE Scope" in section_prompt


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


def test_retrieval_prompts_enforce_latest_disclosure_version():
    retriever = sp.RETRIEVER_SYSTEM_PROMPT
    cypher = sp.CYPHER_BUILDER_SYSTEM_PROMPT
    query = sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT
    schema = QDRANT_QUERY_SCHEMA_PATH.read_text(encoding="utf-8")

    assert "`is_latest_version = true`" in retriever
    assert "d.is_latest_version = $is_latest_version" in cypher
    assert "application이 이 조건을 검증" in cypher
    assert "relationship type, 양쪽 node label과 방향을 schema로 검증" in cypher
    assert "(d:Disclosure)-[:REPORTS]->(e:Event)" in cypher
    assert "application이 `is_latest_version = true` filter를 자동 적용" in query
    assert "`is_latest_version`을 직접 filter로 생성" in query
    assert "automatic_filters" in schema
    assert "is_latest_version" in schema
    assert "llm_output: forbidden" in schema


def test_compactor_prompt_prefers_recall_when_uncertain():
    prompt = sp.QDRANT_POINT_COMPACTOR_SYSTEM_PROMPT

    assert "필요한 item을 제외하는 것을 더 큰 오류" in prompt
    assert "판단이 확실하지 않은 item도 보존" in prompt
    assert "질문과 명백히 무관한 경우에만" in prompt
