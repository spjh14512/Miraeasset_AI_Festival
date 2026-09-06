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
    assert "issuer_universe_tsv" in prompt
    assert "sub_questions" in prompt
    assert "entities" in prompt
    assert "events" in prompt
    assert "intents" in prompt
    assert "periods" in prompt
    assert "requested_facts" in prompt
    assert "synthesis_requirement" in prompt
    assert "corp_code" in prompt
    assert "match_status`를 추정·교정·검증하지 않고" in prompt
    assert "OUT_OF_UNIVERSE" in prompt
    assert "ISSUER가 아닌 TARGET" in prompt
    assert "QuestionAnalyzerOutput" in prompt
    assert "Plan, PlanDraft, retrieval source" in prompt
    assert "Cypher 또는 Qdrant filter를 생성하지 마세요" in prompt
    assert not hasattr(sp, "PLANNER_SYSTEM_PROMPT")


def test_retriever_prompt_matches_current_state_and_qdrant_items():
    prompt = sp.RETRIEVER_SYSTEM_PROMPT

    for value in (
        "sub_questions",
        "scope_candidates",
        "retrieval_results",
        "retrieval_search_count",
        "max_retrieval_search_count",
        "retrieve_search(plan, breadth)",
        "retrieve_correction_history(disclosure_id)",
        "calculate_table_statistic",
        "combine_numeric_results",
        "finish(status, reason, selected_result_ids)",
        "dependencies",
        "scope_id",
        "NO_RESULTS",
        "DUPLICATES_ONLY",
        "omitted_record_count",
    ):
        assert value in prompt
    assert "최대 15회" in prompt
    assert 'breadth="initial"' in prompt
    assert 'breadth="expand"' in prompt
    assert "최대 20개" in prompt
    assert "Answer Generator는 계산하지 않고" in prompt
    assert "계산이 필요한 질문에서 원시 숫자만 선택한 채 finish하지 마세요" in prompt
    assert "같은 Plan이 2회 실패" in prompt
    assert "Company.industry" in prompt
    assert "IT 기업들의 목록" in prompt
    assert 'source="neo4j"' in prompt
    assert "Neo4j 속성·관계만으로 완전히 충족" in prompt
    assert "공시 본문 문장" in prompt
    assert "OUT_OF_UNIVERSE" in prompt
    assert "GLOBAL 검색으로 바꾸거나" in prompt
    assert "create_plan" not in prompt
    assert "modify_plan" not in prompt
    assert "QueryPlan" not in prompt


def test_answer_generator_prompt_matches_compacted_qdrant_output():
    prompt = sp.ANSWER_GENERATOR_SYSTEM_PROMPT

    assert "context" in prompt
    assert "content" in prompt
    assert "table_info.omitted_record_count" in prompt
    assert "row_group" in prompt
    assert "새로운 계산을 직접 수행하지 마세요" in prompt
    assert "derived RetrievalResult" in prompt
    assert "원시 값을 조합해 새로운 숫자를 만들지 마세요" in prompt


def test_builder_prompts_use_current_names():
    cypher = sp.CYPHER_BUILDER_SYSTEM_PROMPT
    qdrant = sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT

    assert cypher
    assert qdrant
    assert not hasattr(sp, "CYPHER_SYSTEM_PROMPT")
    assert not hasattr(sp, "QDRANT_QUERY_SYSTEM_PROMPT")
    assert "query_vector와 limit은 application 소유" in qdrant
    assert "previous_results" in cypher
    assert "recent_failures" in cypher
    assert 'AS "corp_code"' in cypher
    assert "폐쇄형 domain" in cypher
    assert "유사어·상위어·하위어" in cypher
    assert "inline literal" in cypher
    assert "previous_results" in qdrant
    assert "scope" in cypher
    assert "scope" in qdrant
    assert "dependency_results" not in sp.CYPHER_BUILDER_SYSTEM_PROMPT
    assert "dependency_results" not in sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT
    assert "point_kinds" not in sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT
    assert "동일 filter 조합을 절대 재사용" in qdrant
    assert "query_text 변경보다 filter 제거·완화" in sp.RETRIEVER_SYSTEM_PROMPT


def test_narrow_scope_prompt_matches_internal_agent_contract():
    disclosure_prompt = sp.NARROW_SCOPE_DISCLOSURE_SELECTION_SYSTEM_PROMPT
    section_prompt = sp.NARROW_SCOPE_SECTION_SELECTION_SYSTEM_PROMPT

    assert not hasattr(sp, "NARROW_SCOPE_SYSTEM_PROMPT")
    assert "knowledge_hints" in disclosure_prompt
    assert "disclosure_candidates" in disclosure_prompt
    assert "searched_rcept_date_range" in disclosure_prompt
    assert "REPORTING_PERIOD" in disclosure_prompt
    for kind in ("FILING_DATE", "EVENT_DATE", "AS_OF"):
        assert kind in disclosure_prompt
    assert "knowledge_hints" in section_prompt
    assert "section_candidates" in section_prompt
    assert "DISCLOSURE Scope" in section_prompt


def test_qdrant_builder_uses_flat_payload_filter_paths():
    prompt = sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT
    schema = QDRANT_QUERY_SCHEMA_PATH.read_text(encoding="utf-8")

    for content in (prompt, schema):
        assert "query_text" in content

    assert "rcept_date" in schema
    assert "chunking.table_id" in schema
    assert "retrieval_metadata" not in schema
    allowed_keys = schema.split("allowed_keys:", 1)[1].split("condition_rules:", 1)[0]
    assert "base_year" not in allowed_keys
    assert "base_month" not in allowed_keys
    assert "corp_name" in prompt
    for forbidden_filter in ("corp_code", "industry", "sector"):
        assert forbidden_filter in prompt


def test_retrieval_prompts_use_rcept_date_with_month_tolerance():
    retriever = sp.RETRIEVER_SYSTEM_PROMPT
    cypher = sp.CYPHER_BUILDER_SYSTEM_PROMPT
    query = sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT

    for prompt in (retriever, cypher, query):
        assert "rcept_date" in prompt
        assert "최소 1개월" in prompt
        assert "base_year" in prompt
        assert "base_month" in prompt

    assert "Disclosure.rcept_date는 Neo4j date" in cypher
    assert "d.rcept_date >= date($start_date)" in cypher
    assert "d.rcept_date <= date($end_date)" in cypher
    assert "단일 rcept_date exact match" in query


def test_retrieval_prompts_enforce_latest_disclosure_version():
    retriever = sp.RETRIEVER_SYSTEM_PROMPT
    cypher = sp.CYPHER_BUILDER_SYSTEM_PROMPT
    query = sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT
    schema = QDRANT_QUERY_SCHEMA_PATH.read_text(encoding="utf-8")

    assert "is_latest_version=true" in retriever
    assert "d.is_latest_version = $is_latest_version" in cypher
    assert "(d:Disclosure)-[:REPORTS]->(e:Event)" in cypher
    assert "is_latest_version=true" in query
    assert "automatic_filters" in schema
    assert "is_latest_version" in schema
    assert "llm_output: forbidden" in schema


def test_retriever_prompt_limits_correction_history_to_explicit_requests():
    prompt = sp.RETRIEVER_SYSTEM_PROMPT

    assert "정정 이력·정정 전후·변경 내용을 명시적으로 요구할 때만" in prompt
    assert "일반 검색에서 확인한 최신 disclosure_id" in prompt


def test_compactor_prompt_prefers_recall_when_uncertain():
    prompt = sp.QDRANT_POINT_COMPACTOR_SYSTEM_PROMPT

    assert "필요한 item을 제외하는 것을 더 큰 오류" in prompt
    assert "판단이 확실하지 않은 item도 보존" in prompt
    assert "질문과 명백히 무관한 경우에만" in prompt
