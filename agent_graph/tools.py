import json
import re
import statistics
from datetime import datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import InjectedState
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from vector_db.text2vector import HybridEmbedding, text_to_hybrid_vector

from dotenv import load_dotenv
import os

from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models

from . import system_prompts as sp
from .calculation import check_table_completeness, parse_numeric_cell
from .compactor import compact_qdrant_point, point_requires_compaction
from .llm import MAX_LLM_RETRIES, build_output_retry_message, get_llm
from .retrieval_result_parser import (
    extract_qdrant_points,
    parse_and_remove_qdrant_contextual_text,
    parse_neo4j_response,
    parse_qdrant_response,
)
from .state import (
    AgentState,
    AiAnswer,
    AnswerGeneratorOutput,
    Citation,
    Plan,
    PlanDraft,
    RetrievalResult,
)


NEO4J_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "knowledge_graph" / "neo4j_schema.yaml"
)
QDRANT_QUERY_SCHEMA_PATH = Path(__file__).with_name("qdrant_query_schema.yaml")
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DOCUMENT_MANIFEST_PATH = DATA_ROOT / "manifest.jsonl"
CITATION_CONTEXT_QUERY = """
UNWIND $citations AS citation
MATCH (d:Disclosure {id: citation.disclosure_id})
MATCH (d)-[:HAS_SECTION*1..]->(s:Section {id: citation.section_id})
OPTIONAL MATCH (s)-[:HAS_EVIDENCE]->(e:Evidence {id: citation.evidence_id})
RETURN citation.disclosure_id AS disclosure_id,
       citation.section_id AS section_id,
       citation.evidence_id AS requested_evidence_id,
       e.id AS evidence_id,
       s.section_path AS section_path,
       e.heading_path AS heading_path
""".strip()

load_dotenv()
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

qdrant_host = os.getenv("QDRANT_HOST") or "localhost"
qdrant_port = int(os.getenv("QDRANT_PORT") or 6333)
qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME") or "dart_evidence"
qdrant_dense_vector_name = "evidence_dense"
qdrant_sparse_vector_name = "evidence_sparse"
HYBRID_PREFETCH_MULTIPLIER = 5

class CypherQuery(BaseModel):
    """실행 가능한 read-only Cypher와 parameter를 분리한 요청입니다."""

    cypher: str = Field(..., min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class CypherQueryToolArgs(BaseModel):
    """Clova function calling이 지원하는 단순 field로 만든 Cypher 출력입니다."""

    cypher: str = Field(..., min_length=1, description="실행할 read-only Cypher")
    parameters_json: str = Field(
        ...,
        min_length=2,
        description="Cypher parameter를 나타내는 JSON object 문자열",
    )

    @field_validator("parameters_json")
    @classmethod
    def validate_parameters_json(cls, value: str) -> str:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("parameters_json은 유효한 JSON이어야 합니다.") from error
        if not isinstance(parsed, dict):
            raise ValueError("parameters_json은 JSON object여야 합니다.")
        return value

    def to_cypher_query(self) -> CypherQuery:
        return CypherQuery(
            cypher=self.cypher,
            parameters=json.loads(self.parameters_json),
        )


QdrantFilterField = Literal[
    "disclosure_id",
    "section_id",
    "evidence_id",
    "rcept_date",
    "chunking.table_id",
]
QDRANT_STRING_FILTER_FIELDS = {
    "disclosure_id",
    "section_id",
    "evidence_id",
    "rcept_date",
    "chunking.table_id",
}
QDRANT_INTEGER_FILTER_FIELDS: set[str] = set()


class RepeatedNoResultsFilterError(ValueError):
    """NO_RESULTS가 발생한 Qdrant filter 조합을 다시 생성했을 때 발생합니다."""


class QdrantFilter(BaseModel):
    """하나의 indexed Qdrant payload 조건입니다."""

    key: QdrantFilterField = Field(
        description="허용 목록에 포함된 indexed payload 경로",
    )
    match: str | int | None = Field(
        default=None,
        description="정확 일치시킬 string 또는 integer scalar 값. list는 허용하지 않음",
    )
    gt: int | None = Field(default=None, description="현재 payload schema에서는 사용하지 않음")
    gte: int | None = Field(default=None, description="현재 payload schema에서는 사용하지 않음")
    lt: int | None = Field(default=None, description="현재 payload schema에서는 사용하지 않음")
    lte: int | None = Field(default=None, description="현재 payload schema에서는 사용하지 않음")

    @model_validator(mode="after")
    def validate_filter(self) -> "QdrantFilter":
        has_match = self.match is not None
        range_values = {
            "gt": self.gt,
            "gte": self.gte,
            "lt": self.lt,
            "lte": self.lte,
        }
        has_range = any(value is not None for value in range_values.values())
        if has_match == has_range:
            raise ValueError("Qdrant filter에는 match 또는 range 중 하나가 필요합니다.")

        if has_match and self.key in QDRANT_STRING_FILTER_FIELDS:
            if not isinstance(self.match, str):
                raise ValueError(f"{self.key}의 match는 string이어야 합니다.")
        if has_match and self.key in QDRANT_INTEGER_FILTER_FIELDS:
            if not isinstance(self.match, int) or isinstance(self.match, bool):
                raise ValueError(f"{self.key}의 match는 integer여야 합니다.")

        if has_range and self.key not in QDRANT_INTEGER_FILTER_FIELDS:
            raise ValueError("현재 Qdrant payload filter에는 range를 사용할 수 없습니다.")
        if self.gt is not None and self.gte is not None:
            raise ValueError("range filter에는 gt와 gte를 함께 사용할 수 없습니다.")
        if self.lt is not None and self.lte is not None:
            raise ValueError("range filter에는 lt와 lte를 함께 사용할 수 없습니다.")

        lower = self.gt if self.gt is not None else self.gte
        upper = self.lt if self.lt is not None else self.lte
        if lower is not None and upper is not None:
            if lower > upper or (
                lower == upper and (self.gt is not None or self.lt is not None)
            ):
                raise ValueError("range filter의 하한은 상한보다 작아야 합니다.")
        return self


class QdrantQuery(BaseModel):
    """Qdrant vector 또는 filter retrieval 요청입니다."""

    mode: Literal["vector", "filter"] = Field(
        description="의미 검색은 vector, 명시된 식별자만으로 정확 조회하면 filter",
    )
    query_text: str | None = Field(
        default=None,
        description="vector mode에서 embedding할 자연어 검색문. filter mode에서는 null",
    )
    query_vector: SkipJsonSchema[HybridEmbedding | None] = None
    filters: list[QdrantFilter] = Field(
        default_factory=list,
        description="Plan에 명시된 조건으로 만든 indexed payload filter 목록",
    )
    limit: SkipJsonSchema[int] = 5
    score_threshold: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="vector 검색 점수 하한. 근거가 없으면 null이며 filter mode에서는 null",
    )

    @model_validator(mode="after")
    def validate_mode(self) -> "QdrantQuery":
        query_text = self.query_text.strip() if self.query_text else None
        if self.mode == "vector" and query_text is None:
            raise ValueError("vector mode에는 query_text가 필요합니다.")
        if self.mode == "filter" and query_text is not None:
            raise ValueError("filter mode에는 query_text가 없어야 합니다.")
        if self.mode == "filter" and not self.filters:
            raise ValueError("filter mode에는 하나 이상의 filter가 필요합니다.")
        if self.mode == "filter" and self.query_vector is not None:
            raise ValueError("filter mode에는 query_vector를 사용할 수 없습니다.")
        if self.mode == "filter" and self.score_threshold is not None:
            raise ValueError("filter mode에는 score_threshold를 사용할 수 없습니다.")
        if (
            not isinstance(self.limit, int)
            or isinstance(self.limit, bool)
            or not 1 <= self.limit <= 100
        ):
            raise ValueError("limit은 1 이상 100 이하의 정수여야 합니다.")

        filter_keys = [item.key for item in self.filters]
        if len(filter_keys) != len(set(filter_keys)):
            raise ValueError("동일한 key의 filter를 여러 번 사용할 수 없습니다.")
        self.query_text = query_text
        return self


class QdrantQueryToolArgs(BaseModel):
    """CLOVA function calling용 단순 field Qdrant query 출력입니다."""

    mode: Literal["vector", "filter"] = Field(
        description="의미 검색은 vector, 식별자 정확 조회는 filter",
    )
    query_text: str = Field(
        description="vector 검색문. filter mode에서는 빈 문자열",
    )
    filters_json: str = Field(
        ...,
        min_length=2,
        description="Qdrant filter object 목록을 나타내는 JSON array 문자열",
    )
    score_threshold_json: str = Field(
        ...,
        min_length=1,
        description="score threshold 숫자 또는 null을 나타내는 JSON 문자열",
    )

    @field_validator("filters_json")
    @classmethod
    def validate_filters_json(cls, value: str) -> str:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("filters_json은 유효한 JSON이어야 합니다.") from error
        if not isinstance(parsed, list):
            raise ValueError("filters_json은 JSON array여야 합니다.")
        return value

    @field_validator("score_threshold_json")
    @classmethod
    def validate_score_threshold_json(cls, value: str) -> str:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                "score_threshold_json은 유효한 JSON이어야 합니다."
            ) from error
        if parsed is not None and (
            not isinstance(parsed, (int, float)) or isinstance(parsed, bool)
        ):
            raise ValueError("score_threshold_json은 숫자 또는 null이어야 합니다.")
        return value

    def to_qdrant_query(self) -> QdrantQuery:
        return QdrantQuery(
            mode=self.mode,
            query_text=self.query_text.strip() or None,
            filters=json.loads(self.filters_json),
            score_threshold=json.loads(self.score_threshold_json),
        )


@lru_cache(maxsize=1)
def _load_neo4j_schema() -> str:
    return NEO4J_SCHEMA_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _load_qdrant_query_schema() -> str:
    return QDRANT_QUERY_SCHEMA_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _get_cypher_llm() -> Any:
    """공용 LLM에 Cypher structured output을 한 번 binding합니다."""

    return get_llm().with_structured_output(
        CypherQueryToolArgs,
        method="function_calling",
    )


@lru_cache(maxsize=1)
def _get_query_llm() -> Any:
    """공용 LLM에 Qdrant query structured output을 한 번 binding합니다."""

    return get_llm().with_structured_output(
        QdrantQueryToolArgs,
        method="function_calling",
    )


def _resolve_plan_dependencies(
    plan: Plan,
    state: AgentState,
) -> list[RetrievalResult]:
    """Plan에 명시적으로 저장된 dependency를 RetrievalResult로 해석합니다.

    입력 예시:
        plan.dependencies == ["result_plan_1"]
        state["retrieval_results"] == [RetrievalResult(result_id="result_plan_1", ...)]

    출력 예시:
        [RetrievalResult(result_id="result_plan_1", ...)]
    """

    results_by_id = {
        result.result_id: result
        for result in state.get("retrieval_results", [])
    }
    missing = [
        result_id
        for result_id in plan.dependencies
        if result_id not in results_by_id
    ]
    if missing:
        raise ValueError(
            f"Plan dependency RetrievalResult를 찾지 못했습니다: {missing}"
        )
    return [results_by_id[result_id] for result_id in plan.dependencies]


def _build_query_builder_human_message(
    plan: Plan,
    user_question: str,
    dependencies: list[RetrievalResult],
) -> HumanMessage:
    """Builder가 사용할 질문, Plan, 이전 결과를 JSON message로 만듭니다.

    입력 예시:
        plan=Plan(plan_id="plan_2", dependencies=["result_plan_1"], ...)
        user_question="삼성전자의 해당 공시에서 매출액을 알려줘"
        dependencies=[RetrievalResult(result_id="result_plan_1", ...)]

    출력 예시:
        HumanMessage(content='{"user_question": ..., "plan": ..., "previous_results": [...] }')
    """

    return HumanMessage(
        content=json.dumps(
            {
                "user_question": user_question,
                "plan": plan.model_dump(
                    mode="json",
                    exclude={"dependencies"},
                ),
                "previous_results": [
                    result.model_dump(mode="json")
                    for result in dependencies
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

@tool
def retrieve_search(
    plan: PlanDraft,
    state: Annotated[AgentState, InjectedState],
    limit: int = 5,
) -> dict:
    """전달받은 단일 plan을 즉시 실행하고 retrieval 결과를 state에 추가합니다.

    args:
        plan(PlanDraft): 즉시 실행할 source, query, purpose, dependencies
        limit(int): Qdrant에서 조회할 누적 상위 point 수. 최초 검색은 5, 추가 검색은 5 단위로 늘립니다. Neo4j 검색에는 적용하지 않습니다.

    return:
        dict: 자동 할당된 plan_id와 검색 결과를 포함한 state update
    """

    next_plan_seq = state.get("next_plan_seq", 1)
    executable_plan = Plan.from_plan_draft(plan, next_plan_seq)
    dependencies = _resolve_plan_dependencies(executable_plan, state)

    try:
        retrieval_result, previous_point_ids, new_point_ids = _execute_retrieval_plan(
            executable_plan,
            state=state,
            dependencies=dependencies,
            limit=limit,
        )
    except (ValidationError, RepeatedNoResultsFilterError) as error:
        retrieval_result = _build_failed_retrieval_result(
            executable_plan,
            status="INVALID_QUERY",
            error=error,
            stage="query_builder",
        )
        previous_point_ids, new_point_ids = [], []
    except RuntimeError as error:
        retrieval_result = _build_failed_retrieval_result(
            executable_plan,
            status=_classify_execution_error(error),
            error=error,
            stage="query_executor",
        )
        previous_point_ids = (
            list(dict.fromkeys(state.get("retrieved_qdrant_point_ids", [])))
            if executable_plan.source == "qdrant"
            else []
        )
        new_point_ids = []

    retrieval_result = retrieval_result.model_copy(update={
        "metadata": {
            **retrieval_result.metadata,
            "plan_purpose": executable_plan.purpose
        }
    })

    update = {
        "next_plan_seq": next_plan_seq + 1,
        "retrieval_results": [retrieval_result],
    }
    if executable_plan.source == "qdrant":
        update["retrieved_qdrant_point_ids"] = previous_point_ids + new_point_ids
    return update


def _execute_retrieval_plan(
    plan: Plan,
    *,
    state: AgentState,
    dependencies: list[RetrievalResult],
    limit: int,
) -> tuple[RetrievalResult, list[str], list[str]]:
    if plan.source == "neo4j":
        neo4j_cypher = cypher_builder(
            plan,
            user_question=state["question_text"],
            dependencies=dependencies,
        )
        print("-- Neo4j에서 다음 Cypher를 실행합니다.: --", neo4j_cypher, "\n\n")
        try:
            result = cypher_executor(neo4j_cypher, plan.plan_id)
        except RuntimeError as error:
            error.retrieval_query = neo4j_cypher.cypher
            raise
        return result, [], []

    if plan.source != "qdrant":
        raise ValueError("retrieval source가 neo4j 또는 qdrant가 아닙니다.")
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit > 100
        or limit < 5
        or limit % 5 != 0
    ):
        raise ValueError(
            "Qdrant limit은 5 이상 100 이하의 5 배수여야 합니다."
        )

    qdrant_query = query_builder(
        plan,
        user_question=state["question_text"],
        dependencies=dependencies,
    )
    qdrant_query.limit = limit
    print("-- Qdrant에서 Query retrieval을 실행합니다. --\n\n")
    try:
        raw_result = query_executor(qdrant_query)
    except RuntimeError as error:
        error.retrieval_query = qdrant_query.model_dump_json(
            exclude={"query_vector"}
        )
        raise
    raw_points = list(extract_qdrant_points(raw_result))
    previous_point_ids = list(dict.fromkeys(
        state.get("retrieved_qdrant_point_ids", [])
    ))
    seen_point_ids = set(previous_point_ids)
    points = []
    new_point_ids: list[str] = []
    for point in raw_points:
        point_id = getattr(point, "id", None)
        if point_id is None:
            raise ValueError("Qdrant point에 id가 없습니다.")
        point_id = str(point_id)
        if point_id in seen_point_ids:
            continue
        seen_point_ids.add(point_id)
        new_point_ids.append(point_id)
        points.append(point)

    for point in points:
        parse_and_remove_qdrant_contextual_text(point)

    selected_item_ids: dict[str, list[int]] = {}
    for point in points:
        if not point_requires_compaction(point):
            continue
        point_id = getattr(point, "id", None)
        if point_id is None:
            raise ValueError("Compaction 대상 Qdrant point에 id가 없습니다.")
        selected_item_ids[str(point_id)] = compact_qdrant_point(point, plan)
    result = parse_qdrant_response(
        points,
        plan_id=plan.plan_id,
        query=qdrant_query.model_dump_json(exclude={"query_vector"}),
        r_table_detail="records",
        selected_item_ids=selected_item_ids,
        metadata={
            "mode": qdrant_query.mode,
            "requested_limit": limit,
            "raw_point_count": len(raw_points),
            "duplicate_point_count": len(raw_points) - len(points),
        },
    )
    if raw_points and not points:
        result.status = "DUPLICATES_ONLY"
    return result, previous_point_ids, new_point_ids


def _classify_execution_error(error: BaseException) -> str:
    current: BaseException | None = error
    while current is not None:
        name = type(current).__name__.lower()
        message = str(current).lower()
        if isinstance(current, TimeoutError) or "timeout" in name or "timed out" in message:
            return "TIMEOUT"
        if (
            "syntax" in name
            or "clienterror" in name
            or "validation" in name
            or "invalid query" in message
            or "bad request" in message
        ):
            return "INVALID_QUERY"
        current = current.__cause__
    return "ERROR"


def _build_failed_retrieval_result(
    plan: Plan,
    *,
    status: str,
    error: BaseException,
    stage: str,
) -> RetrievalResult:
    return RetrievalResult(
        result_id=f"retrieval:{plan.plan_id}",
        plan_id=plan.plan_id,
        source=plan.source,
        status=status,
        query=getattr(error, "retrieval_query", plan.query),
        items=[],
        result_count=0,
        metadata={
            "failure_stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
    )

def cypher_builder(
    plan: Plan,
    *,
    user_question: str,
    dependencies: list[RetrievalResult],
    neo4j_schema: str | None = None,
    llm: Any | None = None,
) -> CypherQuery:
    """사용자 질문, Plan, dependency 결과로 read-only Cypher를 생성한다."""

    cypher_llm = (
        _get_cypher_llm()
        if llm is None
        else llm.with_structured_output(
            CypherQueryToolArgs,
            method="function_calling",
        )
    )
    schema = neo4j_schema if neo4j_schema is not None else _load_neo4j_schema()
    human_message = _build_query_builder_human_message(
        plan,
        user_question,
        dependencies,
    )
    print("[Cypher Builder human message]:\n" + human_message.content + "\n\n")
    messages = [
        SystemMessage(
            content=sp.CYPHER_BUILDER_SYSTEM_PROMPT.format(
                neo4j_schema=schema
            )
        ),
        human_message,
    ]
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            result = cypher_llm.invoke(messages)
            tool_args = (
                result
                if isinstance(result, CypherQueryToolArgs)
                else CypherQueryToolArgs.model_validate(result)
            )
            query = tool_args.to_cypher_query()
            _validate_latest_disclosure_filter(query)
            return query
        except (ValidationError, ValueError, TypeError) as error:
            if attempt == MAX_LLM_RETRIES:
                raise
            messages.append(HumanMessage(content=build_output_retry_message(
                "CypherQuery",
                error,
            )))


def _validate_latest_disclosure_filter(query: CypherQuery) -> None:
    """Disclosure를 조회하는 Cypher가 최신 공시만 선택하는지 검증합니다.

    입력 예시:
        CypherQuery(
            cypher="MATCH (d:Disclosure) WHERE d.is_latest_version = $latest RETURN d",
            parameters={"latest": True},
        )

    출력 예시:
        유효하면 None, 조건이 없거나 true가 아니면 ValueError
    """

    aliases = set(re.findall(
        r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*Disclosure\b",
        query.cypher,
        flags=re.IGNORECASE,
    ))
    for alias in aliases:
        value_patterns = (
            rf"\b{re.escape(alias)}\s*\.\s*is_latest_version\s*=\s*"
            r"(true|\$[A-Za-z_][A-Za-z0-9_]*)",
            rf"\(\s*{re.escape(alias)}\s*:\s*Disclosure\b[^)]*"
            r"\bis_latest_version\s*:\s*"
            r"(true|\$[A-Za-z_][A-Za-z0-9_]*)",
        )
        values = [
            match.group(1)
            for pattern in value_patterns
            for match in re.finditer(pattern, query.cypher, re.IGNORECASE)
        ]
        if any(
            value.lower() == "true"
            or (
                value.startswith("$")
                and query.parameters.get(value[1:]) is True
            )
            for value in values
        ):
            continue
        raise ValueError(
            "Disclosure 조회에는 최신 공시 조건이 필요합니다: "
            f"{alias}.is_latest_version = true"
        )

def _qdrant_filter_signature(filters: Any) -> tuple[str, ...]:
    """Qdrant filter 순서와 null field를 무시하는 비교용 signature를 만듭니다.

    입력 예시:
        [{"key": "evidence_id", "match": "evidence-1"}]

    출력 예시:
        ('{"key":"evidence_id","match":"evidence-1"}',)
    """

    if not isinstance(filters, list):
        return ()
    normalized = []
    for condition in filters:
        if isinstance(condition, BaseModel):
            condition = condition.model_dump(mode="json", exclude_none=True)
        if not isinstance(condition, dict):
            continue
        normalized.append(json.dumps(
            {key: value for key, value in condition.items() if value is not None},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ))
    return tuple(sorted(normalized))


def _find_repeated_no_results_filters(
    query: QdrantQuery,
    dependencies: list[RetrievalResult],
) -> list[str]:
    """동일 filter 조합으로 NO_RESULTS였던 dependency ID를 반환합니다.

    입력 예시:
        query.filters == [{"key": "evidence_id", "match": "evidence-1"}]

    출력 예시:
        ["retrieval:plan_1"]
    """

    signature = _qdrant_filter_signature(query.filters)
    conflicts = []
    for result in dependencies:
        if result.source != "qdrant" or result.status != "NO_RESULTS":
            continue
        try:
            previous_query = json.loads(result.query)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(previous_query, dict):
            continue
        if _qdrant_filter_signature(previous_query.get("filters")) == signature:
            conflicts.append(result.result_id)
    return conflicts


def query_builder(
    plan: Plan,
    *,
    user_question: str,
    dependencies: list[RetrievalResult],
    qdrant_schema: str | None = None,
    llm: Any | None = None,
) -> QdrantQuery:
    """사용자 질문, Plan, dependency 결과로 Qdrant 요청을 생성한다."""

    query_llm = (
        _get_query_llm()
        if llm is None
        else llm.with_structured_output(
            QdrantQueryToolArgs,
            method="function_calling",
        )
    )
    query_schema = (
        qdrant_schema
        if qdrant_schema is not None
        else _load_qdrant_query_schema()
    )
    query_builder_human_message = _build_query_builder_human_message(
        plan,
        user_question,
        dependencies,
    )
    print(
        "[query builder human message]:\n",
        query_builder_human_message.content,
        "\n\n",
        sep="",
    )
    messages = [
        SystemMessage(
            content=sp.QDRANT_QUERY_BUILDER_SYSTEM_PROMPT.format(
                qdrant_query_schema=query_schema,
            )
        ),
        query_builder_human_message,
    ]
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            result = query_llm.invoke(messages)
            tool_args = (
                result
                if isinstance(result, QdrantQueryToolArgs)
                else QdrantQueryToolArgs.model_validate(result)
            )
            query = tool_args.to_qdrant_query()
            conflicts = _find_repeated_no_results_filters(query, dependencies)
            if conflicts:
                raise RepeatedNoResultsFilterError(
                    "NO_RESULTS가 발생한 filter 조합을 반복해서 생성했습니다: "
                    f"{conflicts}. query_text 변경보다 filter 조건의 제거 또는 "
                    "완화를 우선하세요."
                )
            break
        except (
            ValidationError,
            ValueError,
            TypeError,
        ) as error:
            if attempt == MAX_LLM_RETRIES:
                raise
            messages.append(HumanMessage(content=build_output_retry_message(
                "QdrantQuery",
                error,
            )))

    if query.mode == "vector":
        query.query_vector = text_to_hybrid_vector(query.query_text)
    return query


def cypher_executor(cypher_query: CypherQuery, plan_id: str) -> RetrievalResult:
    
    try:
        with neo4j_driver.session() as session:
            result = list(session.run(
                cypher_query.cypher,
                cypher_query.parameters
            ))
    except Exception as error:
        raise RuntimeError("Neo4j Cypher 쿼리 실행 중 오류 발생!") from error

    return parse_neo4j_response(
        result,
        plan_id=plan_id,
        query=cypher_query.cypher,
        parameters=cypher_query.parameters
    )
    

def query_executor(qdrant_query: QdrantQuery) -> Any:

    must = [models.FieldCondition(
        key="is_latest_version",
        match=models.MatchValue(value=True),
    )]
    for condition in qdrant_query.filters:
        if condition.match is not None:
            must.append(models.FieldCondition(
                key=condition.key,
                match=models.MatchValue(value=condition.match)
            ))
        else:
            must.append(models.FieldCondition(
                key=condition.key,
                range=models.Range(**condition.model_dump(
                    include={"gt", "gte", "lt", "lte"},
                    exclude_none=True
                ))
            ))
    query_filter = models.Filter(must=must) if must else None

    try:
        if qdrant_query.mode == "vector":
            embedding = qdrant_query.query_vector
            if not isinstance(embedding, HybridEmbedding):
                raise ValueError("vector mode requires a hybrid query embedding")
            prefetch_limit = qdrant_query.limit * HYBRID_PREFETCH_MULTIPLIER
            result = qdrant_client.query_points(
                collection_name=qdrant_collection_name,
                prefetch=[
                    models.Prefetch(
                        query=list(embedding.dense),
                        using=qdrant_dense_vector_name,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=list(embedding.sparse.indices),
                            values=list(embedding.sparse.values),
                        ),
                        using=qdrant_sparse_vector_name,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=qdrant_query.limit,
                score_threshold=qdrant_query.score_threshold,
                with_payload=True,
                with_vectors=False
            )
        else:
            result = qdrant_client.scroll(
                collection_name=qdrant_collection_name,
                scroll_filter=query_filter,
                limit=qdrant_query.limit,
                with_payload=True,
                with_vectors=False
            )
    except Exception as error:
        raise RuntimeError("Qdrant 쿼리 실행 중 오류 발생!") from error

    return result

FinishStatus = Literal["COMPLETE", "INSUFFICIENT"]


def _validate_finish_selection(
    status: FinishStatus,
    selected_result_ids: list[str],
    state: AgentState,
) -> None:
    if status == "COMPLETE" and not selected_result_ids:
        raise ValueError("COMPLETE에는 최소 하나의 selected_result_ids가 필요합니다.")
    if any(not result_id.strip() for result_id in selected_result_ids):
        raise ValueError("selected_result_ids에는 빈 ID를 사용할 수 없습니다.")
    if len(selected_result_ids) != len(set(selected_result_ids)):
        raise ValueError("selected_result_ids에는 중복 ID를 사용할 수 없습니다.")

    results_by_id = {
        result.result_id: result
        for result in state.get("retrieval_results", [])
    }
    missing = [
        result_id
        for result_id in selected_result_ids
        if result_id not in results_by_id
    ]
    if missing:
        raise ValueError(f"RetrievalResult를 찾지 못했습니다: {missing}")

    not_success = [
        result_id
        for result_id in selected_result_ids
        if results_by_id[result_id].status != "SUCCESS"
    ]
    if not_success:
        raise ValueError(
            "SUCCESS 상태가 아닌 RetrievalResult는 선택할 수 없습니다: "
            f"{not_success}"
        )


@tool
def finish(
    status: FinishStatus,
    reason: str,
    selected_result_ids: list[str],
    state: Annotated[AgentState, InjectedState]
) -> dict:
    """더 이상의 retrieval을 멈추고 답변을 생성합니다. 종료 원인은 다음 두 가지 중 하나입니다.
        - COMPLETE: 지금까지의 retrieval을 통해 사용자의 질문에 답변하기 위해 필요한 충분한 정보를 얻었음.
        - INSUFFICIENT: 충분한 retrieval을 수행했으나, 사용자의 질문에 답변하기 위한 신뢰도 있는 정보를 얻지 못함. 더 이상의 retrieval은 무의미하다 판단.

    args:
        status(FinishStatus): 'COMPLETE' 또는 'INSUFFICIENT'
        reason(str): status를 그렇게 판단한 이유를 설명하는 간략한 한국어 문장. INSUFFICIENT라면 추가로 유효한 검색 전략이 없는 이유를 설명하며, 추가 검색이 필요하다고 작성하지 않습니다.
        selected_result_ids(list[str]): Answer Generator가 사용할 RetrievalResult ID 목록.
            status와 무관하게 SUCCESS 상태의 RetrievalResult만 선택할 수 있습니다.

    return:
        dict: retrieval_status와 검증된 selected_result_ids를 포함한 state update
    """

    _validate_finish_selection(status, selected_result_ids, state)

    return {
        "retrieval_status": status,
        "retrieval_finish_reason": reason,
        "selected_result_ids": selected_result_ids
    }


CalculationOperation = Literal["sum", "mean", "median", "max", "min", "mode", "stdev"]


class TableTarget(BaseModel):
    """calculate_table_statistic이 참조할 R_TABLE item 하나의 주소입니다.

    result_id는 대상 RetrievalResult, item_index는 그 RetrievalResult.items
    안에서 이 R_TABLE이 위치한 인덱스입니다. 하나의 표가 여러 chunk로 나뉘어
    서로 다른 result_id에 저장된 경우, 이 표를 여러 개 나열해 함께 참조합니다.
    """

    result_id: str = Field(
        ...,
        min_length=1,
        description="참조할 RetrievalResult의 result_id",
    )
    item_index: int = Field(
        ...,
        ge=0,
        strict=True,
        description="RetrievalResult.items 안에서 이 R_TABLE의 위치",
    )

    @field_validator("result_id")
    @classmethod
    def validate_result_id(cls, value: str) -> str:
        result_id = value.strip()
        if not result_id:
            raise ValueError("result_id는 비어 있을 수 없습니다.")
        return result_id


class TableRowSelector(BaseModel):
    """calculate_table_statistic이 표 전체가 아니라 특정 계정과목(행)만 계산하게 하는 선택자입니다.

    지정하지 않으면(기본값 None) 대상 표의 모든 행을 계산 대상으로
    삼습니다(기존 동작과 동일). 지정하면 label_column 열의 값이 공백만
    정규화한 뒤 정확히 일치하는 행만 골라 계산합니다. 부분 일치는
    지원하지 않습니다 — "유동부채"로 찾으면 그 글자를 포함할 뿐인
    "비유동부채" 같은 행은 매치되지 않습니다. labels를 여러 개 지정하면
    (예: ["유동부채", "비유동부채"]) 각각 정확히 한 행과 매치되어야 하며,
    그 값들에 operation(sum 등)을 적용합니다.
    """

    label_column: str = Field(
        ...,
        min_length=1,
        description="행을 식별하는 라벨 열 이름(예: '구분')",
    )
    labels: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "선택할 라벨 값 목록. 각 라벨은 대상 표 전체에서 정확히 "
            "하나의 행과만 일치해야 합니다."
        ),
    )

    @field_validator("label_column")
    @classmethod
    def validate_label_column(cls, value: str) -> str:
        label_column = value.strip()
        if not label_column:
            raise ValueError("label_column은 비어 있을 수 없습니다.")
        return label_column

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, value: list[str]) -> list[str]:
        normalized = [label.strip() for label in value]
        if any(not label for label in normalized):
            raise ValueError("labels에는 빈 문자열을 사용할 수 없습니다.")
        if len(normalized) != len(set(normalized)):
            raise ValueError("labels에는 중복 값을 사용할 수 없습니다.")
        return normalized


def _resolve_table_targets(
    targets: list[TableTarget],
    column: str,
    state: AgentState,
    row_selector: TableRowSelector | None = None,
) -> list[dict[str, Any]]:
    """TableTarget 목록을 실제 R_TABLE item dict 목록으로 해석합니다.

    다음 중 하나라도 어긋나면 ValueError를 발생시켜 Retriever LLM이 tool
    호출을 다시 만들게 합니다(state에 결과를 남기지 않음). 이 조건들은
    LLM이 애초에 존재하지 않거나 사용할 수 없는 대상을 지정한
    구조적 오류이기 때문입니다.

    - result_id가 state에 없음
    - 그 RetrievalResult의 status가 SUCCESS가 아님(finish의 SUCCESS 전용
      선택 규칙과 동일한 원칙)
    - item_index가 범위를 벗어남
    - 가리킨 item이 R_TABLE이 아님
    - 가리킨 item에 column이 없음
    - row_selector가 주어졌는데 가리킨 item에 label_column이 없음

    column/label_column의 "존재 여부"는 item마다 즉시 확인할 수 있는
    구조적 성질이라 여기서 검증합니다. 반면 특정 label이 실제로 몇 개의
    행과 일치하는지는 chunk가 전부 모여야 결론 낼 수 있으므로(다른 chunk에
    있을 수 있음) 여기서 검증하지 않고, check_table_completeness로 완전성을
    확인한 뒤 calculate_table_statistic 본문에서 처리합니다.
    """

    results_by_id = {
        result.result_id: result
        for result in state.get("retrieval_results", [])
    }
    resolved: list[dict[str, Any]] = []
    for target in targets:
        result = results_by_id.get(target.result_id)
        if result is None:
            raise ValueError(f"RetrievalResult를 찾지 못했습니다: {target.result_id}")
        if result.status != "SUCCESS":
            raise ValueError(
                f"{target.result_id}의 status가 SUCCESS가 아니어서 참조할 수 "
                f"없습니다(status={result.status})."
            )
        if target.item_index >= len(result.items):
            raise ValueError(
                f"{target.result_id}에 item_index {target.item_index}가 없습니다"
                f"(item 개수: {len(result.items)})."
            )
        item = result.items[target.item_index]
        if not isinstance(item, dict) or item.get("type") != "r_table":
            raise ValueError(
                f"{target.result_id}의 item {target.item_index}는 R_TABLE이 아닙니다."
            )
        columns = item.get("columns")
        if not isinstance(columns, list) or column not in columns:
            raise ValueError(
                f"{target.result_id}의 item {target.item_index}에 "
                f"'{column}' 열이 없습니다."
            )
        if row_selector is not None and row_selector.label_column not in columns:
            raise ValueError(
                f"{target.result_id}의 item {target.item_index}에 "
                f"label_column '{row_selector.label_column}'이 없습니다."
            )
        resolved.append(item)
    return resolved


def _next_derived_ids(state: AgentState) -> tuple[str, str, int]:
    """계산 결과에 쓸 plan_id/result_id를 next_plan_seq에서 직접 만듭니다.

    검색 계획이 아니므로 Plan.from_plan_draft()는 사용하지 않습니다.
    """

    next_plan_seq = state.get("next_plan_seq", 1)
    plan_id = f"plan_{next_plan_seq}"
    return f"derived:{plan_id}", plan_id, next_plan_seq


def _normalized_id(value: Any) -> str | None:
    """문자열이면 strip한 뒤 비어 있지 않을 때만 반환하고, 아니면 None을 반환합니다."""

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_reference(reference: Any) -> dict[str, str] | None:
    """disclosure/section/evidence id가 담긴 dict를 검증된 인용 정보로 정규화합니다.

    id는 모두 strip 후 비어 있지 않아야 유효한 것으로 인정합니다.
    evidence_id가 있는데 section_id가 없으면(Citation의 계층 규칙 위반)
    유효하지 않은 것으로 처리합니다 — _extract_citations가 이런 조합을
    조용히 버리기 때문에, 여기서 미리 걸러내지 않으면 source_references에는
    들어있지만 실제 답변에는 인용되지 않는 결과가 생깁니다. R_TABLE
    item의 metadata와 derived item의 source_references 원소 모두
    이 함수 하나로 검증합니다.
    """

    if not isinstance(reference, dict):
        return None
    disclosure_id = _normalized_id(reference.get("disclosure_id"))
    if disclosure_id is None:
        return None
    section_id = _normalized_id(reference.get("section_id"))
    evidence_id = _normalized_id(reference.get("evidence_id"))
    if evidence_id is not None and section_id is None:
        return None
    normalized: dict[str, str] = {"disclosure_id": disclosure_id}
    if section_id is not None:
        normalized["section_id"] = section_id
    if evidence_id is not None:
        normalized["evidence_id"] = evidence_id
    return normalized


def _extract_source_reference(item: dict[str, Any]) -> dict[str, str] | None:
    """R_TABLE item의 metadata에서 인용용 disclosure/section/evidence id를 뽑습니다."""

    return _normalize_reference(item.get("metadata"))


def _collect_source_references(
    items: list[dict[str, Any]],
) -> list[dict[str, str]] | None:
    """모든 item에 유효한 인용 정보가 있는지 확인하고 중복 없이 모읍니다.

    items 중 단 하나라도 유효한 인용을 뽑을 수 없으면 전체를 신뢰할 수
    없는 것으로 보고 None을 반환합니다(일부 chunk만 인용이 있다고 해서
    나머지 chunk의 값까지 인용된 것처럼 보이면 안 되기 때문입니다).
    """

    seen: set[tuple[str, str | None, str | None]] = set()
    references: list[dict[str, str]] = []
    for item in items:
        reference = _extract_source_reference(item)
        if reference is None:
            return None
        key = (
            reference["disclosure_id"],
            reference.get("section_id"),
            reference.get("evidence_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        references.append(reference)
    return references


def _merge_derived_source_references(
    items: list[dict[str, Any]],
) -> list[dict[str, str]] | None:
    """derived record item들이 이미 갖고 있는 source_references를 병합합니다.

    R_TABLE item과 달리 derived item(계산/조합 결과)은 인용 정보를
    item["metadata"]가 아니라 item["source_references"]에 형제 key로
    직접 담고 있으므로 원소 각각을 _normalize_reference로 검증합니다
    (strip, 빈 문자열 거부, evidence_id에는 section_id 필수인 계층 규칙
    모두 동일하게 적용). calculate_table_statistic의 citation_check와
    같은 원칙으로, item 중 하나라도 유효한 인용이 없으면(형식이
    비정상이어도) 전체를 None으로 거부합니다 — 입력이 정상적인 SUCCESS
    derived 결과라면 항상 통과해야 하며, 이 조건은 조작되었거나 손상된
    state에 대한 방어입니다.
    """

    seen: set[tuple[str, str | None, str | None]] = set()
    references: list[dict[str, str]] = []
    for item in items:
        raw_references = item.get("source_references")
        if not isinstance(raw_references, list) or not raw_references:
            return None
        item_references: list[dict[str, str]] = []
        for reference in raw_references:
            normalized = _normalize_reference(reference)
            if normalized is None:
                # 하나라도 무효한 reference가 섞여 있으면 이 item 전체를
                # 신뢰할 수 없는 것으로 보고 부분 인용을 허용하지 않는다.
                return None
            item_references.append(normalized)
        for normalized in item_references:
            key = (
                normalized["disclosure_id"],
                normalized.get("section_id"),
                normalized.get("evidence_id"),
            )
            if key in seen:
                continue
            seen.add(key)
            references.append(normalized)
    return references or None


def _collect_table_units(items: list[dict[str, Any]]) -> list[str]:
    """item들의 table_metadata.units에 선언된 표 단위를 중복 없이 모읍니다.

    셀 자체에 단위가 없을 때(예: 표 전체가 "단위: 백만원"이라고만 표시된
    경우) 계산 결과에 붙일 단위를 보완하는 데 사용합니다. 후보가 여러
    개면 이 열에 어떤 단위가 적용되는지 확정할 수 없으므로, 호출부에서
    후보가 정확히 하나일 때만 unit으로 채택해야 합니다.
    """

    units: list[str] = []
    seen: set[str] = set()
    for item in items:
        table_metadata = item.get("table_metadata")
        if not isinstance(table_metadata, dict):
            continue
        declared = table_metadata.get("units")
        if not isinstance(declared, list):
            continue
        for entry in declared:
            if isinstance(entry, str) and entry and entry not in seen:
                seen.add(entry)
                units.append(entry)
    return units


_RECORD_EXTRACTION_ERROR = object()


def _extract_cell_value(record: Any, column: str) -> str | None | object:
    """record에서 column 값을 안전하게 꺼냅니다.

    RetrievalResult.items는 검증되지 않은 dict이므로, record가 dict가
    아니거나 values가 없거나 dict가 아니거나 column key 자체가 없거나
    셀 값이 str/None이 아니면 예외를 던지지 않고 `_RECORD_EXTRACTION_ERROR`
    sentinel을 반환합니다. 정상적으로 존재하는 값(문자열 또는 진짜
    결측인 None)만 그대로 반환합니다.
    """

    if not isinstance(record, dict):
        return _RECORD_EXTRACTION_ERROR
    values = record.get("values")
    if not isinstance(values, dict) or column not in values:
        return _RECORD_EXTRACTION_ERROR
    cell = values[column]
    if cell is not None and not isinstance(cell, str):
        return _RECORD_EXTRACTION_ERROR
    return cell


def _apply_calculation_operation(
    operation: CalculationOperation,
    values: list[Decimal],
) -> tuple[Decimal | None, str | None]:
    """지정한 연산을 values에 적용합니다.

    계산에 성공하면 (결과값, None)을, 계산할 수 없으면 (None, 실패 사유)를
    반환합니다. mode는 동률이 여러 개면 임의로 하나를 고르지 않고
    계산 불가로 처리합니다. stdev는 표본 표준편차(n-1로 나눔)이며
    값이 2개 미만이면 계산 불가로 처리합니다.
    """

    if operation == "sum":
        return sum(values), None
    if operation == "mean":
        return sum(values) / len(values), None
    if operation == "median":
        return statistics.median(values), None
    if operation == "max":
        return max(values), None
    if operation == "min":
        return min(values), None
    if operation == "mode":
        modes = statistics.multimode(values)
        if len(modes) > 1:
            return None, "최빈값이 여러 개로 동률입니다."
        return modes[0], None
    if operation == "stdev":
        if len(values) < 2:
            return None, "표본 표준편차는 값이 2개 이상 있어야 계산할 수 있습니다."
        return statistics.stdev(values), None
    raise ValueError(f"지원하지 않는 operation입니다: {operation}")


def _build_calculation_invalid_result(
    *,
    result_id: str,
    plan_id: str,
    query: str,
    reason: str,
    failure_stage: str,
    source_result_ids: list[str],
) -> RetrievalResult:
    return RetrievalResult(
        result_id=result_id,
        plan_id=plan_id,
        source="derived",
        status="INVALID_INPUT",
        query=query,
        items=[],
        result_count=0,
        metadata={
            "failure_stage": failure_stage,
            "reason": reason,
            "source_result_ids": source_result_ids,
        },
    )


def _validate_calculate_call(
    variable_name: str,
    column: str,
    targets: list[TableTarget],
    state: AgentState,
    row_selector: TableRowSelector | None = None,
) -> list[dict[str, Any]]:
    """calculate_table_statistic의 구조적 오류를 검증하고 해석된 R_TABLE item들을 반환합니다.

    validate_retriever_tool_call(실행 전 사전 검증)과 calculate_table_statistic
    본문이 모두 이 함수를 호출합니다. 두 경로가 서로 다른 검증을 하면,
    사전 검증은 통과했는데 실제 실행에서 ValueError가 나는 경우
    retriever 노드의 재시도 루프를 벗어나 처리되지 않은 예외로 이어질
    수 있으므로 반드시 동일한 검증을 공유해야 합니다.
    """

    if not targets:
        raise ValueError("targets에는 최소 하나의 TableTarget이 필요합니다.")
    if not variable_name.strip():
        raise ValueError("variable_name은 비어 있을 수 없습니다.")
    if not column.strip():
        raise ValueError("column은 비어 있을 수 없습니다.")
    return _resolve_table_targets(targets, column, state, row_selector)


def _select_labeled_records(
    resolved_items: list[dict[str, Any]],
    row_selector: TableRowSelector,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """resolved_items 전체에서 row_selector.labels와 정확히 일치하는 record를 하나씩 찾습니다.

    각 label은 label_column 값을 공백만 정규화한 뒤(strip) 정확히
    비교합니다. 부분 일치는 하지 않습니다. 반환값은
    (선택된 record 목록, 실패 사유)입니다. label이 하나도 매치되지
    않거나 여러 record와 매치되면 자동으로 무시하거나 합치지 않고
    (None, 사유)를 반환합니다 — chunk가 전부 모인 뒤에만 호출되므로,
    여기서 매치가 없다는 것은 그 label이 이 표에 정말 없다는 뜻입니다.

    label_column 값을 읽을 수 없는(record가 dict가 아니거나 값이 없거나
    문자열이 아니거나 공백뿐인) record가 하나라도 있으면 즉시 실패로
    처리합니다. 그런 record를 조용히 건너뛰면, 그 record가 사실은
    요청한 label과 같은 행이었는지 판별할 수 없어 "정확히 한 행과
    일치"한다는 보장이 깨지기 때문입니다.

    입력 예시:
        resolved_items=[{...columns:["구분","당기금액"], records:[
            {"record_index":0,"values":{"구분":"유동부채","당기금액":"100"}},
            {"record_index":1,"values":{"구분":"비유동부채","당기금액":"200"}},
        ]}]
        row_selector=TableRowSelector(label_column="구분", labels=["유동부채"])

    출력 예시:
        ([{"record_index":0,"values":{"구분":"유동부채","당기금액":"100"}}], None)
    """

    matches: dict[str, list[dict[str, Any]]] = {label: [] for label in row_selector.labels}
    for item in resolved_items:
        for record in item.get("records", []):
            cell = _extract_cell_value(record, row_selector.label_column)
            if not isinstance(cell, str) or not cell.strip():
                return None, (
                    f"label_column '{row_selector.label_column}' 값을 읽을 수 "
                    "없는 행이 있어 라벨을 안전하게 선택할 수 없습니다."
                )
            normalized = cell.strip()
            if normalized in matches:
                matches[normalized].append(record)

    missing = [label for label, records in matches.items() if not records]
    if missing:
        return None, (
            f"label_column '{row_selector.label_column}'에서 다음 label을 "
            f"찾지 못했습니다: {missing}"
        )
    duplicated = [label for label, records in matches.items() if len(records) > 1]
    if duplicated:
        return None, (
            f"label_column '{row_selector.label_column}'에서 다음 label이 "
            f"여러 행과 일치해 자동으로 합산하지 않습니다: {duplicated}"
        )
    return [matches[label][0] for label in row_selector.labels], None


@tool
def calculate_table_statistic(
    variable_name: str,
    operation: CalculationOperation,
    column: str,
    targets: list[TableTarget],
    state: Annotated[AgentState, InjectedState],
    row_selector: TableRowSelector | None = None,
) -> dict:
    """검색된 R_TABLE의 한 열에 통계 연산을 적용하고 결과를 새 RetrievalResult로 state에 저장합니다.

    R_TABLE에만 사용할 수 있습니다. LLM은 숫자를 직접 계산하거나
    전달하지 않습니다 — 어떤 열에 어떤 연산을 적용할지만 지정하면, 실제
    값 추출과 연산은 이 tool이 state에서 직접 수행합니다.

    args:
        variable_name(str): 계산 결과에 붙일 사람이 읽을 이름.
            예: "삼성전자 2024년 매출액 합계". 빈 문자열은 허용하지 않습니다.
        operation(CalculationOperation): 'sum', 'mean', 'median', 'max',
            'min', 'mode', 'stdev'(표본 표준편차, 값 2개 이상 필요) 중 하나
        column(str): 집계할 R_TABLE의 열 이름. 대상 표의 columns에
            정확히 존재해야 합니다. 빈 문자열은 허용하지 않습니다.
        targets(list[TableTarget]): 집계 대상 R_TABLE의 위치 목록(최소
            1개). result_id와 item_index로 정확히 존재하는 R_TABLE
            item을 가리켜야 합니다. 하나의 표가 여러 chunk로 나뉘어
            서로 다른 result_id에 저장된 경우에만 여러 개를 나열하세요.
            서로 다른 표(예: 서로 다른 기업·기간의 표)를 섞지 마세요 —
            표마다 각각 calculate_table_statistic을 호출한 뒤 비교
            전용 tool로 비교하세요.
        state(AgentState): InjectedState로 주입되며 LLM에는 보이지
            않습니다. tool 내부에서만 state["retrieval_results"]를
            조회하는 데 사용합니다.
        row_selector(TableRowSelector | None): 지정하면 표 전체가 아니라
            특정 계정과목(행)만 계산 대상으로 삼습니다. label_column(예:
            "구분") 열의 값이 공백만 정규화한 뒤 정확히 일치하는 행만
            골라 그 값들에 operation을 적용합니다. 부분 일치는 하지
            않습니다("유동부채"로 찾아도 "비유동부채"는 매치되지
            않습니다). labels를 여러 개 지정하면(예: 총부채를 구하려고
            ["유동부채", "비유동부채"]) 각각 정확히 한 행과 매치되어야
            합니다. label_column 값을 읽을 수 없는 행이 하나라도 있으면
            "정확히 한 행과 일치"를 보장할 수 없으므로 계산을 중단합니다.
            선택된 행 중 하나라도 값이 결측이면(전체 열 집계와 달리)
            조용히 제외하지 않고 계산을 중단합니다 — 사용자가 명시한
            행을 빠뜨리면 안 되기 때문입니다. 생략하면(기본값 None)
            기존처럼 표의 모든 행을 계산 대상으로 삼고, 결측 행은
            제외한 채 계산합니다.

    return:
        dict: next_plan_seq와 계산 결과가 담긴 새 RetrievalResult 하나를
            포함한 state update.
            - 계산에 성공하면 status="SUCCESS"이고 items에
              {"type": "record", "fields": {변수명/연산/값/단위/사용한
              값 개수, row_selector 사용 시 row_selection:{label_column,
              labels}}, "source_references": [원본 인용 정보]}가
              담깁니다. 셀 자체에 단위가 없으면 표 전체 단위(정확히
              하나로 확정될 때만)로 보완합니다. 대상 item 중 하나라도
              유효한 인용 정보(disclosure_id, 그리고 evidence_id가
              있다면 그에 대응하는 section_id)가 없으면 성공으로
              처리하지 않습니다.
            - 표가 불완전하거나(chunk 누락, 일부 record 제외 등),
              row_selector의 label이 하나도 없거나 여러 행과 일치하거나
              label_column을 읽을 수 없는 행이 있거나, row_selector로
              선택한 행 중 결측값이 있거나, 값을 계산할 수 없거나
              (비정상 값 포함, 단위 혼재, 표 단위 후보가 여러 개라
              확정할 수 없음, 최빈값 동률, 모든 값이 결측 등), record
              구조 자체가 잘못됐거나, 인용할 원본 정보가 없으면
              status="INVALID_INPUT"이고 items는 빈 목록입니다.
              metadata.failure_stage로 원인 단계
              (completeness_check/citation_check/row_selection/
              record_shape/numeric_parsing/unit_check/calculation)를
              구분합니다.
            - result_id가 존재하지 않거나, 그 RetrievalResult의
              status가 SUCCESS가 아니거나, item_index가 존재하지
              않거나, 가리킨 item이 R_TABLE이 아니거나, column이나
              row_selector.label_column이 없거나, targets가 비어
              있거나, variable_name/column이 비어 있으면 ValueError를
              발생시킵니다(state에 결과를 남기지 않고 tool 호출 자체를
              다시 만들어야 합니다). label이 실제로 몇 개와 일치하는지는
              구조적 오류가 아니므로 여기 포함되지 않습니다(위 INVALID_INPUT
              참고).
    """

    resolved_items = _validate_calculate_call(
        variable_name, column, targets, state, row_selector,
    )
    result_id, plan_id, next_plan_seq = _next_derived_ids(state)
    source_result_ids = [target.result_id for target in targets]
    request_query = json.dumps(
        {
            "operation": operation,
            "column": column,
            "targets": [target.model_dump() for target in targets],
            "row_selector": row_selector.model_dump() if row_selector else None,
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    def invalid(reason: str, failure_stage: str) -> dict:
        result = _build_calculation_invalid_result(
            result_id=result_id,
            plan_id=plan_id,
            query=request_query,
            reason=reason,
            failure_stage=failure_stage,
            source_result_ids=source_result_ids,
        )
        return {"next_plan_seq": next_plan_seq + 1, "retrieval_results": [result]}

    completeness = check_table_completeness(resolved_items, column)
    if not completeness.is_complete:
        return invalid(
            completeness.reason or "표가 완전하지 않습니다.",
            "completeness_check",
        )

    source_references = _collect_source_references(resolved_items)
    if not source_references:
        return invalid(
            "계산 결과에 연결할 원본 공시 인용 정보가 없습니다.",
            "citation_check",
        )

    if row_selector is not None:
        records_to_scan, selection_error = _select_labeled_records(
            resolved_items, row_selector,
        )
        if selection_error is not None:
            return invalid(selection_error, "row_selection")
    else:
        records_to_scan = [
            record
            for item in resolved_items
            for record in item.get("records", [])
        ]

    parsed_values: list[Decimal] = []
    units: set[str] = set()
    for record in records_to_scan:
        cell = _extract_cell_value(record, column)
        if cell is _RECORD_EXTRACTION_ERROR:
            return invalid(
                f"'{column}' 값을 가진 record 구조가 올바르지 않습니다.",
                "record_shape",
            )
        parsed = parse_numeric_cell(cell)
        if parsed.is_missing:
            if row_selector is not None:
                return invalid(
                    f"'{column}' 열에서 선택한 행 중 하나가 결측값입니다: {parsed.raw!r}",
                    "numeric_parsing",
                )
            continue
        if parsed.is_invalid:
            return invalid(
                f"'{column}' 열에 숫자로 해석할 수 없는 값이 있습니다: {parsed.raw!r}",
                "numeric_parsing",
            )
        parsed_values.append(parsed.value)
        if parsed.unit is not None:
            units.add(parsed.unit)

    if len(units) > 1:
        return invalid(
            f"'{column}' 열에 서로 다른 단위가 섞여 있습니다: {sorted(units)}",
            "unit_check",
        )

    table_units = _collect_table_units(resolved_items)
    unit = next(iter(units)) if units else None
    if unit is None:
        if len(table_units) > 1:
            return invalid(
                f"'{column}' 열의 단위를 표에서 하나로 확정할 수 없습니다: {table_units}",
                "unit_check",
            )
        if len(table_units) == 1:
            unit = table_units[0]

    if not parsed_values:
        return invalid(
            f"'{column}' 열에 계산할 수 있는 값이 없습니다(모든 값이 결측).",
            "numeric_parsing",
        )

    value, operation_error = _apply_calculation_operation(operation, parsed_values)
    if operation_error is not None:
        return invalid(operation_error, "calculation")

    fields: dict[str, Any] = {
        "variable_name": variable_name,
        "operation": operation,
        "value": str(value),
        "unit": unit,
        "input_count": len(parsed_values),
    }
    if row_selector is not None:
        fields["row_selection"] = {
            "label_column": row_selector.label_column,
            "labels": row_selector.labels,
        }
    result_item = {
        "type": "record",
        "fields": fields,
        "source_references": source_references,
    }
    result = RetrievalResult(
        result_id=result_id,
        plan_id=plan_id,
        source="derived",
        status="SUCCESS",
        query=request_query,
        items=[result_item],
        result_count=1,
        metadata={
            "source_result_ids": source_result_ids,
            "result_kind": RESULT_KIND_NUMERIC_SCALAR,
        },
    )
    return {"next_plan_seq": next_plan_seq + 1, "retrieval_results": [result]}


CombineOperation = Literal[
    "sum", "mean", "difference", "ratio", "percent_ratio", "percent_change",
    "cagr", "ordering",
]
OrderingDirection = Literal["ascending", "descending"]

# result_kind는 derived RetrievalResult가 어떤 모양의 값을 담고 있는지
# 나타내는 구분값입니다. combine_numeric_results는 numeric_scalar만
# 입력으로 받아, numeric_ordering(순위 목록) 결과를 실수로 다시 더하거나
# 나누는 것을 막습니다.
RESULT_KIND_NUMERIC_SCALAR = "numeric_scalar"
RESULT_KIND_NUMERIC_ORDERING = "numeric_ordering"

# operation별 허용되는 targets 개수 범위. 위쪽 경계가 None이면 상한 없음.
_COMBINE_OPERATION_INPUT_COUNTS: dict[CombineOperation, tuple[int, int | None]] = {
    "sum": (2, None),
    "mean": (2, None),
    "difference": (2, 2),
    "ratio": (2, 2),
    "percent_ratio": (2, 2),
    "percent_change": (2, 2),
    "cagr": (2, 2),
    "ordering": (2, None),
}


class NumericResultTarget(BaseModel):
    """combine_numeric_results가 참조할 numeric_scalar 결과 하나의 주소입니다.

    result_id는 대상 derived RetrievalResult, item_index는 그 안에서
    이 숫자 결과가 위치한 인덱스입니다. numeric_scalar 결과는 현재 항상
    item을 하나만 가지므로 보통 0을 사용합니다.
    """

    result_id: str = Field(
        ...,
        min_length=1,
        description="참조할 derived RetrievalResult의 result_id",
    )
    item_index: int = Field(
        default=0,
        ge=0,
        strict=True,
        description="RetrievalResult.items 안에서 이 숫자 결과의 위치(보통 0)",
    )

    @field_validator("result_id")
    @classmethod
    def validate_result_id(cls, value: str) -> str:
        result_id = value.strip()
        if not result_id:
            raise ValueError("result_id는 비어 있을 수 없습니다.")
        return result_id


def _resolve_combine_operand(
    target: NumericResultTarget,
    state: AgentState,
) -> tuple[Decimal, str | None, dict[str, Any]]:
    """combine_numeric_results가 참조할 하나의 numeric_scalar 결과를 해석합니다.

    calculate_table_statistic이나 combine_numeric_results 자신이 만든
    result_kind="numeric_scalar"인 SUCCESS 상태의 derived RetrievalResult만
    허용합니다. numeric_ordering(순위 목록) 결과는 단일 숫자가 아니므로
    거부합니다. 존재하지 않거나 이 조건에 맞지 않으면 ValueError를
    발생시켜 Retriever LLM이 다른 대상을 고르게 합니다.

    반환값은 (값, 단위, 원본 item)입니다. 원본 item은 source_references를
    다시 모으는 데 사용합니다.
    """

    results_by_id = {
        result.result_id: result
        for result in state.get("retrieval_results", [])
    }
    result = results_by_id.get(target.result_id)
    if result is None:
        raise ValueError(f"RetrievalResult를 찾지 못했습니다: {target.result_id}")
    if result.source != "derived":
        raise ValueError(
            f"{target.result_id}는 derived 계산 결과가 아니어서 참조할 수 없습니다"
            f"(source={result.source})."
        )
    if result.status != "SUCCESS":
        raise ValueError(
            f"{target.result_id}의 status가 SUCCESS가 아니어서 참조할 수 없습니다"
            f"(status={result.status})."
        )
    if result.metadata.get("result_kind") != RESULT_KIND_NUMERIC_SCALAR:
        raise ValueError(
            f"{target.result_id}는 숫자 scalar 결과가 아니어서 참조할 수 없습니다"
            f"(result_kind={result.metadata.get('result_kind')!r})."
        )
    if target.item_index >= len(result.items):
        raise ValueError(
            f"{target.result_id}에 item_index {target.item_index}가 없습니다"
            f"(item 개수: {len(result.items)})."
        )

    item = result.items[target.item_index]
    if not isinstance(item, dict) or item.get("type") != "record":
        raise ValueError(
            f"{target.result_id}의 item {target.item_index}는 계산 결과 형식이 아닙니다."
        )
    fields = item.get("fields")
    if not isinstance(fields, dict):
        raise ValueError(f"{target.result_id}에 fields가 없습니다.")
    value_text = fields.get("value")
    if not isinstance(value_text, str):
        raise ValueError(f"{target.result_id}의 value가 문자열이 아닙니다.")
    try:
        value = Decimal(value_text)
    except InvalidOperation as error:
        raise ValueError(f"{target.result_id}의 value를 숫자로 해석할 수 없습니다.") from error

    unit = fields.get("unit")
    if unit is not None and not isinstance(unit, str):
        raise ValueError(f"{target.result_id}의 unit이 문자열이 아닙니다.")
    return value, unit, item


def _validate_combine_call(
    variable_name: str,
    operation: CombineOperation,
    targets: list[NumericResultTarget],
    state: AgentState,
    periods: int | None = None,
) -> list[tuple[Decimal, str | None, dict[str, Any]]]:
    """combine_numeric_results의 구조적 오류를 검증하고 해석된 operand들을 반환합니다.

    validate_retriever_tool_call(실행 전 사전 검증)과 combine_numeric_results
    본문이 모두 이 함수를 호출합니다. _validate_calculate_call과 같은 이유로
    두 경로는 반드시 같은 검증을 공유해야 합니다.
    """

    if not variable_name.strip():
        raise ValueError("variable_name은 비어 있을 수 없습니다.")
    target_keys = [(target.result_id, target.item_index) for target in targets]
    if len(target_keys) != len(set(target_keys)):
        raise ValueError("targets에는 중복된 (result_id, item_index)를 사용할 수 없습니다.")

    min_count, max_count = _COMBINE_OPERATION_INPUT_COUNTS[operation]
    if len(targets) < min_count or (max_count is not None and len(targets) > max_count):
        expected = (
            f"정확히 {min_count}개" if min_count == max_count else f"최소 {min_count}개"
        )
        raise ValueError(
            f"{operation}에는 targets가 {expected} 필요합니다"
            f"(전달된 개수: {len(targets)})."
        )

    if operation == "cagr":
        if not isinstance(periods, int) or isinstance(periods, bool) or periods < 1:
            raise ValueError("cagr에는 1 이상의 정수 periods가 필요합니다.")

    return [_resolve_combine_operand(target, state) for target in targets]


@tool
def combine_numeric_results(
    variable_name: str,
    operation: CombineOperation,
    targets: list[NumericResultTarget],
    state: Annotated[AgentState, InjectedState],
    direction: OrderingDirection = "ascending",
    periods: Annotated[int | None, Field(strict=True)] = None,
) -> dict:
    """calculate_table_statistic 등이 만든 numeric_scalar 결과 여러 개를 조합합니다.

    서로 다른 표에서 각각 계산한 숫자를 합치거나 비교할 때 사용합니다.
    서로 다른 표를 calculate_table_statistic 하나에 직접 섞을 수 없으므로,
    표마다 각각 계산한 뒤 이 tool로 조합하세요. LLM은 어떤 결과들을
    어떤 연산으로 조합할지만 지정하고, 실제 산술은 이 tool이 수행합니다.

    args:
        variable_name(str): 조합 결과에 붙일 사람이 읽을 이름. 빈
            문자열은 허용하지 않습니다.
        operation(CombineOperation):
            - 'sum': targets 전체를 더함(2개 이상)
            - 'mean': targets 전체의 평균(2개 이상). 기초·기말 평균
              같은 지표에 사용
            - 'difference': 정확히 2개, targets[0] - targets[1]
            - 'ratio': 정확히 2개, targets[0] / targets[1]. 배수로
              표현하는 지표(PER, 회전율 등)에 사용
            - 'percent_ratio': 정확히 2개, targets[0] / targets[1] * 100.
              ROI·ROA·ROE처럼 백분율로 표현하는 비율 지표에 사용
              (예: ROA = 당기순이익 / 자산총계 * 100). 기준값(분자)이
              targets[0], 나누는 값(분모)이 targets[1]입니다.
            - 'percent_change': 정확히 2개,
              (targets[1] - targets[0]) / targets[0] * 100.
              기준은 targets[0](이전 값)입니다. percent_ratio와
              혼동하지 마세요 — percent_change는 "증감률"(예: 전년
              대비 몇 % 늘었는지), percent_ratio는 서로 다른 두
              항목의 "비율"(예: 순이익이 자산의 몇 %인지)입니다.
            - 'cagr': 정확히 2개, (targets[1] / targets[0]) ** (1 / periods)
              - 1을 백분율로 반환. targets[0]이 시작 값, targets[1]이
              끝 값이며 periods(기간 수, 정수)를 반드시 지정해야
              합니다. 시작 값과 끝 값이 모두 양수가 아니면(0, 음수,
              부호 전환 포함) 계산할 수 없습니다.
            - 'ordering': targets를 값 기준으로 정렬해 순위를 매김
              (2개 이상). 정렬 방향은 direction으로 지정합니다.
        targets(list[NumericResultTarget]): 조합할 numeric_scalar
            결과의 위치 목록. calculate_table_statistic 또는
            combine_numeric_results가 만든 result_kind="numeric_scalar",
            status="SUCCESS" 결과만 참조할 수 있습니다(numeric_ordering
            결과는 참조 불가). difference/ratio/percent_ratio/
            percent_change/cagr에서는 순서가 결과에 직접 영향을 주므로
            정확히 지정하세요. 동일한 (result_id, item_index) 조합을
            중복해서 넣을 수 없습니다(이중 계산 방지).
        state(AgentState): InjectedState로 주입되며 LLM에는 보이지
            않습니다. tool 내부에서만 state["retrieval_results"]를
            조회하는 데 사용합니다.
        direction(OrderingDirection): operation='ordering'일 때만
            사용하는 정렬 방향('ascending' 또는 'descending', 기본
            'ascending'). 다른 operation에서는 무시됩니다. 값이 같은
            대상은 입력 순서를 그대로 유지합니다(안정 정렬).
        periods(int | None): operation='cagr'일 때만 사용하는 기간 수
            (예: 2020년부터 2024년까지면 4). 1 이상의 정수여야 합니다.
            다른 operation에서는 무시됩니다.

    return:
        dict: next_plan_seq와 조합 결과가 담긴 새 RetrievalResult 하나를
            포함한 state update.
            - sum/mean/difference/ratio/percent_ratio/percent_change/
              cagr이 성공하면 status="SUCCESS", metadata.result_kind=
              "numeric_scalar"이고 items[0].fields에 변수명/연산/값/
              단위/사용한 값 개수가 담깁니다. sum/mean/difference/ratio/
              percent_ratio/percent_change는 입력 단위가 전부 완전히
              같을 때만(모두 None이거나 모두 같은 문자열) 계산하며,
              calculate_table_statistic의 셀 단위 정책과 달리 단위
              없음과 명시된 단위를 같다고 보지 않습니다. sum·mean·
              difference는 그 공통 단위를 그대로 사용합니다. ratio는
              단위를 None으로, percent_ratio·percent_change·cagr는
              "%"로 반환합니다.
            - ordering이 성공하면 status="SUCCESS",
              metadata.result_kind="numeric_ordering"이고
              items[0].fields는 단일 value 대신 {"direction":...,
              "unit":..., "ordered_results": [{"rank":1, "result_id":...,
              "variable_name":..., "value":...}, ...]} 형태입니다.
              rank는 정렬 순서를 나타내는 1부터 시작하는 순번입니다.
            - 모든 경우 입력 대상 전체의 source_references를
              검증·병합한 결과가 함께 담기며, 하나라도 유효한 인용이
              없으면 성공으로 처리하지 않습니다.
            - 입력 단위가 서로 다르거나(비교 가능한 단위끼리만 허용),
              0으로 나누게 되거나, cagr에서 두 값의 부호가 다르거나,
              인용할 원본 정보가 없으면 status="INVALID_INPUT"이고
              items는 빈 목록입니다. metadata.failure_stage로 원인을
              구분합니다.
            - result_id가 존재하지 않거나, numeric_scalar
              result_kind의 SUCCESS 결과가 아니거나, item_index가
              존재하지 않거나, 계산 결과 형식이 아니거나, operation에
              필요한 개수의 targets가 아니거나, cagr인데 1 이상의
              정수 periods가 없거나, 중복 target이 있거나,
              variable_name이 비어 있으면 ValueError를 발생시킵니다.
    """

    operands = _validate_combine_call(variable_name, operation, targets, state, periods)
    values = [operand[0] for operand in operands]
    units = [operand[1] for operand in operands]
    items = [operand[2] for operand in operands]
    source_result_ids = [target.result_id for target in targets]

    result_id, plan_id, next_plan_seq = _next_derived_ids(state)
    request_query = json.dumps(
        {
            "operation": operation,
            "direction": direction if operation == "ordering" else None,
            "periods": periods if operation == "cagr" else None,
            "targets": [target.model_dump() for target in targets],
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    def invalid(reason: str, failure_stage: str) -> dict:
        result = _build_calculation_invalid_result(
            result_id=result_id,
            plan_id=plan_id,
            query=request_query,
            reason=reason,
            failure_stage=failure_stage,
            source_result_ids=source_result_ids,
        )
        return {"next_plan_seq": next_plan_seq + 1, "retrieval_results": [result]}

    source_references = _merge_derived_source_references(items)
    if not source_references:
        return invalid("조합 결과에 연결할 원본 공시 인용 정보가 없습니다.", "citation_check")

    distinct_units = set(units)
    if len(distinct_units) > 1:
        unit_labels = sorted(
            "(없음)" if unit is None else unit for unit in distinct_units
        )
        return invalid(
            f"입력들의 단위가 서로 달라 조합할 수 없습니다: {unit_labels}",
            "unit_check",
        )
    input_unit = units[0]

    if operation == "sum":
        value = sum(values)
        unit = input_unit
    elif operation == "mean":
        value = sum(values) / len(values)
        unit = input_unit
    elif operation == "difference":
        value = values[0] - values[1]
        unit = input_unit
    elif operation == "ratio":
        if values[1] == 0:
            return invalid("두 번째 값이 0이어서 나눌 수 없습니다.", "calculation")
        value = values[0] / values[1]
        unit = None
    elif operation == "percent_ratio":
        if values[1] == 0:
            return invalid("두 번째 값이 0이어서 나눌 수 없습니다.", "calculation")
        value = values[0] / values[1] * 100
        unit = "%"
    elif operation == "percent_change":
        if values[0] == 0:
            return invalid("기준값(첫 번째 값)이 0이어서 증감률을 계산할 수 없습니다.", "calculation")
        value = (values[1] - values[0]) / values[0] * 100
        unit = "%"
    elif operation == "cagr":
        if values[0] <= 0 or values[1] <= 0:
            return invalid(
                "CAGR은 시작 값과 끝 값이 모두 양수여야 계산할 수 있습니다.",
                "calculation",
            )
        growth_ratio = values[1] / values[0]
        try:
            value = (growth_ratio ** (Decimal(1) / periods) - 1) * 100
        except InvalidOperation:
            return invalid("CAGR 계산 중 값을 확정할 수 없습니다.", "calculation")
        unit = "%"
    elif operation == "ordering":
        ranked = sorted(
            zip(source_result_ids, values, items),
            key=lambda entry: entry[1],
            reverse=(direction == "descending"),
        )
        fields = {
            "variable_name": variable_name,
            "operation": operation,
            "direction": direction,
            "unit": input_unit,
            "input_count": len(values),
            "ordered_results": [
                {
                    "rank": rank,
                    "result_id": ranked_result_id,
                    "variable_name": (ranked_item.get("fields") or {}).get("variable_name"),
                    "value": str(ranked_value),
                }
                for rank, (ranked_result_id, ranked_value, ranked_item) in enumerate(ranked, start=1)
            ],
        }
        result_item = {
            "type": "record",
            "fields": fields,
            "source_references": source_references,
        }
        result = RetrievalResult(
            result_id=result_id,
            plan_id=plan_id,
            source="derived",
            status="SUCCESS",
            query=request_query,
            items=[result_item],
            result_count=1,
            metadata={
                "source_result_ids": source_result_ids,
                "result_kind": RESULT_KIND_NUMERIC_ORDERING,
            },
        )
        return {"next_plan_seq": next_plan_seq + 1, "retrieval_results": [result]}
    else:
        raise ValueError(f"지원하지 않는 operation입니다: {operation}")

    result_item = {
        "type": "record",
        "fields": {
            "variable_name": variable_name,
            "operation": operation,
            "value": str(value),
            "unit": unit,
            "input_count": len(values),
        },
        "source_references": source_references,
    }
    result = RetrievalResult(
        result_id=result_id,
        plan_id=plan_id,
        source="derived",
        status="SUCCESS",
        query=request_query,
        items=[result_item],
        result_count=1,
        metadata={
            "source_result_ids": source_result_ids,
            "result_kind": RESULT_KIND_NUMERIC_SCALAR,
        },
    )
    return {"next_plan_seq": next_plan_seq + 1, "retrieval_results": [result]}


# retriever llm에 현재 state를 전달하기 위해 HumanMessage를 생성하는 함수
def build_retriever_human_message(state: AgentState) -> HumanMessage:
    payload = {
        "user_question": state["question_text"],
        "retrieval_results": [
            result.model_dump(mode="json")
            for result in state.get("retrieval_results", [])
        ]
    }
    return HumanMessage(
        content=(
            "현재 상태는 retrieval입니다.\n\n"
            +json.dumps(
                payload,
                ensure_ascii=False,
                indent=2
            )
        )
    )

def _extract_citations(value: Any) -> list[Citation]:
    """중첩된 retrieval item에서 명시적으로 완성된 Citation을 찾습니다.

    입력 예시:
        {"disclosure_id": "d1", "section_id": "s1", "evidence_id": "e1"}

    출력 예시:
        [Citation(disclosure_id="d1", section_id="s1", evidence_id="e1")]
    """

    citations: list[Citation] = []
    if isinstance(value, dict):
        disclosure_id = value.get("disclosure_id")
        section_id = value.get("section_id")
        evidence_id = value.get("evidence_id")
        if isinstance(disclosure_id, str) and disclosure_id.strip():
            try:
                citations.append(Citation(
                    disclosure_id=disclosure_id,
                    section_id=(
                        section_id
                        if isinstance(section_id, str) and section_id.strip()
                        else None
                    ),
                    evidence_id=(
                        evidence_id
                        if isinstance(evidence_id, str) and evidence_id.strip()
                        else None
                    ),
                ))
            except ValueError:
                pass
        for nested_value in value.values():
            citations.extend(_extract_citations(nested_value))
    elif isinstance(value, list):
        for nested_value in value:
            citations.extend(_extract_citations(nested_value))
    return citations


def _find_first_named_value(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if key in keys and nested_value not in (None, "", []):
                return nested_value
        for nested_value in value.values():
            found = _find_first_named_value(nested_value, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested_value in value:
            found = _find_first_named_value(nested_value, keys)
            if found is not None:
                return found
    return None


def _build_answer_context(item: dict[str, Any]) -> str:
    metadata = item.get("metadata")
    retrieval_context = (
        metadata.get("retrieval_context")
        if isinstance(metadata, dict)
        else None
    )
    context_source = (
        retrieval_context
        if isinstance(retrieval_context, dict)
        else item
    )
    corp_name = _find_first_named_value(context_source, {"corp_name"})
    report_name = _find_first_named_value(
        context_source,
        {"report_name", "report_nm"},
    )
    section_path = _find_first_named_value(context_source, {"section_path"})
    heading_path = _find_first_named_value(context_source, {"heading_path"})

    context_parts = [
        value.strip()
        for value in (corp_name, report_name)
        if isinstance(value, str) and value.strip()
    ]
    for path in (section_path, heading_path):
        if isinstance(path, str) and path.strip():
            path = [path]
        if isinstance(path, list):
            for part in path:
                if (
                    isinstance(part, str)
                    and part.strip()
                    and (not context_parts or context_parts[-1] != part.strip())
                ):
                    context_parts.append(part.strip())
    return " > ".join(context_parts)


def _build_answer_result_payload(
    result_id: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    item_type = item.get("type")
    payload: dict[str, Any] = {
        "result_id": result_id,
        "context": _build_answer_context(item),
    }
    if item_type == "text":
        payload["content"] = item.get("content", "")
    elif item_type == "kv_table":
        payload["content"] = {"entries": item.get("entries", [])}
    elif item_type == "r_table":
        payload["content"] = {"records": item.get("records", [])}
    elif item_type == "record" and isinstance(item.get("fields"), dict):
        payload["content"] = item["fields"]
    else:
        payload["content"] = {
            key: value
            for key, value in item.items()
            if key not in {"type", "metadata", "score"}
        }

    if item_type in {"kv_table", "r_table"}:
        table_info = dict(item.get("table_metadata") or {})
        for key in (
            "scope",
            "available_record_count",
            "included_record_count",
            "omitted_record_count",
        ):
            if key in item:
                table_info[key] = item[key]
        if table_info:
            payload["table_info"] = table_info
    return payload


def build_answer_result_map(
    state: AgentState,
) -> dict[str, dict[str, Any]]:
    """선택된 RetrievalResult를 item 단위 Answer Generator 결과로 변환합니다.

    입력 예시:
        selected_result_ids=["retrieval:plan_1"]인 AgentState

    출력 예시:
        {
            "answer_result_1": {
                "payload": {
                    "result_id": "answer_result_1",
                    "context": "삼성전자 > 사업보고서 > 재무제표",
                    "content": "본문"
                },
                "item": {"type": "text", ...}
            }
        }
    """

    results = {
        result.result_id: result
        for result in state.get("retrieval_results", [])
    }
    answer_result_map: dict[str, dict[str, Any]] = {}
    for selected_result_id in state.get("selected_result_ids", []):
        result = results[selected_result_id]
        for item in result.items:
            answer_result_id = f"answer_result_{len(answer_result_map) + 1}"
            answer_result_map[answer_result_id] = {
                "payload": _build_answer_result_payload(
                    answer_result_id,
                    item,
                ),
                "item": item,
            }
    return answer_result_map


def resolve_answer_draft(
    state: AgentState,
    draft: AnswerGeneratorOutput,
) -> AiAnswer:
    """AnswerGeneratorOutput의 result ID를 검증된 Citation으로 변환합니다.

    입력 예시:
        AnswerGeneratorOutput(answer="답변", used_result_ids=["answer_result_1"])

    출력 예시:
        AiAnswer(answer="답변", citation=[Citation(disclosure_id="d1")])
    """

    answer_result_map = build_answer_result_map(state)
    unknown_ids = [
        result_id
        for result_id in draft.used_result_ids
        if result_id not in answer_result_map
    ]
    if unknown_ids:
        raise ValueError(
            "허용되지 않은 answer result ID입니다: "
            + ", ".join(unknown_ids)
        )

    selected_citations: list[Citation] = []
    seen_result_ids: set[str] = set()
    seen_citations: set[tuple[str, str | None, str | None]] = set()
    for result_id in draft.used_result_ids:
        if result_id in seen_result_ids:
            continue
        seen_result_ids.add(result_id)
        for citation in _extract_citations(answer_result_map[result_id]["item"]):
            key = (
                citation.disclosure_id,
                citation.section_id,
                citation.evidence_id,
            )
            if key in seen_citations:
                continue
            seen_citations.add(key)
            selected_citations.append(citation)

    return AiAnswer(
        answer=draft.answer,
        citation=selected_citations
    )


@lru_cache(maxsize=4)
def _load_jsonl_index(path: Path, key: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            item_key = item.get(key)
            if isinstance(item_key, str) and item_key:
                index[item_key] = item
    return index


def format_citations(
    citations: list[Citation],
    *,
    document_manifest_path: Path = DOCUMENT_MANIFEST_PATH,
    driver: Any | None = None,
    style: Literal["sentence", "path"] = "sentence",
) -> list[str]:
    """Citation을 사용자에게 보여줄 문서·섹션 단위 문장으로 변환합니다.

    입력 예시:
        [Citation(
            disclosure_id="d20240306000686",
            section_id="d20240306000686:src0:s27",
            evidence_id="d20240306000686:src0:s27:e8"
        )]

    출력 예시:
        ["LG유플러스가 2024년 3월 6일에 발행한 「사업보고서 (2023.12)」"
         "(접수번호 20240306000686)의 「IV. 이사의 경영진단 및 분석의견」 "
         "섹션 중 「3. 재무상태 및 영업실적(연결기준) > 가. 연결 재무상태」를 "
         "근거로 사용했습니다."]
    """

    if style not in {"sentence", "path"}:
        raise ValueError(f"지원하지 않는 citation 표시 형식입니다: {style}")

    document_manifest_path = Path(document_manifest_path).resolve()
    documents = _load_jsonl_index(document_manifest_path, "rcept_no")

    citation_requests: list[dict[str, str | None]] = []
    seen_requests: set[tuple[str, str, str | None]] = set()
    for citation in citations:
        if citation.section_id is None:
            continue
        request_key = (
            citation.disclosure_id,
            citation.section_id,
            citation.evidence_id,
        )
        if request_key in seen_requests:
            continue
        seen_requests.add(request_key)
        citation_requests.append({
            "disclosure_id": citation.disclosure_id,
            "section_id": citation.section_id,
            "evidence_id": citation.evidence_id,
        })

    citation_contexts: dict[
        tuple[str, str, str | None],
        tuple[list[str], list[str] | None],
    ] = {}
    if citation_requests:
        citation_driver = neo4j_driver if driver is None else driver
        try:
            with citation_driver.session() as session:
                records = list(session.run(
                    CITATION_CONTEXT_QUERY,
                    {"citations": citation_requests},
                ))
        except Exception as error:
            raise RuntimeError("Neo4j citation context 조회 중 오류가 발생했습니다.") from error
        for record in records:
            disclosure_id = record["disclosure_id"]
            section_id = record["section_id"]
            requested_evidence_id = record["requested_evidence_id"]
            evidence_id = record["evidence_id"]
            section_path = record["section_path"]
            heading_path = record["heading_path"]
            if (
                isinstance(disclosure_id, str)
                and isinstance(section_id, str)
                and (
                    requested_evidence_id is None
                    or isinstance(requested_evidence_id, str)
                )
                and (
                    requested_evidence_id is None
                    or evidence_id == requested_evidence_id
                )
                and isinstance(section_path, list)
                and section_path
                and all(isinstance(part, str) and part for part in section_path)
                and (
                    heading_path is None
                    or (
                        isinstance(heading_path, list)
                        and all(
                            isinstance(part, str) and part
                            for part in heading_path
                        )
                    )
                )
            ):
                citation_contexts[(
                    disclosure_id,
                    section_id,
                    requested_evidence_id,
                )] = (section_path, heading_path or None)

    formatted: list[str] = []
    seen: set[
        tuple[str, tuple[str, ...] | None, tuple[str, ...] | None]
    ] = set()
    for citation in citations:
        disclosure_id = citation.disclosure_id
        if (
            len(disclosure_id) != 15
            or not disclosure_id.startswith("d")
            or not disclosure_id[1:].isdigit()
        ):
            raise ValueError(f"잘못된 disclosure_id 형식입니다: {disclosure_id}")
        rcept_no = disclosure_id[1:]
        section_path: list[str] | None = None
        heading_path: list[str] | None = None
        if citation.section_id is not None:
            context = citation_contexts.get((
                citation.disclosure_id,
                citation.section_id,
                citation.evidence_id,
            ))
            if context is None:
                raise ValueError(
                    f"Neo4j에서 citation context를 찾지 못했습니다: "
                    f"{citation.section_id}, {citation.evidence_id}"
                )
            section_path, heading_path = context
        key = (
            rcept_no,
            tuple(section_path) if section_path is not None else None,
            tuple(heading_path) if heading_path is not None else None,
        )
        if key in seen:
            continue
        seen.add(key)

        document = documents.get(rcept_no)
        if document is None:
            raise ValueError(f"manifest에서 공시를 찾지 못했습니다: {rcept_no}")
        corp_name = str(document.get("corp_name", "")).strip()
        report_name = str(document.get("report_nm", "")).strip()
        rcept_date = str(document.get("rcept_dt", "")).strip()
        if not corp_name or not report_name:
            raise ValueError(f"공시 metadata가 불완전합니다: {rcept_no}")

        if style == "path":
            path = [f"{report_name}({rcept_no})"]
            if section_path is not None:
                path.extend(section_path)
            if heading_path is not None:
                path.extend(heading_path)
            formatted.append("[" + " > ".join(path) + "]")
            continue

        try:
            published_at = datetime.strptime(rcept_date, "%Y%m%d")
        except ValueError as error:
            raise ValueError(f"잘못된 rcept_dt 형식입니다: {rcept_date}") from error

        sentence = (
            f"{corp_name}가 {published_at.year}년 {published_at.month}월 "
            f"{published_at.day}일에 발행한 「{report_name}」"
            f"(접수번호 {rcept_no})"
        )
        if section_path is not None:
            sentence += "의 「" + " > ".join(section_path) + "」 섹션"
            if heading_path is not None:
                sentence += " 중 「" + " > ".join(heading_path) + "」를 근거로 사용했습니다."
                formatted.append(sentence)
                continue
        formatted.append(sentence + "입니다.")

    return formatted


# answer generator llm에 현재 state를 전달하기 위해 HumanMessage를 생성하는 함수
def build_answer_generator_human_message(state: AgentState) -> HumanMessage:
    """질문과 item 단위 검색 결과를 Answer Generator 입력으로 변환합니다.

    입력 예시:
        selected_result_ids=["retrieval:plan_1"]

    출력 예시:
        HumanMessage(content='아래 입력을 근거로 ...\n\n[입력]\n\n{...}\n\n[출력]\n\n...')
    """

    answer_result_map = build_answer_result_map(state)
    payload = {
        "user_question": state["question_text"],
        "retrieval_finish_reason": state.get("retrieval_finish_reason"),
        "retrieval_results": [
            value["payload"]
            for value in answer_result_map.values()
        ],
    }

    json_dump = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )
    return HumanMessage(
        content=(
            "아래 입력을 근거로 최종 answer와 used_result_ids를 생성하고\n\n"
            "AnswerGeneratorOutput 형식으로 반환하세요.\n\n\n"
            "[입력]\n\n"
            f"{json_dump}\n\n\n"
            "[출력]\n\n"
            "입력으로 제공된 question과 retrieval_results만을 근거로 최종 답변을 생성하세요.\n\n"
            "- answer에는 사용자의 질문에 직접 답하세요.\n"
            "- retrieval_results에 없는 사실을 추측해서 추가하지 마세요.\n"
            "- used_result_ids에는 retrieval_results에서 실제 답변 생성에 사용한 "
            "항목의 result_id만 선택하세요.\n"
            "- retrieval_results가 비어 있어도 반드시 AnswerGeneratorOutput 형식으로 "
            "반환하세요.\n"
            "- retrieval_results가 비어 있으면 answer에는 검색 결과만으로 확인할 수 "
            "없다고 명시하고 used_result_ids는 빈 목록으로 반환하세요."
        )
    )

# 하나의 형식으로 모든 tool call을 처리하기 위한 interface 함수
def validate_retriever_tool_call(state: AgentState, tool_call: dict) -> None:
    """Retriever의 tool schema와 state 참조를 실행 전에 검증합니다."""

    if not isinstance(tool_call, dict):
        raise ValueError("tool call은 object 형식이어야 합니다.")
    name = tool_call.get("name")
    args = tool_call.get("args", {})
    if name == "retrieve_search":
        validated = retrieve_search.tool_call_schema.model_validate(args)
        executable_plan = Plan.from_plan_draft(
            validated.plan,
            state.get("next_plan_seq", 1),
        )
        _resolve_plan_dependencies(executable_plan, state)
        return
    if name == "finish":
        validated = finish.tool_call_schema.model_validate(args)
        _validate_finish_selection(
            validated.status,
            validated.selected_result_ids,
            state,
        )
        return
    if name == "calculate_table_statistic":
        validated = calculate_table_statistic.tool_call_schema.model_validate(args)
        _validate_calculate_call(
            validated.variable_name,
            validated.column,
            validated.targets,
            state,
            validated.row_selector,
        )
        return
    if name == "combine_numeric_results":
        validated = combine_numeric_results.tool_call_schema.model_validate(args)
        _validate_combine_call(
            validated.variable_name,
            validated.operation,
            validated.targets,
            state,
            validated.periods,
        )
        return
    raise ValueError(f"지원하지 않는 tool call입니다: {name}")


def execute_tool_call(state: AgentState, tool_call: dict) -> dict:

    name = tool_call["name"]
    args = tool_call["args"]

    if name == "retrieve_search":
        return retrieve_search.invoke({**args, "state": state})
    if name == "finish":
        return finish.invoke({**args, "state": state})
    if name == "calculate_table_statistic":
        return calculate_table_statistic.invoke({**args, "state": state})
    if name == "combine_numeric_results":
        return combine_numeric_results.invoke({**args, "state": state})

    raise ValueError(f"지원하지 않는 tool call입니다: {name}")
