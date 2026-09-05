import json
import re
import statistics
from datetime import datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import (
    BaseModel,
    ConfigDict,
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
import yaml

from . import system_prompts as sp
from .compactor import compact_qdrant_point, point_requires_compaction
from .llm import (
    CYPHER_BUILDER_MAX_COMPLETION_TOKENS,
    MAX_LLM_RETRIES,
    NARROW_SCOPE_MAX_COMPLETION_TOKENS,
    QDRANT_QUERY_BUILDER_MAX_COMPLETION_TOKENS,
    bind_structured_output,
    build_output_retry_message,
    get_llm,
    invoke_with_rate_limit_retry,
)
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
    QuestionAnalysis,
    RetrievalResult,
    Scope,
    ScopeDraft,
    SubQuestion,
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
QDRANT_KNOWLEDGE_COLLECTION_NAME = (
    os.getenv("QDRANT_KNOWLEDGE_COLLECTION_NAME") or "knowledge_base"
)
QDRANT_KNOWLEDGE_DENSE_VECTOR_NAME = "knowledge_dense"
QDRANT_KNOWLEDGE_SPARSE_VECTOR_NAME = "knowledge_sparse"
SCOPE_KNOWLEDGE_LIMIT = 5
SCOPE_DISCLOSURE_RESULT_LIMIT = 50
SCOPE_SECTION_RESULT_LIMIT = 500
SCOPE_DOC_GROUPS = ("periodic", "major", "exchange", "holding")
SCOPE_KNOWLEDGE_HINT_FIELDS = (
    "knowledge_type",
    "name",
    "aliases",
    "description",
    "datasets",
    "categories",
)

class CypherQuery(BaseModel):
    """실행 가능한 read-only Cypher와 parameter를 분리한 요청입니다."""

    cypher: str = Field(..., min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class CypherQueryToolArgs(BaseModel):
    """CLOVA Structured Outputs용 단순 field로 만든 Cypher 출력입니다."""

    cypher: str = Field(..., min_length=1, description="실행할 read-only Cypher")
    parameters_json: str = Field(
        ...,
        min_length=2,
        description="Cypher parameter를 나타내는 JSON object 문자열",
    )

    @field_validator("parameters_json")
    @classmethod
    def validate_parameters_json(cls, value: str) -> str:
        """Cypher parameters 문자열이 JSON object인지 검증합니다."""

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("parameters_json은 유효한 JSON이어야 합니다.") from error
        if not isinstance(parsed, dict):
            raise ValueError("parameters_json은 JSON object여야 합니다.")
        return value

    def to_cypher_query(self) -> CypherQuery:
        """LLM 출력 schema를 실행 가능한 CypherQuery로 변환합니다."""

        return CypherQuery(
            cypher=self.cypher,
            parameters=json.loads(self.parameters_json),
        )


class DisclosureSelection(BaseModel):
    """실제 공시 후보 중 SubQuestion과 관련된 공시 ID를 선택합니다."""

    model_config = ConfigDict(extra="forbid")

    selected_disclosure_ids: list[str]
    reason: str = Field(description="공시 후보를 선택하거나 제외한 간단한 이유")

    @field_validator("selected_disclosure_ids")
    @classmethod
    def normalize_ids(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("selected_disclosure_ids에는 빈 값을 사용할 수 없습니다.")
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def normalize_reason(self) -> "DisclosureSelection":
        self.reason = self.reason.strip() or "관련 공시 후보를 선택했습니다."
        return self


class SectionSelection(BaseModel):
    """선택된 공시의 실제 Section 후보 중 관련 Section ID를 선택합니다."""

    model_config = ConfigDict(extra="forbid")

    selected_section_ids: list[str]
    reason: str = Field(description="Section 후보를 선택하거나 제외한 간단한 이유")

    @field_validator("selected_section_ids")
    @classmethod
    def normalize_ids(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("selected_section_ids에는 빈 값을 사용할 수 없습니다.")
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def normalize_reason(self) -> "SectionSelection":
        self.reason = self.reason.strip() or "관련 Section 후보를 선택했습니다."
        return self


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
        """Qdrant filter의 값 유형과 범위 조건을 검증합니다."""

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
        """검색 mode에 맞게 query와 filter 조합을 검증합니다."""

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
    """CLOVA Structured Outputs용 단순 field Qdrant query 출력입니다."""

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
        """Qdrant filter 문자열이 JSON array인지 검증합니다."""

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
        """score threshold 문자열이 숫자 또는 null인지 검증합니다."""

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
        """LLM 출력 schema를 실행 가능한 QdrantQuery로 변환합니다."""

        return QdrantQuery(
            mode=self.mode,
            query_text=self.query_text.strip() or None,
            filters=json.loads(self.filters_json),
            score_threshold=json.loads(self.score_threshold_json),
        )


@lru_cache(maxsize=1)
def _load_neo4j_schema() -> str:
    """Neo4j schema 문서를 한 번 읽어 캐시합니다."""

    return NEO4J_SCHEMA_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _load_neo4j_relationship_contract() -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Neo4j schema에서 relationship별 허용 source와 target label을 읽습니다."""

    schema = yaml.safe_load(_load_neo4j_schema())
    return {
        relation["type"]: (
            tuple(relation["endpoints"]["source"]),
            tuple(relation["endpoints"]["target"]),
        )
        for relation in schema["relations"].values()
    }


@lru_cache(maxsize=1)
def _load_qdrant_query_schema() -> str:
    """Qdrant query schema 문서를 한 번 읽어 캐시합니다."""

    return QDRANT_QUERY_SCHEMA_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _get_cypher_llm() -> Any:
    """공용 LLM에 Cypher structured output을 한 번 binding합니다."""

    return bind_structured_output(
        get_llm(CYPHER_BUILDER_MAX_COMPLETION_TOKENS),
        CypherQueryToolArgs,
    )


@lru_cache(maxsize=1)
def _get_query_llm() -> Any:
    """공용 LLM에 Qdrant query structured output을 한 번 binding합니다."""

    return bind_structured_output(
        get_llm(QDRANT_QUERY_BUILDER_MAX_COMPLETION_TOKENS),
        QdrantQueryToolArgs,
    )


@lru_cache(maxsize=1)
def _get_disclosure_selection_llm() -> Any:
    """공용 LLM에 공시 후보 선택 schema를 binding합니다."""

    return bind_structured_output(
        get_llm(NARROW_SCOPE_MAX_COMPLETION_TOKENS),
        DisclosureSelection,
    )


@lru_cache(maxsize=1)
def _get_section_selection_llm() -> Any:
    """공용 LLM에 Section 후보 선택 schema를 binding합니다."""

    return bind_structured_output(
        get_llm(NARROW_SCOPE_MAX_COMPLETION_TOKENS),
        SectionSelection,
    )


def assign_subquestion_ids(analysis: QuestionAnalysis) -> QuestionAnalysis:
    """Question Analyzer 결과의 SubQuestion에 순서 기반 ID를 할당합니다."""

    return analysis.model_copy(update={
        "sub_questions": [
            subquestion.model_copy(update={
                "subquestion_id": f"subquestion_{index}",
            })
            for index, subquestion in enumerate(
                analysis.sub_questions,
                start=1,
            )
        ]
    })


def _find_subquestion(state: AgentState, subquestion_id: str) -> SubQuestion:
    """state에서 application이 할당한 ID와 일치하는 SubQuestion을 찾습니다."""

    analysis_value = state.get("question_analysis")
    if analysis_value is None:
        raise ValueError("question_analysis가 없습니다.")
    analysis = QuestionAnalysis.model_validate(analysis_value)
    for subquestion in analysis.sub_questions:
        if subquestion.subquestion_id == subquestion_id:
            return subquestion
    raise ValueError(f"SubQuestion을 찾지 못했습니다: {subquestion_id}")


def _scope_knowledge_query_text(subquestion: SubQuestion) -> str:
    """SubQuestion의 분석 필드를 Knowledge Base 검색문으로 합칩니다."""

    values = [
        subquestion.question,
        *subquestion.intents,
        *subquestion.requested_facts,
    ]
    for event in subquestion.events:
        if event.event_type:
            values.append(event.event_type)
        values.extend(event.candidate_event_types)
    return " ".join(dict.fromkeys(value.strip() for value in values if value.strip()))


def _has_scope_knowledge_hint_value(value: Any) -> bool:
    """Knowledge Base hint에서 비어 있지 않은 값만 판별합니다."""

    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def search_scope_knowledge(
    subquestion: SubQuestion,
    *,
    limit: int = SCOPE_KNOWLEDGE_LIMIT,
) -> list[dict[str, Any]]:
    """Knowledge Base collection을 hybrid search하고 payload 목록을 반환합니다."""

    query_text = _scope_knowledge_query_text(subquestion)
    embedding = text_to_hybrid_vector(query_text)
    try:
        response = qdrant_client.query_points(
            collection_name=QDRANT_KNOWLEDGE_COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=list(embedding.dense),
                    using=QDRANT_KNOWLEDGE_DENSE_VECTOR_NAME,
                    limit=limit * HYBRID_PREFETCH_MULTIPLIER,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(embedding.sparse.indices),
                        values=list(embedding.sparse.values),
                    ),
                    using=QDRANT_KNOWLEDGE_SPARSE_VECTOR_NAME,
                    limit=limit * HYBRID_PREFETCH_MULTIPLIER,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as error:
        raise RuntimeError("Knowledge Base 검색 중 오류가 발생했습니다.") from error

    hints = []
    for point in extract_qdrant_points(response):
        payload = dict(getattr(point, "payload", None) or {})
        hint = {
            field_name: value
            for field_name in SCOPE_KNOWLEDGE_HINT_FIELDS
            if _has_scope_knowledge_hint_value(
                value := payload.get(field_name)
            )
        }
        score = getattr(point, "score", None)
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            hint["score"] = float(score)
        hints.append(hint)
    return hints


def _cypher_node_labels_by_alias(cypher: str) -> dict[str, set[str]]:
    """Cypher에 선언된 node alias별 label 집합을 수집합니다."""

    labels_by_alias: dict[str, set[str]] = {}
    for match in re.finditer(
        r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*"
        r"((?:\s*:\s*[A-Za-z_][A-Za-z0-9_]*)+)",
        cypher,
    ):
        labels_by_alias.setdefault(match.group(1), set()).update(
            re.findall(r":\s*([A-Za-z_][A-Za-z0-9_]*)", match.group(2))
        )
    return labels_by_alias


def _cypher_node_pattern_labels(
    pattern: str,
    labels_by_alias: dict[str, set[str]],
) -> set[str]:
    """하나의 node pattern에서 직접 또는 alias로 확인되는 label을 반환합니다."""

    label_match = re.match(
        r"\(\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*)?"
        r"((?:\s*:\s*[A-Za-z_][A-Za-z0-9_]*)+)",
        pattern,
    )
    labels = (
        set(re.findall(r":\s*([A-Za-z_][A-Za-z0-9_]*)", label_match.group(1)))
        if label_match
        else set()
    )
    alias_match = re.match(r"\(\s*([A-Za-z_][A-Za-z0-9_]*)", pattern)
    if alias_match:
        labels.update(labels_by_alias.get(alias_match.group(1), set()))
    return labels


def _matches_relationship_endpoint(
    actual_labels: set[str],
    allowed_endpoints: tuple[str, ...],
) -> bool:
    """복합 label을 포함한 실제 node label이 schema endpoint와 호환되는지 확인합니다."""

    if not actual_labels:
        return False
    return any(
        actual_labels.issubset(expected_labels)
        or expected_labels.issubset(actual_labels)
        for endpoint in allowed_endpoints
        if (expected_labels := set(endpoint.split(":")))
    )


def _validate_cypher_relationships(query: CypherQuery) -> None:
    """Cypher relationship type, endpoint label과 방향을 Neo4j schema로 검증합니다."""

    if re.search(r"\)\s*(?:-->|<--|--)\s*\(", query.cypher):
        raise ValueError("Cypher relationship에는 schema의 relationship type을 명시해야 합니다.")

    contract = _load_neo4j_relationship_contract()
    labels_by_alias = _cypher_node_labels_by_alias(query.cypher)
    node_pattern = r"\([^()\r\n]*\)"
    relationship_pattern = re.compile(
        rf"(?=(?P<left>{node_pattern})\s*"
        r"(?P<left_connector><-|-)\s*"
        r"\[(?P<body>[^\]]*)\]\s*"
        r"(?P<right_connector>->|-)\s*"
        rf"(?P<right>{node_pattern}))"
    )

    for match in relationship_pattern.finditer(query.cypher):
        left_connector = match.group("left_connector")
        right_connector = match.group("right_connector")
        if left_connector not in {"-", "<-"}:
            continue
        type_match = re.match(
            r"\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*)?:\s*"
            r"([A-Za-z_][A-Za-z0-9_]*)",
            match.group("body"),
        )
        if type_match is None:
            raise ValueError("Cypher relationship에는 relationship type이 필요합니다.")
        relationship_type = type_match.group(1)
        endpoints = contract.get(relationship_type)
        if endpoints is None:
            raise ValueError(
                f"Neo4j schema에 없는 relationship type입니다: {relationship_type}"
            )
        if left_connector == "-" and right_connector == "->":
            source_pattern = match.group("left")
            target_pattern = match.group("right")
        elif left_connector == "<-" and right_connector == "-":
            source_pattern = match.group("right")
            target_pattern = match.group("left")
        else:
            raise ValueError(
                f"Cypher relationship 방향을 하나로 명시해야 합니다: {relationship_type}"
            )

        source_labels = _cypher_node_pattern_labels(source_pattern, labels_by_alias)
        target_labels = _cypher_node_pattern_labels(target_pattern, labels_by_alias)
        allowed_sources, allowed_targets = endpoints
        if not (
            _matches_relationship_endpoint(source_labels, allowed_sources)
            and _matches_relationship_endpoint(target_labels, allowed_targets)
        ):
            actual_source = ":".join(sorted(source_labels)) or "unknown"
            actual_target = ":".join(sorted(target_labels)) or "unknown"
            raise ValueError(
                "Cypher relationship endpoint 또는 방향이 Neo4j schema와 일치하지 "
                f"않습니다: {relationship_type}은 "
                f"{'/'.join(allowed_sources)} -> {'/'.join(allowed_targets)}, "
                f"현재 query는 {actual_source} -> {actual_target}"
            )


def _execute_scope_cypher(
    query: CypherQuery,
    *,
    stage: str,
) -> list[dict[str, Any]]:
    """고정된 Scope Cypher를 실행하고 후보 item 목록으로 변환합니다."""

    try:
        with neo4j_driver.session() as session:
            records = list(session.run(query.cypher, query.parameters))
    except Exception as error:
        raise RuntimeError("Scope 후보 Neo4j 조회 중 오류가 발생했습니다.") from error
    return parse_neo4j_response(
        records,
        plan_id=f"narrow_scope_{stage}",
        query=query.cypher,
        parameters=query.parameters,
    ).items


def _collect_named_strings(value: Any, field_name: str) -> set[str]:
    """중첩된 Neo4j 결과에서 특정 field 이름의 문자열 값을 수집합니다."""

    values: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == field_name:
                if isinstance(nested, str) and nested.strip():
                    values.add(nested.strip())
                elif isinstance(nested, list):
                    values.update(
                        item.strip()
                        for item in nested
                        if isinstance(item, str) and item.strip()
                    )
            values.update(_collect_named_strings(nested, field_name))
    elif isinstance(value, list):
        for nested in value:
            values.update(_collect_named_strings(nested, field_name))
    return values


def _collect_disclosure_section_pairs(value: Any) -> set[tuple[str, str]]:
    """같은 Neo4j record에 함께 반환된 disclosure-section 관계를 수집합니다."""

    pairs: set[tuple[str, str]] = set()
    if isinstance(value, dict):
        disclosure_id = value.get("disclosure_id")
        section_id = value.get("section_id")
        if (
            isinstance(disclosure_id, str)
            and disclosure_id.strip()
            and isinstance(section_id, str)
            and section_id.strip()
        ):
            pairs.add((disclosure_id.strip(), section_id.strip()))
        for nested in value.values():
            pairs.update(_collect_disclosure_section_pairs(nested))
    elif isinstance(value, list):
        for nested in value:
            pairs.update(_collect_disclosure_section_pairs(nested))
    return pairs


def _validate_scope_selection(
    scope: ScopeDraft,
    search_results: list[dict[str, Any]],
) -> None:
    """LLM이 선택한 모든 Scope 값이 실제 Neo4j 결과에 있었는지 검증합니다."""

    observed = {
        "corp_names": _collect_named_strings(search_results, "corp_name"),
        "corp_codes": _collect_named_strings(search_results, "corp_code"),
        "disclosure_ids": _collect_named_strings(search_results, "disclosure_id"),
        "section_ids": _collect_named_strings(search_results, "section_id"),
    }
    for field_name, selected_values in (
        ("corp_names", scope.corp_names),
        ("corp_codes", scope.corp_codes),
        ("disclosure_ids", scope.disclosure_ids),
        ("section_ids", scope.section_ids),
    ):
        unknown = sorted(set(selected_values) - observed[field_name])
        if unknown:
            raise ValueError(
                f"Neo4j 검색 결과에서 확인되지 않은 {field_name}입니다: {unknown}"
            )
    if scope.level == "SECTION":
        observed_pairs = _collect_disclosure_section_pairs(search_results)
        unlinked_sections = [
            section_id
            for section_id in scope.section_ids
            if not any(
                disclosure_id in scope.disclosure_ids
                and observed_section_id == section_id
                for disclosure_id, observed_section_id in observed_pairs
            )
        ]
        if unlinked_sections:
            raise ValueError(
                "선택한 disclosure와의 상위 관계가 확인되지 않은 section_ids입니다: "
                f"{unlinked_sections}"
            )


def _scope_issuer_names(subquestion: SubQuestion) -> list[str]:
    """SubQuestion에서 공시 발행회사로 해석된 기업명을 추출합니다."""

    names = [
        (entity.canonical_name or entity.mention).strip()
        for entity in subquestion.entities
        if "ISSUER" in entity.roles
        and entity.match_status != "NOT_FOUND"
        and (entity.canonical_name or entity.mention).strip()
    ]
    return list(dict.fromkeys(names))


def _scope_doc_groups(hints: list[dict[str, Any]]) -> list[str]:
    """가장 높은 순위의 유효한 Knowledge hint에서 doc_group을 선택합니다."""

    allowed = set(SCOPE_DOC_GROUPS)
    for hint in hints:
        datasets = hint.get("datasets")
        if not isinstance(datasets, list):
            continue
        groups = [
            value.strip()
            for value in datasets
            if isinstance(value, str) and value.strip() in allowed
        ]
        if groups:
            return list(dict.fromkeys(groups))
    return list(SCOPE_DOC_GROUPS)


def _scope_candidate_records(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Neo4j record parser 결과에서 LLM에게 필요한 fields만 추출합니다."""

    return [
        dict(fields)
        for item in items
        if isinstance(item, dict)
        and isinstance((fields := item.get("fields")), dict)
    ]


def _disclosure_scope_query(
    corp_names: list[str],
    doc_groups: list[str],
) -> CypherQuery:
    """기업과 doc_group으로 최신 Disclosure 후보 목록을 조회합니다."""

    return CypherQuery(
        cypher=(
            "MATCH (c:Company)-[:PUBLISHES]->(d:Disclosure) "
            "WHERE c.corp_name IN $corp_names "
            "AND d.doc_group IN $doc_groups "
            "AND d.is_latest_version = true "
            "RETURN c.corp_name AS corp_name, c.corp_code AS corp_code, "
            "d.id AS disclosure_id, d.report_name AS report_name, "
            "d.doc_group AS doc_group, d.rcept_date AS rcept_date "
            "ORDER BY d.rcept_date DESC LIMIT $limit"
        ),
        parameters={
            "corp_names": corp_names,
            "doc_groups": doc_groups,
            "limit": SCOPE_DISCLOSURE_RESULT_LIMIT,
        },
    )


def _section_scope_query(disclosure_ids: list[str]) -> CypherQuery:
    """선택된 Disclosure 아래의 전체 Section 후보 목록을 조회합니다."""

    return CypherQuery(
        cypher=(
            "MATCH (c:Company)-[:PUBLISHES]->(d:Disclosure) "
            "MATCH (d)-[:HAS_SECTION*1..]->(s:Section) "
            "WHERE d.id IN $disclosure_ids "
            "AND d.is_latest_version = true "
            "RETURN c.corp_name AS corp_name, c.corp_code AS corp_code, "
            "d.id AS disclosure_id, d.report_name AS report_name, "
            "s.id AS section_id, s.title AS section_title, "
            "s.section_path AS section_path, s.order_in_doc AS order_in_doc "
            "ORDER BY d.rcept_date DESC, s.order_in_doc LIMIT $limit"
        ),
        parameters={
            "disclosure_ids": disclosure_ids,
            "limit": SCOPE_SECTION_RESULT_LIMIT,
        },
    )


def _narrow_scope_human_message(
    *,
    stage: str,
    subquestion: SubQuestion,
    hints: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    doc_groups: list[str] | None = None,
    selected_disclosure_ids: list[str] | None = None,
) -> HumanMessage:
    """각 선택 단계에 SubQuestion, hint와 실제 Neo4j 후보를 명시합니다."""

    payload: dict[str, Any] = {
        "stage": stage,
        "subquestion": subquestion.model_dump(mode="json"),
        "knowledge_hints": hints,
    }
    if doc_groups is not None:
        payload["searched_doc_groups"] = doc_groups
        payload["disclosure_candidates"] = candidates
    else:
        payload["selected_disclosure_ids"] = selected_disclosure_ids or []
        payload["section_candidates"] = candidates
    return HumanMessage(content=json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        default=str,
    ))


def _print_narrow_scope_human_message(message: HumanMessage) -> None:
    """narrow_scope가 LLM에 새로 전달하는 HumanMessage만 출력합니다."""

    print(f"[narrow_scope human message]:\n{message.content}\n")


def _invoke_scope_selection(
    runnable: Any,
    *,
    output_model: type[BaseModel],
    system_prompt: str,
    human_message: HumanMessage,
    selected_field: str,
    available_ids: set[str],
) -> BaseModel:
    """후보 선택 출력을 검증하고 오류가 있으면 같은 단계에서 재호출합니다."""

    messages = [SystemMessage(content=system_prompt), human_message]
    _print_narrow_scope_human_message(human_message)
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            response = invoke_with_rate_limit_retry(runnable, messages)
            selection = (
                response
                if isinstance(response, output_model)
                else output_model.model_validate(response)
            )
            selected_ids = getattr(selection, selected_field)
            unknown_ids = sorted(set(selected_ids) - available_ids)
            if unknown_ids:
                raise ValueError(
                    f"Neo4j 후보 목록에 없는 {selected_field}입니다: {unknown_ids}"
                )
            return selection
        except (ValidationError, ValueError, TypeError, AttributeError) as error:
            if attempt == MAX_LLM_RETRIES:
                raise
            retry_message = HumanMessage(content=build_output_retry_message(
                output_model.__name__,
                error,
            ))
            messages.append(retry_message)
            _print_narrow_scope_human_message(retry_message)
    raise RuntimeError("Scope 후보 선택 결과를 생성하지 못했습니다.")


def run_narrow_scope_agent(
    subquestion_id: str,
    state: AgentState,
    *,
    llm: Any | None = None,
) -> Scope:
    """공시 목록과 Section 목록을 순서대로 좁혀 Scope를 생성합니다."""

    normalized_id = subquestion_id.strip()
    if not normalized_id:
        raise ValueError("subquestion_id는 비어 있을 수 없습니다.")
    subquestion = _find_subquestion(state, normalized_id)
    hints = search_scope_knowledge(subquestion)

    corp_names = _scope_issuer_names(subquestion)
    if not corp_names:
        return Scope.from_scope_draft(
            ScopeDraft(
                level="GLOBAL",
                reason="공시 발행회사로 확인할 ISSUER entity가 없습니다.",
            ),
            subquestion_id=normalized_id,
        )

    doc_groups = _scope_doc_groups(hints)
    disclosure_items = _execute_scope_cypher(
        _disclosure_scope_query(corp_names, doc_groups),
        stage="disclosures",
    )
    disclosure_candidates = _scope_candidate_records(disclosure_items)
    available_disclosure_ids = _collect_named_strings(
        disclosure_candidates,
        "disclosure_id",
    )
    if not available_disclosure_ids:
        return Scope.from_scope_draft(
            ScopeDraft(
                level="GLOBAL",
                reason="해당 기업과 doc_group에 일치하는 최신 공시를 찾지 못했습니다.",
            ),
            subquestion_id=normalized_id,
        )

    disclosure_llm = (
        _get_disclosure_selection_llm()
        if llm is None
        else bind_structured_output(llm, DisclosureSelection)
    )
    disclosure_selection = _invoke_scope_selection(
        disclosure_llm,
        output_model=DisclosureSelection,
        system_prompt=sp.NARROW_SCOPE_DISCLOSURE_SELECTION_SYSTEM_PROMPT,
        human_message=_narrow_scope_human_message(
            stage="DISCLOSURE_SELECTION",
            subquestion=subquestion,
            hints=hints,
            candidates=disclosure_candidates,
            doc_groups=doc_groups,
        ),
        selected_field="selected_disclosure_ids",
        available_ids=available_disclosure_ids,
    )
    selected_disclosure_ids = disclosure_selection.selected_disclosure_ids
    selected_disclosure_candidates = [
        candidate
        for candidate in disclosure_candidates
        if candidate.get("disclosure_id") in selected_disclosure_ids
    ]
    company_candidates = selected_disclosure_candidates or disclosure_candidates
    corp_names_found = sorted(_collect_named_strings(company_candidates, "corp_name"))
    corp_codes_found = sorted(_collect_named_strings(company_candidates, "corp_code"))
    if not selected_disclosure_ids:
        level = "COMPANY" if corp_names_found else "GLOBAL"
        return Scope.from_scope_draft(
            ScopeDraft(
                level=level,
                corp_names=corp_names_found if level == "COMPANY" else [],
                corp_codes=corp_codes_found if level == "COMPANY" else [],
                reason=disclosure_selection.reason,
            ),
            subquestion_id=normalized_id,
        )

    section_items = _execute_scope_cypher(
        _section_scope_query(selected_disclosure_ids),
        stage="sections",
    )
    section_candidates = _scope_candidate_records(section_items)
    available_section_ids = _collect_named_strings(section_candidates, "section_id")
    if not available_section_ids:
        scope_draft = ScopeDraft(
            level="DISCLOSURE",
            corp_names=corp_names_found,
            corp_codes=corp_codes_found,
            disclosure_ids=selected_disclosure_ids,
            reason="관련 공시는 확인했지만 하위 Section을 찾지 못했습니다.",
        )
        _validate_scope_selection(scope_draft, disclosure_items)
        return Scope.from_scope_draft(scope_draft, subquestion_id=normalized_id)

    section_llm = (
        _get_section_selection_llm()
        if llm is None
        else bind_structured_output(llm, SectionSelection)
    )
    section_selection = _invoke_scope_selection(
        section_llm,
        output_model=SectionSelection,
        system_prompt=sp.NARROW_SCOPE_SECTION_SELECTION_SYSTEM_PROMPT,
        human_message=_narrow_scope_human_message(
            stage="SECTION_SELECTION",
            subquestion=subquestion,
            hints=hints,
            candidates=section_candidates,
            selected_disclosure_ids=selected_disclosure_ids,
        ),
        selected_field="selected_section_ids",
        available_ids=available_section_ids,
    )
    selected_section_ids = section_selection.selected_section_ids
    if not selected_section_ids:
        scope_draft = ScopeDraft(
            level="DISCLOSURE",
            corp_names=corp_names_found,
            corp_codes=corp_codes_found,
            disclosure_ids=selected_disclosure_ids,
            reason=section_selection.reason,
        )
        _validate_scope_selection(scope_draft, disclosure_items)
        return Scope.from_scope_draft(scope_draft, subquestion_id=normalized_id)

    disclosure_section_pairs = _collect_disclosure_section_pairs(section_candidates)
    parent_disclosure_ids = sorted({
        disclosure_id
        for disclosure_id, section_id in disclosure_section_pairs
        if section_id in selected_section_ids
    })
    scope_draft = ScopeDraft(
        level="SECTION",
        corp_names=corp_names_found,
        corp_codes=corp_codes_found,
        disclosure_ids=parent_disclosure_ids,
        section_ids=selected_section_ids,
        reason=section_selection.reason,
    )
    _validate_scope_selection(scope_draft, disclosure_items + section_items)
    return Scope.from_scope_draft(scope_draft, subquestion_id=normalized_id)


def _resolve_plan_scope(plan: Plan, state: AgentState) -> Scope:
    """Plan.scope_id를 state의 계층형 Scope 객체로 해석합니다."""

    scopes = [
        scope if isinstance(scope, Scope) else Scope.model_validate(scope)
        for scope in state.get("scope_candidates", [])
    ]
    scopes_by_id = {scope.scope_id: scope for scope in scopes}
    scope = scopes_by_id.get(plan.scope_id)
    if scope is None:
        raise ValueError(f"Plan Scope를 찾지 못했습니다: {plan.scope_id}")
    return scope


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
    scope: Scope | None = None,
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
                    exclude={"dependencies", "scope_id"},
                ),
                "scope": scope.model_dump(mode="json") if scope else None,
                "previous_results": [
                    result.model_dump(mode="json")
                    for result in dependencies
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

def _execute_retrieval_plan(
    plan: Plan,
    *,
    state: AgentState,
    dependencies: list[RetrievalResult],
    limit: int,
) -> tuple[RetrievalResult, list[str], list[str]]:
    """Plan의 source에 맞는 검색을 실행하고 정규화된 결과를 반환합니다."""

    scope = _resolve_plan_scope(plan, state)
    if plan.source == "neo4j":
        neo4j_cypher = cypher_builder(
            plan,
            user_question=state["question_text"],
            dependencies=dependencies,
            scope=scope,
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
        scope=scope,
    )
    qdrant_query.limit = limit
    print("-- Qdrant에서 Query retrieval을 실행합니다. --\n\n")
    try:
        raw_result = query_executor(qdrant_query, scope=scope)
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
    """검색 실행 예외를 RetrievalResult status로 분류합니다."""

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
    """검색 실패 정보를 빈 RetrievalResult로 변환합니다."""

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
    scope: Scope | None = None,
    neo4j_schema: str | None = None,
    llm: Any | None = None,
) -> CypherQuery:
    """사용자 질문, Plan, dependency 결과로 read-only Cypher를 생성한다."""

    cypher_llm = (
        _get_cypher_llm()
        if llm is None
        else bind_structured_output(llm, CypherQueryToolArgs)
    )
    schema = neo4j_schema if neo4j_schema is not None else _load_neo4j_schema()
    human_message = _build_query_builder_human_message(
        plan,
        user_question,
        dependencies,
        scope,
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
            result = invoke_with_rate_limit_retry(cypher_llm, messages)
            tool_args = (
                result
                if isinstance(result, CypherQueryToolArgs)
                else CypherQueryToolArgs.model_validate(result)
            )
            query = tool_args.to_cypher_query()
            _validate_latest_disclosure_filter(query)
            _validate_cypher_relationships(query)
            _validate_cypher_scope_filter(query, scope)
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


def _validate_cypher_scope_filter(
    query: CypherQuery,
    scope: Scope | None,
) -> None:
    """Scope가 있으면 Cypher가 가장 구체적인 ID 목록으로 제한되는지 검증합니다."""

    if scope is None:
        return
    if scope.level == "GLOBAL":
        return
    if scope.level == "COMPANY":
        label = "Company"
        property_names = ("corp_name", "corp_code")
        expected_by_property = {
            "corp_name": scope.corp_names,
            "corp_code": scope.corp_codes,
        }
    elif scope.level == "DISCLOSURE":
        label = "Disclosure"
        property_names = ("id",)
        expected_by_property = {"id": scope.disclosure_ids}
    else:
        label = "Section"
        property_names = ("id",)
        expected_by_property = {"id": scope.section_ids}
    aliases = re.findall(
        rf"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*{label}\b",
        query.cypher,
        flags=re.IGNORECASE,
    )
    for alias in aliases:
        for property_name in property_names:
            expected_values = expected_by_property[property_name]
            if not expected_values:
                continue
            patterns = (
                rf"\b{re.escape(alias)}\s*\.\s*{property_name}\s+IN\s+"
                rf"\$([A-Za-z_][A-Za-z0-9_]*)",
                rf"\b{re.escape(alias)}\s*\.\s*{property_name}\s*=\s*"
                rf"\$([A-Za-z_][A-Za-z0-9_]*)",
                rf"\(\s*{re.escape(alias)}\s*:\s*{label}\b[^)]*"
                rf"\b{property_name}\s*:\s*\$([A-Za-z_][A-Za-z0-9_]*)",
            )
            for pattern in patterns:
                for match in re.finditer(pattern, query.cypher, re.IGNORECASE):
                    parameter = query.parameters.get(match.group(1))
                    if parameter == expected_values or parameter in expected_values:
                        return
    raise ValueError(
        f"Scope가 지정된 Cypher는 {label}을 선택된 Scope 값으로 제한해야 합니다: "
        f"{scope.scope_id}"
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
    scope: Scope | None = None,
    qdrant_schema: str | None = None,
    llm: Any | None = None,
) -> QdrantQuery:
    """사용자 질문, Plan, dependency 결과로 Qdrant 요청을 생성한다."""

    query_llm = (
        _get_query_llm()
        if llm is None
        else bind_structured_output(llm, QdrantQueryToolArgs)
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
        scope,
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
            result = invoke_with_rate_limit_retry(query_llm, messages)
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
    """Read-only Cypher를 실행하고 Neo4j 응답을 RetrievalResult로 변환합니다."""

    
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
    

def query_executor(
    qdrant_query: QdrantQuery,
    *,
    scope: Scope | None = None,
) -> Any:
    """Qdrant query를 실행하며 선택된 Scope를 application filter로 적용합니다."""


    must = [models.FieldCondition(
        key="is_latest_version",
        match=models.MatchValue(value=True),
    )]
    if scope is not None and scope.level != "GLOBAL":
        if scope.level == "COMPANY":
            scope_key = "corp_name"
            scope_ids = scope.corp_names
        elif scope.level == "DISCLOSURE":
            scope_key = "disclosure_id"
            scope_ids = scope.disclosure_ids
        else:
            scope_key = "section_id"
            scope_ids = scope.section_ids
        must.append(models.FieldCondition(
            key=scope_key,
            match=models.MatchAny(any=scope_ids),
        ))
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
    """Retriever가 선택한 종료 상태와 result ID 목록을 검증합니다."""

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

    non_success = [
        result_id
        for result_id in selected_result_ids
        if results_by_id[result_id].status != "SUCCESS"
    ]
    if non_success:
        raise ValueError(
            f"SUCCESS 상태가 아닌 RetrievalResult는 선택할 수 없습니다: {non_success}"
        )


# retriever llm에 현재 state를 전달하기 위해 HumanMessage를 생성하는 함수
def build_retriever_human_message(state: AgentState) -> HumanMessage:
    """현재 retrieval state를 Retriever용 HumanMessage로 직렬화합니다."""

    question_analysis_value = state.get("question_analysis")
    question_analysis = (
        QuestionAnalysis.model_validate(question_analysis_value)
        if question_analysis_value is not None
        else None
    )
    payload = {
        "user_question": state["question_text"],
        "sub_questions": (
            [SubQuestion.model_validate(sub_question).model_dump(mode="json") for sub_question in question_analysis.sub_questions]
            if question_analysis is not None
            else None
        ),
        "scope_candidates": [
            (
                scope if isinstance(scope, Scope) else Scope.model_validate(scope)
            ).model_dump(mode="json")
            for scope in state.get("scope_candidates", [])
        ],
        "retrieval_results": [
            result.model_dump(mode="json")
            for result in state.get("retrieval_results", [])
        ]
    }
    return HumanMessage(content=json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    ))

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
    """중첩 구조에서 지정한 이름을 가진 첫 번째 값을 찾습니다."""

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
    """Retrieval item의 문서 계층 정보를 자연어 context로 조합합니다."""

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
    """Retrieval item 하나를 Answer Generator 입력 형식으로 축약합니다."""

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
    """JSONL 파일을 지정 key 기준의 lookup dictionary로 읽습니다."""

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

    # tools가 utils를 import하므로 실행 시점에 불러와 순환 import를 피합니다.
    from .tools import (
        calculate_table_statistic,
        combine_numeric_results,
        finish,
        retrieve_search,
    )

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
        _resolve_plan_scope(executable_plan, state)
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
    if name == "finish":
        validated = finish.tool_call_schema.model_validate(args)
        _validate_finish_selection(
            validated.status,
            validated.selected_result_ids,
            state,
        )
        return
    raise ValueError(f"지원하지 않는 tool call입니다: {name}")


def execute_tool_call(state: AgentState, tool_call: dict) -> dict:
    """검증된 Retriever tool call을 실제 LangChain tool에 전달합니다."""

    # tools가 utils를 import하므로 실행 시점에 불러와 순환 import를 피합니다.
    from .tools import (
        calculate_table_statistic,
        combine_numeric_results,
        finish,
        retrieve_search,
    )

    name = tool_call["name"]
    args = tool_call["args"]

    if name == "retrieve_search":
        return retrieve_search.invoke({**args, "state": state})
    if name == "calculate_table_statistic":
        return calculate_table_statistic.invoke({**args, "state": state})
    if name == "combine_numeric_results":
        return combine_numeric_results.invoke({**args, "state": state})
    if name == "finish":
        return finish.invoke({**args, "state": state})

    raise ValueError(f"지원하지 않는 tool call입니다: {name}")


# section_id 를 이용해 element_path를 찾아 해당 공시 원문을 가져오는 함수
def get_section_element(section_id: str):
    """section ID로 공시 원문 element를 조회하기 위한 확장 지점입니다."""

    pass


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
        """Table target의 result ID를 정규화하고 빈 값을 거부합니다."""

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
        """행 선택에 사용할 column 이름을 정규화합니다."""

        label_column = value.strip()
        if not label_column:
            raise ValueError("label_column은 비어 있을 수 없습니다.")
        return label_column

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, value: list[str]) -> list[str]:
        """행 label을 정규화하고 빈 값과 중복을 거부합니다."""

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
    """계산 실패 사유를 INVALID_INPUT RetrievalResult로 변환합니다."""

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
        """Numeric result target의 result ID를 정규화합니다."""

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
