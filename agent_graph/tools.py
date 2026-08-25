import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from vector_db.text2vector import text_to_vector

from dotenv import load_dotenv
import os

from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models

from . import system_prompts as sp
from .llm import get_llm
from .retrieval_result_parser import parse_neo4j_response, parse_qdrant_response
from .state import (
    AgentState,
    AiAnswer,
    AnswerGeneratorOutput,
    Citation,
    EvidenceSelection,
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
qdrant_vector_name = "evidence_dense"

class CypherQuery(BaseModel):
    """실행 가능한 read-only Cypher와 parameter를 분리한 요청입니다."""

    cypher: str = Field(..., min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


QdrantPointKind = Literal["TEXT", "KV_TABLE", "R_TABLE"]
QdrantFilterField = Literal[
    "retrieval_metadata.evidence_id",
    "retrieval_metadata.corp_name",
    "retrieval_metadata.corp_code",
    "retrieval_metadata.industry",
    "retrieval_metadata.sector",
    "retrieval_metadata.base_year",
    "retrieval_metadata.base_month",
    "retrieval_metadata.table_id",
]
QDRANT_STRING_FILTER_FIELDS = {
    "retrieval_metadata.evidence_id",
    "retrieval_metadata.corp_name",
    "retrieval_metadata.corp_code",
    "retrieval_metadata.industry",
    "retrieval_metadata.sector",
    "retrieval_metadata.table_id",
}
QDRANT_INTEGER_FILTER_FIELDS = {
    "retrieval_metadata.base_year",
    "retrieval_metadata.base_month",
}


class QdrantFilter(BaseModel):
    """하나의 indexed Qdrant payload 조건입니다."""

    key: QdrantFilterField = Field(
        description="허용 목록에 포함된 indexed retrieval_metadata 경로",
    )
    match: str | int | None = Field(
        default=None,
        description="정확 일치시킬 string 또는 integer scalar 값. list는 허용하지 않음",
    )
    gt: int | None = Field(default=None, description="초과 조건. 연도 또는 월에만 사용")
    gte: int | None = Field(default=None, description="이상 조건. 연도 또는 월에만 사용")
    lt: int | None = Field(default=None, description="미만 조건. 연도 또는 월에만 사용")
    lte: int | None = Field(default=None, description="이하 조건. 연도 또는 월에만 사용")

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
            raise ValueError("range filter는 base_year와 base_month에만 사용할 수 있습니다.")
        if self.gt is not None and self.gte is not None:
            raise ValueError("range filter에는 gt와 gte를 함께 사용할 수 없습니다.")
        if self.lt is not None and self.lte is not None:
            raise ValueError("range filter에는 lt와 lte를 함께 사용할 수 없습니다.")

        if self.key == "retrieval_metadata.base_month":
            values = [
                value
                for value in (self.match, *range_values.values())
                if value is not None
            ]
            if any(not isinstance(value, int) or not 1 <= value <= 12 for value in values):
                raise ValueError("base_month 값은 1 이상 12 이하여야 합니다.")

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
    query_vector: SkipJsonSchema[list[float] | None] = None
    point_kinds: list[QdrantPointKind] = Field(
        default_factory=lambda: ["TEXT", "KV_TABLE", "R_TABLE"],
        min_length=1,
        description=(
            "검색할 point 종류. point_kind는 filter가 아니라 반드시 이 필드에 지정"
        ),
    )
    filters: list[QdrantFilter] = Field(
        default_factory=list,
        description="Plan에 명시된 조건으로 만든 indexed payload filter 목록",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="반환할 최대 검색 결과 수",
    )
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
        if len(self.point_kinds) != len(set(self.point_kinds)):
            raise ValueError("point_kinds에는 중복 값을 사용할 수 없습니다.")

        filter_keys = [item.key for item in self.filters]
        if len(filter_keys) != len(set(filter_keys)):
            raise ValueError("동일한 key의 filter를 여러 번 사용할 수 없습니다.")
        if "retrieval_metadata.table_id" in filter_keys and self.point_kinds != ["R_TABLE"]:
            raise ValueError("table_id filter에는 point_kinds=['R_TABLE']을 사용해야 합니다.")

        self.query_text = query_text
        return self


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
        CypherQuery,
        method="function_calling",
    )


@lru_cache(maxsize=1)
def _get_query_llm() -> Any:
    """공용 LLM에 Qdrant query structured output을 한 번 binding합니다."""

    return get_llm().with_structured_output(
        QdrantQuery,
        method="function_calling",
    )

@tool
def retrieve_search(
    plan_id: str,
    state: Annotated[AgentState, InjectedState],
) -> dict:
    """지정한 plan을 실행하고 retrieval 결과가 포함된 state update를 반환합니다.

    args:
        plan_id(str): 실행할 Plan의 ID

    return:
        dict: 실행 완료된 plan을 제거하고 검색 결과를 추가한 state update
    """

    plans = state["plans"]
    idx = next(
        (i for i, plan in enumerate(plans) if plan.plan_id == plan_id),
        None,
    )
    if idx is None:
        raise ValueError(f"plan id가 '{plan_id}'인 Plan 객체를 찾지 못했습니다.")

    plan = plans[idx]

    if (plan.source == "neo4j"):
        # Neo4j Retrieval
        neo4j_cypher = cypher_builder(plan)
        print("-- Neo4j에서 다음 Cypher를 실행합니다.: --", neo4j_cypher, "\n\n")
        retrieval_result = cypher_executor(neo4j_cypher, plan.plan_id)

    elif (plan.source == "qdrant"):
        # Qdrant Retrieval
        qdrant_query = query_builder(plan)
        print("-- Qdrant에서 다음 Query를 실행합니다.: --", qdrant_query, "\n\n")
        retrieval_result = query_executor(qdrant_query, plan.plan_id)

    else:
        raise ValueError("retrieval source가 neo4j 또는 qdrant가 아닙니다.")

    retrieval_result = retrieval_result.model_copy(update={
        "metadata": {
            **retrieval_result.metadata,
            "plan_purpose": plan.purpose
        }
    })

    return {
        "plans": plans[:idx] + plans[idx + 1:],
        "retrieval_results": [retrieval_result],
    }

def cypher_builder(
    plan: Plan,
    *,
    neo4j_schema: str | None = None,
    llm: Any | None = None,
) -> CypherQuery:
    """입력받은 Plan으로부터 read-only Neo4j Cypher 요청을 생성한다."""

    cypher_llm = (
        _get_cypher_llm()
        if llm is None
        else llm.with_structured_output(
            CypherQuery,
            method="function_calling",
        )
    )
    schema = neo4j_schema if neo4j_schema is not None else _load_neo4j_schema()
    result = cypher_llm.invoke(
        [
            SystemMessage(content = sp.CYPHER_SYSTEM_PROMPT.format(neo4j_schema=schema)),
            HumanMessage(content = plan.model_dump_json()),
        ]
    )

    return (
        result
        if isinstance(result, CypherQuery)
        else CypherQuery.model_validate(result)
    )

def query_builder(
    plan: Plan,
    *,
    qdrant_schema: str | None = None,
    llm: Any | None = None,
) -> QdrantQuery:
    """입력받은 Plan으로부터 Qdrant retrieval 요청을 생성한다."""

    query_llm = (
        _get_query_llm()
        if llm is None
        else llm.with_structured_output(
            QdrantQuery,
            method="function_calling",
        )
    )
    query_schema = (
        qdrant_schema
        if qdrant_schema is not None
        else _load_qdrant_query_schema()
    )
    result = query_llm.invoke(
        [
            SystemMessage(
                content=sp.QDRANT_QUERY_SYSTEM_PROMPT.format(
                    qdrant_query_schema=query_schema,
                )
            ),
            HumanMessage(content = plan.model_dump_json())
        ]
    )

    query = (
        result
        if isinstance(result, QdrantQuery)
        else QdrantQuery.model_validate(result)
    )
    if query.mode == "vector":
        query.query_vector = text_to_vector(query.query_text)
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
    

def query_executor(qdrant_query: QdrantQuery, plan_id: str) -> RetrievalResult:

    must = [
        models.FieldCondition(
            key="retrieval_metadata.point_kind",
            match=models.MatchAny(any=qdrant_query.point_kinds)
        )
    ]
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
    query_filter = models.Filter(must=must)

    try:
        if qdrant_query.mode == "vector":
            result = qdrant_client.query_points(
                collection_name=qdrant_collection_name,
                query=qdrant_query.query_vector,
                using=qdrant_vector_name,
                query_filter=query_filter,
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

    include_r_table_records = any(
        condition.key == "retrieval_metadata.table_id"
        for condition in qdrant_query.filters
    )
    return parse_qdrant_response(
        result,
        plan_id=plan_id,
        query=qdrant_query.model_dump_json(exclude={"query_vector"}),
        r_table_detail="records" if include_r_table_records else "summary",
        metadata={"mode": qdrant_query.mode}
    )

@tool
def create_plan(
    new_plan: PlanDraft,
    position: str,
    state: Annotated[AgentState, InjectedState]
) -> dict:
    """추가적인 Plan 생성이 필요할 때 사용합니다. 새로운 Plan이 추가된 state update를 반환합니다.

    args:
        new_plan(PlanDraft): 새롭게 추가할 Plan의 내용입니다. plan_id는 tool에서 자동으로 생성하고 관리하므로 절대 임의로 생성하지 마세요.
        position(str): 새로운 Plan을 추가할 위치입니다. Plan은 기본적으로 앞에 위치할수록 우선순위가 높습니다. 맨 앞에 추가하고 싶다면 이 값을 "HEAD"로 지정하세요. plans에 포함된 특정 Plan의 바로 뒤에 두고 싶다면 이 값을 그 Plan의 plan_id로 지정하세요.

    return:
        dict: plans에 new_plan이 추가된 state update
    """

    plans = list(state.get("plans", []))

    next_plan_seq = state.get("next_plan_seq", 100)
    plan = Plan.from_plan_draft(new_plan, next_plan_seq)
    next_plan_seq += 1

    position = position.strip()
    if position == "HEAD":
        insert_at = 0
    else:
        anchor = next(
            (
                index
                for index, existing_plan in enumerate(plans)
                if existing_plan.plan_id == position
            ),
            None,
        )
        if anchor is None:
            raise ValueError(f"position plan을 찾지 못했습니다: {position}")
        insert_at = anchor + 1

    plans.insert(insert_at, plan)
    return { "plans": plans, "next_plan_seq": next_plan_seq }

@tool
def modify_plan(
    plan_id: str,
    modified_plan: PlanDraft,
    state: Annotated[AgentState, InjectedState]
) -> dict:
    """아직 실행되지 않은 plan의 내용을 수정합니다.

    args:
        plan_id(str): 수정할 Plan의 plan_id입니다.
        modified_plan(PlanDraft): 변경할 source, query, purpose입니다. plan_id는 기존 값을 유지합니다.

    return:
        dict: 지정한 Plan이 수정된 plans state update
    """

    plans = list(state.get("plans", []))
    anchor = next(
        (
            index
            for index, existing_plan in enumerate(plans)
            if existing_plan.plan_id == plan_id
        ),
        None,
    )
    if anchor is None:
        raise ValueError(f"plan을 찾지 못했습니다: {plan_id}")

    plans[anchor] = Plan(
        plan_id=plan_id,
        **modified_plan.model_dump()
    )
    return { "plans": plans }

@tool
def delete_plan(
    plan_id: str,
    state: Annotated[AgentState, InjectedState]
) -> dict:
    """아직 실행되지 않은 plan들 중 더 이상 필요 없다고 판단되는 plan을 제거합니다.

    args:
        plan_id(str): 제거할 plan의 plan_id 입니다.

    return:
        dict: 해당 plan 제거가 반영된 state update
    """

    plans = state.get("plans", [])
    anchor = next(
        (
            index
            for index, existing_plan in enumerate(plans)
            if existing_plan.plan_id == plan_id
        ),
        None,
    )
    if anchor is None:
        pass
    else:
        plans.pop(anchor)
        print(f"[{plan_id}] 플랜을 제거했습니다.")

    return { "plans": plans }

FinishStatus = Literal["COMPLETE", "INSUFFICIENT"]
@tool
def finish(
    status: FinishStatus,
    reason: str,
    selected_evidence: list[EvidenceSelection],
    state: Annotated[AgentState, InjectedState]
) -> dict:
    """남아 있는 plan과 관계 없이 더 이상의 retrieval을 멈추고 답변을 생성합니다. 종료 원인은 다음 두 가지 중 하나입니다.
        - COMPLETE: 지금까지의 retrieval을 통해 사용자의 질문에 답변하기 위해 필요한 충분한 정보를 얻었음.
        - INSUFFICIENT: 충분한 retrieval을 수행했으나, 사용자의 질문에 답변하기 위한 신뢰도 있는 정보를 얻지 못함. 더 이상의 retrieval은 무의미하다 판단.

    args:
        status(FinishStatus): 'COMPLETE' 또는 'INSUFFICIENT'
        reason(str): status를 그렇게 판단한 이유와 사고 과정을 포함한 간략한 한국어 문장
        selected_evidence(list[EvidenceSelection]): Answer Generator가 사용할 result와 item 범위

    return:
        dict: retrieval_status와 검증된 selected_evidence를 포함한 state update
    """

    if status == "COMPLETE" and not selected_evidence:
        raise ValueError("COMPLETE에는 최소 하나의 selected_evidence가 필요합니다.")

    results = {
        result.result_id: result
        for result in state.get("retrieval_results", [])
    }
    for selection in selected_evidence:
        result = results.get(selection.result_id)
        if result is None:
            raise ValueError(
                f"RetrievalResult를 찾지 못했습니다: {selection.result_id}"
            )
        if selection.item_indexes is None:
            continue
        invalid_indexes = [
            index
            for index in selection.item_indexes
            if index >= len(result.items)
        ]
        if invalid_indexes:
            raise ValueError(
                f"item index 범위를 벗어났습니다: {selection.result_id} "
                f"{invalid_indexes}"
            )

    return {
        "retrieval_status": status,
        "retrieval_finish_reason": reason,
        "selected_evidence": selected_evidence
    }
    

# retriever llm에 현재 state를 전달하기 위해 HumanMessage를 생성하는 함수
def build_retriever_human_message(state: AgentState) -> HumanMessage:
    payload = {
        "user_question": state["question_text"],
        "plans": [plan.model_dump(mode="json") for plan in state["plans"]],
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


def build_citation_candidates(state: AgentState) -> list[dict[str, Any]]:
    """선택된 retrieval item에서 LLM이 선택할 수 있는 Citation 후보를 만듭니다.

    입력 예시:
        selected_evidence가 disclosure_id, section_id, evidence_id를 포함한 item을 선택한 AgentState

    출력 예시:
        [{
            "reference_id": "C1",
            "source_item_reference_id": "R1-I1",
            "citation": {
                "disclosure_id": "d1",
                "section_id": "s1",
                "evidence_id": "e1"
            }
        }]
    """

    results = {
        result.result_id: result
        for result in state.get("retrieval_results", [])
    }
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for result_number, selection in enumerate(
        state.get("selected_evidence", []),
        start=1
    ):
        result = results[selection.result_id]
        item_indexes = (
            range(len(result.items))
            if selection.item_indexes is None
            else selection.item_indexes
        )
        for item_index in item_indexes:
            for citation in _extract_citations(result.items[item_index]):
                key = (
                    citation.disclosure_id,
                    citation.section_id,
                    citation.evidence_id,
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({
                    "reference_id": f"C{len(candidates) + 1}",
                    "source_item_reference_id": (
                        f"R{result_number}-I{item_index + 1}"
                    ),
                    "citation": citation.model_dump(
                        mode="json",
                        exclude_none=True
                    )
                })
    return candidates


def resolve_answer_draft(
    state: AgentState,
    draft: AnswerGeneratorOutput,
) -> AiAnswer:
    """AnswerGeneratorOutput의 reference ID를 검증된 Citation으로 변환합니다.

    입력 예시:
        AnswerGeneratorOutput(answer="답변", citation_reference_ids=["C1"])

    출력 예시:
        AiAnswer(answer="답변", citation=[Citation(disclosure_id="d1")])
    """

    citation_map = {
        candidate["reference_id"]: Citation.model_validate(candidate["citation"])
        for candidate in build_citation_candidates(state)
    }
    if not citation_map:
        return AiAnswer(answer=draft.answer, citation=[])

    unknown_ids = [
        reference_id
        for reference_id in draft.citation_reference_ids
        if reference_id not in citation_map
    ]
    if unknown_ids:
        raise ValueError(
            "허용되지 않은 citation reference입니다: "
            + ", ".join(unknown_ids)
        )

    selected_citations: list[Citation] = []
    seen_reference_ids: set[str] = set()
    for reference_id in draft.citation_reference_ids:
        if reference_id in seen_reference_ids:
            continue
        seen_reference_ids.add(reference_id)
        selected_citations.append(citation_map[reference_id])

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
    """질문과 선택된 retrieval evidence를 Answer Generator 입력으로 변환합니다.

    입력 예시:
        selected_evidence=[{
            "result_id": "retrieval:plan_1",
            "item_indexes": [0]
        }]

    출력 예시:
        HumanMessage(content='{"user_question": "...", "selected_retrieval_results": [...]}')
    """

    results = {
        result.result_id: result
        for result in state.get("retrieval_results", [])
    }
    selected_results = []
    for result_number, selection in enumerate(
        state.get("selected_evidence", []),
        start=1
    ):
        result = results[selection.result_id]
        item_indexes = (
            range(len(result.items))
            if selection.item_indexes is None
            else selection.item_indexes
        )
        result_payload = result.model_dump(mode="json", exclude={"items"})
        result_payload["selection_reason"] = selection.reason
        result_payload["items"] = [
            {
                **result.items[item_index],
                "item_reference_id": f"R{result_number}-I{item_index + 1}",
                "item_index": item_index
            }
            for item_index in item_indexes
        ]
        selected_results.append(result_payload)

    payload = {
        "user_question": state["question_text"],
        "normalized_question": state["question_analysis"].normalized_question,
        "decision": state["question_analysis"].decision,
        "retrieval_status": (state["retrieval_status"] if state["question_analysis"].decision == "retrieve" else None),
        "retrieval_finish_reason": (state["retrieval_finish_reason"] if state["question_analysis"].decision == "retrieve" else None),
        "selected_retrieval_results": selected_results,
        "citation_candidates": build_citation_candidates(state)
    }

    return HumanMessage(
        content=json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )

# 하나의 형식으로 모든 tool call을 처리하기 위한 interface 함수
def execute_tool_call(state: AgentState, tool_call: dict) -> dict:

    name = tool_call["name"]
    args = tool_call["args"]

    if name == "retrieve_search":
        return retrieve_search.invoke({**args, "state": state})
    if name == "create_plan":
        return create_plan.invoke({**args, "state": state})
    if name == "modify_plan":
        return modify_plan.invoke({**args, "state": state})
    if name == "delete_plan":
        return delete_plan.invoke({**args, "state": state})
    if name == "finish":
        return finish.invoke({**args, "state": state})

    raise ValueError(f"지원하지 않는 tool call입니다: {name}")
