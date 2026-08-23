from typing import Annotated, Any, Literal
from typing_extensions import NotRequired, Required, TypedDict
from pydantic import BaseModel, Field, model_validator

RetrievalSource = Literal["qdrant", "neo4j"]
QuestionDecision = Literal["retrieve", "direct", "clarify"]


class QueryPlan(BaseModel):
    """
    QueryPlan은 사용자의 질문을 분석하여 생성된 검색 계획을 나타낸다.
    각 QueryPlan은 특정 검색 소스에서 수행될 검색 쿼리와 관련된 정보를 포함한다.
    """
    plan_id: str
    source: RetrievalSource
    query: str
    purpose: str
    filters: dict[str, Any] = Field(default_factory=dict)


class QuestionAnalysis(BaseModel):
    """질문의 처리 경로와 필요한 retrieval 계획을 나타낸다."""

    decision: QuestionDecision
    normalized_question: str = Field(..., min_length=1)
    decision_reason: str = Field(
        ...,
        min_length=1,
        description="Routing 결정을 설명하는 짧은 근거",
    )
    query_plans: list[QueryPlan] = Field(default_factory=list)
    clarification_question: str | None = None

    @model_validator(mode="after")
    def validate_decision_payload(self) -> "QuestionAnalysis":
        if self.decision == "retrieve" and not self.query_plans:
            raise ValueError("retrieve 결정에는 query plan이 필요합니다.")
        if self.decision != "retrieve" and self.query_plans:
            raise ValueError(
                "direct 또는 clarify 결정에는 query plan이 없어야 합니다."
            )
        if self.decision == "clarify" and not self.clarification_question:
            raise ValueError(
                "clarify 결정에는 clarification_question이 필요합니다."
            )
        if self.decision != "clarify" and self.clarification_question is not None:
            raise ValueError(
                "clarify가 아닌 결정에는 clarification_question이 없어야 합니다."
            )
        return self


class RetrievalHit(BaseModel):
    """
    RetrievalHit은 검색 결과로 반환된 단일 항목을 나타낸다.
    """
    hit_id: str
    plan_id: str
    source: RetrievalSource

    evidence_id: str | None = None
    section_id: str | None = None
    table_id: str | None = None
    record_index: int | None = None

    score: float | None = None
    snippet: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def merge_hits(
    left: list[RetrievalHit],
    right: list[RetrievalHit],
) -> list[RetrievalHit]:
    """
    두 개의 RetrievalHit 리스트를 병합한다. 동일한 hit_id를 가진 항목이 존재할 경우,
    score가 더 높은 항목을 우선적으로 선택한다.
    """
    merged = {hit.hit_id: hit for hit in left}

    for hit in right:
        previous = merged.get(hit.hit_id)
        if previous is None or (hit.score or 0) > (previous.score or 0):
            merged[hit.hit_id] = hit

    return sorted(merged.values(), key=lambda hit: hit.hit_id)



class Citation(BaseModel):
    """
    Citation은 검색 결과에서 특정 정보를 참조할 때 사용되는 인용 정보를 나타낸다.
    """
    evidence_id: str
    section_id: str
    disclosure_id: str


class AgentState(TypedDict, total=False):

    # Input
    question_id: Required[str]
    question_text: Required[str]

    # Question analysis and planning
    question_analysis: NotRequired[QuestionAnalysis]

    # Parallel retrieval accumulation
    retrieval_hits: NotRequired[
        Annotated[list[RetrievalHit], merge_hits]
    ]

    # Reranking / evidence selection
    selected_hits: NotRequired[list[RetrievalHit]]

    # Output
    answer: NotRequired[str]
    citations: NotRequired[list[Citation]]

    # Observability / recovery
    errors: NotRequired[list[str]]
    retry_counts: NotRequired[dict[str, int]]
