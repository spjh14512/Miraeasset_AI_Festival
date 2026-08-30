from typing import Annotated, Any, Literal
from typing_extensions import NotRequired, Required, TypedDict
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

RetrievalSource = Literal["qdrant", "neo4j"]
RetrievalResultSource = Literal["qdrant", "neo4j", "derived"]
RetrievalResultStatus = Literal[
    "SUCCESS",
    "NO_RESULTS",
    "ERROR",
    "TIMEOUT",
    "DUPLICATES_ONLY",
    "INVALID_QUERY",
    "INVALID_INPUT",
]
QuestionDecision = Literal["retrieve", "direct", "clarify"]


class PlanDraft(BaseModel):
    """
    PlanDraft는 LLM이 생성하는 검색 계획의 내용입니다.
    plan_id는 application이 Plan으로 변환할 때 할당합니다.
    """

    source: RetrievalSource = Field(description="검색에 사용할 retrieval source")
    query: str = Field(..., min_length=1, description="검색할 자연어 정보 요구")
    purpose: str = Field(..., min_length=1, description="검색 결과가 필요한 이유")
    dependencies: list[str] = Field(..., description="이 Plan의 query 생성에 사용할 기존 RetrievalResult의 result_id 목록")

    @field_validator("dependencies")
    @classmethod
    def validate_dependencies(cls, value: list[str]) -> list[str]:
        dependencies = [result_id.strip() for result_id in value]
        if any(not result_id for result_id in dependencies):
            raise ValueError("dependencies에는 비어 있는 result_id를 사용할 수 없습니다.")
        if len(dependencies) != len(set(dependencies)):
            raise ValueError("dependencies에는 중복 result_id를 사용할 수 없습니다.")
        return dependencies


class Plan(PlanDraft):
    """application이 고유 ID를 할당한 실행 가능한 검색 계획입니다."""

    plan_id: SkipJsonSchema[str]

    @classmethod
    def from_plan_draft(cls, plan_draft: PlanDraft, seq: int):
        return cls(
            **plan_draft.model_dump(exclude={"plan_id"}),
            plan_id="plan_" + str(seq),
        )


class QuestionAnalysis(BaseModel):
    """질문의 처리 경로와 정규화 결과를 나타낸다."""

    decision: QuestionDecision
    normalized_question: str = Field(..., min_length=1)
    decision_reason: str = Field(
        ...,
        min_length=1,
        description="Routing 결정을 설명하는 짧은 근거",
    )
    clarification_question: str | None = None

    @model_validator(mode="after")
    def validate_decision_payload(self) -> "QuestionAnalysis":
        if self.decision == "clarify" and not self.clarification_question:
            raise ValueError(
                "clarify 결정에는 clarification_question이 필요합니다."
            )
        if self.decision != "clarify" and self.clarification_question is not None:
            raise ValueError(
                "clarify가 아닌 결정에는 clarification_question이 없어야 합니다."
            )
        return self


class PlannerOutput(BaseModel):
    """Planner가 한 번의 호출로 생성하는 질문 분석 결과입니다."""

    model_config = ConfigDict(extra="forbid")

    question_analysis: QuestionAnalysis


class RetrievalResult(BaseModel):
    """
    RetrievalResult는 하나의 retrieval 실행 결과를 나타낸다.
    """
    result_id: str
    plan_id: str
    source: RetrievalResultSource
    status: RetrievalResultStatus | None = None
    query: str
    items: list[dict[str, Any]] = Field(default_factory=list)
    result_count: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def set_default_status(self) -> "RetrievalResult":
        if self.status is None:
            self.status = "SUCCESS" if self.result_count > 0 else "NO_RESULTS"
        return self


def merge_results(
    left: list[RetrievalResult],
    right: list[RetrievalResult],
) -> list[RetrievalResult]:
    """
    두 개의 RetrievalResult 리스트를 병합한다. 동일한 result_id가 다시 들어오면
    나중에 실행된 결과로 교체한다.
    """
    merged = {result.result_id: result for result in left}

    for result in right:
        merged[result.result_id] = result

    return sorted(merged.values(), key=lambda result: result.result_id)



class Citation(BaseModel):
    """
    Citation은 검색 결과에서 특정 정보를 참조할 때 사용되는 인용 정보를 나타낸다.
    """
    disclosure_id: str
    section_id: str | None = None
    evidence_id: str | None = None

    @model_validator(mode="after")
    def validate_id_hierarchy(self) -> "Citation":
        if self.evidence_id is not None and self.section_id is None:
            raise ValueError(
                "evidence_id를 사용하려면 section_id가 필요합니다."
            )
        return self


class AiAnswer(BaseModel):
    """
    Ai의 답변과 답변에 사용한 인용 정보를 포함한 클래스이다.
    """
    answer: str
    citation: list[Citation]


class AnswerGeneratorOutput(BaseModel):
    """검색 결과가 충분하거나 부족한 모든 경우에 최종 답변을 제출하는 function입니다.

    근거가 없으면 확인할 수 없다는 답변과 빈 used_result_ids 목록을 반환합니다.
    """

    answer: str = Field(..., description="사용자에게 제공할 최종 답변 메시지")
    used_result_ids: list[str] = Field(
        ...,
        description=(
            "답변 생성에 실제로 사용한 retrieval_results의 result_id 목록"
        ),
        default_factory=list,
    )


RetrievalStatus = Literal["CONTINUE", "COMPLETE", "INSUFFICIENT"]

class AgentState(TypedDict, total=False):

    # Input
    question_id: Required[str]
    question_text: Required[str]

    # Question analysis and planning
    question_analysis: NotRequired[QuestionAnalysis]
    next_plan_seq: NotRequired[int]

    # Parallel retrieval accumulation
    retrieval_results: NotRequired[
        Annotated[list[RetrievalResult], merge_results]
    ]
    retrieved_qdrant_point_ids: NotRequired[list[str]]
    retrieval_status: NotRequired[RetrievalStatus]
    retrieval_finish_reason: NotRequired[str]
    selected_result_ids: NotRequired[list[str]]

    # Output
    ai_answer: AiAnswer

    # Observability / recovery
    errors: NotRequired[list[str]]
    retry_counts: NotRequired[dict[str, int]]
