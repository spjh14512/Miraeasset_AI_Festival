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
EntityRole = Literal[
    "ISSUER",
    "TARGET",
    "COUNTERPARTY",
    "SUBSIDIARY",
    "INVESTEE",
    "SHAREHOLDER",
    "OTHER",
]
EntityMatchStatus = Literal["MATCHED", "AMBIGUOUS", "NOT_FOUND", "UNKNOWN"]
EventConfidence = Literal["HIGH", "MEDIUM", "LOW"]
QuestionIntent = Literal[
    "DECISION",
    "PLAN",
    "EXECUTION",
    "RESULT",
    "STATUS",
    "CHANGE",
    "HISTORY",
    "TREND",
    "AMOUNT",
    "DETAIL",
    "EXISTENCE",
    "COMPARISON",
    "UNKNOWN",
]
PeriodKind = Literal[
    "FILING_DATE",
    "REPORTING_PERIOD",
    "EVENT_DATE",
    "AS_OF",
    "RELATIVE_DOCUMENT",
    "OTHER",
]
PeriodGranularity = Literal["DATE", "MONTH", "YEAR", "RANGE", "UNKNOWN"]
ScopeLevel = Literal["GLOBAL", "COMPANY", "DISCLOSURE", "SECTION"]


class PlanDraft(BaseModel):
    """
    PlanDraft는 LLM이 생성하는 검색 계획의 내용입니다.
    plan_id는 application이 Plan으로 변환할 때 할당합니다.
    """

    source: RetrievalSource = Field(description="검색에 사용할 retrieval source")
    query: str = Field(..., min_length=1, description="검색할 자연어 정보 요구")
    purpose: str = Field(..., min_length=1, description="검색 결과가 필요한 이유")
    dependencies: list[str] = Field(..., description="이 Plan의 query 생성에 사용할 기존 RetrievalResult의 result_id 목록")
    scope_id: str = Field(
        ...,
        min_length=1,
        description=(
            "application이 생성한 Scope의 scope_id. 범위를 좁힐 수 없는 경우에도 "
            "GLOBAL Scope의 ID를 사용"
        ),
    )

    @field_validator("dependencies")
    @classmethod
    def validate_dependencies(cls, value: list[str]) -> list[str]:
        dependencies = [result_id.strip() for result_id in value]
        if any(not result_id for result_id in dependencies):
            raise ValueError("dependencies에는 비어 있는 result_id를 사용할 수 없습니다.")
        if len(dependencies) != len(set(dependencies)):
            raise ValueError("dependencies에는 중복 result_id를 사용할 수 없습니다.")
        return dependencies

    @field_validator("scope_id")
    @classmethod
    def validate_scope_id(cls, value: str) -> str:
        scope_id = value.strip()
        if not scope_id:
            raise ValueError("scope_id는 비어 있을 수 없습니다.")
        return scope_id


class Plan(PlanDraft):
    """application이 고유 ID를 할당한 실행 가능한 검색 계획입니다."""

    plan_id: SkipJsonSchema[str]

    @classmethod
    def from_plan_draft(cls, plan_draft: PlanDraft, seq: int):
        return cls(
            **plan_draft.model_dump(exclude={"plan_id"}),
            plan_id="plan_" + str(seq),
        )


class EntityMention(BaseModel):
    """질문에 등장한 기업 등 검색 대상 entity를 나타냅니다."""

    mention: str = Field(..., min_length=1)
    roles: list[EntityRole] = Field(..., min_length=1)
    canonical_name: str | None = None
    match_status: EntityMatchStatus = "UNKNOWN"

    @field_validator("mention")
    @classmethod
    def normalize_mention(cls, value: str) -> str:
        return value.strip()

    @field_validator("roles")
    @classmethod
    def deduplicate_roles(cls, value: list[EntityRole]) -> list[EntityRole]:
        return list(dict.fromkeys(value))


class EventAnalysis(BaseModel):
    """질문이 가리키는 사건 유형 후보와 분석 확신도를 나타냅니다."""

    event_type: str | None = None
    candidate_event_types: list[str] = Field(default_factory=list)
    confidence: EventConfidence = "LOW"

    @field_validator("candidate_event_types")
    @classmethod
    def normalize_candidates(cls, value: list[str]) -> list[str]:
        candidates = [candidate.strip() for candidate in value if candidate.strip()]
        return list(dict.fromkeys(candidates))


class PeriodAnalysis(BaseModel):
    """질문에 명시되거나 암시된 기간 표현과 의미를 나타냅니다."""

    expression: str = Field(..., min_length=1)
    kind: PeriodKind
    normalized_value: str | None = None
    granularity: PeriodGranularity = "UNKNOWN"

    @field_validator("expression")
    @classmethod
    def normalize_expression(cls, value: str) -> str:
        return value.strip()


class SubQuestion(BaseModel):
    """독립적으로 검색 가능한 하나의 정보 요구를 나타냅니다."""

    subquestion_id: SkipJsonSchema[str | None] = None
    question: str = Field(..., min_length=1)
    entities: list[EntityMention] = Field(default_factory=list)
    events: list[EventAnalysis] = Field(default_factory=list)
    intents: list[QuestionIntent] = Field(..., min_length=1)
    periods: list[PeriodAnalysis] = Field(default_factory=list)
    requested_facts: list[str] = Field(..., min_length=1)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        return value.strip()

    @field_validator("intents", "requested_facts")
    @classmethod
    def deduplicate_non_empty_values(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        if not normalized:
            raise ValueError("하나 이상의 값이 필요합니다.")
        return list(dict.fromkeys(normalized))


class ScopeDraft(BaseModel):
    """narrow_scope 내부 LLM이 선택한 계층형 검색 범위입니다."""

    level: ScopeLevel
    corp_names: list[str] = Field(default_factory=list)
    corp_codes: list[str] = Field(default_factory=list)
    disclosure_ids: list[str] = Field(default_factory=list)
    section_ids: list[str] = Field(default_factory=list)
    reason: str = Field(..., min_length=1)

    @field_validator("reason")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Scope의 필수 문자열은 비어 있을 수 없습니다.")
        return normalized

    @field_validator("corp_names", "corp_codes", "disclosure_ids", "section_ids")
    @classmethod
    def normalize_ids(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("Scope 목록에는 빈 값을 사용할 수 없습니다.")
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def validate_level(self) -> "ScopeDraft":
        if self.level == "GLOBAL":
            if any((self.corp_names, self.corp_codes, self.disclosure_ids, self.section_ids)):
                raise ValueError("GLOBAL Scope에는 범위 식별자를 사용할 수 없습니다.")
        elif self.level == "COMPANY":
            if not self.corp_names:
                raise ValueError("COMPANY Scope에는 corp_names가 필요합니다.")
            if self.disclosure_ids or self.section_ids:
                raise ValueError("COMPANY Scope에는 공시·섹션 ID를 사용할 수 없습니다.")
        elif self.level == "DISCLOSURE":
            if not self.disclosure_ids:
                raise ValueError("DISCLOSURE Scope에는 disclosure_ids가 필요합니다.")
            if self.section_ids:
                raise ValueError("DISCLOSURE Scope에는 section_ids를 사용할 수 없습니다.")
        elif self.level == "SECTION":
            if not self.disclosure_ids or not self.section_ids:
                raise ValueError(
                    "SECTION Scope에는 disclosure_ids와 section_ids가 모두 필요합니다."
                )
        return self


class Scope(ScopeDraft):
    """Application이 ID와 원본 SubQuestion을 연결한 검색 범위 후보입니다."""

    scope_id: SkipJsonSchema[str] = Field(..., min_length=1)
    subquestion_id: str = Field(..., min_length=1)

    @field_validator("scope_id", "subquestion_id")
    @classmethod
    def normalize_identity(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Scope 식별자는 비어 있을 수 없습니다.")
        return normalized

    @classmethod
    def from_scope_draft(
        cls,
        scope_draft: ScopeDraft,
        *,
        subquestion_id: str,
    ) -> "Scope":
        normalized_subquestion_id = subquestion_id.strip()
        if not normalized_subquestion_id:
            raise ValueError("subquestion_id는 비어 있을 수 없습니다.")
        return cls(
            **scope_draft.model_dump(),
            scope_id=f"scope_{normalized_subquestion_id}",
            subquestion_id=normalized_subquestion_id,
        )


def merge_scopes(left: list[Scope], right: list[Scope]) -> list[Scope]:
    """scope_id가 같은 후보는 최신 값으로 교체하며 Scope 목록을 병합합니다."""

    validated_left = [
        scope if isinstance(scope, Scope) else Scope.model_validate(scope)
        for scope in left
    ]
    validated_right = [
        scope if isinstance(scope, Scope) else Scope.model_validate(scope)
        for scope in right
    ]
    merged = {scope.scope_id: scope for scope in validated_left}
    for scope in validated_right:
        merged[scope.scope_id] = scope
    return sorted(merged.values(), key=lambda scope: scope.scope_id)


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
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    synthesis_requirement: str | None = None

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


class QuestionAnalyzerOutput(BaseModel):
    """Question Analyzer가 한 번의 호출로 생성하는 분석 결과입니다."""

    model_config = ConfigDict(extra="forbid")

    question_analysis: QuestionAnalysis

    @model_validator(mode="after")
    def validate_analysis(self) -> "QuestionAnalyzerOutput":
        analysis = self.question_analysis
        if analysis.decision == "retrieve" and not analysis.sub_questions:
            raise ValueError("retrieve 결정에는 하나 이상의 sub_questions가 필요합니다.")
        if len(analysis.sub_questions) > 1 and not analysis.synthesis_requirement:
            raise ValueError(
                "여러 sub_questions를 생성한 경우 synthesis_requirement가 필요합니다."
            )
        return self


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


class AnswerValidatorOutput(BaseModel):
    """answer_generator 초안을 근거와 대조해 문제를 식별하는 검증 결과입니다.

    문제가 없으면 세 필드 모두 비어 있습니다. 판정만 담당하며 답변을
    직접 고치지 않습니다 — 수정은 application이 결정론적으로 수행합니다.
    """

    model_config = ConfigDict(extra="forbid")

    unsupported_claims: list[str] = Field(
        default_factory=list,
        description=(
            "제공된 근거로 뒷받침되지 않는, answer에 실제로 등장하는 문장 그대로. "
            "수치·비교·순위 표현이 근거와 다른 경우도 포함합니다."
        ),
    )
    missing_requested_facts: list[str] = Field(
        default_factory=list,
        description="sub_questions의 requested_facts 중 answer가 다루지 않은 항목",
    )
    incomplete_evidence_note: str | None = Field(
        default=None,
        description=(
            "70개사 코퍼스 안에서 답변 도출에 필수적인데 검색 근거에 "
            "포함되지 않은 데이터가 있으면 그 사유. 없으면 null."
        ),
    )


RetrievalStatus = Literal["CONTINUE", "COMPLETE", "INSUFFICIENT"]

class AgentState(TypedDict, total=False):

    # Input
    question_id: Required[str]
    question_text: Required[str]

    # Question analysis and planning
    question_analysis: NotRequired[QuestionAnalysis]
    next_plan_seq: NotRequired[int]
    scope_candidates: NotRequired[Annotated[list[Scope], merge_scopes]]

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
