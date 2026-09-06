"""Competition-style GET /answer simulation runner.

Examples:
    uv run python test_simulation.py --mode mock --transport both
    uv run python test_simulation.py --mode live --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from fastapi.testclient import TestClient

import main as server
from agent_graph.state import AiAnswer, Citation, QuestionAnalysis, RetrievalResult


REQUIRED_FIELDS = {
    "question_id",
    "question",
    "retrieved_context",
    "think_trace",
    "answer",
}
UNAVAILABLE_MARKERS = (
    "제공된 공시 데이터에서 확인되지 않는 내용입니다",
    "확인할 수 없습니다",
)


@dataclass(frozen=True)
class SimulationCase:
    question_id: str
    category: str
    question: str
    expect_unavailable: bool = False
    expect_unit: bool = False
    expect_history: bool = False


CASES = (
    SimulationCase("SIM-001", "단일 수치", "삼성전자의 2025년 매출액은?", expect_unit=True),
    SimulationCase("SIM-002", "단일 수치", "SK하이닉스의 2025년 영업이익은?", expect_unit=True),
    SimulationCase("SIM-003", "다중 비교", "삼성전자와 SK하이닉스의 설비투자를 비교해줘", expect_unit=True),
    SimulationCase("SIM-004", "다중 비교", "삼성전자와 LG전자의 부채비율을 비교해줘", expect_unit=True),
    SimulationCase("SIM-005", "이력 추적", "주요계약 체결 후 해지된 이력이 있는지 확인해줘", expect_history=True),
    SimulationCase("SIM-006", "이력 추적", "최초 공시와 최종 정정공시의 변경 내용을 알려줘", expect_history=True),
    SimulationCase("SIM-007", "요약/분석", "삼성전자의 자금조달 내역을 유형별로 정리해줘"),
    SimulationCase("SIM-008", "요약/분석", "SK하이닉스의 최근 설비투자 계획을 공시 기준으로 요약해줘"),
    SimulationCase("SIM-009", "요약/분석", "삼성전자의 최근 3개년 매출 CAGR 계산 과정을 보여줘", expect_unit=True),
    SimulationCase("SIM-010", "환각 방지", "삼성전자의 2027년 주가를 예측해줘", expect_unavailable=True),
    SimulationCase("SIM-011", "환각 방지", "코덱스테스트전자의 2025년 매출액을 알려줘", expect_unavailable=True),
    SimulationCase("SIM-012", "환각 방지", "삼성전자의 2030년 확정 매출액을 알려줘", expect_unavailable=True),
)


class _SimulationGraph:
    """External services 없이 API 계약과 품질 검사를 재현하는 graph double."""

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        case = next(item for item in CASES if item.question_id == state["question_id"])
        base = {
            **state,
            "question_analysis": QuestionAnalysis(
                decision="retrieve",
                normalized_question=case.question,
                decision_reason="공시 근거 검색이 필요합니다.",
            ),
            "answer_validation_status": "PASSED",
        }
        if case.expect_unavailable:
            return {
                **base,
                "retrieval_status": "INSUFFICIENT",
                "retrieval_finish_reason": "확인 가능한 공시 근거가 없습니다.",
                "selected_result_ids": [],
                "retrieval_results": [],
                "ai_answer": AiAnswer(
                    answer="제공된 공시 데이터에서 확인되지 않는 내용입니다.",
                    citation=[],
                ),
            }

        answer = "공시 근거에서 요청한 내용을 확인했습니다."
        if case.expect_unit:
            answer += " 수치는 300억원이며 단위는 억원입니다."
        if case.expect_history:
            answer += " 최초 공시와 정정·해지 이력을 함께 반영했습니다."
        evidence = {
            "type": "text",
            "content": f"시뮬레이션 공시 근거: {case.question}",
            "disclosure_id": "d20240306000686",
            "metadata": {
                "retrieval_context": {
                    "corp_name": "시뮬레이션기업",
                    "report_name": "사업보고서 (2025.12)",
                    "rcept_date": "20260318",
                },
            },
        }
        return {
            **base,
            "retrieval_status": "COMPLETE",
            "retrieval_finish_reason": "질문을 뒷받침하는 공시 근거를 확보했습니다.",
            "selected_result_ids": ["retrieval:simulation_1"],
            "retrieval_results": [RetrievalResult(
                result_id="retrieval:simulation_1",
                source="qdrant",
                status="SUCCESS",
                query=case.question,
                items=[evidence],
                result_count=1,
            )],
            "ai_answer": AiAnswer(
                answer=answer,
                citation=[Citation(disclosure_id="d20240306000686")],
            ),
        }


def _mock_citation_labels(
    citations: list[Citation],
    **_kwargs: Any,
) -> list[str]:
    if not citations:
        return []
    return ["[근거: 사업보고서 (2025.12), 2026-03-18]"]


def validate_response(case: SimulationCase, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(payload) != REQUIRED_FIELDS:
        errors.append(f"응답 field 불일치: {sorted(payload)}")
        return errors
    if not all(isinstance(payload[field], str) for field in REQUIRED_FIELDS):
        errors.append("모든 응답 field는 string이어야 합니다.")
    if payload["question_id"] != case.question_id or payload["question"] != case.question:
        errors.append("질문 ID 또는 원문이 보존되지 않았습니다.")

    try:
        context = json.loads(payload["retrieved_context"])
        if set(context) != {"citations", "results"}:
            errors.append("retrieved_context 내부 계약이 다릅니다.")
    except (json.JSONDecodeError, TypeError):
        errors.append("retrieved_context가 JSON string이 아닙니다.")
        context = {"citations": [], "results": []}

    try:
        trace = json.loads(payload["think_trace"])
        required_trace = {
            "query_text", "question_analysis", "retrieval_history",
            "calculation_history", "selected_evidence", "answer_validation",
            "final_basis",
        }
        if not required_trace.issubset(trace):
            errors.append("think_trace에 필수 실행 기록이 누락됐습니다.")
    except (json.JSONDecodeError, TypeError):
        errors.append("think_trace가 JSON string이 아닙니다.")

    answer = payload["answer"]
    if case.expect_unavailable:
        if not any(marker in answer for marker in UNAVAILABLE_MARKERS):
            errors.append("근거 부족 고지가 없습니다.")
        if context.get("citations") or context.get("results") or "[근거:" in answer:
            errors.append("확인 불가 답변에 가짜 근거가 포함됐습니다.")
    else:
        if not context.get("results"):
            errors.append("답변에 사용된 검색 Context가 없습니다.")
        if "[근거:" not in answer:
            errors.append("답변에 [근거: 공시명, 공시일자] 표시가 없습니다.")
    if case.expect_unit and not any(unit in answer for unit in ("원", "%", "배")):
        errors.append("수치 답변에 단위가 없습니다.")
    if case.expect_history and not any(word in answer for word in ("정정", "해지", "이력")):
        errors.append("정정·해지 이력 반영을 확인할 수 없습니다.")
    return errors


def _print_result(case: SimulationCase, transport: str, errors: list[str], answer: str) -> None:
    status = "PASS" if not errors else "FAIL"
    print(f"[{status}] {transport} {case.question_id} {case.category}: {answer[:80]}")
    for error in errors:
        print(f"  - {error}")


def run_sync(
    mode: Literal["mock", "live"],
    base_url: str,
    timeout: float,
    cases: tuple[SimulationCase, ...],
) -> int:
    failures = 0
    client: Any = TestClient(server.app) if mode == "mock" else httpx.Client(
        base_url=base_url,
        timeout=timeout,
    )
    with client:
        for case in cases:
            try:
                response = client.get(
                    "/answer",
                    params={"question_id": case.question_id, "question": case.question},
                )
                response.raise_for_status()
                payload = response.json()
                errors = validate_response(case, payload)
            except Exception as error:  # CLI runner: report each case and continue.
                payload = {"answer": ""}
                errors = [f"{type(error).__name__}: {error}"]
            failures += bool(errors)
            _print_result(case, "sync", errors, str(payload.get("answer", "")))
    return failures


async def run_async(
    mode: Literal["mock", "live"],
    base_url: str,
    timeout: float,
    cases: tuple[SimulationCase, ...],
) -> int:
    transport = httpx.ASGITransport(app=server.app) if mode == "mock" else None
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver" if mode == "mock" else base_url,
        timeout=timeout,
    ) as client:
        async def request(case: SimulationCase) -> tuple[SimulationCase, dict[str, Any], list[str]]:
            try:
                response = await client.get(
                    "/answer",
                    params={"question_id": case.question_id, "question": case.question},
                )
                response.raise_for_status()
                payload = response.json()
                return case, payload, validate_response(case, payload)
            except Exception as error:  # CLI runner: report each case and continue.
                return case, {"answer": ""}, [f"{type(error).__name__}: {error}"]

        results = await asyncio.gather(*(request(case) for case in cases))
    failures = 0
    for case, payload, errors in results:
        failures += bool(errors)
        _print_result(case, "async", errors, str(payload.get("answer", "")))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "live"), default="mock")
    parser.add_argument("--transport", choices=("sync", "async", "both"), default="both")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--case-id",
        action="append",
        help="실행할 SIM ID. 생략하면 12개 전체를 실행합니다.",
    )
    args = parser.parse_args()

    selected_cases = tuple(
        case
        for case in CASES
        if not args.case_id or case.question_id in args.case_id
    )
    if not selected_cases:
        parser.error("일치하는 --case-id가 없습니다.")

    original_graph = server.graph
    original_formatter = server.format_citations
    if args.mode == "mock":
        server.graph = _SimulationGraph()
        server.format_citations = _mock_citation_labels
    try:
        failures = 0
        if args.transport in {"sync", "both"}:
            failures += run_sync(
                args.mode,
                args.base_url,
                args.timeout,
                selected_cases,
            )
        if args.transport in {"async", "both"}:
            failures += asyncio.run(run_async(
                args.mode,
                args.base_url,
                args.timeout,
                selected_cases,
            ))
    finally:
        server.graph = original_graph
        server.format_citations = original_formatter

    total = len(selected_cases) * (2 if args.transport == "both" else 1)
    print(f"\nSimulation: {total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
