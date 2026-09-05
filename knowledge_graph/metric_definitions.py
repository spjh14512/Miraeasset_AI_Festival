"""Seed MetricDefinition nodes for Tier 3 cross-company financial screening.

이 파일은 R_TABLE에서 추출하거나 계산해 낼 재무 지표의 정의(공식·단위·
방향성)를 코드로 고정합니다. 값 자체(MetricObservation)는 별도의 배치
추출기가 채우며, 이 파일은 "그 값이 무엇을 뜻하는지"만 정의합니다.

계산 지표(value_origin=calculated)의 required_metrics는 배치 추출기가
계산 순서를 정하는 데 쓰는 힌트일 뿐입니다. "같은 지표를 어느 기간에서
몇 개 가져와야 하는지"(예: ROA의 평균자산은 당기말·전기말 두 시점의
total_assets가 필요함)는 이 표에 표현하지 않고, 배치 추출기의 지표별
계산 함수(예: compute_roa)에 명시적으로 하드코딩합니다. combine_numeric_
results가 operation마다 Python 함수로 계산 로직을 직접 구현하는 것과
동일한 원칙입니다.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

from knowledge_graph.insertDSE import load_schema

# ROA = 당기순이익 ÷ 평균자산 × 100, 평균자산 = (전기말 자산총계 + 당기말 자산총계) / 2
# 배치 추출기는 이 지표를 계산할 때 다음 3개 관측치를 가져와야 합니다.
#   - net_income, fiscal_year=Y, period_type=annual        (1개)
#   - total_assets, fiscal_year=Y (당기말)                  (1개)
#   - total_assets, fiscal_year=Y-1 (전기말)                (1개, 같은 지표의 이전 연도 값)
METRIC_DEFINITIONS: list[dict[str, Any]] = [
    {
        "id": "revenue",
        "name": "매출액",
        "formula": None,
        "unit": "백만원",
        "higher_is_better": True,
        "formula_version": None,
        "required_metrics": None,
    },
    {
        "id": "operating_income",
        "name": "영업이익",
        "formula": None,
        "unit": "백만원",
        "higher_is_better": True,
        "formula_version": None,
        "required_metrics": None,
    },
    {
        "id": "net_income",
        "name": "당기순이익",
        "formula": None,
        "unit": "백만원",
        "higher_is_better": True,
        "formula_version": None,
        "required_metrics": None,
    },
    {
        "id": "total_assets",
        "name": "자산총계",
        "formula": None,
        "unit": "백만원",
        "higher_is_better": None,
        "formula_version": None,
        "required_metrics": None,
    },
    {
        "id": "total_equity",
        "name": "자본총계",
        "formula": None,
        "unit": "백만원",
        "higher_is_better": None,
        "formula_version": None,
        "required_metrics": None,
    },
    {
        "id": "roa",
        "name": "ROA(총자산이익률)",
        "formula": "당기순이익 / 평균자산 * 100, 평균자산 = (전기말 자산총계 + 당기말 자산총계) / 2",
        "unit": "%",
        "higher_is_better": True,
        "formula_version": "v1",
        "required_metrics": ["net_income", "total_assets"],
    },
]


def _required_text(value: Any, *, field: str, definition_id: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"MetricDefinition {field} is required for {definition_id!r}")
    return text


def metric_definition_rows(
    definitions: Sequence[dict[str, Any]] = METRIC_DEFINITIONS,
) -> list[dict[str, Any]]:
    """seed 목록을 검증하고 Neo4j 삽입용으로 정리합니다."""

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for entry in definitions:
        definition_id = _required_text(
            entry.get("id"), field="id", definition_id=str(entry.get("id"))
        )
        if definition_id in seen_ids:
            raise ValueError(f"Duplicate MetricDefinition id: {definition_id}")
        seen_ids.add(definition_id)

        name = _required_text(entry.get("name"), field="name", definition_id=definition_id)
        higher_is_better = entry.get("higher_is_better")
        if higher_is_better is not None and not isinstance(higher_is_better, bool):
            raise ValueError(
                f"MetricDefinition higher_is_better must be boolean or null: {definition_id}"
            )
        required_metrics = entry.get("required_metrics")
        if required_metrics is not None and not (
            isinstance(required_metrics, list)
            and all(isinstance(item, str) and item.strip() for item in required_metrics)
        ):
            raise ValueError(
                f"MetricDefinition required_metrics must be a list of non-empty strings: {definition_id}"
            )

        rows.append(
            {
                "id": definition_id,
                "name": name,
                "formula": entry.get("formula"),
                "unit": entry.get("unit"),
                "higher_is_better": higher_is_better,
                "formula_version": entry.get("formula_version"),
                "required_metrics": required_metrics,
            }
        )
    return rows


def insert_metric_definitions(
    driver: Driver,
    rows: Sequence[dict[str, Any]],
    *,
    database: str | None,
) -> None:
    """MetricDefinition을 자연키(id)로 MERGE upsert합니다."""

    if not rows:
        return

    driver.verify_connectivity()
    with driver.session(database=database) as session:
        session.run(
            """
            CREATE CONSTRAINT metric_definition_id IF NOT EXISTS
            FOR (definition:MetricDefinition) REQUIRE definition.id IS UNIQUE
            """
        ).consume()
        session.run(
            """
            UNWIND $rows AS row
            MERGE (definition:MetricDefinition {id: row.id})
            SET definition.name = row.name,
                definition.formula = row.formula,
                definition.unit = row.unit,
                definition.higher_is_better = row.higher_is_better,
                definition.formula_version = row.formula_version,
                definition.required_metrics = row.required_metrics
            """,
            rows=list(rows),
        ).consume()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed MetricDefinition nodes for Tier 3 financial screening."
    )
    parser.add_argument(
        "--schema", type=Path, default=Path("knowledge_graph/neo4j_schema.yaml")
    )
    parser.add_argument("--database", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    schema = load_schema(args.schema)
    if "MetricDefinition" not in schema["entities"]:
        raise ValueError("Missing entity schema: MetricDefinition")

    rows = metric_definition_rows()
    summary = {"metric_definitions": len(rows), "ids": [row["id"] for row in rows]}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if args.dry_run:
        return 0

    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not all((uri, username, password)):
        raise RuntimeError(
            "NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must be configured"
        )
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        insert_metric_definitions(driver, rows, database=args.database)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["METRIC_DEFINITIONS", "metric_definition_rows", "insert_metric_definitions"]
