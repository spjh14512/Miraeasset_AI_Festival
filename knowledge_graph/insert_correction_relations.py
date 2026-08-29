"""Add CORRECTS relationships between existing Disclosure nodes in Neo4j."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase


RESOLVABLE_STATUSES = {"FOUND", "RECOVERED"}


def _jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid correction JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Correction record must be an object at {path}:{line_number}"
                )
            yield line_number, value


def correction_relation_rows(path: Path) -> list[dict[str, Any]]:
    """Build correction chains from resolved correction.v1 records."""
    relations: dict[str, dict[str, Any]] = {}
    for line_number, record in _jsonl(path):
        if record.get("schema_version") != "correction.v1":
            raise ValueError(
                f"Unsupported correction schema at {path}:{line_number}: "
                f"{record.get('schema_version')!r}"
            )
        source_document = record.get("source_document")
        correction = record.get("correction")
        if correction is None:
            continue
        if not isinstance(source_document, dict) or not isinstance(correction, dict):
            raise ValueError(
                f"Invalid correction structure at {path}:{line_number}"
            )
        target_rcept_no = correction.get("target_rcept_no")
        if target_rcept_no is None:
            continue
        if record.get("status") not in RESOLVABLE_STATUSES:
            raise ValueError(
                f"Resolved target has invalid status at {path}:{line_number}"
            )
        source_rcept_no = str(source_document.get("rcept_no", "")).strip()
        target_rcept_no = str(target_rcept_no).strip()
        if not source_rcept_no or not target_rcept_no:
            raise ValueError(
                f"Resolved correction requires both receipts at {path}:{line_number}"
            )
        if source_rcept_no == target_rcept_no:
            raise ValueError(f"A disclosure cannot correct itself: {source_rcept_no}")

        row = {
            "source_id": f"d{source_rcept_no}",
            "root_target_id": f"d{target_rcept_no}",
            "source_rcept_no": source_rcept_no,
            "correction_date": correction.get("correction_date"),
            "original_submission_date": correction.get("original_submission_date"),
            "target_document_name": correction.get("target_document_name"),
            "reason": correction.get("reason"),
        }
        previous = relations.get(source_rcept_no)
        if (
            previous is not None
            and previous["root_target_id"] != row["root_target_id"]
        ):
            raise ValueError(
                f"Conflicting original disclosures for correction {source_rcept_no}"
            )
        relations[source_rcept_no] = row

    by_root: dict[str, list[dict[str, Any]]] = {}
    for row in relations.values():
        by_root.setdefault(row["root_target_id"], []).append(row)

    chained: list[dict[str, Any]] = []
    for root_target_id in sorted(by_root):
        previous_id = root_target_id
        chain = sorted(
            by_root[root_target_id],
            key=lambda row: row["source_rcept_no"],
        )
        for row in chain:
            chained.append({
                key: value
                for key, value in {
                    **row,
                    "target_id": previous_id,
                }.items()
                if key not in {"root_target_id", "source_rcept_no"}
            })
            previous_id = row["source_id"]
    return chained


def _chunks(
    rows: Sequence[dict[str, Any]], size: int
) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def insert_correction_relations(
    driver: Driver,
    rows: Sequence[dict[str, Any]],
    *,
    database: str | None,
    batch_size: int,
) -> None:
    """Create relationships only; Disclosure node properties are never changed."""
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not rows:
        return

    driver.verify_connectivity()
    with driver.session(database=database) as session:
        missing = session.run(
            """
            UNWIND $rows AS row
            OPTIONAL MATCH (source:Disclosure {id: row.source_id})
            OPTIONAL MATCH (target:Disclosure {id: row.target_id})
            WITH row, source, target
            WHERE source IS NULL OR target IS NULL
            RETURN collect({source_id: row.source_id, target_id: row.target_id}) AS rows
            """,
            rows=list(rows),
        ).single(strict=True)["rows"]
        if missing:
            raise RuntimeError(
                "CORRECTS endpoints must already exist as Disclosure nodes: "
                f"{json.dumps(missing, ensure_ascii=False, sort_keys=True)}"
            )

        invalid = session.run(
            """
            UNWIND $rows AS row
            MATCH (source:Disclosure {id: row.source_id})
            MATCH (target:Disclosure {id: row.target_id})
            OPTIONAL MATCH (source_company:Company)-[:PUBLISHES]->(source)
            OPTIONAL MATCH (target_company:Company)-[:PUBLISHES]->(target)
            WITH row, source, target,
                 collect(DISTINCT source_company.corp_code) AS source_companies,
                 collect(DISTINCT target_company.corp_code) AS target_companies
            WHERE coalesce(source.is_correction, false) = false
               OR none(code IN source_companies WHERE code IN target_companies)
            RETURN collect({source_id: row.source_id, target_id: row.target_id}) AS rows
            """,
            rows=list(rows),
        ).single(strict=True)["rows"]
        if invalid:
            raise RuntimeError(
                "CORRECTS must connect a correction to a previous version from the same company: "
                f"{json.dumps(invalid, ensure_ascii=False, sort_keys=True)}"
            )

        involved_ids = sorted({
            node_id
            for row in rows
            for node_id in (row["source_id"], row["target_id"])
        })
        target_ids = {row["target_id"] for row in rows}
        latest_ids = sorted({row["source_id"] for row in rows} - target_ids)
        session.run(
            """
            UNWIND $ids AS id
            MATCH (disclosure:Disclosure {id: id})
            SET disclosure.is_latest_version = false
            """,
            ids=involved_ids,
        ).consume()
        session.run(
            """
            UNWIND $ids AS id
            MATCH (disclosure:Disclosure {id: id})
            SET disclosure.is_latest_version = true
            """,
            ids=latest_ids,
        ).consume()

        for batch in _chunks(rows, batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MATCH (source:Disclosure {id: row.source_id})
                OPTIONAL MATCH (source)-[existing:CORRECTS]->(existing_target:Disclosure)
                WHERE existing_target.id <> row.target_id
                DELETE existing
                """,
                rows=batch,
            ).consume()
            session.run(
                """
                UNWIND $rows AS row
                MATCH (source:Disclosure {id: row.source_id})
                MATCH (target:Disclosure {id: row.target_id})
                MERGE (source)-[relation:CORRECTS]->(target)
                SET relation.correction_date = row.correction_date,
                    relation.original_submission_date = row.original_submission_date,
                    relation.target_document_name = row.target_document_name,
                    relation.reason = row.reason
                """,
                rows=batch,
            ).consume()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert CORRECTS relationships between existing Disclosure nodes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/correction/manifest.jsonl"),
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--database", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    rows = correction_relation_rows(args.input)
    print(json.dumps({"correction_relations": len(rows)}, ensure_ascii=False))
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
        insert_correction_relations(
            driver,
            rows,
            database=args.database,
            batch_size=args.batch_size,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
