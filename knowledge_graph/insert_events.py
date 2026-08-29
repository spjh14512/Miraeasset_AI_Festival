"""Insert extracted Event nodes and their Evidence relationships into Neo4j."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

from knowledge_graph.insertDSE import load_schema


def _jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid Event JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Event record must be an object at {path}:{line_number}")
            yield line_number, value


def _required_text(value: Any, *, field: str, location: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Event {field} is required at {location}")
    return text


def event_rows(path: Path) -> list[dict[str, Any]]:
    """Validate event.v1 intermediates and flatten them for Neo4j insertion."""
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, record in _jsonl(path):
        location = f"{path}:{line_number}"
        if record.get("schema_version") != "event.v1":
            raise ValueError(
                f"Unsupported Event schema at {location}: "
                f"{record.get('schema_version')!r}"
            )
        source = record.get("source_document")
        event = record.get("event")
        relations = record.get("relations")
        if not all(isinstance(value, Mapping) for value in (source, event, relations)):
            raise ValueError(f"Invalid Event structure at {location}")
        assert isinstance(source, Mapping)
        assert isinstance(event, Mapping)
        assert isinstance(relations, Mapping)
        if source.get("is_latest_version") is not True:
            raise ValueError(f"Event source is not the latest disclosure at {location}")

        event_id = _required_text(event.get("id"), field="id", location=location)
        disclosure_id = _required_text(
            source.get("disclosure_id"), field="disclosure_id", location=location
        )
        if event_id in seen_ids:
            raise ValueError(f"Duplicate Event id at {location}: {event_id}")
        seen_ids.add(event_id)

        reports = relations.get("reports")
        supports = relations.get("is_supported_by")
        if not isinstance(reports, Mapping) or not isinstance(supports, list):
            raise ValueError(f"Invalid Event relations at {location}")
        if (
            reports.get("type") != "REPORTS"
            or reports.get("source_id") != disclosure_id
            or reports.get("target_id") != event_id
        ):
            raise ValueError(f"Invalid REPORTS direction at {location}")

        evidence_ids: list[str] = []
        for relation in supports:
            if not isinstance(relation, Mapping):
                raise ValueError(f"Invalid IS_SUPPORTED_BY row at {location}")
            evidence_id = _required_text(
                relation.get("target_id"), field="Evidence id", location=location
            )
            if (
                relation.get("type") != "IS_SUPPORTED_BY"
                or relation.get("source_id") != event_id
            ):
                raise ValueError(f"Invalid IS_SUPPORTED_BY direction at {location}")
            if evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
        if not evidence_ids:
            raise ValueError(f"Event requires supporting Evidence at {location}")

        event_date_text = _required_text(
            event.get("event_date"), field="event_date", location=location
        )
        try:
            event_date = date.fromisoformat(event_date_text)
        except ValueError as exc:
            raise ValueError(
                f"Event event_date must use YYYY-MM-DD at {location}: "
                f"{event_date_text!r}"
            ) from exc

        rows.append(
            {
                "id": event_id,
                "event_type": _required_text(
                    event.get("event_type"), field="event_type", location=location
                ),
                "event_subtype": _required_text(
                    event.get("event_subtype"), field="event_subtype", location=location
                ),
                "event_date": event_date,
                "content": _required_text(
                    event.get("content"), field="content", location=location
                ),
                "disclosure_id": disclosure_id,
                "evidence_ids": evidence_ids,
            }
        )
    return rows


def _chunks(
    rows: Sequence[dict[str, Any]], size: int
) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def insert_events(
    driver: Driver,
    rows: Sequence[dict[str, Any]],
    *,
    database: str | None,
    batch_size: int,
) -> None:
    """Upsert Events after verifying every existing graph endpoint."""
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not rows:
        return

    driver.verify_connectivity()
    with driver.session(database=database) as session:
        invalid_disclosures = session.run(
            """
            UNWIND $rows AS row
            OPTIONAL MATCH (disclosure:Disclosure {id: row.disclosure_id})
            WITH row, disclosure
            WHERE disclosure IS NULL OR disclosure.is_latest_version <> true
            RETURN collect(row.disclosure_id) AS rows
            """,
            rows=list(rows),
        ).single(strict=True)["rows"]
        if invalid_disclosures:
            raise RuntimeError(
                "REPORTS source must be an existing latest Disclosure: "
                f"{json.dumps(invalid_disclosures, ensure_ascii=False)}"
            )

        support_rows = [
            {"event_id": row["id"], "evidence_id": evidence_id}
            for row in rows
            for evidence_id in row["evidence_ids"]
        ]
        missing_evidence = session.run(
            """
            UNWIND $rows AS row
            OPTIONAL MATCH (evidence:Evidence {id: row.evidence_id})
            WITH row, evidence
            WHERE evidence IS NULL
            RETURN collect(row.evidence_id) AS rows
            """,
            rows=support_rows,
        ).single(strict=True)["rows"]
        if missing_evidence:
            raise RuntimeError(
                "IS_SUPPORTED_BY target must be an existing Evidence: "
                f"{json.dumps(missing_evidence, ensure_ascii=False)}"
            )

        session.run(
            """
            CREATE CONSTRAINT event_id IF NOT EXISTS
            FOR (event:Event) REQUIRE event.id IS UNIQUE
            """
        ).consume()
        for batch in _chunks(rows, batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MERGE (event:Event {id: row.id})
                SET event.event_type = row.event_type,
                    event.event_subtype = row.event_subtype,
                    event.event_date = row.event_date,
                    event.content = row.content
                WITH row, event
                MATCH (disclosure:Disclosure {id: row.disclosure_id})
                MERGE (disclosure)-[:REPORTS]->(event)
                """,
                rows=batch,
            ).consume()
            session.run(
                """
                UNWIND $rows AS row
                MATCH (event:Event {id: row.id})
                OPTIONAL MATCH (event)-[existing:IS_SUPPORTED_BY]->(old:Evidence)
                WHERE NOT old.id IN row.evidence_ids
                DELETE existing
                """,
                rows=batch,
            ).consume()

        for batch in _chunks(support_rows, batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MATCH (event:Event {id: row.event_id})
                MATCH (evidence:Evidence {id: row.evidence_id})
                MERGE (event)-[:IS_SUPPORTED_BY]->(evidence)
                """,
                rows=batch,
            ).consume()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert Event, REPORTS, and IS_SUPPORTED_BY into Neo4j."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/event/manifest.jsonl")
    )
    parser.add_argument(
        "--schema", type=Path, default=Path("knowledge_graph/neo4j_schema.yaml")
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--database", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    schema = load_schema(args.schema)
    for entity in ("Event",):
        if entity not in schema["entities"]:
            raise ValueError(f"Missing entity schema: {entity}")
    for relation in ("REPORTS", "IS_SUPPORTED_BY"):
        if relation not in schema["relations"]:
            raise ValueError(f"Missing relation schema: {relation}")

    rows = event_rows(args.input)
    summary = {
        "events": len(rows),
        "reports": len(rows),
        "is_supported_by": sum(len(row["evidence_ids"]) for row in rows),
    }
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
        insert_events(
            driver,
            rows,
            database=args.database,
            batch_size=args.batch_size,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["event_rows", "insert_events"]
