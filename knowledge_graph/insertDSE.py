"""Insert Disclosure, Section, and Evidence data into Neo4j.

The allowed labels and properties are read from ``knowledge_graph/neo4j_schema.yaml``.
Disclosures are sampled evenly across the four disclosure groups.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase


DISCLOSURE_GROUPS = ("periodic", "major", "holding", "exchange")
VALID_STATUSES = {"SUCCESS", "RECOVERED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert schema-compliant Disclosure/Section/Evidence data into Neo4j."
    )
    parser.add_argument("--limit", type=int, required=True, help="Total disclosures to sample")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed for reproducible random sampling",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--schema", type=Path, default=Path("knowledge_graph/neo4j_schema.yaml")
    )
    parser.add_argument("--database", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_schema(path: Path) -> dict[str, Any]:
    schema = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise ValueError("neo4j_schema.yaml must contain a mapping")

    entities = schema.get("entities")
    relations = schema.get("relations")
    if not isinstance(entities, dict) or not isinstance(relations, dict):
        raise ValueError("neo4j_schema.yaml requires entities and relations mappings")

    for name in ("Disclosure", "Section", "Evidence"):
        if name not in entities:
            raise ValueError(f"Missing entity schema: {name}")
    for name in ("HAS_SECTION", "HAS_EVIDENCE"):
        if name not in relations:
            raise ValueError(f"Missing relation schema: {name}")
    return schema


def schema_properties(schema: Mapping[str, Any], entity: str) -> set[str]:
    properties = schema["entities"][entity].get("properties", [])
    if isinstance(properties, list):
        return {str(value) for value in properties}
    if isinstance(properties, dict):
        return {str(value) for value in properties}
    raise ValueError(f"Invalid properties definition for {entity}")


def filter_properties(
    schema: Mapping[str, Any], entity: str, values: Mapping[str, Any]
) -> dict[str, Any]:
    allowed = schema_properties(schema, entity)
    return {key: value for key, value in values.items() if key in allowed and value is not None}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def valid_manifest_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Keep the last valid manifest row for each (doc_group, rcept_no)."""
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = (str(row.get("doc_group", "")), str(row.get("rcept_no", "")))
        if (
            all(key)
            and row.get("status") in VALID_STATUSES
            and not row.get("error")
        ):
            rows[key] = row
    return rows


def disclosure_metadata_rows(data_root: Path) -> dict[str, dict[str, Any]]:
    """Index the source document manifest by rcept_no."""
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(data_root / "manifest.jsonl"):
        rcept_no = str(row.get("rcept_no", ""))
        corp_code = str(row.get("corp_code", ""))
        if not rcept_no or not corp_code:
            continue
        previous = rows.get(rcept_no)
        if previous is not None and str(previous.get("corp_code")) != corp_code:
            raise ValueError(
                f"Conflicting corp_code values for rcept_no {rcept_no}"
            )
        rows[rcept_no] = row
    return rows


def parse_rcept_date(value: Any, *, rcept_no: str):
    text = str(value or "").strip()
    try:
        return datetime.strptime(text, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(
            f"rcept_dt must use YYYYMMDD for disclosure {rcept_no}: {value!r}"
        ) from exc


def allocation_by_group(limit: int) -> dict[str, int]:
    if limit < 1:
        raise ValueError("--limit must be positive")
    quotient, remainder = divmod(limit, len(DISCLOSURE_GROUPS))
    return {
        group: quotient + (index < remainder)
        for index, group in enumerate(DISCLOSURE_GROUPS)
    }


def select_disclosures(
    data_root: Path, *, limit: int, random_seed: int | None
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    section_rows = valid_manifest_rows(
        data_root / "canonical_section" / "manifest.jsonl"
    )
    evidence_rows = valid_manifest_rows(
        data_root / "evidence_fragment" / "manifest.jsonl"
    )
    common_keys = section_rows.keys() & evidence_rows.keys()
    candidates: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)

    for group, rcept_no in common_keys:
        if group not in DISCLOSURE_GROUPS:
            continue
        section_row = section_rows[(group, rcept_no)]
        evidence_row = evidence_rows[(group, rcept_no)]
        section_path = data_root / str(section_row.get("output_path", ""))
        evidence_paths = [
            data_root / str(path) for path in evidence_row.get("output_paths", [])
        ]
        if section_path.is_file() and evidence_paths and all(
            path.is_file() for path in evidence_paths
        ):
            candidates[group].append((section_row, evidence_row))

    rng = random.Random(random_seed)
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for group, count in allocation_by_group(limit).items():
        group_candidates = sorted(
            candidates[group], key=lambda rows: str(rows[0]["rcept_no"])
        )
        if len(group_candidates) < count:
            raise ValueError(
                f"Not enough complete disclosures for {group}: "
                f"required={count}, available={len(group_candidates)}"
            )
        selected.extend(rng.sample(group_candidates, count))
    return selected


def graph_id(source_id: str, rcept_no: str) -> str:
    prefixes = (f"section:{rcept_no}", f"evidence:{rcept_no}")
    for prefix in prefixes:
        if source_id.startswith(prefix):
            return f"d{rcept_no}{source_id[len(prefix):]}"
    raise ValueError(f"Unexpected source ID for {rcept_no}: {source_id}")


def flattened_headers(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for header in value:
        if isinstance(header, list):
            result.append(" > ".join(str(part) for part in header if part))
        else:
            result.append(str(header))
    return result


def flattened_keys(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for field in value:
        if not isinstance(field, dict):
            continue
        key_parts: list[str] = []
        key_paths = field.get("key_paths", [])
        if isinstance(key_paths, list):
            for path in key_paths:
                if isinstance(path, list):
                    key_parts.extend(str(part) for part in path if part)
                elif path:
                    key_parts.append(str(path))
        result.append(" > ".join(key_parts))
    return result


def joined_caption(value: Any) -> str | None:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value) or None
    return str(value) if value else None


def build_rows(
    schema: Mapping[str, Any],
    data_root: Path,
    selected: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    disclosure_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {
        "disclosures": [],
        "sections": [],
        "texts": [],
        "tables": [],
    }

    for section_manifest, evidence_manifest in selected:
        rcept_no = str(section_manifest["rcept_no"])
        document_metadata = disclosure_metadata.get(rcept_no)
        if document_metadata is None:
            raise ValueError(
                f"Selected disclosure is missing from data/manifest.jsonl: {rcept_no}"
            )
        corp_code = str(document_metadata.get("corp_code", ""))
        if not corp_code:
            raise ValueError(f"corp_code is missing for disclosure {rcept_no}")
        disclosure_id = f"d{rcept_no}"
        disclosure_properties = filter_properties(
                schema,
                "Disclosure",
                {
                    "id": disclosure_id,
                    "doc_group": str(section_manifest["doc_group"]),
                    "source_path": str(section_manifest.get("source_path", "")),
                    "n_sections": 0,
                },
            )
        disclosure_properties.update(
            {
                "report_name": str(document_metadata.get("report_nm", "")),
                "rcept_date": parse_rcept_date(
                    document_metadata.get("rcept_dt"),
                    rcept_no=rcept_no,
                ),
                "_corp_code": corp_code,
            }
        )
        rows["disclosures"].append(disclosure_properties)

        section_document = json.loads(
            (data_root / str(section_manifest["output_path"])).read_text(
                encoding="utf-8"
            )
        )
        known_sections: set[str] = set()
        for section in section_document.get("sections", []):
            section_id = graph_id(str(section["section_id"]), rcept_no)
            known_sections.add(section_id)
            parent_id = section.get("parent_section_id")
            rows["sections"].append(
                {
                    **filter_properties(
                        schema,
                        "Section",
                        {
                            "id": section_id,
                            "title": section.get("title"),
                            "section_path": section.get("section_path", []),
                            "element_path": section.get("element_path"),
                            "order_in_doc": int(section.get("order", 0)),
                            "n_evidence": 0,
                        },
                    ),
                    "_disclosure_id": disclosure_id,
                    "_parent_id": (
                        graph_id(str(parent_id), rcept_no) if parent_id else None
                    ),
                }
            )

        for output_path in evidence_manifest.get("output_paths", []):
            fragment = json.loads(
                (data_root / str(output_path)).read_text(encoding="utf-8")
            )
            section_id = graph_id(str(fragment["section_id"]), rcept_no)
            if section_id not in known_sections:
                raise ValueError(f"Evidence references unknown section: {section_id}")
            for evidence in fragment.get("evidence_list", []):
                evidence_id = graph_id(str(evidence["evidence_id"]), rcept_no)
                payload = evidence.get("payload", {})
                common = filter_properties(
                    schema,
                    "Evidence",
                    {
                        "id": evidence_id,
                        "order_in_section": int(evidence.get("order", 0)),
                        "heading_path": payload.get("heading_path", []),
                    },
                )
                evidence_type = str(evidence.get("evidence_type", ""))
                if evidence_type == "TEXT":
                    subtype = filter_properties(
                        schema, "Text", {"text": payload.get("text", "")}
                    )
                    rows["texts"].append(
                        {**common, **subtype, "_section_id": section_id}
                    )
                elif evidence_type == "TABLE":
                    raw_type = str(evidence.get("table_type", ""))
                    table_type = {
                        "KV_TABLE": "KV-table",
                        "R_TABLE": "R-table",
                    }.get(raw_type, raw_type)
                    table_values: dict[str, Any] = {
                        "table_type": table_type,
                        "title": payload.get("title"),
                        "captions": joined_caption(payload.get("captions")),
                        "notes": payload.get("notes", []),
                        "units": payload.get("units", []),
                    }
                    if raw_type == "R_TABLE":
                        table_values["headers"] = flattened_headers(
                            payload.get("headers", [])
                        )
                        table_values["n_rows"] = int(
                            payload.get("record_count", 0)
                        )
                    elif raw_type == "KV_TABLE":
                        fields = payload.get("fields", [])
                        table_values["keys"] = flattened_keys(fields)
                        table_values["n_entries"] = (
                            len(fields) if isinstance(fields, list) else 0
                        )
                    subtype = filter_properties(
                        schema,
                        "Table",
                        table_values,
                    )
                    rows["tables"].append(
                        {**common, **subtype, "_section_id": section_id}
                    )
                else:
                    raise ValueError(f"Unsupported evidence type: {evidence_type}")
    return rows


def chunks(rows: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def run_batched(session: Any, query: str, rows: Sequence[dict[str, Any]], size: int) -> None:
    for batch in chunks(rows, size):
        session.run(query, rows=batch).consume()


def insert_rows(
    driver: Driver,
    rows: Mapping[str, list[dict[str, Any]]],
    *,
    database: str | None,
    batch_size: int,
) -> None:
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    driver.verify_connectivity()
    with driver.session(database=database) as session:
        corp_codes = sorted(
            {str(row["_corp_code"]) for row in rows["disclosures"]}
        )
        missing_record = session.run(
            """
            UNWIND $corp_codes AS corp_code
            OPTIONAL MATCH (company:Company {corp_code: corp_code})
            WITH corp_code, count(company) AS company_count
            WHERE company_count = 0
            RETURN collect(corp_code) AS missing_corp_codes
            """,
            corp_codes=corp_codes,
        ).single(strict=True)
        missing_corp_codes = list(missing_record["missing_corp_codes"])
        if missing_corp_codes:
            raise RuntimeError(
                "Company nodes must be inserted before Disclosure nodes. "
                f"Missing corp_code values: {', '.join(missing_corp_codes)}"
            )

        for label in ("Disclosure", "Section", "Evidence"):
            session.run(
                f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
            ).consume()

        run_batched(
            session,
            """
            UNWIND $rows AS row
            MERGE (n:Disclosure {id: row.id})
            SET n += properties(row), n._corp_code = null
            REMOVE n.base_year, n.base_month
            """,
            rows["disclosures"],
            batch_size,
        )
        run_batched(
            session,
            """
            UNWIND $rows AS row
            MATCH (company:Company {corp_code: row._corp_code})
            MATCH (disclosure:Disclosure {id: row.id})
            MERGE (company)-[:PUBLISHES]->(disclosure)
            """,
            rows["disclosures"],
            batch_size,
        )
        run_batched(
            session,
            """
            UNWIND $rows AS row
            MERGE (n:Section {id: row.id})
            SET n += properties(row), n._disclosure_id = null, n._parent_id = null
            """,
            rows["sections"],
            batch_size,
        )
        for key, subtype in (("texts", "Text"), ("tables", "Table")):
            run_batched(
                session,
                f"""
                UNWIND $rows AS row
                MERGE (n:Evidence:{subtype} {{id: row.id}})
                SET n += properties(row), n._section_id = null
                """,
                rows[key],
                batch_size,
            )

        root_sections = [row for row in rows["sections"] if not row["_parent_id"]]
        child_sections = [row for row in rows["sections"] if row["_parent_id"]]
        run_batched(
            session,
            """
            UNWIND $rows AS row
            MATCH (d:Disclosure {id: row._disclosure_id}), (s:Section {id: row.id})
            MERGE (d)-[:HAS_SECTION]->(s)
            """,
            root_sections,
            batch_size,
        )
        run_batched(
            session,
            """
            UNWIND $rows AS row
            MATCH (p:Section {id: row._parent_id}), (s:Section {id: row.id})
            MERGE (p)-[:HAS_SECTION]->(s)
            """,
            child_sections,
            batch_size,
        )
        for key in ("texts", "tables"):
            run_batched(
                session,
                """
                UNWIND $rows AS row
                MATCH (s:Section {id: row._section_id}), (e:Evidence {id: row.id})
                MERGE (s)-[:HAS_EVIDENCE]->(e)
                """,
                rows[key],
                batch_size,
            )

        disclosure_ids = [row["id"] for row in rows["disclosures"]]
        session.run(
            """
            UNWIND $ids AS id
            MATCH (d:Disclosure {id: id})-[:HAS_SECTION*]->(s:Section)
            WITH d, count(DISTINCT s) AS n_sections
            SET d.n_sections = n_sections
            """,
            ids=disclosure_ids,
        ).consume()
        session.run(
            """
            UNWIND $ids AS id
            MATCH (d:Disclosure {id: id})-[:HAS_SECTION*]->(s:Section)
            OPTIONAL MATCH (s)-[:HAS_EVIDENCE]->(e:Evidence)
            WITH s, count(DISTINCT e) AS n_evidence
            SET s.n_evidence = n_evidence
            """,
            ids=disclosure_ids,
        ).consume()


def summary(
    selected: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    rows: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_group = {group: 0 for group in DISCLOSURE_GROUPS}
    for section_manifest, _ in selected:
        by_group[str(section_manifest["doc_group"])] += 1
    return {
        "disclosures": len(rows["disclosures"]),
        "sections": len(rows["sections"]),
        "evidences": len(rows["texts"]) + len(rows["tables"]),
        "texts": len(rows["texts"]),
        "tables": len(rows["tables"]),
        "publishers": len(
            {str(row["_corp_code"]) for row in rows["disclosures"]}
        ),
        "by_group": by_group,
        "rcept_nos": [str(item[0]["rcept_no"]) for item in selected],
    }


def main() -> int:
    args = parse_args()
    schema = load_schema(args.schema)
    selected = select_disclosures(
        args.data_root, limit=args.limit, random_seed=args.random_seed
    )
    disclosure_metadata = disclosure_metadata_rows(args.data_root)
    rows = build_rows(schema, args.data_root, selected, disclosure_metadata)
    print(json.dumps(summary(selected, rows), ensure_ascii=False, indent=2))
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
        insert_rows(
            driver,
            rows,
            database=args.database,
            batch_size=args.batch_size,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
