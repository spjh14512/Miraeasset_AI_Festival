from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from vector_db.insert_points import (
    assemble_qdrant_points,
    embed_point_inputs_batch,
    load_qdrant_schema,
    upsert_point_dicts,
)
from vector_db.point_builder import PointInput, build_point_inputs
from vector_db.r_table_strategy_selector import RTableEmbeddingStrategySelector


DATA_ROOT = Path("data")
SCHEMA_PATH = Path("vector_db/qdrant_schema.yaml")
BACKUP_ROOT = Path("tmp/qdrant_backups")
MAX_BYTES = 32 * 1024
HARD_MAX_BYTES = 64 * 1024
TABLE_ID_PATTERN = re.compile(
    r"^rtable:(?P<rcept_no>\d+):src(?P<source_index>\d+):"
    r"s(?P<section_order>\d+):t\d+$"
)


def compact_size(value: object) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def client_from_env() -> QdrantClient:
    load_dotenv()
    return QdrantClient(
        host=os.environ["QDRANT_HOST"],
        port=int(os.environ["QDRANT_PORT"]),
    )


def scroll_r_table_points(client: QdrantClient, collection: str) -> list[Any]:
    result = []
    offset = None
    point_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="retrieval_metadata.point_kind",
                match=models.MatchValue(value="R_TABLE"),
            )
        ]
    )
    while True:
        page, offset = client.scroll(
            collection_name=collection,
            scroll_filter=point_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        result.extend(page)
        if offset is None:
            return result


def find_targets(points: list[Any]) -> list[Any]:
    return [
        point
        for point in points
        if compact_size((point.payload or {}).get("canonical")) > MAX_BYTES
    ]


def load_document_rows() -> dict[str, dict[str, Any]]:
    result = {}
    with (DATA_ROOT / "manifest.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            result[str(row["rcept_no"])] = row
    return result


def document_context(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "corp_name": row.get("corp_name"),
        "corp_code": row.get("corp_code"),
        "industry": row.get("industry"),
        "sector": row.get("sector"),
        "report_nm": row.get("report_nm"),
        "base_year": row.get("base_year"),
        "base_month": row.get("base_month"),
    }


def build_replacements(
    targets: list[Any],
) -> tuple[list[PointInput], list[dict[str, Any]]]:
    documents = load_document_rows()
    selector = RTableEmbeddingStrategySelector()
    json_cache: dict[Path, dict[str, Any]] = {}
    replacements: list[PointInput] = []
    table_summaries = []

    def load_json(path: Path) -> dict[str, Any]:
        if path not in json_cache:
            json_cache[path] = json.loads(path.read_text(encoding="utf-8"))
        return json_cache[path]

    for target in sorted(
        targets,
        key=lambda item: str((item.payload or {})["retrieval_metadata"]["table_id"]),
    ):
        old_payload = target.payload or {}
        old_metadata = old_payload["retrieval_metadata"]
        table_id = str(old_metadata["table_id"])
        match = TABLE_ID_PATTERN.fullmatch(table_id)
        if match is None:
            raise RuntimeError(f"Invalid target table_id: {table_id}")
        values = match.groupdict()
        document = documents.get(values["rcept_no"])
        if document is None:
            raise RuntimeError(f"Missing document manifest: {values['rcept_no']}")
        doc_group = str(document["doc_group"])
        fragment_path = (
            DATA_ROOT
            / "evidence_fragment"
            / doc_group
            / values["rcept_no"]
            / (
                f"src{values['source_index']}__"
                f"s{values['section_order']}.json"
            )
        )
        section_path = (
            DATA_ROOT
            / "canonical_section"
            / doc_group
            / f"sections_{values['rcept_no']}.json"
        )
        fragment = load_json(fragment_path)
        section_document = load_json(section_path)
        evidence_matches = [
            evidence
            for evidence in fragment.get("evidence_list", [])
            if evidence.get("payload", {}).get("table_id") == table_id
        ]
        if len(evidence_matches) != 1:
            raise RuntimeError(
                f"Expected one current Evidence for {table_id}, "
                f"found {len(evidence_matches)}"
            )
        evidence = evidence_matches[0]
        if evidence.get("table_type") != "R_TABLE":
            raise RuntimeError(f"Current Evidence is not R_TABLE: {table_id}")
        section_matches = [
            section
            for section in section_document.get("sections", [])
            if section.get("section_id") == fragment.get("section_id")
        ]
        if len(section_matches) != 1:
            raise RuntimeError(
                f"Expected one current Section for {table_id}, "
                f"found {len(section_matches)}"
            )
        source_records = [
            record
            for record in fragment.get("records", [])
            if record.get("table_id") == table_id
        ]
        filtered_fragment = {
            "schema_version": fragment.get("schema_version"),
            "section_id": fragment.get("section_id"),
            "evidence_list": [evidence],
            "records": source_records,
        }
        point_inputs = build_point_inputs(
            filtered_fragment,
            document_context=document_context(document),
            section_context=section_matches[0],
            r_table_strategy_selector=selector,
        )
        if not point_inputs:
            raise RuntimeError(f"No replacement Points generated: {table_id}")

        expected_indexes = sorted(
            int(record["record_index"]) for record in source_records
        )
        actual_indexes = []
        chunk_sizes = []
        overflow_chunks = 0
        overflow_details = []
        for point_input in point_inputs:
            metadata = point_input.payload["retrieval_metadata"]
            if metadata.get("point_kind") != "R_TABLE":
                raise RuntimeError(f"Unexpected replacement Point kind: {table_id}")
            if metadata.get("table_id") != table_id:
                raise RuntimeError(f"Replacement table_id mismatch: {table_id}")
            canonical = point_input.payload["canonical"]
            records = canonical["records"]
            actual_indexes.extend(
                int(record["record_index"]) for record in records
            )
            size_bytes = compact_size(canonical)
            chunk_sizes.append(size_bytes)
            if size_bytes > MAX_BYTES:
                base_size = compact_size({**canonical, "records": []})
                if size_bytes > HARD_MAX_BYTES:
                    raise RuntimeError(
                        f"Replacement chunk exceeds 64 KiB hard limit: {table_id}"
                    )
                overflow_chunks += 1
                overflow_details.append(
                    {
                        "size_bytes": size_bytes,
                        "base_size_bytes": base_size,
                        "record_indexes": [
                            int(record["record_index"]) for record in records
                        ],
                    }
                )
        if actual_indexes != expected_indexes:
            raise RuntimeError(f"Replacement record coverage mismatch: {table_id}")
        if len(actual_indexes) != len(set(actual_indexes)):
            raise RuntimeError(f"Duplicate replacement records: {table_id}")
        if len(point_inputs) > 1:
            chunk_indexes = [
                int(item.payload["retrieval_metadata"]["chunk_index"])
                for item in point_inputs
            ]
            chunk_counts = {
                int(item.payload["retrieval_metadata"]["chunk_count"])
                for item in point_inputs
            }
            if chunk_indexes != list(range(len(point_inputs))):
                raise RuntimeError(f"Invalid replacement chunk indexes: {table_id}")
            if chunk_counts != {len(point_inputs)}:
                raise RuntimeError(f"Invalid replacement chunk count: {table_id}")

        replacements.extend(point_inputs)
        table_summaries.append(
            {
                "table_id": table_id,
                "old_point_id": str(target.id),
                "old_evidence_id": old_metadata.get("evidence_id"),
                "new_evidence_id": point_inputs[0].payload[
                    "retrieval_metadata"
                ]["evidence_id"],
                "source_records": len(source_records),
                "replacement_points": len(point_inputs),
                "canonical_size_min": min(chunk_sizes),
                "canonical_size_max": max(chunk_sizes),
                "overflow_chunks": overflow_chunks,
                "overflow_details": overflow_details,
                "unique_contextual_texts": len(
                    {item.contextual_text for item in point_inputs}
                ),
            }
        )

    if len({item.id for item in replacements}) != len(replacements):
        raise RuntimeError("Replacement Point IDs must be globally unique")
    return replacements, table_summaries


def point_record_to_json(point: Any) -> dict[str, Any]:
    return {
        "id": str(point.id),
        "vector": point.vector,
        "payload": point.payload,
    }


def write_backup(
    client: QdrantClient,
    collection: str,
    targets: list[Any],
    *,
    timestamp: str,
) -> Path:
    target_ids = [str(point.id) for point in targets]
    records = client.retrieve(
        collection_name=collection,
        ids=target_ids,
        with_payload=True,
        with_vectors=True,
    )
    if {str(point.id) for point in records} != set(target_ids):
        raise RuntimeError("Could not retrieve every target Point for backup")
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    path = BACKUP_ROOT / f"large_r_tables_before_{timestamp}.json"
    path.write_text(
        json.dumps(
            {
                "collection": collection,
                "created_at": timestamp,
                "point_count": len(records),
                "points": [point_record_to_json(point) for point in records],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def verify_replacements(
    client: QdrantClient,
    collection: str,
    replacements: list[PointInput],
) -> None:
    expected_by_id = {item.id: item for item in replacements}
    records = client.retrieve(
        collection_name=collection,
        ids=list(expected_by_id),
        with_payload=True,
        with_vectors=True,
    )
    if {str(point.id) for point in records} != set(expected_by_id):
        raise RuntimeError("Not every replacement Point was retrieved after upsert")
    for point in records:
        expected = expected_by_id[str(point.id)]
        if point.payload != expected.payload:
            raise RuntimeError(f"Replacement payload mismatch: {point.id}")
        vector = point.vector
        if not isinstance(vector, dict):
            raise RuntimeError(f"Replacement named vector missing: {point.id}")
        dense = vector.get("evidence_dense")
        if not isinstance(dense, list) or len(dense) != 1024:
            raise RuntimeError(f"Replacement vector dimension mismatch: {point.id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_qdrant_schema(SCHEMA_PATH)
    client = client_from_env()
    try:
        all_r_table_points = scroll_r_table_points(client, schema.collection_name)
        targets = find_targets(all_r_table_points)
        if len(targets) != 31:
            raise RuntimeError(
                f"Expected exactly 31 current targets, found {len(targets)}"
            )
        target_table_ids = {
            str(point.payload["retrieval_metadata"]["table_id"])
            for point in targets
        }
        points_by_table: dict[str, list[str]] = defaultdict(list)
        for point in all_r_table_points:
            table_id = str(point.payload["retrieval_metadata"].get("table_id"))
            if table_id in target_table_ids:
                points_by_table[table_id].append(str(point.id))
        non_singleton_tables = {
            table_id: point_ids
            for table_id, point_ids in points_by_table.items()
            if len(point_ids) != 1
        }
        if non_singleton_tables:
            raise RuntimeError(
                "Target tables already have unexpected Point counts: "
                f"{non_singleton_tables}"
            )

        replacements, table_summaries = build_replacements(targets)
        old_ids = {str(point.id) for point in targets}
        replacement_ids = {item.id for item in replacements}
        collisions = client.retrieve(
            collection_name=schema.collection_name,
            ids=list(replacement_ids - old_ids),
            with_payload=True,
            with_vectors=False,
        )
        if collisions:
            raise RuntimeError(
                "Replacement Point IDs already exist outside the target set: "
                + ", ".join(str(point.id) for point in collisions)
            )

        summary = {
            "execute": args.execute,
            "target_tables": len(targets),
            "old_points": len(old_ids),
            "replacement_points": len(replacements),
            "replacement_unique_contexts": len(
                {item.embedding_cache_key for item in replacements}
            ),
            "evidence_id_changes": sum(
                item["old_evidence_id"] != item["new_evidence_id"]
                for item in table_summaries
            ),
            "overflow_chunks": sum(
                item["overflow_chunks"] for item in table_summaries
            ),
            "replacement_count_distribution": dict(
                sorted(
                    Counter(
                        item["replacement_points"] for item in table_summaries
                    ).items()
                )
            ),
            "tables": table_summaries,
        }
        if not args.execute:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = write_backup(
            client,
            schema.collection_name,
            targets,
            timestamp=timestamp,
        )
        embeddings = embed_point_inputs_batch(replacements)
        points = assemble_qdrant_points(replacements, embeddings)
        inserted = upsert_point_dicts(
            client,
            schema,
            points,
            batch_size=100,
        )
        if inserted != len(replacements):
            raise RuntimeError("Unexpected replacement upsert count")
        verify_replacements(client, schema.collection_name, replacements)

        old_ids_to_delete = sorted(old_ids - replacement_ids)
        if old_ids_to_delete:
            client.delete(
                collection_name=schema.collection_name,
                points_selector=models.PointIdsList(points=old_ids_to_delete),
                wait=True,
            )
        deleted_records = client.retrieve(
            collection_name=schema.collection_name,
            ids=old_ids_to_delete,
            with_payload=False,
            with_vectors=False,
        )
        if deleted_records:
            raise RuntimeError("Some replaced legacy Points were not deleted")
        verify_replacements(client, schema.collection_name, replacements)

        final_points = scroll_r_table_points(client, schema.collection_name)
        final_by_table: dict[str, set[str]] = defaultdict(set)
        for point in final_points:
            table_id = str(point.payload["retrieval_metadata"].get("table_id"))
            if table_id in target_table_ids:
                final_by_table[table_id].add(str(point.id))
        expected_by_table: dict[str, set[str]] = defaultdict(set)
        for item in replacements:
            expected_by_table[
                str(item.payload["retrieval_metadata"]["table_id"])
            ].add(item.id)
        if dict(final_by_table) != dict(expected_by_table):
            raise RuntimeError("Final target table Point sets do not match replacements")

        summary.update(
            {
                "backup_path": str(backup_path),
                "inserted": inserted,
                "deleted_old_points": len(old_ids_to_delete),
                "final_r_table_points": len(final_points),
            }
        )
        report_path = BACKUP_ROOT / f"large_r_tables_report_{timestamp}.json"
        report_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["report_path"] = str(report_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
