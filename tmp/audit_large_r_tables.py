from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


MAX_BYTES = 32 * 1024
DATA_ROOT = Path("data")
TABLE_ID_PATTERN = re.compile(
    r"^rtable:(?P<rcept_no>\d+):src(?P<source_index>\d+):s(?P<section_order>\d+):t\d+$"
)


def compact_size(value: object) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def main() -> None:
    load_dotenv()
    client = QdrantClient(
        host=os.environ["QDRANT_HOST"],
        port=int(os.environ["QDRANT_PORT"]),
    )
    points = []
    offset = None
    point_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="retrieval_metadata.point_kind",
                match=models.MatchValue(value="R_TABLE"),
            )
        ]
    )
    try:
        while True:
            page, offset = client.scroll(
                collection_name="dart_evidence",
                scroll_filter=point_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(page)
            if offset is None:
                break
    finally:
        client.close()

    targets = []
    for point in points:
        payload = point.payload or {}
        size_bytes = compact_size(payload.get("canonical"))
        if size_bytes <= MAX_BYTES:
            continue
        metadata = payload.get("retrieval_metadata", {})
        targets.append(
            {
                "point_id": str(point.id),
                "evidence_id": metadata.get("evidence_id"),
                "table_id": metadata.get("table_id"),
                "size_bytes": size_bytes,
                "row_start_index": metadata.get("row_start_index"),
                "row_end_index": metadata.get("row_end_index"),
            }
        )

    document_rows = {}
    with (DATA_ROOT / "manifest.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            document_rows[str(row["rcept_no"])] = row

    source_matches = []
    for target in targets:
        table_id = str(target["table_id"])
        match = TABLE_ID_PATTERN.fullmatch(table_id)
        if match is None:
            source_matches.append({**target, "source_status": "invalid_table_id"})
            continue
        values = match.groupdict()
        document = document_rows.get(values["rcept_no"])
        if document is None:
            source_matches.append(
                {**target, "source_status": "missing_document_manifest"}
            )
            continue
        fragment_path = (
            DATA_ROOT
            / "evidence_fragment"
            / str(document["doc_group"])
            / values["rcept_no"]
            / (
                f"src{values['source_index']}__"
                f"s{values['section_order']}.json"
            )
        )
        if not fragment_path.is_file():
            source_matches.append(
                {
                    **target,
                    "source_status": "missing_fragment",
                    "fragment_path": str(fragment_path),
                }
            )
            continue
        fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
        matches = [
            evidence
            for evidence in fragment.get("evidence_list", [])
            if evidence.get("payload", {}).get("table_id") == table_id
        ]
        if len(matches) != 1:
            source_matches.append(
                {
                    **target,
                    "source_status": "table_id_match_count",
                    "match_count": len(matches),
                    "fragment_path": str(fragment_path),
                }
            )
            continue
        evidence = matches[0]
        source_matches.append(
            {
                **target,
                "source_status": "matched",
                "source_evidence_id": evidence.get("evidence_id"),
                "source_table_type": evidence.get("table_type"),
                "fragment_path": str(fragment_path),
            }
        )

    print(
        json.dumps(
            {
                "r_table_points": len(points),
                "targets": len(targets),
                "distinct_tables": len(
                    {item["table_id"] for item in targets}
                ),
                "source_status_counts": {
                    status: sum(
                        item["source_status"] == status
                        for item in source_matches
                    )
                    for status in sorted(
                        {item["source_status"] for item in source_matches}
                    )
                },
                "items": sorted(
                    source_matches,
                    key=lambda item: (str(item["table_id"]), item["point_id"]),
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
