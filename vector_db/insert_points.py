"""Build Evidence Points and insert a reproducible disclosure sample into Qdrant."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models

from knowledge_graph.insertDSE import (
    DISCLOSURE_GROUPS,
    read_jsonl,
    select_disclosures,
    valid_manifest_rows,
)
from vector_db.point_builder import (
    SPARSE_VECTOR_NAME,
    VECTOR_NAME,
    PointInput,
    assemble_qdrant_points,
    build_point_inputs,
)
from vector_db.text2vector import HybridEmbedding, texts_to_hybrid_vectors


BatchVectorizer = Callable[[Sequence[str]], list[HybridEmbedding]]
LatestVersionLookup = Callable[[Sequence[str]], dict[str, bool]]


@dataclass(frozen=True, slots=True)
class PayloadIndex:
    field_name: str
    field_schema: models.PayloadSchemaType


@dataclass(frozen=True, slots=True)
class QdrantSchema:
    collection_name: str
    vector_name: str
    sparse_vector_name: str
    vector_dimension: int
    distance: models.Distance
    payload_indexes: tuple[PayloadIndex, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and insert sampled Evidence Points into Qdrant."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--limit", type=int)
    selection.add_argument(
        "--all",
        action="store_true",
        help="Insert every disclosure with complete Canonical and Evidence outputs.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Use the same value as insertDSE.py to select the same disclosures.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("vector_db/qdrant_schema.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _required_mapping(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = source.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"qdrant schema requires a mapping at {key}")
    return value


def _required_text(source: Mapping[str, Any], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"qdrant schema requires non-empty text at {key}")
    return value.strip()


def load_qdrant_schema(path: Path) -> QdrantSchema:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("qdrant schema must contain a mapping")

    collection = _required_mapping(raw, "collection")
    point = _required_mapping(raw, "point")
    vectors = _required_mapping(point, "vectors")
    dense = _required_mapping(vectors, "dense")
    sparse = _required_mapping(vectors, "sparse")

    collection_name = _required_text(collection, "name")
    vector_name = _required_text(dense, "name")
    if vector_name != VECTOR_NAME:
        raise ValueError(
            f"Point builder vector name mismatch: {VECTOR_NAME} != {vector_name}"
        )
    sparse_vector_name = _required_text(sparse, "name")
    if sparse_vector_name != SPARSE_VECTOR_NAME:
        raise ValueError(
            "Point builder sparse vector name mismatch: "
            f"{SPARSE_VECTOR_NAME} != {sparse_vector_name}"
        )

    vector_dimension = dense.get("dimension")
    if not isinstance(vector_dimension, int) or isinstance(vector_dimension, bool):
        raise ValueError("qdrant vector dimension must be an integer")
    if vector_dimension < 1:
        raise ValueError("qdrant vector dimension must be positive")

    distance_name = _required_text(dense, "distance").upper()
    distance_by_name = {
        "COSINE": models.Distance.COSINE,
        "DOT": models.Distance.DOT,
        "EUCLID": models.Distance.EUCLID,
        "MANHATTAN": models.Distance.MANHATTAN,
    }
    if distance_name not in distance_by_name:
        raise ValueError(f"Unsupported Qdrant distance: {distance_name}")

    payload = _required_mapping(raw, "payload")
    payload_fields = _required_mapping(payload, "fields")
    payload_type_map = {
        "keyword": models.PayloadSchemaType.KEYWORD,
        "integer": models.PayloadSchemaType.INTEGER,
        "boolean": models.PayloadSchemaType.BOOL,
    }
    indexes: list[PayloadIndex] = []

    def collect_indexes(fields: Mapping[str, Any]) -> None:
        for field_name, definition in fields.items():
            if not isinstance(definition, Mapping):
                raise ValueError(f"Invalid payload field definition: {field_name}")
            nested_fields = definition.get("fields")
            if nested_fields is not None:
                if not isinstance(nested_fields, Mapping):
                    raise ValueError(f"Invalid nested payload fields: {field_name}")
                collect_indexes(nested_fields)
            if not definition.get("payload_index"):
                continue
            field_type = definition.get("type")
            index_path = definition.get("index_path")
            if field_type not in payload_type_map or not isinstance(index_path, str):
                raise ValueError(f"Unsupported payload index definition: {field_name}")
            indexes.append(
                PayloadIndex(
                    field_name=index_path,
                    field_schema=payload_type_map[field_type],
                )
            )

    collect_indexes(payload_fields)

    return QdrantSchema(
        collection_name=collection_name,
        vector_name=vector_name,
        sparse_vector_name=sparse_vector_name,
        vector_dimension=vector_dimension,
        distance=distance_by_name[distance_name],
        payload_indexes=tuple(indexes),
    )


def ensure_collection(client: QdrantClient, schema: QdrantSchema) -> None:
    """Create the collection and declared payload indexes without recreating data."""
    if not client.collection_exists(schema.collection_name):
        client.create_collection(
            collection_name=schema.collection_name,
            vectors_config={
                schema.vector_name: models.VectorParams(
                    size=schema.vector_dimension,
                    distance=schema.distance,
                )
            },
            sparse_vectors_config={
                schema.sparse_vector_name: models.SparseVectorParams()
            },
        )
    else:
        collection = client.get_collection(schema.collection_name)
        config = getattr(collection, "config", None)
        params = getattr(config, "params", None)
        sparse_vectors = getattr(params, "sparse_vectors", None)
        sparse_vector_names = (
            set(sparse_vectors)
            if isinstance(sparse_vectors, Mapping)
            else set()
        )
        if schema.sparse_vector_name not in sparse_vector_names:
            client.create_vector_name(
                collection_name=schema.collection_name,
                vector_name=schema.sparse_vector_name,
                vector_name_config=models.SparseVectorNameConfig(
                    sparse=models.SparseVectorConfig(),
                ),
                wait=True,
            )

    for payload_index in schema.payload_indexes:
        client.create_payload_index(
            collection_name=schema.collection_name,
            field_name=payload_index.field_name,
            field_schema=payload_index.field_schema,
            wait=True,
        )


def _document_manifest_rows(data_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(data_root / "manifest.jsonl"):
        key = (str(row.get("doc_group", "")), str(row.get("rcept_no", "")))
        if all(key):
            rows[key] = row
    return rows


def selected_disclosures(
    data_root: Path,
    *,
    limit: int,
    random_seed: int | None,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Reuse insertDSE sampling and attach each selected document manifest."""
    selected = select_disclosures(
        data_root,
        limit=limit,
        random_seed=random_seed,
    )
    document_rows = _document_manifest_rows(data_root)
    result: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for section_manifest, evidence_manifest in selected:
        key = (
            str(section_manifest["doc_group"]),
            str(section_manifest["rcept_no"]),
        )
        document_manifest = document_rows.get(key)
        if document_manifest is None:
            raise ValueError(
                "Selected disclosure is missing from data/manifest.jsonl: "
                f"{key[0]}/{key[1]}"
            )
        result.append((section_manifest, evidence_manifest, document_manifest))
    return result


def all_disclosures(
    data_root: Path,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Select every complete disclosure in stable group/receipt order."""
    section_rows = valid_manifest_rows(
        data_root / "canonical_section" / "manifest.jsonl"
    )
    evidence_rows = valid_manifest_rows(
        data_root / "evidence_fragment" / "manifest.jsonl"
    )
    document_rows = _document_manifest_rows(data_root)
    selected: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for doc_group, rcept_no in sorted(section_rows.keys() & evidence_rows.keys()):
        if doc_group not in DISCLOSURE_GROUPS:
            continue
        section_manifest = section_rows[(doc_group, rcept_no)]
        evidence_manifest = evidence_rows[(doc_group, rcept_no)]
        section_path = data_root / str(section_manifest.get("output_path", ""))
        evidence_paths = [
            data_root / str(path)
            for path in evidence_manifest.get("output_paths", [])
        ]
        if (
            not section_path.is_file()
            or not evidence_paths
            or not all(path.is_file() for path in evidence_paths)
        ):
            continue
        document_manifest = document_rows.get((doc_group, rcept_no))
        if document_manifest is None:
            raise ValueError(
                "Complete disclosure is missing from data/manifest.jsonl: "
                f"{doc_group}/{rcept_no}"
            )
        selected.append(
            (section_manifest, evidence_manifest, document_manifest)
        )
    return selected


def _document_context(
    document_manifest: Mapping[str, Any],
    *,
    is_latest_version: bool,
) -> dict[str, Any]:
    if not isinstance(is_latest_version, bool):
        raise ValueError("is_latest_version must be a boolean")
    return {
        "corp_name": document_manifest.get("corp_name"),
        "report_nm": document_manifest.get("report_nm"),
        "rcept_date": document_manifest.get("rcept_dt"),
        "is_latest_version": is_latest_version,
    }


def latest_version_statuses_from_neo4j(
    disclosure_ids: Sequence[str],
) -> dict[str, bool]:
    """Read authoritative latest-version flags for Qdrant payload creation."""
    unique_ids = sorted(set(disclosure_ids))
    if not unique_ids:
        return {}

    load_dotenv()
    uri = os.getenv("NEO4J_URI", "").strip()
    username = os.getenv("NEO4J_USERNAME", "").strip()
    password = os.getenv("NEO4J_PASSWORD", "").strip()
    database = os.getenv("NEO4J_DATABASE", "").strip() or None
    if not all((uri, username, password)):
        raise RuntimeError(
            "NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must be configured"
        )

    statuses: dict[str, bool] = {}
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            for id_batch in _chunks(unique_ids, 1000):
                result = session.run(
                    """
                    UNWIND $ids AS id
                    MATCH (disclosure:Disclosure {id: id})
                    RETURN disclosure.id AS id,
                           disclosure.is_latest_version AS is_latest_version
                    """,
                    ids=list(id_batch),
                )
                for record in result:
                    disclosure_id = record["id"]
                    is_latest_version = record["is_latest_version"]
                    if not isinstance(disclosure_id, str):
                        raise ValueError("Neo4j Disclosure.id must be a string")
                    if not isinstance(is_latest_version, bool):
                        raise ValueError(
                            "Neo4j Disclosure.is_latest_version must be a boolean: "
                            f"{disclosure_id}"
                        )
                    statuses[disclosure_id] = is_latest_version

    missing_ids = sorted(set(unique_ids) - statuses.keys())
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        suffix = " ..." if len(missing_ids) > 10 else ""
        raise ValueError(
            f"Neo4j is missing {len(missing_ids)} selected Disclosure nodes: "
            f"{preview}{suffix}"
        )
    return statuses


def iter_fragment_point_inputs(
    data_root: Path,
    section_manifest: Mapping[str, Any],
    evidence_manifest: Mapping[str, Any],
    document_manifest: Mapping[str, Any],
    *,
    is_latest_version: bool,
) -> Iterator[tuple[Path, list[PointInput]]]:
    section_document = json.loads(
        (data_root / str(section_manifest["output_path"])).read_text(
            encoding="utf-8"
        )
    )
    sections = section_document.get("sections")
    if not isinstance(sections, list):
        raise ValueError("Canonical section document requires a sections list")
    section_by_id = {
        str(section["section_id"]): section
        for section in sections
        if isinstance(section, Mapping) and section.get("section_id")
    }
    document_context = _document_context(
        document_manifest,
        is_latest_version=is_latest_version,
    )

    for output_path in evidence_manifest.get("output_paths", []):
        fragment_path = data_root / str(output_path)
        fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
        section_id = str(fragment.get("section_id", ""))
        section_context = section_by_id.get(section_id)
        if section_context is None:
            raise ValueError(f"Evidence references unknown section: {section_id}")
        yield fragment_path, build_point_inputs(
            fragment,
            document_context=document_context,
            section_context=section_context,
        )


def embed_point_inputs_batch(
    point_inputs: Sequence[PointInput],
    *,
    vectorizer: BatchVectorizer = texts_to_hybrid_vectors,
) -> dict[str, HybridEmbedding]:
    """Embed unique contextual texts with the local batch vectorizer."""
    unique_texts: dict[str, str] = {}
    for point_input in point_inputs:
        if not isinstance(point_input, PointInput):
            raise ValueError("point_inputs must contain PointInput items")
        existing = unique_texts.get(point_input.embedding_cache_key)
        if existing is not None and existing != point_input.contextual_text:
            raise ValueError("Embedding cache key collision detected")
        unique_texts.setdefault(
            point_input.embedding_cache_key,
            point_input.contextual_text,
        )

    keys = list(unique_texts)
    if not keys:
        return {}
    vectors = vectorizer([unique_texts[key] for key in keys])
    if not isinstance(vectors, list) or len(vectors) != len(keys):
        raise ValueError("Batch vectorizer returned an unexpected vector count")
    return dict(zip(keys, vectors, strict=True))


def _chunks(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    if size < 1:
        raise ValueError("--batch-size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def upsert_point_dicts(
    client: QdrantClient,
    schema: QdrantSchema,
    points: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> int:
    inserted = 0
    for point_batch in _chunks(points, batch_size):
        client.upsert(
            collection_name=schema.collection_name,
            wait=True,
            points=[
                models.PointStruct(
                    id=point["id"],
                    vector=point["vector"],
                    payload=point["payload"],
                )
                for point in point_batch
            ],
        )
        inserted += len(point_batch)
    return inserted


def _qdrant_client_from_env() -> QdrantClient:
    load_dotenv()
    host = os.getenv("QDRANT_HOST", "").strip()
    port_text = os.getenv("QDRANT_PORT", "").strip()
    if not host or not port_text:
        raise RuntimeError("QDRANT_HOST and QDRANT_PORT must be configured")
    try:
        port = int(port_text)
    except ValueError as error:
        raise RuntimeError("QDRANT_PORT must be an integer") from error
    if not 1 <= port <= 65535:
        raise RuntimeError("QDRANT_PORT must be between 1 and 65535")
    return QdrantClient(host=host, port=port)


def run(
    *,
    data_root: Path,
    schema: QdrantSchema,
    limit: int | None,
    random_seed: int | None,
    batch_size: int,
    dry_run: bool,
    client: QdrantClient | None = None,
    vectorizer: BatchVectorizer = texts_to_hybrid_vectors,
    latest_version_lookup: LatestVersionLookup = latest_version_statuses_from_neo4j,
) -> dict[str, Any]:
    selected = (
        all_disclosures(data_root)
        if limit is None
        else selected_disclosures(
            data_root,
            limit=limit,
            random_seed=random_seed,
        )
    )
    disclosure_ids = [f"d{item[0]['rcept_no']}" for item in selected]
    latest_version_statuses = latest_version_lookup(disclosure_ids)
    resolved_client = client
    owns_client = False
    if not dry_run:
        if resolved_client is None:
            resolved_client = _qdrant_client_from_env()
            owns_client = True
        ensure_collection(resolved_client, schema)

    point_counts: Counter[str] = Counter()
    fragment_count = 0
    inserted = 0
    try:
        for section_manifest, evidence_manifest, document_manifest in selected:
            disclosure_id = f"d{section_manifest['rcept_no']}"
            for _, point_inputs in iter_fragment_point_inputs(
                data_root,
                section_manifest,
                evidence_manifest,
                document_manifest,
                is_latest_version=latest_version_statuses[disclosure_id],
            ):
                fragment_count += 1
                for point_input in point_inputs:
                    point_counts[str(point_input.payload["point_kind"])] += 1
                if dry_run or not point_inputs:
                    continue
                embeddings = embed_point_inputs_batch(
                    point_inputs,
                    vectorizer=vectorizer,
                )
                points = assemble_qdrant_points(point_inputs, embeddings)
                inserted += upsert_point_dicts(
                    resolved_client,
                    schema,
                    points,
                    batch_size=batch_size,
                )
    finally:
        if owns_client and resolved_client is not None:
            resolved_client.close()

    rcept_nos = [str(item[0]["rcept_no"]) for item in selected]
    return {
        "dry_run": dry_run,
        "disclosures": len(selected),
        "fragments": fragment_count,
        "points": sum(point_counts.values()),
        "inserted": inserted,
        "point_kinds": dict(sorted(point_counts.items())),
        "rcept_nos": rcept_nos,
    }


def main() -> int:
    args = parse_args()
    schema = load_qdrant_schema(args.schema)
    result = run(
        data_root=args.data_root,
        schema=schema,
        limit=args.limit,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
