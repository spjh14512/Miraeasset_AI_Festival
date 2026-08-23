from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from qdrant_client import models

import vector_db.insert_points as insert_points
from vector_db.point_builder import PointInput


def _schema() -> insert_points.QdrantSchema:
    return insert_points.QdrantSchema(
        collection_name="dart_evidence",
        vector_name="evidence_dense",
        vector_dimension=1024,
        distance=models.Distance.COSINE,
        payload_indexes=(
            insert_points.PayloadIndex(
                "retrieval_metadata.point_kind",
                models.PayloadSchemaType.KEYWORD,
            ),
        ),
    )


class _Client:
    def __init__(self, *, exists=False):
        self.exists = exists
        self.created = []
        self.indexes = []
        self.upserts = []
        self.closed = False

    def collection_exists(self, collection_name):
        return self.exists

    def create_collection(self, **kwargs):
        self.created.append(kwargs)

    def create_payload_index(self, **kwargs):
        self.indexes.append(kwargs)

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def close(self):
        self.closed = True


def test_loads_current_qdrant_schema():
    schema = insert_points.load_qdrant_schema(
        Path("vector_db/qdrant_schema.yaml")
    )

    assert schema.collection_name == "dart_evidence"
    assert schema.vector_name == "evidence_dense"
    assert schema.vector_dimension == 1024
    assert schema.distance == models.Distance.COSINE
    assert insert_points.PayloadIndex(
        "retrieval_metadata.evidence_id",
        models.PayloadSchemaType.KEYWORD,
    ) in schema.payload_indexes


def test_ensure_collection_creates_named_vector_and_payload_indexes():
    client = _Client()
    schema = _schema()

    insert_points.ensure_collection(client, schema)

    assert client.created[0]["collection_name"] == "dart_evidence"
    vector = client.created[0]["vectors_config"]["evidence_dense"]
    assert vector.size == 1024
    assert vector.distance == models.Distance.COSINE
    assert client.indexes == [
        {
            "collection_name": "dart_evidence",
            "field_name": "retrieval_metadata.point_kind",
            "field_schema": models.PayloadSchemaType.KEYWORD,
            "wait": True,
        }
    ]


def test_existing_collection_is_not_recreated():
    client = _Client(exists=True)

    insert_points.ensure_collection(client, _schema())

    assert client.created == []
    assert len(client.indexes) == 1


def test_selected_disclosures_reuses_insert_dse_selection(monkeypatch):
    with tempfile.TemporaryDirectory(dir="tmp") as temp_dir:
        data_root = Path(temp_dir) / "data"
        data_root.mkdir()
        _assert_selected_disclosures_reuses_insert_dse_selection(
            data_root,
            monkeypatch,
        )


def _assert_selected_disclosures_reuses_insert_dse_selection(
    data_root,
    monkeypatch,
):
    (data_root / "manifest.jsonl").write_text(
        json.dumps(
            {
                "doc_group": "periodic",
                "rcept_no": "20250000000001",
                "corp_name": "회사",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    section = {"doc_group": "periodic", "rcept_no": "20250000000001"}
    evidence = {"doc_group": "periodic", "rcept_no": "20250000000001"}
    captured = {}

    def fake_select(root, *, limit, random_seed):
        captured.update(root=root, limit=limit, random_seed=random_seed)
        return [(section, evidence)]

    monkeypatch.setattr(insert_points, "select_disclosures", fake_select)

    selected = insert_points.selected_disclosures(
        data_root,
        limit=1,
        random_seed=37,
    )

    assert captured == {"root": data_root, "limit": 1, "random_seed": 37}
    assert selected[0][:2] == (section, evidence)
    assert selected[0][2]["corp_name"] == "회사"


def test_batch_embedding_deduplicates_contextual_text():
    first = PointInput("p1", "동일 문서", "cache-key", {})
    second = PointInput("p2", "동일 문서", "cache-key", {})
    captured = []

    def vectorizer(texts):
        captured.append(texts)
        return [[0.5, 0.25] for _ in texts]

    result = insert_points.embed_point_inputs_batch(
        [first, second],
        vectorizer=vectorizer,
    )

    assert captured == [["동일 문서"]]
    assert result == {"cache-key": [0.5, 0.25]}


def test_upsert_uses_named_vectors_and_requested_batch_size():
    client = _Client(exists=True)
    points = [
        {
            "id": f"00000000-0000-0000-0000-00000000000{index}",
            "vector": {"evidence_dense": [0.0, 1.0]},
            "payload": {"order": index},
        }
        for index in range(3)
    ]

    inserted = insert_points.upsert_point_dicts(
        client,
        _schema(),
        points,
        batch_size=2,
    )

    assert inserted == 3
    assert [len(call["points"]) for call in client.upserts] == [2, 1]
    assert client.upserts[0]["collection_name"] == "dart_evidence"
    assert client.upserts[0]["wait"] is True
    assert client.upserts[0]["points"][0].vector == {
        "evidence_dense": [0.0, 1.0]
    }


@pytest.mark.parametrize("batch_size", [0, -1])
def test_upsert_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch-size"):
        insert_points.upsert_point_dicts(
            _Client(),
            _schema(),
            [],
            batch_size=batch_size,
        )
