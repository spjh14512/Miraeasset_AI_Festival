from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import models

import vector_db.insert_points as insert_points
from vector_db.point_builder import PointInput
from vector_db.text2vector import HybridEmbedding, SparseEmbedding


def _schema() -> insert_points.QdrantSchema:
    return insert_points.QdrantSchema(
        collection_name="dart_evidence",
        vector_name="evidence_dense",
        sparse_vector_name="evidence_sparse",
        vector_dimension=1024,
        distance=models.Distance.COSINE,
        payload_indexes=(
            insert_points.PayloadIndex(
                "point_kind",
                models.PayloadSchemaType.KEYWORD,
            ),
        ),
    )


class _Client:
    def __init__(self, *, exists=False, sparse_exists=True):
        self.exists = exists
        self.sparse_exists = sparse_exists
        self.created = []
        self.created_vector_names = []
        self.indexes = []
        self.upserts = []
        self.closed = False

    def collection_exists(self, collection_name):
        return self.exists

    def create_collection(self, **kwargs):
        self.created.append(kwargs)

    def get_collection(self, collection_name):
        sparse_vectors = (
            {"evidence_sparse": models.SparseVectorParams()}
            if self.sparse_exists
            else {}
        )
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(sparse_vectors=sparse_vectors)
            )
        )

    def create_vector_name(self, **kwargs):
        self.created_vector_names.append(kwargs)

    def create_payload_index(self, **kwargs):
        self.indexes.append(kwargs)

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def close(self):
        self.closed = True


def test_main_skips_thermal_guard_when_cuda_is_unavailable(monkeypatch, capsys):
    args = SimpleNamespace(
        schema=Path("schema.yaml"),
        disable_gpu_thermal_guard=False,
        gpu_pause_temperature=82,
        gpu_resume_temperature=72,
        gpu_temperature_poll_seconds=10.0,
        data_root=Path("data"),
        limit=None,
        random_seed=None,
        batch_size=128,
        embedding_buffer_size=512,
        dry_run=True,
        resume_from=None,
    )
    captured = {}
    monkeypatch.setattr(insert_points, "parse_args", lambda: args)
    monkeypatch.setattr(insert_points, "load_qdrant_schema", lambda path: _schema())
    monkeypatch.setattr(insert_points, "cuda_is_available", lambda: False)
    monkeypatch.setattr(
        insert_points,
        "GpuThermalGuard",
        lambda **kwargs: pytest.fail("thermal guard must not be created"),
    )

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(insert_points, "run", fake_run)

    assert insert_points.main() == 0
    assert captured["thermal_guard"] is None
    assert "using CPU embedding" in capsys.readouterr().out


def test_loads_current_qdrant_schema():
    schema = insert_points.load_qdrant_schema(
        Path("vector_db/qdrant_schema.yaml")
    )

    assert schema.collection_name == "dart_evidence"
    assert schema.vector_name == "evidence_dense"
    assert schema.sparse_vector_name == "evidence_sparse"
    assert schema.vector_dimension == 1024
    assert schema.distance == models.Distance.COSINE
    assert insert_points.PayloadIndex(
        "evidence_id",
        models.PayloadSchemaType.KEYWORD,
    ) in schema.payload_indexes
    assert insert_points.PayloadIndex(
        "disclosure_id",
        models.PayloadSchemaType.KEYWORD,
    ) in schema.payload_indexes
    assert insert_points.PayloadIndex(
        "section_id",
        models.PayloadSchemaType.KEYWORD,
    ) in schema.payload_indexes
    assert insert_points.PayloadIndex(
        "is_latest_version",
        models.PayloadSchemaType.BOOL,
    ) in schema.payload_indexes


def test_document_context_includes_latest_version_flag():
    context = insert_points._document_context(
        {
            "corp_name": "회사",
            "report_nm": "사업보고서",
            "rcept_dt": "20250318",
        },
        is_latest_version=False,
    )

    assert context["is_latest_version"] is False


def test_document_context_rejects_non_boolean_latest_version_flag():
    with pytest.raises(ValueError, match="is_latest_version must be a boolean"):
        insert_points._document_context({}, is_latest_version=1)


def test_ensure_collection_creates_named_vector_and_payload_indexes():
    client = _Client()
    schema = _schema()

    insert_points.ensure_collection(client, schema)

    assert client.created[0]["collection_name"] == "dart_evidence"
    vector = client.created[0]["vectors_config"]["evidence_dense"]
    assert vector.size == 1024
    assert vector.distance == models.Distance.COSINE
    assert "evidence_sparse" in client.created[0]["sparse_vectors_config"]
    assert client.indexes == [
        {
            "collection_name": "dart_evidence",
            "field_name": "point_kind",
            "field_schema": models.PayloadSchemaType.KEYWORD,
            "wait": True,
        }
    ]


def test_existing_collection_is_not_recreated():
    client = _Client(exists=True)

    insert_points.ensure_collection(client, _schema())

    assert client.created == []
    assert client.created_vector_names == []
    assert len(client.indexes) == 1


def test_existing_collection_adds_missing_sparse_vector_name():
    client = _Client(exists=True, sparse_exists=False)

    insert_points.ensure_collection(client, _schema())

    assert len(client.created_vector_names) == 1
    call = client.created_vector_names[0]
    assert call["collection_name"] == "dart_evidence"
    assert call["vector_name"] == "evidence_sparse"
    assert isinstance(
        call["vector_name_config"],
        models.SparseVectorNameConfig,
    )


def test_selected_disclosures_reuses_insert_dse_selection(monkeypatch):
    with tempfile.TemporaryDirectory(dir="tmp") as temp_dir:
        data_root = Path(temp_dir) / "data"
        data_root.mkdir()
        _assert_selected_disclosures_reuses_insert_dse_selection(
            data_root,
            monkeypatch,
        )


def test_all_disclosures_selects_every_complete_output():
    with tempfile.TemporaryDirectory(dir="tmp") as temp_dir:
        data_root = Path(temp_dir) / "data"
        canonical_root = data_root / "canonical_section"
        evidence_root = data_root / "evidence_fragment"
        canonical_root.mkdir(parents=True)
        evidence_root.mkdir(parents=True)
        (canonical_root / "document.json").write_text("{}", encoding="utf-8")
        (evidence_root / "fragment.json").write_text("{}", encoding="utf-8")
        row_key = {"doc_group": "periodic", "rcept_no": "20250000000001"}
        (canonical_root / "manifest.jsonl").write_text(
            json.dumps({
                **row_key,
                "status": "SUCCESS",
                "output_path": "canonical_section/document.json",
            }) + "\n",
            encoding="utf-8",
        )
        (evidence_root / "manifest.jsonl").write_text(
            json.dumps({
                **row_key,
                "status": "SUCCESS",
                "output_paths": ["evidence_fragment/fragment.json"],
            }) + "\n",
            encoding="utf-8",
        )
        (data_root / "manifest.jsonl").write_text(
            json.dumps({**row_key, "corp_name": "회사"}) + "\n",
            encoding="utf-8",
        )

        selected = insert_points.all_disclosures(data_root)

        assert len(selected) == 1
        assert selected[0][0]["rcept_no"] == "20250000000001"
        assert selected[0][2]["corp_name"] == "회사"


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
        return [
            HybridEmbedding(
                dense=(0.5, 0.25),
                sparse=SparseEmbedding(indices=(1,), values=(0.8,)),
            )
            for _ in texts
        ]

    result = insert_points.embed_point_inputs_batch(
        [first, second],
        vectorizer=vectorizer,
    )

    assert captured == [["동일 문서"]]
    assert result == {
        "cache-key": HybridEmbedding(
            dense=(0.5, 0.25),
            sparse=SparseEmbedding(indices=(1,), values=(0.8,)),
        )
    }


def _point_input(index: int, text: str) -> PointInput:
    return PointInput(
        id=f"00000000-0000-0000-0000-{index:012d}",
        contextual_text=text,
        embedding_cache_key=f"cache-{index}",
        payload={"point_kind": "TEXT", "order": index},
    )


def _hybrid_embedding(index: int) -> HybridEmbedding:
    return HybridEmbedding(
        dense=(float(index),),
        sparse=SparseEmbedding(indices=(index + 1,), values=(0.5,)),
    )


def test_batch_embedding_sorts_unique_texts_by_token_length():
    captured = []

    def vectorizer(texts):
        captured.append(list(texts))
        return [_hybrid_embedding(index) for index, _ in enumerate(texts)]

    result = insert_points.embed_point_inputs_batch(
        [
            _point_input(1, "long text"),
            _point_input(2, "x"),
            _point_input(3, "medium"),
        ],
        vectorizer=vectorizer,
        token_counter=len,
    )

    assert captured == [["x", "medium", "long text"]]
    assert set(result) == {"cache-1", "cache-2", "cache-3"}


def test_ingestion_batches_across_fragment_boundaries_and_flushes_remainders():
    client = _Client(exists=True)
    vectorizer_calls = []

    def vectorizer(texts):
        vectorizer_calls.append(list(texts))
        return [_hybrid_embedding(index) for index, _ in enumerate(texts)]

    inserted = insert_points.ingest_point_inputs(
        (_point_input(index, str(7 - index)) for index in range(7)),
        client=client,
        schema=_schema(),
        embedding_buffer_size=4,
        upload_batch_size=3,
        vectorizer=vectorizer,
        token_counter=lambda text: int(text),
    )

    assert inserted == 7
    assert vectorizer_calls == [["4", "5", "6", "7"], ["1", "2", "3"]]
    assert [len(call["points"]) for call in client.upserts] == [3, 3, 1]
    assert [
        point.payload["order"]
        for call in client.upserts
        for point in call["points"]
    ] == list(range(7))
    assert [
        point.vector["evidence_dense"]
        for call in client.upserts
        for point in call["points"]
    ] == [[3.0], [2.0], [1.0], [0.0], [2.0], [1.0], [0.0]]


def test_ingestion_checks_thermal_guard_before_each_embedding_window():
    client = _Client(exists=True)

    class ThermalGuard:
        def __init__(self):
            self.calls = 0

        def wait_until_ready(self):
            self.calls += 1

    thermal_guard = ThermalGuard()
    inserted = insert_points.ingest_point_inputs(
        [_point_input(index, str(index)) for index in range(5)],
        client=client,
        schema=_schema(),
        embedding_buffer_size=2,
        upload_batch_size=5,
        vectorizer=lambda texts: [
            _hybrid_embedding(index) for index, _ in enumerate(texts)
        ],
        token_counter=int,
        thermal_guard=thermal_guard,
    )

    assert inserted == 5
    assert thermal_guard.calls == 3


def test_ingestion_propagates_background_writer_errors():
    class FailingClient(_Client):
        def upsert(self, **kwargs):
            raise RuntimeError("upload failed")

    with pytest.raises(RuntimeError, match="upload failed"):
        insert_points.ingest_point_inputs(
            [_point_input(1, "text")],
            client=FailingClient(exists=True),
            schema=_schema(),
            embedding_buffer_size=1,
            upload_batch_size=1,
            vectorizer=lambda texts: [_hybrid_embedding(0)],
            token_counter=len,
        )


@pytest.mark.parametrize(
    ("embedding_buffer_size", "upload_batch_size", "message"),
    [
        (0, 1, "embedding-buffer-size"),
        (1, 0, "batch-size"),
    ],
)
def test_ingestion_rejects_invalid_buffer_sizes(
    embedding_buffer_size,
    upload_batch_size,
    message,
):
    with pytest.raises(ValueError, match=message):
        insert_points.ingest_point_inputs(
            [],
            client=_Client(),
            schema=_schema(),
            embedding_buffer_size=embedding_buffer_size,
            upload_batch_size=upload_batch_size,
            vectorizer=lambda texts: [],
            token_counter=len,
        )


def test_upsert_uses_named_vectors_and_requested_batch_size():
    client = _Client(exists=True)
    points = [
        {
            "id": f"00000000-0000-0000-0000-00000000000{index}",
            "vector": {
                "evidence_dense": [0.0, 1.0],
                "evidence_sparse": {"indices": [1], "values": [0.5]},
            },
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
        "evidence_dense": [0.0, 1.0],
        "evidence_sparse": models.SparseVector(indices=[1], values=[0.5]),
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
