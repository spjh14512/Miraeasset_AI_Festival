from __future__ import annotations

from types import SimpleNamespace

import pytest

import vector_db.text2vector as text2vector


@pytest.fixture(autouse=True)
def embedding_environment(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "bge-m3")
    monkeypatch.setenv("VECTOR_DIMENSION", "1024")
    monkeypatch.setattr(text2vector, "load_dotenv", lambda: None)
    text2vector._load_model.cache_clear()
    yield


class _Model:
    def __init__(self, *, dimension=1024):
        self.dimension = dimension
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {
            "dense_vecs": [
                [float(index) for index in range(self.dimension)] for _ in texts
            ]
        }


def test_texts_to_vectors_uses_requested_local_batch_settings(monkeypatch):
    model = _Model()
    captured = {}

    def fake_load_model(model_name, model_cache_dir):
        captured["model_name"] = model_name
        captured["model_cache_dir"] = model_cache_dir
        return model

    monkeypatch.setattr(text2vector, "_load_model", fake_load_model)

    result = text2vector.texts_to_vectors(["첫 번째 문서", "두 번째 문서"])

    assert captured["model_name"] == "BAAI/bge-m3"
    assert captured["model_cache_dir"].endswith("data\\model_cache")
    assert len(result) == 2
    assert len(result[0]) == 1024
    assert model.calls == [
        (
            ["첫 번째 문서", "두 번째 문서"],
            {
                "batch_size": 32,
                "max_length": 3000,
                "return_dense": True,
                "return_sparse": False,
                "return_colbert_vecs": False,
            },
        )
    ]


def test_text_to_vector_preserves_single_text_api(monkeypatch):
    model = _Model()
    monkeypatch.setattr(
        text2vector,
        "_load_model",
        lambda model_name, model_cache_dir: model,
    )

    result = text2vector.text_to_vector("contextual text")

    assert len(result) == 1024
    assert model.calls[0][0] == ["contextual text"]


def test_build_model_requires_cuda(monkeypatch):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(text2vector, "_torch_module", lambda: torch)

    with pytest.raises(text2vector.LocalEmbeddingError, match="CUDA is required"):
        text2vector._build_model("BAAI/bge-m3", "D:/model-cache")


def test_build_model_uses_cuda_zero_and_fp16(monkeypatch):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    captured = {}

    class FakeModel:
        def __init__(self, model_name, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

    monkeypatch.setattr(text2vector, "_torch_module", lambda: torch)
    monkeypatch.setattr(text2vector, "_bge_m3_model_class", lambda: FakeModel)

    text2vector._build_model("BAAI/bge-m3", "D:/model-cache")

    assert captured == {
        "model_name": "BAAI/bge-m3",
        "kwargs": {
            "devices": "cuda:0",
            "use_fp16": True,
            "normalize_embeddings": True,
            "cache_dir": "D:/model-cache",
        },
    }


@pytest.mark.parametrize("value", ["", "   ", None, 123])
def test_rejects_empty_or_non_string_text(value):
    with pytest.raises(ValueError, match="non-empty string"):
        text2vector.text_to_vector(value)


@pytest.mark.parametrize("value", [[], "text", ["valid", ""], [123]])
def test_rejects_invalid_text_batches(value):
    with pytest.raises(ValueError):
        text2vector.texts_to_vectors(value)


def test_rejects_wrong_embedding_dimension(monkeypatch):
    monkeypatch.setattr(
        text2vector,
        "_load_model",
        lambda model_name, model_cache_dir: _Model(dimension=3),
    )

    with pytest.raises(text2vector.LocalEmbeddingError, match="expected 1024, got 3"):
        text2vector.text_to_vector("contextual text")


@pytest.mark.parametrize("dimension", ["invalid", "0", "-1"])
def test_rejects_invalid_configured_dimension(monkeypatch, dimension):
    monkeypatch.setenv("VECTOR_DIMENSION", dimension)

    with pytest.raises(text2vector.LocalEmbeddingError, match="positive integer"):
        text2vector.text_to_vector("contextual text")


def test_wraps_model_inference_error(monkeypatch):
    class BrokenModel:
        def encode(self, texts, **kwargs):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(
        text2vector,
        "_load_model",
        lambda model_name, model_cache_dir: BrokenModel(),
    )

    with pytest.raises(text2vector.LocalEmbeddingError, match="embedding failed"):
        text2vector.text_to_vector("contextual text")
