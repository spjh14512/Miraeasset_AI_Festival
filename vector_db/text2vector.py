"""Convert contextual text to dense vectors with local BGE-M3 inference."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


MODEL_NAME_ENV = "MODEL_NAME"
VECTOR_DIMENSION_ENV = "VECTOR_DIMENSION"
MODEL_CACHE_DIR_ENV = "EMBEDDING_MODEL_CACHE_DIR"

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_MODEL_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "model_cache"
CUDA_DEVICE = "cuda:0"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MAX_LENGTH = 3000


class LocalEmbeddingError(RuntimeError):
    """Raised when local BGE-M3 inference cannot produce valid embeddings."""


@dataclass(frozen=True, slots=True)
class _EmbeddingSettings:
    model_name: str
    vector_dimension: int
    model_cache_dir: str


def _required_environment(name: str) -> str:
    value = os.getenv(name)
    if not isinstance(value, str) or not value.strip():
        raise LocalEmbeddingError(f"Required environment variable is missing: {name}")
    return value.strip()


def _load_settings() -> _EmbeddingSettings:
    load_dotenv()

    model_name = _required_environment(MODEL_NAME_ENV)
    if model_name.casefold() == "bge-m3":
        model_name = DEFAULT_MODEL_NAME

    dimension_text = _required_environment(VECTOR_DIMENSION_ENV)
    try:
        vector_dimension = int(dimension_text)
    except ValueError as error:
        raise LocalEmbeddingError(
            f"{VECTOR_DIMENSION_ENV} must be a positive integer"
        ) from error
    if vector_dimension <= 0:
        raise LocalEmbeddingError(
            f"{VECTOR_DIMENSION_ENV} must be a positive integer"
        )

    cache_dir_text = os.getenv(MODEL_CACHE_DIR_ENV, "").strip()
    cache_dir = Path(cache_dir_text).expanduser() if cache_dir_text else DEFAULT_MODEL_CACHE_DIR
    if not cache_dir.is_absolute():
        cache_dir = Path(__file__).resolve().parents[1] / cache_dir

    return _EmbeddingSettings(
        model_name=model_name,
        vector_dimension=vector_dimension,
        model_cache_dir=str(cache_dir.resolve()),
    )


def _torch_module() -> Any:
    try:
        import torch
    except ImportError as error:
        raise LocalEmbeddingError("PyTorch is required for local embedding") from error
    return torch


def _bge_m3_model_class() -> Any:
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as error:
        raise LocalEmbeddingError(
            "FlagEmbedding is required for local embedding"
        ) from error
    return BGEM3FlagModel


def _build_model(model_name: str, model_cache_dir: str) -> Any:
    torch = _torch_module()
    if not torch.cuda.is_available():
        raise LocalEmbeddingError(
            f"CUDA is required but unavailable for device {CUDA_DEVICE}"
        )

    model_class = _bge_m3_model_class()
    try:
        return model_class(
            model_name,
            devices=CUDA_DEVICE,
            use_fp16=True,
            normalize_embeddings=True,
            cache_dir=model_cache_dir,
        )
    except Exception as error:
        raise LocalEmbeddingError(
            f"Failed to load local embedding model: {model_name}"
        ) from error


@lru_cache(maxsize=1)
def _load_model(model_name: str, model_cache_dir: str) -> Any:
    """Load BGE-M3 once and reuse it across embedding batches."""
    return _build_model(model_name, model_cache_dir)


def _validated_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise ValueError("texts must be a non-empty sequence of strings")

    result = list(texts)
    if not result:
        raise ValueError("texts must be a non-empty sequence of strings")
    if any(not isinstance(text, str) or not text.strip() for text in result):
        raise ValueError("each text must be a non-empty string")
    return result


def _parse_dense_vectors(
    result: Any,
    *,
    expected_count: int,
    expected_dimension: int,
) -> list[list[float]]:
    if not isinstance(result, dict):
        raise LocalEmbeddingError("FlagEmbedding returned an invalid result")

    dense_vectors = result.get("dense_vecs")
    if hasattr(dense_vectors, "tolist"):
        dense_vectors = dense_vectors.tolist()
    if not isinstance(dense_vectors, list) or len(dense_vectors) != expected_count:
        raise LocalEmbeddingError(
            "FlagEmbedding returned an unexpected number of dense vectors"
        )

    parsed: list[list[float]] = []
    for vector in dense_vectors:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if not isinstance(vector, list) or not all(
            isinstance(value, Real) and not isinstance(value, bool)
            for value in vector
        ):
            raise LocalEmbeddingError(
                "FlagEmbedding dense_vecs must contain numeric arrays"
            )
        if len(vector) != expected_dimension:
            raise LocalEmbeddingError(
                "FlagEmbedding returned an unexpected embedding dimension: "
                f"expected {expected_dimension}, got {len(vector)}"
            )
        parsed.append([float(value) for value in vector])
    return parsed


def texts_to_vectors(texts: Sequence[str]) -> list[list[float]]:
    """Embed contextual texts locally in batches on the configured CUDA GPU."""
    validated_texts = _validated_texts(texts)
    settings = _load_settings()
    model = _load_model(settings.model_name, settings.model_cache_dir)

    try:
        result = model.encode(
            validated_texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            max_length=EMBEDDING_MAX_LENGTH,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
    except Exception as error:
        raise LocalEmbeddingError("Local BGE-M3 embedding failed") from error

    return _parse_dense_vectors(
        result,
        expected_count=len(validated_texts),
        expected_dimension=settings.vector_dimension,
    )


def text_to_vector(text: str) -> list[float]:
    """Embed one contextual text while preserving the existing callable API."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    return texts_to_vectors([text])[0]


__all__ = [
    "CUDA_DEVICE",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_MAX_LENGTH",
    "LocalEmbeddingError",
    "text_to_vector",
    "texts_to_vectors",
]
