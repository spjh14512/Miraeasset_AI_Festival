"""Count BGE-M3 tokens locally with the HuggingFace tokenizer."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from transformers import AutoTokenizer


DEFAULT_BGE_M3_TOKENIZER_MODEL = "BAAI/bge-m3"


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str) -> Any:
    """Load and reuse one tokenizer instance per HuggingFace model ID."""
    return AutoTokenizer.from_pretrained(model_name)


def count_bge_m3_tokens(
    text: str,
    *,
    model_name: str = DEFAULT_BGE_M3_TOKENIZER_MODEL,
) -> int:
    """Return the untruncated BGE-M3 input length including special tokens."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")

    tokenizer = _load_tokenizer(model_name.strip())
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=False,
        verbose=False,
    )
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in token_ids
    ):
        raise RuntimeError("BGE-M3 tokenizer returned invalid token IDs")
    return len(token_ids)


__all__ = [
    "DEFAULT_BGE_M3_TOKENIZER_MODEL",
    "count_bge_m3_tokens",
]
