from __future__ import annotations

import pytest

from vector_db import bgem3_token_counter


@pytest.fixture(autouse=True)
def _clear_tokenizer_cache():
    bgem3_token_counter._load_tokenizer.cache_clear()
    yield
    bgem3_token_counter._load_tokenizer.cache_clear()


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, text, *, add_special_tokens, truncation, verbose):
        self.calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
                "truncation": truncation,
                "verbose": verbose,
            }
        )
        return [0, 10, 20, 2]


def test_counts_special_tokens_without_truncation_and_reuses_tokenizer(monkeypatch):
    tokenizer = _FakeTokenizer()
    load_calls = []

    def fake_from_pretrained(model_name):
        load_calls.append(model_name)
        return tokenizer

    monkeypatch.setattr(
        bgem3_token_counter.AutoTokenizer,
        "from_pretrained",
        fake_from_pretrained,
    )

    assert bgem3_token_counter.count_bge_m3_tokens("첫 번째 문장") == 4
    assert bgem3_token_counter.count_bge_m3_tokens("두 번째 문장") == 4
    assert load_calls == ["BAAI/bge-m3"]
    assert tokenizer.calls == [
        {
            "text": "첫 번째 문장",
            "add_special_tokens": True,
            "truncation": False,
            "verbose": False,
        },
        {
            "text": "두 번째 문장",
            "add_special_tokens": True,
            "truncation": False,
            "verbose": False,
        },
    ]


def test_uses_the_requested_huggingface_model_id(monkeypatch):
    tokenizer = _FakeTokenizer()
    load_calls = []

    monkeypatch.setattr(
        bgem3_token_counter.AutoTokenizer,
        "from_pretrained",
        lambda model_name: load_calls.append(model_name) or tokenizer,
    )

    count = bgem3_token_counter.count_bge_m3_tokens(
        "contextual text",
        model_name=" local/bge-m3 ",
    )

    assert count == 4
    assert load_calls == ["local/bge-m3"]


@pytest.mark.parametrize("text", [None, "", "   ", 123])
def test_rejects_invalid_text(text):
    with pytest.raises(ValueError, match="text must be a non-empty string"):
        bgem3_token_counter.count_bge_m3_tokens(text)


@pytest.mark.parametrize("model_name", [None, "", "   ", 123])
def test_rejects_invalid_model_name(model_name):
    with pytest.raises(ValueError, match="model_name must be a non-empty string"):
        bgem3_token_counter.count_bge_m3_tokens(
            "contextual text",
            model_name=model_name,
        )


def test_rejects_invalid_token_ids(monkeypatch):
    class InvalidTokenizer:
        def encode(self, text, *, add_special_tokens, truncation, verbose):
            return [0, "invalid", 2]

    monkeypatch.setattr(
        bgem3_token_counter.AutoTokenizer,
        "from_pretrained",
        lambda model_name: InvalidTokenizer(),
    )

    with pytest.raises(RuntimeError, match="invalid token IDs"):
        bgem3_token_counter.count_bge_m3_tokens("contextual text")
